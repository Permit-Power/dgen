"""
Measure how the modelled batteries actually dispatch.

Why this exists
---------------
The model simulates full hourly battery dispatch for every agent, then throws it
away: `financial_functions` drops every hourly array before `agent_outputs` is
written (see the `drop_cols` tuple there and the `drop_list` in `dgen_model`), so
the only battery fields that survive a run are `batt_kw` and `batt_kwh` --
nameplate, not behaviour. `batt_dispatch_helpers.dispatch_export_diags` was
written to report exactly these statistics but is never called.

Rather than reimplement the dispatch setup (and risk diverging from it), this
re-runs the model's own `calc_system_size_and_performance` on a sample of agents
and snapshots the PySAM Battery outputs from inside it, by wrapping
`calc_system_performance`. Whatever the model does, this measures.

The agent's merged cost/financing fields are taken from a completed run's
`agent_outputs`, and the fields that run dropped (tariff_dict, wholesale_prices)
are joined back from the input agent pickle. So the sampled agents carry the same
prices, tariffs and financing the real run gave them.

Usage
-----
Start the Cloud SQL proxy, then:

    python battery_dispatch_stats.py --schema diffusion_results_..._baseline \
        --states NJ IL VA NY MD PA MA --per-state 40 --out /tmp/dispatch

    python battery_dispatch_stats.py --list-schemas
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import psycopg2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

MIN_SOC_PCT = 10.0        # financial_functions sets batt.batt_minimum_SOC = 10
DISCHARGE_EPS_KW = 1e-3   # below this an hour does not count as "discharging"
MIDDAY = (11, 15)         # inclusive hour-of-day range, matches dispatch_export_diags


# ----------------------------------------------------------------------------
# Snapshotting the battery from inside the model's own sizing routine
# ----------------------------------------------------------------------------

_SNAP: dict = {}


def _install_probe():
    """
    Wrap financial_functions.calc_system_performance so that every PV+battery
    evaluation leaves a snapshot behind.

    calc_system_size_and_performance evaluates PV-only many times (the optimizer),
    then makes exactly ONE PV+battery call at kW* as its final step. So the last
    en_batt=True snapshot is the battery run at the chosen PV size -- the one that
    produced the agent's reported batt_kw/batt_kwh.
    """
    import financial_functions as ff

    if getattr(ff.calc_system_performance, "_dispatch_probe", False):
        return ff

    original = ff.calc_system_performance

    def probed(kw, pv, utilityrate, loan, batt, costs, agent, rate_switch_table,
               en_batt=True, batt_dispatch='price_signal_forecast'):
        out = original(kw, pv, utilityrate, loan, batt, costs, agent,
                       rate_switch_table, en_batt, batt_dispatch)
        if en_batt:
            _SNAP.clear()
            _SNAP.update(_harvest(kw, pv, batt, utilityrate))
        return out

    probed._dispatch_probe = True
    ff.calc_system_performance = probed
    return ff


def _series(batt, name, n):
    """Pull an hourly output off the Battery module, zero-filled if absent."""
    try:
        a = np.asarray(getattr(batt.Outputs, name), dtype=float).ravel()
    except Exception:
        return np.zeros(n)
    if a.size == 0:
        return np.zeros(n)
    return a[:n] if a.size >= n else np.pad(a, (0, n - a.size))


def _harvest(kw, pv, batt, utilityrate) -> dict:
    """Snapshot the hourly dispatch of one PV+battery evaluation."""
    # Rebuild the AC generation series exactly as calc_system_performance does
    # (inv_eff = 0.96 applied to the per-kW DC profile scaled by kw).
    gen = np.asarray(pv['generation_hourly'], dtype=float) * float(kw) * 0.96
    load = np.asarray(pv['consumption_hourly'], dtype=float)
    n = int(min(gen.size, load.size))
    gen, load = gen[:n], load[:n]

    return {
        "kw": float(kw),
        "n_hours": n,
        "gen": gen,
        "load": load,
        "batt_to_load": _series(batt, "batt_to_load", n),
        "batt_to_grid": _series(batt, "batt_to_grid", n),
        "system_to_batt": _series(batt, "system_to_batt", n),
        "system_to_grid": _series(batt, "system_to_grid", n),
        "grid_to_batt": _series(batt, "grid_to_batt", n),
        "batt_SOC": _series(batt, "batt_SOC", n),
        "batt_kwh": float(getattr(batt.Outputs, "batt_bank_installed_capacity", 0.0) or 0.0),
        "bill_w_batt_yr1": float(utilityrate.Outputs.export().get("utility_bill_w_sys_year1", 0.0)),
        "bill_wo_sys_yr1": float(utilityrate.Outputs.export().get("utility_bill_wo_sys_year1", 0.0)),
    }


# ----------------------------------------------------------------------------
# Turning one snapshot into statistics
# ----------------------------------------------------------------------------

def _yr1(agent, field, default=np.nan):
    """First analysis-year value out of one of the agent's stored bill arrays."""
    v = agent.get(field)
    try:
        a = np.asarray(v, dtype=float).ravel()
    except Exception:
        return default
    if a.size == 0:
        return default
    # SAM bill arrays lead with a year-0 entry that is 0 for utility bills.
    return float(a[1]) if a.size > 1 and a[0] == 0 else float(a[0])


def summarize(snap: dict, agent: pd.Series) -> dict:
    n = snap["n_hours"]
    b2l, b2g = snap["batt_to_load"], snap["batt_to_grid"]
    s2b, s2g = snap["system_to_batt"], snap["system_to_grid"]
    g2b, soc = snap["grid_to_batt"], snap["batt_SOC"]
    gen, load = snap["gen"], snap["load"]

    discharge = b2l + b2g
    active = discharge > DISCHARGE_EPS_KW
    hod = np.arange(n) % 24
    midday = (hod >= MIDDAY[0]) & (hod <= MIDDAY[1])

    kwh = snap["batt_kwh"]
    usable = kwh * (1.0 - MIN_SOC_PCT / 100.0)
    annual_discharge = float(discharge.sum())

    # Contiguous runs of discharging hours -> how long a discharge lasts.
    runs, cur = [], 0
    for a in active:
        if a:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)

    surplus = np.maximum(gen - load, 0.0)
    surplus_mid = float(surplus[midday].sum())

    # Days on which the battery discharged at all.
    days = n // 24
    day_active = active[: days * 24].reshape(days, 24).any(axis=1) if days else np.array([])

    return {
        "agent_id": agent.get("agent_id"),
        "state_abbr": agent.get("state_abbr"),
        "system_kw": float(agent.get("system_kw", np.nan)),
        "batt_kw": float(agent.get("batt_kw", np.nan)),
        "batt_kwh": kwh,

        # throughput
        "annual_discharge_kwh": annual_discharge,
        "annual_charge_kwh": float(s2b.sum() + g2b.sum()),
        "equiv_full_cycles_nameplate": annual_discharge / kwh if kwh > 0 else np.nan,
        "equiv_full_cycles_usable": annual_discharge / usable if usable > 0 else np.nan,

        # how often / how long
        "hours_discharging": int(active.sum()),
        "days_discharging": int(day_active.sum()),
        "pct_days_discharging": float(day_active.mean() * 100) if days else np.nan,
        "mean_discharge_run_hours": float(np.mean(runs)) if runs else 0.0,
        "median_discharge_run_hours": float(np.median(runs)) if runs else 0.0,
        "max_discharge_run_hours": int(max(runs)) if runs else 0,
        "mean_discharge_kw_when_active": float(discharge[active].mean()) if active.any() else 0.0,
        "peak_discharge_kw": float(discharge.max()) if n else 0.0,

        # where the energy goes
        "batt_to_load_kwh": float(b2l.sum()),
        "batt_to_grid_kwh": float(b2g.sum()),
        "pct_discharge_to_load": float(b2l.sum() / annual_discharge * 100) if annual_discharge > 0 else np.nan,
        "grid_charge_kwh": float(g2b.sum()),

        # PV surplus handling
        "pv_surplus_kwh": float(surplus.sum()),
        "pv_to_batt_kwh": float(s2b.sum()),
        "pv_to_grid_kwh": float(s2g.sum()),
        "midday_capture_frac": float(s2b[midday].sum() / surplus_mid) if surplus_mid > 1e-9 else np.nan,

        # state of charge
        "mean_soc_pct": float(soc.mean()) if n else np.nan,
        "hours_soc_above_95": int((soc >= 95.0).sum()),
        "hours_soc_at_floor": int((soc <= MIN_SOC_PCT + 1.0).sum()),

        # value -- the battery-attributable share of bill savings is the gap
        # between the PV-only bill and the PV+battery bill at the SAME PV size.
        # Both arrays are set by calc_system_size_and_performance itself.
        "bill_wo_sys_yr1": _yr1(agent, "utility_bill_wo_sys_pv_only", snap["bill_wo_sys_yr1"]),
        "bill_pv_only_yr1": _yr1(agent, "utility_bill_w_sys_pv_only", np.nan),
        "bill_pv_batt_yr1": _yr1(agent, "utility_bill_w_sys_pv_batt", snap["bill_w_batt_yr1"]),
        "batt_bill_savings_yr1": (_yr1(agent, "utility_bill_w_sys_pv_only", np.nan)
                                  - _yr1(agent, "utility_bill_w_sys_pv_batt", np.nan)),
        "hour_of_day_discharge_kwh": [float(discharge[hod == h].sum()) for h in range(24)],
    }


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def _like(name: str, pattern: str) -> bool:
    """Minimal SQL-LIKE: % matches any run of characters."""
    import re as _re
    rx = "^" + "".join(".*" if ch == "%" else _re.escape(ch) for ch in pattern) + "$"
    return _re.match(rx, name) is not None


def connect(args):
    import psycopg2
    return psycopg2.connect(host=args.host, port=args.port, dbname=args.db,
                            user=args.user, password=args.password)


def _write_outputs(rows, out_dir):
    """Write the per-agent table and the hour-of-day profile. Safe to call repeatedly."""
    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    if df.empty:
        return df
    hod = pd.DataFrame(df.pop("hour_of_day_discharge_kwh").tolist(),
                       columns=[f"h{h:02d}" for h in range(24)])
    hod.insert(0, "state_abbr", df["state_abbr"].values)
    df.to_csv(os.path.join(out_dir, "dispatch_by_agent.csv"), index=False)
    hod.groupby("state_abbr").mean().to_csv(os.path.join(out_dir, "dispatch_hour_of_day.csv"))
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--schema", help="a single schema, for --describe")
    ap.add_argument("--schema-template", default="diffusion_results_baseline_{st}_2040_a5_%",
                    help="LIKE pattern per state; {st} is the lowercased abbreviation")
    ap.add_argument("--list-schemas", action="store_true")
    ap.add_argument("--describe", action="store_true", help="print agent_outputs columns and exit")
    ap.add_argument("--states", nargs="+", default=["NJ", "IL", "VA", "NY", "MD", "PA", "MA"])
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--per-state", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agents", default="../input_agents/agent_df_base_res_national_updated_tariffs_2026.pkl")
    ap.add_argument("--out", default="/tmp/dispatch")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="5432")
    ap.add_argument("--db", default="dgendb")
    ap.add_argument("--user", default="postgres")
    ap.add_argument("--password", default="postgres")
    args = ap.parse_args()

    con = connect(args)

    if args.list_schemas:
        q = ("select schema_name from information_schema.schemata "
             "where schema_name like 'diffusion_results%' order by schema_name")
        print(pd.read_sql(q, con).to_string())
        return 0

    if args.describe:
        if not args.schema:
            ap.error("--describe needs --schema")
        cols = pd.read_sql(
            f"select column_name, data_type from information_schema.columns "
            f"where table_schema='{args.schema}' and table_name='agent_outputs' "
            f"order by column_name", con)
        print(cols.to_string())
        return 0

    # Each state is its own run schema, so resolve one per state. --schema-template
    # is a LIKE pattern with {st} standing in for the lowercased abbreviation; when
    # several match, the newest (lexically last, since names end in a timestamp) wins.
    all_schemas = pd.read_sql(
        "select schema_name from information_schema.schemata "
        "where schema_name like 'diffusion_results%' order by schema_name", con
    )["schema_name"].tolist()

    frames = []
    for st in args.states:
        pat = args.schema_template.format(st=st.lower())
        hits = [s_ for s_ in all_schemas if _like(s_, pat)]
        if not hits:
            print(f"  !! {st.upper()}: no schema matching {pat} -- skipped")
            continue
        sch = hits[-1]
        df_st = pd.read_sql(
            f'select * from "{sch}".agent_outputs where year = {args.year}', con)
        print(f"  {st.upper()}: {len(df_st):>6} rows from {sch}")
        frames.append(df_st)

    if not frames:
        raise SystemExit("no schemas resolved for any requested state")
    outs = pd.concat(frames, ignore_index=True)
    print(f"agent_outputs: {len(outs)} rows, {outs.state_abbr.nunique()} states")

    src = pd.read_pickle(args.agents)
    # agent_outputs already carries tilt/azimuth/solar_re_9809_gid and every cost and
    # financing field. Only the two the run drops before writing need joining back.
    join_cols = [c for c in ("tariff_dict", "wholesale_prices") if c in src.columns]
    key = "bldg_id" if "bldg_id" in outs.columns and "bldg_id" in src.columns else None
    if key is None:
        raise SystemExit("cannot join agent pickle to agent_outputs: no bldg_id on both")
    src_small = src[[key] + [c for c in join_cols if c != key]].drop_duplicates(key)
    merged = outs.merge(src_small, on=key, how="inner", suffixes=("", "_src"))
    print(f"joined to agent pickle: {len(merged)} rows")

    rng = np.random.default_rng(args.seed)
    picks = []
    for st, g in merged.groupby("state_abbr"):
        take = min(args.per_state, len(g))
        picks.append(g.iloc[rng.choice(len(g), size=take, replace=False)])
    sample = pd.concat(picks, ignore_index=True)
    print(f"sampling {len(sample)} agents ({args.per_state}/state requested)")

    ff = _install_probe()
    import agent_mutation.elec as elec
    rate_switch_table = elec.get_rate_switch_table(con)

    rows, failures = [], 0
    for i, (_, agent) in enumerate(sample.iterrows(), 1):
        # Slow dispatch modes (retail-rate) can outlive the Cloud SQL connection's
        # idle timeout; a dropped connection is retried once on a fresh one.
        for attempt in (1, 2):
            try:
                _SNAP.clear()
                done = ff.calc_system_size_and_performance(con, agent, None, rate_switch_table)
                if not _SNAP:
                    failures += 1
                else:
                    rows.append(summarize(_SNAP, done))
                break
            except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
                if attempt == 1:
                    print(f"  agent {i}: connection dropped ({type(e).__name__}); reconnecting")
                    try:
                        con.close()
                    except Exception:
                        pass
                    try:
                        con = connect(args)
                    except Exception as e2:
                        failures += 1
                        print(f"  agent {i}: reconnect failed ({type(e2).__name__}); skipping")
                        break
                    continue
                failures += 1
                print(f"  agent {i} failed after reconnect: {type(e).__name__}: {e}")
            except Exception as e:
                failures += 1
                if failures <= 5:
                    print(f"  agent {i} failed: {type(e).__name__}: {e}")
                break
        if i % 25 == 0:
            print(f"  {i}/{len(sample)} ({failures} failed)")
            _write_outputs(rows, args.out)   # checkpoint so a crash keeps what's done

    if not rows:
        raise SystemExit("no agents produced dispatch output")

    df = _write_outputs(rows, args.out)
    print(f"\n{len(df)} agents succeeded, {failures} failed")
    print(f"wrote {args.out}/dispatch_by_agent.csv and dispatch_hour_of_day.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
