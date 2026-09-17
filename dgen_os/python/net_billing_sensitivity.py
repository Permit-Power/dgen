"""
How much does the net-metering vs net-billing assumption matter?

Background
----------
`process_tariff` sets `ER.ur_metering_option = 2 if FORCE_NET_BILLING else <tariff value>`,
and essentially every agent tariff carries 0 (net metering). `FORCE_NET_BILLING` is False,
so every dGen result to date was produced under full retail net metering, and the wholesale
export series the model builds and passes to SAM as `ur_ts_sell_rate` is never consulted.
(Verified: scaling `wholesale_prices` 0x/1x/10x moves bill and NPV by exactly $0.00.)

This runs the SAME agents through the model's own sizing routine twice -- once as
configured today, once with FORCE_NET_BILLING = True -- so the difference isolates the
assumption. Payback is what drives adoption (it sets `max_market_share` via the NREL
lookup curve), so the payback shift and the resulting ceiling shift are the headline.

Usage
-----
    python net_billing_sensitivity.py --per-state 30 --out /tmp/nb
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_STATES = ["NY", "PA", "IL", "NJ", "MA", "MD", "VA", "CA", "AZ", "TX"]


def connect(a):
    import psycopg2
    return psycopg2.connect(host=a.host, port=a.port, dbname=a.db,
                            user=a.user, password=a.password)


def _like(name, pattern):
    import re
    rx = "^" + "".join(".*" if c == "%" else re.escape(c) for c in pattern) + "$"
    return re.match(rx, name) is not None


def load_mms_curve(con):
    """payback (rounded to 0.1yr) -> max_market_share, residential host-owned."""
    q = ("select metric_value as payback_period, max_market_share "
         "from diffusion_template.max_market_curves_to_model "
         "where metric='payback_period' and business_model='host_owned' and sector_abbr='res'")
    c = pd.read_sql(q, con)
    c["k"] = (c.payback_period.astype(float) * 100).round().astype(int)
    return dict(zip(c.k, c.max_market_share.astype(float)))


def mms_of(payback, curve):
    if payback is None or not np.isfinite(payback):
        return np.nan
    lo, hi = min(curve), max(curve)
    k = int(round(float(payback) * 100))
    return curve.get(min(max(k, lo), hi), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", default=DEFAULT_STATES)
    ap.add_argument("--schema-template", default="diffusion_results_baseline_{st}_2040_a5_%")
    ap.add_argument("--year", type=int, default=2026)
    ap.add_argument("--per-state", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agents", default="../input_agents/agent_df_base_res_national_updated_tariffs_2026.pkl")
    ap.add_argument("--out", default="/tmp/nb")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", default="5432")
    ap.add_argument("--db", default="dgendb"); ap.add_argument("--user", default="postgres")
    ap.add_argument("--password", default="postgres")
    a = ap.parse_args()

    con = connect(a)
    import financial_functions as ff
    import agent_mutation.elec as elec
    rst = elec.get_rate_switch_table(con)
    curve = load_mms_curve(con)
    print(f"max_market_share curve: {len(curve)} points, "
          f"payback {min(curve)/100:.1f}-{max(curve)/100:.1f} yr")

    all_s = pd.read_sql("select schema_name from information_schema.schemata "
                        "where schema_name like 'diffusion_results%' order by schema_name",
                        con)["schema_name"].tolist()
    src = pd.read_pickle(a.agents)
    join = [c for c in ("tariff_dict", "wholesale_prices") if c in src.columns]
    src_small = src[["bldg_id"] + join].drop_duplicates("bldg_id")

    rng = np.random.default_rng(a.seed)
    frames = []
    for st in a.states:
        hits = [s for s in all_s if _like(s, a.schema_template.format(st=st.lower()))]
        if not hits:
            print(f"  !! {st}: no schema -- skipped"); continue
        d = pd.read_sql(f'select * from "{hits[-1]}".agent_outputs where year={a.year}', con)
        d = d.merge(src_small, on="bldg_id", how="inner")
        if not len(d):
            print(f"  !! {st}: no join -- skipped"); continue
        take = min(a.per_state, len(d))
        frames.append(d.iloc[rng.choice(len(d), size=take, replace=False)])
        print(f"  {st}: {take} agents from {hits[-1]}")
    sample = pd.concat(frames, ignore_index=True)
    print(f"\nrunning {len(sample)} agents x 2 configurations\n")

    rows, fail = [], 0
    for i, (_, agent) in enumerate(sample.iterrows(), 1):
        try:
            res = {}
            for label, force in (("nem", False), ("nb", True)):
                ff.FORCE_NET_BILLING = force        # module global, read at call time
                out = ff.calc_system_size_and_performance(con, agent.copy(), None, rst)
                res[label] = out
            n, b = res["nem"], res["nb"]
            rows.append({
                "state_abbr": agent["state_abbr"], "agent_id": agent.get("agent_id"),
                "pv_kw_nem": float(n["system_kw"]), "pv_kw_nb": float(b["system_kw"]),
                "payback_nem": float(n["payback_period"]), "payback_nb": float(b["payback_period"]),
                "npv_nem": float(n["npv"]), "npv_nb": float(b["npv"]),
                "mms_nem": mms_of(n["payback_period"], curve),
                "mms_nb": mms_of(b["payback_period"], curve),
            })
        except Exception as e:
            fail += 1
            if fail <= 5:
                print(f"  agent {i} failed: {type(e).__name__}: {e}")
        if i % 25 == 0:
            print(f"  {i}/{len(sample)} ({fail} failed)")
    ff.FORCE_NET_BILLING = False                     # leave the module as we found it

    df = pd.DataFrame(rows)
    df["payback_delta"] = df.payback_nb - df.payback_nem
    df["mms_ratio"] = df.mms_nb / df.mms_nem.replace(0, np.nan)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "net_billing_sensitivity.csv"), index=False)

    g = df.groupby("state_abbr").agg(
        n=("agent_id", "count"),
        payback_nem=("payback_nem", "median"), payback_nb=("payback_nb", "median"),
        payback_delta=("payback_delta", "median"),
        mms_nem=("mms_nem", "mean"), mms_nb=("mms_nb", "mean"),
        mms_ratio=("mms_ratio", "mean"),
        pv_kw_nem=("pv_kw_nem", "mean"), pv_kw_nb=("pv_kw_nb", "mean"),
    ).round(3)
    pd.set_option("display.width", 200)
    print("\n=== median payback / mean adoption ceiling, net metering vs net billing ===")
    print(g.to_string())
    print(f"\nNATIONAL-ish: median payback {df.payback_nem.median():.1f} -> "
          f"{df.payback_nb.median():.1f} yr ({df.payback_delta.median():+.1f})")
    print(f"mean ceiling {df.mms_nem.mean():.4f} -> {df.mms_nb.mean():.4f} "
          f"(x{df.mms_nb.mean()/df.mms_nem.mean():.3f})")
    print(f"\n{len(df)} agents succeeded, {fail} failed -> {a.out}/net_billing_sensitivity.csv")


if __name__ == "__main__":
    raise SystemExit(main())
