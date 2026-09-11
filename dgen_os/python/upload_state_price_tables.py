"""
Upload a one-off study's PV cost curves to their OWN Cloud SQL tables.

Why this exists
---------------
The model reads cost trajectories from fixed table names, and the historical way
to run a state-specific price study was to overwrite the shared tables
(`pv_plus_batt_baseline`, `pv_plus_batt_dollar_per_watt`). That destroys the
state-keyed LBNL baseline and the $1/W policy tables other runs depend on.

Instead this writes dedicated tables, and the job points at them with the
`PV_PRICE_TABLE_*` / `PV_PLUS_BATT_TABLE_*` env vars (see `config.py`). Nothing
shared is touched, so studies can coexist.

Usage
-----
Start the Cloud SQL proxy, then:

    python upload_state_price_tables.py --states PA OH
    python upload_state_price_tables.py --states PA --dry-run

Expects, for each state, the repo's standard pair in input_data/pv_plus_batt_prices/:
    pv_plus_batt_prices_FY23_<ST>_baseline.csv
    pv_plus_batt_prices_FY23_<ST>_policy.csv
Build both from the SAME template (pv_plus_batt_prices_FY23_mid.csv) and edit only
system_capex_per_kw_res for 2026-2040. Copying the policy arm from
FY23_dollar_watt.csv instead leaves the two arms with different battery costs
(~2x at 2030), which shows up as a policy effect but is really just the template.

Creates, per state and arm:
    pv_price_<st>_<arm>
    pv_plus_batt_<st>_<arm>
both keyed by state_abbr so the price merge stays explicit (see
`agent_mutation.elec.apply_pv_prices`, which warns when it has to fall back to a
national merge).
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from sqlalchemy import create_engine, text

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from input_data_functions import stacked_sectors  # noqa: E402

PRICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "..", "input_data", "pv_plus_batt_prices")
SCHEMA = "diffusion_shared"

# Columns of the existing shared tables, so the new ones are drop-in compatible.
PV_PRICE_COLS = ["state_abbr", "year", "system_capex_per_kw",
                 "system_om_per_kw", "system_variable_om_per_kw", "sector_abbr"]
PV_BATT_COLS = ["state_abbr", "year", "tech", "batt_replace_frac_kw", "batt_replace_frac_kwh",
                "system_capex_per_kw", "batt_capex_per_kwh", "batt_capex_per_kw",
                "linear_constant", "batt_om_per_kw", "batt_om_per_kwh", "sector_abbr"]


def build(state: str, study: str, arm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pv_price_df, pv_plus_batt_df) for one state/arm, ready to upload."""
    # Repo convention is pv_plus_batt_prices_FY23_<ST>_<arm>.csv (one pair per state).
    # A --study tag is only needed when one state has more than one study.
    names = ([f"pv_plus_batt_prices_{state.upper()}_{study}_{arm}.csv"] if study else []) + \
            [f"pv_plus_batt_prices_FY23_{state.upper()}_{arm}.csv"]
    for n in names:
        src = os.path.join(PRICE_DIR, n)
        if os.path.exists(src):
            break
    else:
        raise FileNotFoundError(f"none of {names} in {PRICE_DIR}")

    wide = pd.read_csv(src).dropna(axis=1, how="all")
    long = stacked_sectors(wide)
    long["state_abbr"] = state.upper()

    # The agent file is residential-only; keep the table to what the run will use.
    long = long[long["sector_abbr"] == "res"].copy()

    pv_batt = long.reindex(columns=[c for c in PV_BATT_COLS if c in long.columns]).copy()

    # PV-only table mirrors the same capex so the two can never disagree. (The
    # economics actually read system_capex_per_kw_combined from the pv+batt table;
    # costs['system_capex_per_kw'] is assembled but never used.)
    pv_price = long.reindex(columns=[c for c in PV_PRICE_COLS if c in long.columns]).copy()
    for missing, default in (("system_om_per_kw", 0.0), ("system_variable_om_per_kw", 0)):
        if missing not in pv_price.columns:
            pv_price[missing] = default
    pv_price = pv_price[PV_PRICE_COLS]

    return pv_price, pv_batt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", default="", help="optional tag when a state has >1 study; default uses the repo convention pv_plus_batt_prices_FY23_<ST>_<arm>.csv")
    ap.add_argument("--states", nargs="+", required=True, help="state abbreviations, e.g. PA OH")
    ap.add_argument("--arms", nargs="+", default=["baseline", "policy"])
    ap.add_argument("--dry-run", action="store_true", help="build and report, upload nothing")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default="5432")
    ap.add_argument("--db", default="dgendb")
    ap.add_argument("--user", default="postgres")
    ap.add_argument("--password", default="postgres")
    args = ap.parse_args()

    engine = None
    if not args.dry_run:
        engine = create_engine(
            f"postgresql+psycopg2://{args.user}:{args.password}@{args.host}:{args.port}/{args.db}")

    made = []
    for st in args.states:
        for arm in args.arms:
            pv_price, pv_batt = build(st, args.study, arm)
            tag = f"{st.lower()}_{args.study}_{arm}" if args.study else f"{st.lower()}_{arm}"
            targets = [(f"pv_price_{tag}", pv_price), (f"pv_plus_batt_{tag}", pv_batt)]

            for name, df in targets:
                capex = df["system_capex_per_kw"]
                print(f"{name:<44} rows={len(df):>3}  years={df.year.min()}-{df.year.max()}  "
                      f"2026=${capex[df.year == 2026].iloc[0]:,.0f}  "
                      f"2040=${capex[df.year == 2040].iloc[0]:,.0f}")
                if engine is not None:
                    df.to_sql(name, engine, schema=SCHEMA, if_exists="replace", index=False)
                made.append(name)

    if engine is not None:
        with engine.connect() as c:
            for name in made:
                n = c.execute(text(f'select count(*) from {SCHEMA}."{name}"')).scalar()
                assert n > 0, f"{name} uploaded empty"
        engine.dispose()
        print(f"\nuploaded {len(made)} tables to {SCHEMA}, all non-empty")
        print("\nPoint a job at them with, e.g.:")
        st = args.states[0].lower()
        sfx = f"{st}_{args.study}" if args.study else st
        print(f'  PV_PRICE_TABLE_BASELINE: "pv_price_{sfx}_baseline"')
        print(f'  PV_PRICE_TABLE_POLICY: "pv_price_{sfx}_policy"')
        print(f'  PV_PLUS_BATT_TABLE_BASELINE: "pv_plus_batt_{sfx}_baseline"')
        print(f'  PV_PLUS_BATT_TABLE_POLICY: "pv_plus_batt_{sfx}_policy"')
    else:
        print(f"\ndry run: {len(made)} tables would be created, nothing uploaded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
