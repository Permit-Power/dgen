"""
Check every year-indexed input covers a target model horizon, before running.

Why
---
The model horizon is set at submit time. Extending it past what the input data
covers does not fail loudly: a price or load-growth table that stops at 2040
either drops the extra years on a merge, leaving agents with null costs, or
silently carries the last value forward. Either way the run completes and the
numbers look plausible.

That is the same failure mode as everything else in this repo's history: an
assumption quietly not what anyone believed, with nothing in the output to say
so. A national run is several hours, so finding out afterwards is expensive.

Run this before changing the horizon:

    python check_end_year.py --end-year 2041

It reports every table in diffusion_shared carrying a `year` column, whether it
reaches the target, and exits non-zero if any does not.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utility_functions as utilfunc

#: Tables whose coverage genuinely gates a run. Others are reported but do not
#: fail the check, since plenty of year-indexed tables are unused by a given
#: scenario (older price vintages, alternative trajectories).
CRITICAL = (
    'pv_price_', 'batt_prices_', 'pv_plus_batt_',
    'load_growth', 'elec_prices', 'input_elec_prices',
    'itc_options', 'main_itc', 'nem_', 'carbon', 'wholesale',
)


def _is_critical(name: str) -> bool:
    return any(k in name for k in CRITICAL)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--end-year', type=int, required=True)
    ap.add_argument('--start-year', type=int, default=2026)
    ap.add_argument('--schema', default='diffusion_shared')
    ap.add_argument('--all', action='store_true',
                    help='fail on any short table, not just the critical ones')
    ap.add_argument('--conn', default=os.environ.get(
        'PG_CONN_STRING',
        'host=127.0.0.1 port=5432 dbname=dgendb user=postgres password=postgres'))
    ap.add_argument('--role', default=os.environ.get('PG_ROLE', 'postgres'))
    args = ap.parse_args()

    con, _ = utilfunc.make_con(args.conn, args.role)
    tables = pd.read_sql(
        "select table_name from information_schema.columns "
        f"where table_schema = '{args.schema}' and column_name = 'year' "
        "order by table_name", con)['table_name'].tolist()

    print(f'checking {len(tables)} year-indexed tables in {args.schema} '
          f'for coverage of {args.start_year}-{args.end_year}\n')

    short_critical, short_other, unreadable = [], [], []
    for name in tables:
        try:
            r = pd.read_sql(f'select min(year) a, max(year) b '
                            f'from {args.schema}."{name}"', con).iloc[0]
            lo = int(r.a) if pd.notna(r.a) else None
            hi = int(r.b) if pd.notna(r.b) else None
        except Exception as e:
            unreadable.append((name, repr(e)[:70]))
            continue
        if hi is None:
            unreadable.append((name, 'empty'))
            continue
        if hi < args.end_year:
            (short_critical if _is_critical(name) else short_other).append((name, lo, hi))

    if short_critical:
        print(f'{len(short_critical)} CRITICAL table(s) do not reach {args.end_year}:')
        for n, lo, hi in short_critical:
            print(f'   {n:52s} {lo}-{hi}')
        print()
    if short_other:
        print(f'{len(short_other)} other table(s) stop short (often unused vintages):')
        for n, lo, hi in short_other:
            print(f'   {n:52s} {lo}-{hi}')
        print()
    if unreadable:
        print(f'{len(unreadable)} table(s) could not be read:')
        for n, why in unreadable:
            print(f'   {n:52s} {why}')
        print()

    fail = short_critical or (short_other if args.all else [])
    if not fail:
        print(f'OK: every table that gates a run covers {args.end_year}.')
        print('Note this checks COVERAGE only. It cannot tell you whether the '
              'values in the extra years are intentional or an artefact of '
              'whatever extrapolation produced them.')
        return 0
    print(f'FAIL: {len(fail)} table(s) would not cover a {args.end_year} run. '
          f'Extend them before submitting, or the run will complete with nulls '
          f'or a silently carried-forward last value.')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
