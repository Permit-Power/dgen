"""
Compare a net-billing national run against its net-metering twin.

Context
-------
Every dGen result before this one credited exports at FULL RETAIL: `process_tariff`
takes `ur_metering_option` from each agent's own tariff, essentially all of which
carry 0 (net metering), so the wholesale export series the model builds and passes
to SAM as `ur_ts_sell_rate` was never consulted. Confirmed by scaling
`wholesale_prices` 0x/1x/10x and watching bill and NPV move by exactly $0.00.
`FORCE_NET_BILLING=1` sets ur_metering_option = 2, activating those wholesale prices.

Schemas are tagged `_a<pct>` (net metering) and `_a<pct>_nb` (net billing), so the two
arms are matched by state and scenario.

IMPORTANT: a schema exists from the moment its job STARTS, so an in-progress run looks
like a finished one with small numbers. Every schema is checked for all 15 model years
before it is used; partial ones are excluded and reported separately. Skipping that
check produced obviously wrong results (states showing zero policy capacity and
negative deltas).

Usage
-----
    python compare_net_billing.py                 # summary
    python compare_net_billing.py --by-state      # add the per-state table
    python compare_net_billing.py --csv out.csv
"""
from __future__ import annotations

import argparse
import re

import pandas as pd

N_YEARS = 15


def connect(a):
    import psycopg2
    return psycopg2.connect(host=a.host, port=a.port, dbname=a.db,
                            user=a.user, password=a.password)


def index_schemas(all_s, tag):
    """(state, arm) -> newest schema carrying `tag`."""
    out = {}
    for s in all_s:
        m = re.match(r'diffusion_results_(baseline|policy)_([a-z]{2})_2040'
                     + tag + r'_\d{8}_\d+$', s)
        if m:
            out.setdefault((m.group(2), m.group(1)), []).append(s)
    return {k: sorted(v)[-1] for k, v in out.items()}


def read(con, schema):
    """2040 cumulative kW and adopters, or None if the run is not finished."""
    ny = pd.read_sql(f'select count(distinct year) n from "{schema}".agent_outputs', con).iloc[0].n
    if int(ny or 0) != N_YEARS:
        return None
    r = pd.read_sql(f'select sum(system_kw_cum) k, sum(number_of_adopters) a '
                    f'from "{schema}".agent_outputs where year = 2040', con).iloc[0]
    return float(r.k or 0), float(r.a or 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attach-pct', default='5')
    ap.add_argument('--by-state', action='store_true')
    ap.add_argument('--csv')
    ap.add_argument('--host', default='127.0.0.1'); ap.add_argument('--port', default='5432')
    ap.add_argument('--db', default='dgendb'); ap.add_argument('--user', default='postgres')
    ap.add_argument('--password', default='postgres')
    a = ap.parse_args()

    con = connect(a)
    all_s = pd.read_sql("select schema_name from information_schema.schemata "
                        "where schema_name like 'diffusion_results%' order by schema_name",
                        con)['schema_name'].tolist()
    nb = index_schemas(all_s, rf'_a{a.attach_pct}_nb')
    nem = index_schemas(all_s, rf'_a{a.attach_pct}')

    rows, partial = [], []
    for st in sorted({s for (s, _) in nb}):
        keys = [(st, 'baseline'), (st, 'policy')]
        if not all(k in nb and k in nem for k in keys):
            partial.append((st, 'missing schema')); continue
        vals = [read(con, nb[k]) for k in keys] + [read(con, nem[k]) for k in keys]
        if any(v is None for v in vals):
            partial.append((st, 'run incomplete')); continue
        (nbb, nbba), (nbp, nbpa), (nmb, nmba), (nmp, nmpa) = vals
        rows.append({
            'state': st.upper(),
            'nem_baseline_GW': nmb / 1e6, 'nb_baseline_GW': nbb / 1e6,
            'nem_policy_GW': nmp / 1e6, 'nb_policy_GW': nbp / 1e6,
            'nem_incremental_GW': (nmp - nmb) / 1e6, 'nb_incremental_GW': (nbp - nbb) / 1e6,
            'nem_incremental_adopters_M': (nmpa - nmba) / 1e6,
            'nb_incremental_adopters_M': (nbpa - nbba) / 1e6,
        })

    d = pd.DataFrame(rows)
    if not len(d):
        print('no states complete in both runs yet'); return 0
    d['incremental_ratio'] = d.nb_incremental_GW / d.nem_incremental_GW

    pd.set_option('display.width', 220)
    if a.by_state:
        print(d.round(3).sort_values('nem_incremental_GW', ascending=False).to_string(index=False))
        print()
    print(f'states complete in BOTH runs: {len(d)}')
    if partial:
        print(f'excluded (still running): {len(partial)} -> '
              f'{", ".join(s.upper() for s, _ in partial)}')
    print()
    for label, nm, nbv in (('BASELINE   ', 'nem_baseline_GW', 'nb_baseline_GW'),
                           ('POLICY     ', 'nem_policy_GW', 'nb_policy_GW'),
                           ('INCREMENTAL', 'nem_incremental_GW', 'nb_incremental_GW')):
        x, y = d[nm].sum(), d[nbv].sum()
        print(f'  {label}  net metering {x:8.2f} GW   net billing {y:8.2f} GW   ratio {y/x:6.3f}')
    x, y = d.nem_incremental_adopters_M.sum(), d.nb_incremental_adopters_M.sum()
    print(f'  ADOPTERS     net metering {x:8.2f} M    net billing {y:8.2f} M    ratio {y/x:6.3f}')

    if a.csv:
        d.round(4).to_csv(a.csv, index=False)
        print(f'\nwrote {a.csv}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
