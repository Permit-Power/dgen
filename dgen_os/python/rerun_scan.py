"""
Re-evaluate a finished run's agents and emit the columns the run itself dropped.

Why
---
dGen strips most per-agent detail before writing results. Two things that later
turned out to matter are gone from every completed run:

  * the yearly bill arrays (utility_bill_wo_sys_pv_only, utility_bill_w_sys_pv_only,
    cf_energy_value_pv_only), which downstream bill-savings work needs;
  * the self-consumption and export split, which grid-impact work needs and which
    cannot be recovered from the state hourly aggregates, since those hold only a
    single blended net-load series mixing adopters with non-adopters.

The split is now recorded by the model on every future run, but existing runs
predate that. This script closes the gap for results already produced.

How
---
It calls the model's own calc_system_size_and_performance for each agent-year,
so whatever the model does, this reproduces, including the system size it
chooses. Nothing is reimplemented and nothing can drift. It costs more than
evaluating at the stored size, but it reproduces the published run exactly rather
than approximately, which matters when the output is meant to describe results
that have already been reported.

Output layout mirrors the existing Drive exports so consumers only change a path:

    {out}/{STATE}/{run_name}/{scenario}.csv

Running
-------
Locally, against the Cloud SQL proxy:

    python rerun_scan.py --states PA --out /tmp/scan

On Cloud Batch, one task per state, reading its state from the task index:

    python rerun_scan.py --state-file /tmp/states.csv --task-index ${BATCH_TASK_INDEX} \
        --out /tmp/scan --gcs-bucket dgen-assets --gcs-prefix netbilling_scan

Set FORCE_NET_BILLING=1 to match a net-billing run. The script refuses to start
if that flag disagrees with the schema it was pointed at, since silently scanning
a net-billing run under net-metering rules would produce plausible, wrong numbers.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from functools import partial

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import utility_functions as utilfunc

#: Columns the consumers need. Everything else the model produces is dropped
#: here to keep the files a manageable size.
KEEP = [
    # identity and weights
    'agent_id', 'bldg_id', 'state_abbr', 'sector_abbr', 'year',
    'new_adopters', 'customers_in_bin', 'developable_agent_weight',
    'number_of_adopters', 'tariff_id', 'tariff_name',
    # system
    'system_kw', 'batt_kw', 'batt_kwh', 'annual_energy_production_kwh',
    # financing needed to interpret the arrays
    'real_discount_rate', 'inflation_rate', 'economic_lifetime_yrs',
    # the arrays the run dropped
    'utility_bill_wo_sys_pv_only', 'utility_bill_w_sys_pv_only',
    'cf_energy_value_pv_only', 'cf_discounted_savings_pv_only',
    'utility_bill_wo_sys_pv_batt', 'utility_bill_w_sys_pv_batt',
    'cf_energy_value_pv_batt', 'cf_discounted_savings_pv_batt',
    # the self-consumption split
    'annual_generation_kwh',
    'exported_kwh_pv_only', 'self_consumed_kwh_pv_only',
    'exported_kwh_pv_batt', 'self_consumed_kwh_pv_batt',
    'batt_roundtrip_loss_kwh', 'grid_charge_kwh',
]

#: Postgres array literal, matching the format the existing exports use.
def _pg_array(v) -> str:
    try:
        a = np.asarray(v, dtype=float).ravel()
    except Exception:
        return ''
    return '{' + ','.join(repr(float(x)) for x in a) + '}'


ARRAY_COLS = {c for c in KEEP if c.startswith(('utility_bill', 'cf_'))}


# ---------------------------------------------------------------------------
# Profile cache: what stops every worker needing its own DB connection
# ---------------------------------------------------------------------------
#
# calc_system_size_and_performance hits the database twice per agent-year, for
# the load profile and the solar resource. With a worker per core that meant
# cores x tasks connections against Cloud SQL, which at full fan-out exhausted
# the instance's connection slots and took the whole job down.
#
# Both fetches are simple keyed lookups whose results do not vary within a run:
# the load profile by (bldg_id, sector, state), the solar resource by
# (gid, tilt, azimuth). So the parent reads every distinct profile once, in two
# queries, and the workers serve from that. Workers then need no connection at
# all, and the per-agent round trip disappears, which is also faster.
#
# The load profile is cached RAW. The model scales it per agent-year by
# load_kwh_per_customer_in_bin, which changes with load growth, so the scaling
# has to stay per-agent rather than being baked into the cache.

_LOAD_CACHE: dict = {}
_SOLAR_CACHE: dict = {}


class _NoConn:
    """Stands in for a DB connection. Only cur.close() is ever called on it."""
    def cursor(self):
        class _C:
            def close(self_inner): pass
            def execute(self_inner, *a, **k):
                raise RuntimeError('worker tried to run SQL; the cache should have served it')
        return _C()
    def close(self): pass


def build_profile_cache(con, merged: pd.DataFrame) -> tuple[int, int]:
    """Read every distinct load and solar profile this state needs, in two queries."""
    global _LOAD_CACHE, _SOLAR_CACHE
    _LOAD_CACHE, _SOLAR_CACHE = {}, {}

    sector = str(merged['sector_abbr'].iloc[0])
    state = str(merged['state_abbr'].iloc[0])
    ids = sorted({int(b) for b in merged['bldg_id'].dropna().unique()})
    if ids:
        # Values are inlined rather than parameterised because the local driver
        # (psycopg2) and the in-container driver (pg8000) disagree on how to
        # adapt a Python list to a SQL array. Every value here came out of the
        # database a moment ago; bldg_ids are cast to int and the sector and
        # state are checked against a whitelist, so nothing user-supplied
        # reaches the query.
        if sector not in ('res', 'com', 'ind') or not state.isalpha() or len(state) != 2:
            raise ValueError(f'unexpected sector/state: {sector!r}/{state!r}')
        q = (f"SELECT bldg_id, kwh_load_profile FROM "
             f"diffusion_load_profiles.{sector}stock_load_profiles "
             f"WHERE sector_abbr = '{sector}' AND state_abbr = '{state}' "
             f"AND bldg_id IN ({','.join(str(int(i)) for i in ids)})")
        df = pd.read_sql(q, con, coerce_float=False)
        for _i, r in df.iterrows():
            _LOAD_CACHE[int(r['bldg_id'])] = r['kwh_load_profile']

    trip = merged[['solar_re_9809_gid', 'tilt', 'azimuth']].drop_duplicates()
    if len(trip):
        gids = sorted({str(g) for g in trip['solar_re_9809_gid'].dropna().unique()})
        safe = [g for g in gids if str(g).replace('.', '').replace('-', '').isdigit()]
        if len(safe) != len(gids):
            raise ValueError('unexpected solar_re_9809_gid value')
        q = ("SELECT solar_re_9809_gid, tilt, azimuth, cf FROM "
             "diffusion_resource_solar.solar_resource_hourly "
             f"WHERE solar_re_9809_gid IN ({','.join(chr(39)+str(g)+chr(39) for g in safe)})")
        df = pd.read_sql(q, con, coerce_float=False)
        for _i, r in df.iterrows():
            _SOLAR_CACHE[(str(r['solar_re_9809_gid']), str(r['tilt']), str(r['azimuth']))] = r['cf']
    return len(_LOAD_CACHE), len(_SOLAR_CACHE)


def install_cache_patches():
    """Serve the two per-agent fetches from the cache instead of the database."""
    import agent_mutation.elec as elec

    def _load(con, agent):
        raw = _LOAD_CACHE.get(int(agent.loc['bldg_id']))
        if raw is None:
            raise KeyError(f"load profile missing from cache for bldg_id "
                           f"{agent.loc['bldg_id']}")
        a = np.asarray(raw, dtype='float64')
        total = np.float64(agent.loc['load_kwh_per_customer_in_bin'])
        return pd.DataFrame({'consumption_hourly': [a / a.sum() * total],
                             'load_kwh_per_customer_in_bin': [total]})

    def _solar(con, agent):
        k = (str(agent.loc['solar_re_9809_gid']), str(agent.loc['tilt']),
             str(agent.loc['azimuth']))
        raw = _SOLAR_CACHE.get(k)
        if raw is None:
            raise KeyError(f'solar resource missing from cache for {k}')
        return pd.DataFrame({'solar_cf_profile': [np.asarray(raw, dtype='float64')],
                             'scale_offset': [1e6]})

    elec.get_and_apply_agent_load_profiles = _load
    elec.get_and_apply_normalized_hourly_resource_solar = _solar


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _init_worker():
    """
    No database connection here, deliberately. Workers serve profiles from the
    cache the parent built, which is inherited through fork. Opening one
    connection per worker is what exhausted Cloud SQL's connection slots.
    """
    install_cache_patches()


def _scan_chunk(chunk: pd.DataFrame, rate_switch_table):
    """Re-evaluate every agent-year in the chunk; return the kept columns."""
    import financial_functions as ff
    out, failed = [], 0
    conn = _NoConn()
    for _idx, row in chunk.iterrows():
        try:
            a = ff.calc_system_size_and_performance(
                conn, row.copy(), None, rate_switch_table)
            out.append({c: a.get(c) for c in KEEP})
        except Exception:
            failed += 1
    return pd.DataFrame(out), failed


# ---------------------------------------------------------------------------
# Per state
# ---------------------------------------------------------------------------

def resolve_schema(con, state: str, scenario: str, pattern: str) -> str | None:
    q = ("select schema_name from information_schema.schemata "
         "where schema_name like %s order by schema_name desc limit 1")
    like = pattern.format(scenario=scenario, st=state.lower())
    df = pd.read_sql(q, con, params=(like,))
    return df['schema_name'][0] if len(df) else None


def scan_state(con, dsn, role, state, scenario, schema, agents_pkl, years, cores,
               rate_switch_table):
    outs = pd.read_sql(f'select * from "{schema}".agent_outputs', con)
    if years:
        outs = outs[outs.year.isin(years)]
    if not len(outs):
        return None, f'{state}/{scenario}: no rows'

    src = pd.read_pickle(agents_pkl)
    join = [c for c in ('tariff_dict', 'wholesale_prices') if c in src.columns]
    key = 'bldg_id' if 'bldg_id' in outs.columns and 'bldg_id' in src.columns else None
    if key is None:
        return None, f'{state}/{scenario}: no join key to the agent file'
    merged = outs.merge(src[[key] + join].drop_duplicates(key), on=key, how='inner')
    if len(merged) != len(outs):
        print(f'  ! {state}/{scenario}: {len(outs) - len(merged)} row(s) lost joining '
              f'to the agent file', flush=True)

    n_load, n_solar = build_profile_cache(con, merged)
    print(f'  {state}/{scenario}: cached {n_load} load and {n_solar} solar profiles',
          flush=True)

    t0 = time.time()
    if cores and cores > 1:
        from multiprocessing import get_context
        # Split by positional index rather than handing the frame to
        # np.array_split, which returns ndarrays on this pandas/numpy pairing
        # and loses the DataFrame interface the worker needs.
        idx = np.array_split(np.arange(len(merged)), min(cores * 4, max(1, len(merged))))
        chunks = [merged.iloc[i] for i in idx if len(i)]
        with get_context('fork').Pool(cores, initializer=_init_worker) as pool:
            parts = pool.map(partial(_scan_chunk, rate_switch_table=rate_switch_table),
                             chunks)
        df = pd.concat([p for p, _ in parts], ignore_index=True)
        failed = sum(f for _, f in parts)
    else:
        install_cache_patches()
        df, failed = _scan_chunk(merged, rate_switch_table)

    for c in ARRAY_COLS:
        if c in df.columns:
            df[c] = df[c].apply(_pg_array)
    note = (f'{state}/{scenario}: {len(df)} rows, {failed} failed, '
            f'{time.time() - t0:.0f}s')
    return df, note


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--states', help='comma separated, e.g. PA,NY')
    ap.add_argument('--state-file', help='CSV of ABBR,FullName, one per line')
    ap.add_argument('--task-index', type=int, help='row of --state-file to run')
    ap.add_argument('--scenarios', default='baseline,policy')
    ap.add_argument('--years', default='', help='comma separated; blank = all')
    ap.add_argument('--schema-pattern',
                    default='diffusion_results_{scenario}_{st}_2040_a5_nb_%')
    ap.add_argument('--run-name', default='synapse_netbilling')
    ap.add_argument('--agents', default='../input_agents/agent_df_base_res_national_updated_tariffs_2026.pkl')
    ap.add_argument('--out', default='/tmp/scan')
    ap.add_argument('--cores', type=int, default=int(os.environ.get('LOCAL_CORES', '0')) or None)
    ap.add_argument('--conn', default=os.environ.get('PG_CONN_STRING',
                    'host=127.0.0.1 port=5432 dbname=dgendb user=postgres password=postgres'))
    ap.add_argument('--role', default=os.environ.get('PG_ROLE', 'postgres'))
    ap.add_argument('--gcs-bucket'); ap.add_argument('--gcs-prefix', default='')
    args = ap.parse_args()

    if args.state_file and args.task_index is not None:
        # Rows are "ABBR" or "ABBR,scenario". Carrying the scenario in the row
        # lets one task cover one state-scenario pair, which halves the longest
        # task. Wall clock is set by the biggest single task, and California has
        # roughly eighty times the agents of DC, so splitting the largest unit of
        # work matters more than raising parallelism once quota allows the full
        # fan-out.
        line = open(args.state_file).read().splitlines()[args.task_index]
        parts = [p.strip() for p in line.split(',')]
        states = [parts[0].upper()]
        if len(parts) > 1 and parts[1].lower() in ('baseline', 'policy'):
            args.scenarios = parts[1].lower()
    elif args.states:
        states = [s.strip().upper() for s in args.states.split(',') if s.strip()]
    else:
        ap.error('need --states or --state-file with --task-index')

    years = [int(y) for y in args.years.split(',') if y.strip()] if args.years else None
    scenarios = [s.strip() for s in args.scenarios.split(',') if s.strip()]

    con, _cur = utilfunc.make_con(args.conn, args.role)

    # Refuse to scan a net-billing run under net-metering rules, or the reverse.
    # The output would be plausible and wrong, which is the failure mode this
    # whole line of work exists to prevent.
    import financial_functions as ff
    nb_flag = bool(getattr(ff, 'FORCE_NET_BILLING', False))
    schema_is_nb = '_nb_' in args.schema_pattern
    if nb_flag != schema_is_nb:
        print(f'REFUSING: FORCE_NET_BILLING={nb_flag} but the schema pattern '
              f'{"names" if schema_is_nb else "does not name"} a net-billing run '
              f'({args.schema_pattern}). Set the env var to match, or change the '
              f'pattern.', flush=True)
        return 2
    print(f'net billing: {nb_flag} | states: {states} | scenarios: {scenarios} | '
          f'years: {years or "all"} | cores: {args.cores}', flush=True)

    import agent_mutation.elec as elec
    rate_switch_table = elec.get_rate_switch_table(con)

    written = []
    for st in states:
        for sc in scenarios:
            schema = resolve_schema(con, st, sc, args.schema_pattern)
            if not schema:
                print(f'  ! {st}/{sc}: no schema matching '
                      f'{args.schema_pattern.format(scenario=sc, st=st.lower())}', flush=True)
                continue
            df, note = scan_state(con, args.conn, args.role, st, sc, schema,
                                  args.agents, years, args.cores, rate_switch_table)
            print('  ' + note, flush=True)
            if df is None or not len(df):
                continue
            d = os.path.join(args.out, st, args.run_name)
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, f'{sc}.csv')
            df.to_csv(p, index=False)
            written.append(p)

    if args.gcs_bucket and written:
        try:
            from google.cloud import storage
            client = storage.Client()
            bucket = client.bucket(args.gcs_bucket)
            for p in written:
                rel = os.path.relpath(p, args.out)
                blob = bucket.blob(os.path.join(args.gcs_prefix, rel) if args.gcs_prefix else rel)
                blob.upload_from_filename(p)
                print(f'  uploaded gs://{args.gcs_bucket}/{blob.name}', flush=True)
        except Exception as e:
            print(f'  ! GCS upload failed: {e!r}', flush=True)
            return 1

    print(f'wrote {len(written)} file(s)', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
