"""
Build the Synapse solar+storage deliverable from exported agent-level run data.

Reproduces the 13-column layout of dGen_synapse_solar_storage_<pct>pct_attachment.xlsx
exactly, so Synapse's code loads it unchanged, and appends per-household savings columns.

Column construction (validated against the Sep-2025 files):
    new_solar_adopters    sum(new_adopters)
    new_solar_kw_dc       sum(new_system_kw)
    new_storage_adopters  sum(batt_adopters_added_this_year)
    new_storage_kwh       sum(new_batt_kwh)
    capex_pv_usd          sum(new_system_kw * system_capex_per_kw_combined)
    capex_storage_usd     sum(new_batt_kwh  * batt_capex_per_kwh_combined)
    down_payment_usd      0.30 * capex_total        (financing is 70% debt)
    loan_payments_usd     each install-year cohort's cf_debt_payment_total rolled forward
                          over its loan and summed by calendar year, truncated at 2040. The
                          model never sets SAM's loan_term, so the loan is SAM's residential
                          default of 25 years (the methodology doc says 20). Year 0 of the
                          array is zero, which is why 2026 shows no payments.
    out_of_pocket_usd     down_payment_usd + loan_payments_usd

Savings columns (new):
    per-household averages for the two cohorts, which are INDEPENDENT of the attachment
    rate -- `calc_system_size_and_performance` computes both PV-only and PV+battery
    economics for every agent regardless of it (verified byte-identical between the 5%
    and 75% runs). Only the mix changes. Nominal series is recovered from the stored
    discounted one as value * (1 + nominal_rate)^k, where nominal_rate is SAM's
    (1 + real_discount_rate) * (1 + inflation_rate) - 1 = 7.625%.

Usage
-----
    python build_synapse_file.py --run-dir <dir with <ST>/<run_id>/{baseline,policy}.csv> \
        --out out.xlsx --attach-pct 5 [--rate 0.05]
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

USECOLS = ['agent_id', 'state_abbr', 'sector_abbr', 'year', 'new_adopters', 'new_system_kw',
           'system_kw', 'batt_kw', 'batt_kwh', 'batt_adopters_added_this_year', 'new_batt_kw',
           'new_batt_kwh', 'system_capex_per_kw_combined', 'batt_capex_per_kwh_combined',
           'cf_debt_payment_total_pv_only', 'cf_debt_payment_total_pv_batt',
           'cf_discounted_savings_pv_only', 'cf_discounted_savings_pv_batt',
           'real_discount_rate', 'inflation_rate', 'initial_batt_kw', 'initial_batt_kwh',
           'batt_kw_cum_last_year', 'batt_kwh_cum_last_year', 'storage_attachment_rate']

Y0, Y1, LOAN_YEARS, DOWN_PAYMENT_FRACTION = 2026, 2040, 25, 0.30

# Cumulative SOLAR (adopters and kW, including the pre-2026 fleet) is identical across
# attachment rates, so it is joined from a state-year export of the run rather than
# recomputed. Cumulative STORAGE does vary with the rate and is computed from the agent
# frame below, so a synthesized scenario gets its own value.
CUM_SOLAR_COLS = ['number_of_adopters', 'system_kw_cum']


def parse_arr(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return np.array([float(x) if x is not None else 0.0 for x in v])
    s = str(v).strip()
    if s in ('', 'nan', 'None'):
        return np.zeros(1)
    if s[0] in '{[':
        s = s[1:-1]
    return np.array([float(x) for x in s.split(',') if x.strip() not in ('', 'NULL', 'None')]) \
        if s else np.zeros(1)


def load_run(run_dir, run_id, scenario):
    # NOTE: must pin the run_id folder. A '*/*/scenario.csv' glob also matches the other
    # runs exported beside it (run_all_states_updated_tariffs, linear_to_1w,
    # synapse_attachrate_75, ...) and silently sums them -- which inflated every column
    # by 400-800x the first time this was run.
    frames = []
    pattern = os.path.join(run_dir, '*', run_id, f'{scenario}.csv')
    files = sorted(glob.glob(pattern))
    print(f'  {scenario}: {len(files)} state files from {run_id}')
    for f in files:
        head = pd.read_csv(f, nrows=0).columns
        use = [c for c in USECOLS if c in head]
        d = pd.read_csv(f, usecols=use)
        frames.append(d)
    if not frames:
        raise SystemExit(f'no {scenario}.csv found matching {pattern}')
    return pd.concat(frames, ignore_index=True)


def build(df, scenario):
    df = df.copy()
    for c in ('new_adopters', 'new_system_kw', 'new_batt_kwh', 'batt_adopters_added_this_year',
              'system_capex_per_kw_combined', 'batt_capex_per_kwh_combined',
              'real_discount_rate', 'inflation_rate'):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)

    df['capex_pv_usd'] = df.new_system_kw * df.system_capex_per_kw_combined
    df['capex_storage_usd'] = df.new_batt_kwh * df.batt_capex_per_kwh_combined

    agg = (df.groupby(['state_abbr', 'year'], as_index=False)
             .agg(new_solar_adopters=('new_adopters', 'sum'),
                  new_solar_kw_dc=('new_system_kw', 'sum'),
                  new_storage_adopters=('batt_adopters_added_this_year', 'sum'),
                  new_storage_kwh=('new_batt_kwh', 'sum'),
                  capex_pv_usd=('capex_pv_usd', 'sum'),
                  capex_storage_usd=('capex_storage_usd', 'sum')))
    agg['capex_total_usd'] = agg.capex_pv_usd + agg.capex_storage_usd
    agg['down_payment_usd'] = DOWN_PAYMENT_FRACTION * agg.capex_total_usd

    # Loan payments: roll each install-year cohort's debt-service array forward.
    debt = {}
    for r in df.itertuples(index=False):
        n_all = float(r.new_adopters or 0)
        n_bat = float(r.batt_adopters_added_this_year or 0)
        n_pv = max(n_all - n_bat, 0.0)
        if n_all <= 0:
            continue
        a_o = parse_arr(r.cf_debt_payment_total_pv_only)
        a_b = parse_arr(r.cf_debt_payment_total_pv_batt)
        y0 = int(r.year)
        for k in range(min(LOAN_YEARS, max(len(a_o), len(a_b)))):
            y = y0 + k
            if y > Y1:
                break
            v = (abs(a_o[k]) * n_pv if k < len(a_o) else 0.0) + \
                (abs(a_b[k]) * n_bat if k < len(a_b) else 0.0)
            if v:
                debt[(r.state_abbr, y)] = debt.get((r.state_abbr, y), 0.0) + v
    agg['loan_payments_usd'] = [debt.get((s, y), 0.0) for s, y in zip(agg.state_abbr, agg.year)]
    agg['out_of_pocket_usd'] = agg.down_payment_usd + agg.loan_payments_usd

    # Per-household savings for the two cohorts (independent of attachment rate).
    rows = []
    for r in df.itertuples(index=False):
        if float(r.new_adopters or 0) <= 0:
            continue
        # SAM discounts cf_discounted_savings at the NOMINAL rate, not the real one:
        #   nominal = (1 + real_discount_rate) * (1 + inflation_rate) - 1
        # Verified against the run's own utility_bill arrays -- un-discounting at the
        # nominal rate reproduces (bill_without - bill_with) exactly, year by year,
        # while using the real rate understates it by 1.025^k (85% by year 25).
        real = float(r.real_discount_rate or 0.05)
        infl = float(getattr(r, 'inflation_rate', 0.025) or 0.025)
        rate = (1 + real) * (1 + infl) - 1
        o, b = parse_arr(r.cf_discounted_savings_pv_only), parse_arr(r.cf_discounted_savings_pv_batt)
        nom_o = np.array([o[k] * (1 + rate) ** k for k in range(len(o))])
        nom_b = np.array([b[k] * (1 + rate) ** k for k in range(len(b))])
        def y1(a):
            nz = a[np.abs(a) > 1e-9]
            return float(nz[0]) if nz.size else 0.0
        rows.append((r.state_abbr, int(r.year), float(r.new_adopters),
                     y1(nom_o), y1(nom_b), float(nom_o.sum()), float(nom_b.sum()),
                     float(o.sum()), float(b.sum())))
    sv = pd.DataFrame(rows, columns=['state_abbr', 'year', 'w', 'so_y1', 'ss_y1',
                                     'so_life', 'ss_life', 'so_disc', 'ss_disc'])
    for c in ('so_y1', 'ss_y1', 'so_life', 'ss_life', 'so_disc', 'ss_disc'):
        sv[c] = sv[c] * sv.w                       # weight by adopters, then divide
    sv = sv.groupby(['state_abbr', 'year'], as_index=False).sum()
    for c in ('so_y1', 'ss_y1', 'so_life', 'ss_life', 'so_disc', 'ss_disc'):
        sv[c] = sv[c] / sv.w.replace(0, np.nan)
    sv = sv.rename(columns={
        'so_y1': 'avg_savings_solar_only_yr1_usd',
        'ss_y1': 'avg_savings_solar_storage_yr1_usd',
        'so_life': 'avg_savings_solar_only_25yr_nominal_usd',
        'ss_life': 'avg_savings_solar_storage_25yr_nominal_usd',
        'so_disc': 'avg_savings_solar_only_25yr_discounted_usd',
        'ss_disc': 'avg_savings_solar_storage_25yr_discounted_usd'}).drop(columns='w')

    # Cumulative storage kWh: present after synthesis; otherwise reconstruct it.
    if 'batt_kwh_cum' in df.columns:
        cum_b = df.groupby(['state_abbr', 'year'], as_index=False)['batt_kwh_cum'].sum()
    else:
        df['_bk'] = (pd.to_numeric(df.get('batt_kwh_cum_last_year', 0), errors='coerce').fillna(0.0)
                     + pd.to_numeric(df.get('new_batt_kwh', 0), errors='coerce').fillna(0.0))
        cum_b = (df.groupby(['state_abbr', 'year'], as_index=False)['_bk'].sum()
                   .rename(columns={'_bk': 'batt_kwh_cum'}))
    cum_b = cum_b.rename(columns={'batt_kwh_cum': 'total_storage_kwh'})

    out = agg.merge(sv, on=['state_abbr', 'year'], how='left').merge(
        cum_b, on=['state_abbr', 'year'], how='left')
    out.insert(2, 'scenario', scenario)
    out['new_storage_adopters'] = out.new_storage_adopters.round().astype(int)
    return out


def attach_cum_solar(out, state_year_csv, scenario):
    """Join cumulative solar adopters/kW (incl. the pre-2026 fleet) from a state-year export."""
    sy = pd.read_csv(state_year_csv)
    sy = sy[sy.scenario == scenario][['state_abbr', 'year'] + CUM_SOLAR_COLS]
    sy = sy.rename(columns={'number_of_adopters': 'total_solar_adopters',
                            'system_kw_cum': 'total_solar_kw_dc'})
    return out.merge(sy, on=['state_abbr', 'year'], how='left')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--run-id', required=True, help='the run_id subfolder, e.g. synapse_attachrate_5')
    ap.add_argument('--out', required=True)
    ap.add_argument('--attach-pct', required=True)
    ap.add_argument('--state-year', default=None,
                    help='state-year export supplying cumulative solar adopters/kW')
    ap.add_argument('--rate', type=float, default=None,
                    help='synthesize this flat attachment rate instead of using the run as-is')
    a = ap.parse_args()

    parts = []
    for scen in ('baseline', 'policy'):
        df = load_run(a.run_dir, a.run_id, scen)
        if a.rate is not None:
            import sys
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from synthesize_attachment_scenario import synthesize_rate
            df = synthesize_rate(df, a.rate)
        b = build(df, scen)
        if a.state_year:
            b = attach_cum_solar(b, a.state_year, scen)
        parts.append(b)
    data = pd.concat(parts, ignore_index=True).sort_values(['state_abbr', 'scenario', 'year'])
    # Keep the original 13 columns first and in order, then cumulative, then savings.
    base = ['state_abbr', 'year', 'scenario', 'new_solar_adopters', 'new_solar_kw_dc',
            'new_storage_adopters', 'new_storage_kwh', 'capex_pv_usd', 'capex_storage_usd',
            'capex_total_usd', 'down_payment_usd', 'loan_payments_usd', 'out_of_pocket_usd']
    cum = [c for c in ('total_solar_adopters', 'total_solar_kw_dc', 'total_storage_kwh')
           if c in data.columns]
    rest = [c for c in data.columns if c not in base + cum]
    data = data[base + cum + rest]

    print(f'{len(data)} rows, {data.state_abbr.nunique()} states, '
          f'{data.year.min()}-{data.year.max()}')
    data.to_csv(a.out.replace('.xlsx', '.csv'), index=False)
    print(f'wrote {a.out.replace(".xlsx", ".csv")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
