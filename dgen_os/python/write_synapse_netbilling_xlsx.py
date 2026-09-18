"""Write the three Synapse NET_BILLING xlsx deliverables (README tab + data tab) from the
nb2_{5,75,100}pct.csv files produced by build_synapse_file.py. Set NB_CSV_DIR to their folder.
"""
import os
import pandas as pd, numpy as np
D=os.environ.get('NB_CSV_DIR', '.')   # folder holding nb2_{5,75,100}pct.csv
OUT=os.environ.get('NB_XLSX_OUT', '/Users/wael/Library/CloudStorage/GoogleDrive-wael@permitpower.org/Shared drives/'
     'PP (All)/Research/$1 watt solar/2025/Results/Updated tariffs')

def readme(pct, derived, frac_note):
    L = [
 f'dGen residential solar + storage. Storage attachment rate: {pct}% of new solar adopters.',
 '49 states (lower 48 + DC), 2026-2040. One row per state x year x scenario.',
 np.nan,
 'scenario: baseline = business as usual; policy = $1/W installed solar cost in 2026, declining',
 '  to $0.75/W by 2040. The policy case also carries a lower storage price path: $800/kWh in',
 '  2026 falling to $390/kWh by 2040, against $1,199/kWh falling to $890/kWh in the baseline.',
 np.nan,
 'Export compensation: NET BILLING. Exported energy is paid the hourly county-level wholesale',
 'price rather than netted against consumption at the full retail rate. This differs from the',
 'earlier version of these files, which used full retail net metering. Solar adoption is lower',
 'here as a result, and storage is worth more.',
 np.nan,
 'Columns',
 '  new_solar_adopters, new_solar_kw_dc      new solar installs that year (kW is DC nameplate)',
 '  new_storage_adopters, new_storage_kwh    of those, the ones that also install storage',
 '  capex_pv_usd, capex_storage_usd          full installed cost of what was installed that year',
 '  capex_total_usd                          capex_pv_usd + capex_storage_usd',
 '  down_payment_usd                         30% paid in cash at install',
 '  loan_payments_usd                        annual payments on the financed 70%',
 '  out_of_pocket_usd                        down_payment_usd + loan_payments_usd',
 '  total_solar_adopters, total_solar_kw_dc  ALL solar on roofs that year, including systems',
 '                                           installed before 2026 (35.7 GW nationally)',
 '  total_storage_kwh                        ALL storage installed that year, including the',
 '                                           4.3 GWh already in place before 2026',
 '  avg_savings_solar_only_yr1_usd           average first-year utility bill savings for one',
 '                                           household installing solar only that year',
 '  avg_savings_solar_storage_yr1_usd        same, for one household installing solar + storage',
 '  avg_savings_*_25yr_nominal_usd           25-year total per household, in as-spent dollars',
 '  avg_savings_*_25yr_discounted_usd        25-year total per household, in present value',
 '                                           (7.6% nominal = 5% real + 2.5% inflation, SAM',
 '                                           convention)',
 np.nan,
 'Notes',
 '  Capex is gross: no tax credits, no netting of financing. There is no federal ITC in these runs.',
 '  loan_payments_usd is 0 in 2026 because a 2026 install makes its first annual payment in 2027.',
 '  It accumulates as each year adds a new cohort of 25-year loans (7% interest), so a 2040',
 '  cohort is still paying in 2064; the column only covers payments falling in 2026-2040.',
 '  Solar cost basis is the LBNL Tracking the Sun 2025 state median where the sample supports it',
 '  (13 states), otherwise the national median of $3.62/W. Storage is $1,199/kWh nationally in',
 '  the 2026 baseline; see the scenario line above for both price paths.',
 '  Storage attachment does not affect solar adoption, so the solar columns are the same in the',
 '  5%, 75% and 100% versions of this file.',
 '  The savings columns are per household, not portfolio totals, and average over the households',
 '  installing in that year. They also do not depend on the attachment rate, so they too are the',
 '  same in all three versions.',
 '  Under net billing, solar + storage households generally save MORE than solar-only households:',
 '  energy kept onsite displaces the full retail rate, while exported energy earns only wholesale.',
 '  The national adopter-weighted uplift is about 10% in year one. It is small (1-5%) in flat-rate,',
 '  low-price states (OK, KS, NE, AR, LA) and NEGATIVE in California, where adding storage moves',
 '  the household onto a different tariff in the model. Treat the CA storage savings with care.',
 '  The solar + storage savings include a small resiliency value of $5-10 per year.',
 '  new_* columns count only what is installed that year; total_* columns are the whole fleet.',
 '  Use new_* for spending and jobs (a 2019 panel creates no 2030 spend); use total_* for',
 '  generation or grid impact. Summing new_solar_kw_dc over 2026-2040 gives 35.7 GW less than',
 '  the 2040 total_solar_kw_dc, which is exactly the pre-existing fleet.',
 '  Savings are blank where a state-year has no adopters (12 rows: DC, NV, UT early baseline',
 '  years, where net billing leaves baseline adoption at zero). There is no household to',
 '  average over, so the cell is empty rather than zero.',
 '  Battery power is 0.67x the solar kW and capacity is 1.34x the solar kW, a 2-hour battery.',
    ]
    if frac_note:
        L += ['  Storage adopters can exceed solar adopters by up to 0.5 in a row: solar adopters are',
              '  fractional (agents are population-weighted) and storage adopters are a whole count.']
    if derived:
        L += [np.nan,
 f'  This {pct}% case is derived from the 5% model run rather than modeled separately. Storage',
 '  attachment does not affect solar adoption or any per-agent economics, so the derivation',
 '  reproduces a real run exactly; re-deriving the 5% case from itself reproduced every battery',
 '  column of the actual run (adopter counts exact, kW/kWh to floating-point noise).']
    return pd.DataFrame({0: L})

for pct, src in ((5,'nb2_5pct'), (75,'nb2_75pct'), (100,'nb2_100pct')):
    d = pd.read_csv(f'{D}/{src}.csv')
    frac = bool((d.new_storage_adopters > d.new_solar_adopters).any())
    p = f'{OUT}/dGen_synapse_solar_storage_{pct}pct_attachment_NET_BILLING.xlsx'
    with pd.ExcelWriter(p, engine='openpyxl') as w:
        readme(pct, derived=(pct != 5), frac_note=frac).to_excel(w, sheet_name='README',
                                                                 index=False, header=False)
        d.to_excel(w, sheet_name='data', index=False)
    print(f"wrote {pct:>3}%  rows={len(d)}  cols={d.shape[1]}  frac_note={frac}")
