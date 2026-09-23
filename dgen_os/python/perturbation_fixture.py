"""
A self-contained agent for perturbation testing.

``calc_system_size_and_performance`` needs a database, because it fetches the
load profile and the solar resource. The economics underneath it,
``calc_system_performance``, takes those as plain arrays, so everything that
actually turns assumptions into money can be exercised offline.

This module builds a valid agent, cost dict, hourly profiles and PySAM stack,
mirroring the setup ``calc_system_size_and_performance`` performs before it
calls the economics. A perturbation test wants a VALID agent, not a
representative one, so the profiles here are synthetic and deterministic. The
tariff is real, lifted from a Pennsylvania residential agent, because tariff
structure is fiddly enough that a hand-made one would not exercise the same
code paths.

Everything is seeded and fixed, so a change in output means a change in the
model, not a change in the fixture.
"""

from __future__ import annotations

import contextlib
import json
import os

import numpy as np
import pandas as pd

HOURS = 8760
_HERE = os.path.dirname(os.path.abspath(__file__))
TARIFF_FIXTURE = os.path.join(_HERE, 'test_fixtures', 'residential_tariff_pa.json')


# ---------------------------------------------------------------------------
# Hourly profiles
# ---------------------------------------------------------------------------

def _load_profile(annual_kwh: float = 10_000.0) -> np.ndarray:
    """
    A plausible residential load: overnight base, morning and evening humps,
    mild summer air-conditioning. Shape matters because the tariff is
    time-of-use, so a flat profile would not exercise the TOU periods.
    """
    h = np.arange(HOURS)
    hour = h % 24
    doy = h // 24
    daily = (0.55
             + 0.35 * np.exp(-0.5 * ((hour - 7.5) / 1.8) ** 2)
             + 0.85 * np.exp(-0.5 * ((hour - 19.0) / 2.4) ** 2))
    seasonal = 1.0 + 0.28 * np.cos(2 * np.pi * (doy - 200) / 365.0)
    p = daily * seasonal
    return p * (annual_kwh / p.sum())


def _solar_per_kw(naep: float = 1300.0) -> np.ndarray:
    """Normalised generation per kW-dc: a clipped sinusoid by hour and season."""
    h = np.arange(HOURS)
    hour = h % 24
    doy = h // 24
    day = np.clip(np.sin(np.pi * (hour - 6.2) / 11.6), 0.0, None)
    seasonal = 1.0 + 0.32 * np.cos(2 * np.pi * (doy - 172) / 365.0)
    p = day * seasonal
    return p * (naep / p.sum())


def _wholesale_prices(mean_usd_per_kwh: float = 0.030) -> np.ndarray:
    """
    Hourly wholesale export price, peaking in the evening. Only consulted under
    net billing -- which is the entire point of the test that perturbs it.
    """
    h = np.arange(HOURS)
    hour = h % 24
    shape = 0.75 + 0.85 * np.exp(-0.5 * ((hour - 19.0) / 2.6) ** 2)
    return shape * (mean_usd_per_kwh / shape.mean())


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

#: Baseline financing and cost assumptions. Values are the fork's own defaults,
#: so a perturbation moves away from a configuration the model really uses.
BASE = {
    'economic_lifetime_yrs':   25,
    'down_payment_fraction':   0.70,
    'tax_rate':                0.2574,
    'inflation_rate':          0.025,
    'loan_interest_rate':      0.07,
    'real_discount_rate':      0.05,
    'pv_degradation_factor':   0.005,
    'elec_price_multiplier':   1.0,
    'value_of_resiliency_usd': 7.5,
    'system_capex_per_kw_combined': 3465.0,
    'batt_capex_per_kwh_combined':  1199.0,
    'system_om_per_kw':             18.0,
    'system_variable_om_per_kw':    0.0,
    'batt_capex_per_kw_combined':   0.0,
    'batt_om_per_kw_combined':      0.0,
    'batt_om_per_kwh_combined':     0.0,
    'linear_constant_combined':     0.0,
    'cap_cost_multiplier':          1.0,
    'annual_load_kwh':              10_000.0,
    'naep':                         1300.0,
    'wholesale_mean_usd_per_kwh':   0.030,
    'rate_escalation_pct':          3.0,
}


def load_tariff() -> dict:
    with open(TARIFF_FIXTURE) as fh:
        return json.load(fh)


def build(overrides: dict | None = None):
    """
    Build (agent, costs, pv, stack) for one evaluation.

    `overrides` replaces any key in BASE, plus these fixture-level knobs:
      ur_metering_option   -- 0 net metering, 2 net billing
      tariff_energy_scale  -- multiply every energy rate in the tariff
    """
    o = dict(BASE)
    o.update(overrides or {})

    tariff = load_tariff()
    if 'ur_metering_option' in o:
        tariff['ur_metering_option'] = o['ur_metering_option']
    if o.get('tariff_energy_scale', 1.0) != 1.0:
        mat = np.asarray(tariff['ur_ec_tou_mat'], dtype=float)
        mat[:, 4] *= float(o['tariff_energy_scale'])       # buy rate column
        tariff['ur_ec_tou_mat'] = mat.tolist()

    agent = pd.Series({
        'agent_id': 1,
        'state_abbr': 'PA',
        'eia_id': -1,               # matches nothing in the rate-switch table
        'nem_system_kw_limit': 1e6,
        'sector_abbr': 'res',
        'tariff_dict': json.dumps(tariff),
        'wholesale_prices': _wholesale_prices(o['wholesale_mean_usd_per_kwh']),
        'elec_price_multiplier':   o['elec_price_multiplier'],
        'value_of_resiliency_usd': o['value_of_resiliency_usd'],
        'economic_lifetime_yrs':   o['economic_lifetime_yrs'],
        'down_payment_fraction':   o['down_payment_fraction'],
        'tax_rate':                o['tax_rate'],
        'inflation_rate':          o['inflation_rate'],
        'loan_interest_rate':      o['loan_interest_rate'],
        'real_discount_rate':      o['real_discount_rate'],
        'pv_degradation_factor':   o['pv_degradation_factor'],
        'naep':                    o['naep'],
    })

    costs = {
        'system_capex_per_kw_combined':       o['system_capex_per_kw_combined'],
        'system_om_per_kw_combined':          o['system_om_per_kw'],
        'system_variable_om_per_kw_combined': o['system_variable_om_per_kw'],
        'system_om_per_kw':                   o['system_om_per_kw'],
        'system_variable_om_per_kw':          o['system_variable_om_per_kw'],
        'batt_capex_per_kwh_combined':        o['batt_capex_per_kwh_combined'],
        'batt_capex_per_kw_combined':         o['batt_capex_per_kw_combined'],
        'batt_om_per_kw_combined':            o['batt_om_per_kw_combined'],
        'batt_om_per_kwh_combined':           o['batt_om_per_kwh_combined'],
        'linear_constant_combined':           o['linear_constant_combined'],
        'cap_cost_multiplier':                o['cap_cost_multiplier'],
    }

    pv = {
        'consumption_hourly': _load_profile(o['annual_load_kwh']),
        'generation_hourly':  _solar_per_kw(o['naep']),
    }
    return agent, costs, pv, o


#: An empty rate-switch table with the columns apply_rate_switch requires. Empty
#: on purpose: a tariff switch on adoption would change the counterfactual bill
#: and make a perturbation's effect impossible to attribute cleanly.
_RATE_SWITCH_COLS = ['tech', 'rate_id_alias', 'json', 'eia_id', 'res_com',
                     'min_kw_limit', 'max_kw_limit', 'one_time_charge']


def empty_rate_switch_table() -> pd.DataFrame:
    return pd.DataFrame(columns=_RATE_SWITCH_COLS)


def npv(kw: float = 7.0, en_batt: bool = False, overrides: dict | None = None) -> float:
    """
    Net present value at a fixed system size, with everything else held still.

    Size is fixed on purpose. The optimiser has a 2 kW tolerance, so letting it
    choose would add a step function big enough to swamp the response being
    measured.

    The PySAM driver is held in a local until the call returns: batt,
    utilityrate and loan share its memory, and letting it be collected early
    segfaults the process.
    """
    import financial_functions as ff

    agent, costs, pv, o = build(overrides)
    driver, batt, utilityrate, loan, market_flag = ff._init_pv_batt_stack('res')
    loan.FinancialParameters.market = market_flag

    utilityrate.Lifetime.inflation_rate = agent.loc['inflation_rate'] * 100
    utilityrate.Lifetime.analysis_period = agent.loc['economic_lifetime_yrs']
    utilityrate.Lifetime.system_use_lifetime_output = 0
    utilityrate.SystemOutput.degradation = [agent.loc['pv_degradation_factor'] * 100]
    utilityrate.ElectricityRates.rate_escalation = [o['rate_escalation_pct']]

    ts_sell = (np.asarray(agent.loc['wholesale_prices'], dtype=float).ravel()
               * agent.loc['elec_price_multiplier'])
    tariff_dict = ff.normalize_tariff(agent.loc['tariff_dict'], net_sell_rate_scalar=0.0)
    utilityrate = ff.process_tariff(utilityrate, tariff_dict, 0.0, ts_sell_rate=ts_sell)

    loan.FinancialParameters.analysis_period = agent.loc['economic_lifetime_yrs']
    loan.FinancialParameters.debt_fraction = agent.loc['down_payment_fraction'] * 100
    loan.FinancialParameters.federal_tax_rate = [(agent.loc['tax_rate'] * 100) * 0.7]
    loan.FinancialParameters.inflation_rate = agent.loc['inflation_rate'] * 100
    loan.FinancialParameters.loan_rate = agent.loc['loan_interest_rate'] * 100
    loan.FinancialParameters.property_tax_rate = 0
    loan.FinancialParameters.real_discount_rate = agent.loc['real_discount_rate'] * 100
    loan.FinancialParameters.salvage_percentage = 0
    loan.FinancialParameters.state_tax_rate = [(agent.loc['tax_rate'] * 100) * 0.3]

    # calc_system_performance returns the NEGATIVE npv, because the optimiser
    # minimises it.
    out = ff.calc_system_performance(kw, pv, utilityrate, loan, batt, costs,
                                     agent, empty_rate_switch_table(),
                                     en_batt, 'price_signal_forecast')
    assert driver is not None or driver is None       # keep `driver` alive to here
    return -float(out)


@contextlib.contextmanager
def configured_stack(overrides: dict | None = None, kw: float = 7.0, en_batt: bool = False):
    """
    Yield (agent, costs, loan, utilityrate, batt) after one full evaluation, with
    every object still configured and executed.

    Use this to interrogate SAM directly. The driver is held for the life of the
    with-block on purpose: the other objects share its memory, and touching them
    after it is collected segfaults the process.
    """
    import financial_functions as ff

    agent, costs, pv, o = build(overrides)
    driver, batt, utilityrate, loan, market_flag = ff._init_pv_batt_stack('res')
    loan.FinancialParameters.market = market_flag
    utilityrate.Lifetime.inflation_rate = agent.loc['inflation_rate'] * 100
    utilityrate.Lifetime.analysis_period = agent.loc['economic_lifetime_yrs']
    utilityrate.Lifetime.system_use_lifetime_output = 0
    utilityrate.SystemOutput.degradation = [agent.loc['pv_degradation_factor'] * 100]
    utilityrate.ElectricityRates.rate_escalation = [o['rate_escalation_pct']]
    ts_sell = (np.asarray(agent.loc['wholesale_prices'], dtype=float).ravel()
               * agent.loc['elec_price_multiplier'])
    tariff_dict = ff.normalize_tariff(agent.loc['tariff_dict'], net_sell_rate_scalar=0.0)
    utilityrate = ff.process_tariff(utilityrate, tariff_dict, 0.0, ts_sell_rate=ts_sell)
    fp = loan.FinancialParameters
    fp.analysis_period = agent.loc['economic_lifetime_yrs']
    fp.debt_fraction = agent.loc['down_payment_fraction'] * 100
    fp.federal_tax_rate = [(agent.loc['tax_rate'] * 100) * 0.7]
    fp.inflation_rate = agent.loc['inflation_rate'] * 100
    fp.loan_rate = agent.loc['loan_interest_rate'] * 100
    fp.property_tax_rate = 0
    fp.real_discount_rate = agent.loc['real_discount_rate'] * 100
    fp.salvage_percentage = 0
    fp.state_tax_rate = [(agent.loc['tax_rate'] * 100) * 0.3]
    ff.calc_system_performance(kw, pv, utilityrate, loan, batt, costs, agent,
                               empty_rate_switch_table(), en_batt,
                               'price_signal_forecast')
    try:
        yield agent, costs, loan, utilityrate, batt
    finally:
        del driver          # explicit: nothing may touch the stack after this
