"""
Perturbation tests: change an input, assert the answer moves, and moves sensibly.

This is the general control. The manifest's named checks catch errors we have
already made, and its plausibility bands catch values of the wrong magnitude.
Neither notices an input that is wired to nothing, wired to the wrong thing, or
scaled wrongly, as long as it looks reasonable sitting still.

Perturbation does, and it needs nobody to have anticipated the bug:

  * no response at all  -> the input is dead or is being overridden
  * response far too small -> a unit error
  * response in the wrong direction -> miswiring

This runs entirely offline. calc_system_size_and_performance needs a database
for load and solar profiles, but the economics underneath it takes those as
arrays, so perturbation_fixture builds a valid agent and calls that directly.
System size is pinned, because the optimiser's 2 kW tolerance would otherwise
add a step change larger than the effect being measured.

Run:  python dgen_os/python/test_input_response.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import perturbation_fixture as pf
    IMPORT_ERROR = None
except Exception as _e:                                   # pragma: no cover
    pf = None
    IMPORT_ERROR = _e

#: A response smaller than this is treated as no response. Well above float
#: noise, well below every genuine effect measured here (the smallest real one
#: is about $1,000).
MATERIAL_USD = 100.0


def _delta(overrides, en_batt=False):
    """NPV change from baseline for one perturbation, in dollars."""
    return pf.npv(en_batt=en_batt, overrides=overrides) - pf.npv(en_batt=en_batt)


# ---------------------------------------------------------------------------
# Costs: more expensive must mean worse
# ---------------------------------------------------------------------------

def test_pv_capex_increase_lowers_npv():
    d = _delta({'system_capex_per_kw_combined': pf.BASE['system_capex_per_kw_combined'] * 1.2})
    assert d < -MATERIAL_USD, f'PV capex +20% moved NPV by {d:+.0f}, expected a clear fall'


def test_battery_capex_increase_lowers_npv_in_the_battery_arm():
    d = _delta({'batt_capex_per_kwh_combined': pf.BASE['batt_capex_per_kwh_combined'] * 1.5},
               en_batt=True)
    assert d < -MATERIAL_USD, f'battery capex +50% moved NPV by {d:+.0f}'


def test_battery_capex_does_nothing_without_a_battery():
    """A battery cost must not reach the solar-only case."""
    d = _delta({'batt_capex_per_kwh_combined': pf.BASE['batt_capex_per_kwh_combined'] * 5})
    assert abs(d) < 1e-6, f'battery capex leaked into the PV-only arm by {d:+.0f}'


# ---------------------------------------------------------------------------
# Financing
# ---------------------------------------------------------------------------

def test_discount_rate_increase_lowers_npv():
    d = _delta({'real_discount_rate': 0.07})
    assert d < -MATERIAL_USD, f'discount rate 5% -> 7% moved NPV by {d:+.0f}'


def test_loan_rate_increase_lowers_npv():
    d = _delta({'loan_interest_rate': 0.10})
    assert d < -MATERIAL_USD, f'loan rate 7% -> 10% moved NPV by {d:+.0f}'


def test_down_payment_fraction_changes_npv():
    """
    Borrowing more at 7% against a 5% discount rate should hurt. The field is
    named for the down payment but is passed to SAM as debt_fraction, so this
    also pins down which way round it is.
    """
    d = _delta({'down_payment_fraction': 0.40})
    assert abs(d) > MATERIAL_USD, f'down payment 0.70 -> 0.40 moved NPV by {d:+.0f}'


def test_tax_rate_raises_npv_through_interest_deduction():
    """Mortgage interest is deductible in this configuration, so a higher
    marginal rate is worth more, not less."""
    d = _delta({'tax_rate': 0.40})
    assert d > MATERIAL_USD, f'tax rate 25.7% -> 40% moved NPV by {d:+.0f}'


# ---------------------------------------------------------------------------
# Energy and prices
# ---------------------------------------------------------------------------

def test_higher_retail_rates_raise_npv():
    d = _delta({'tariff_energy_scale': 1.2})
    assert d > MATERIAL_USD, f'retail energy rates +20% moved NPV by {d:+.0f}'


def test_more_generation_raises_npv():
    d = _delta({'naep': pf.BASE['naep'] * 1.2})
    assert d > MATERIAL_USD, f'annual yield +20% moved NPV by {d:+.0f}'


def test_faster_degradation_lowers_npv():
    d = _delta({'pv_degradation_factor': 0.02})
    assert d < -MATERIAL_USD, f'degradation 0.5% -> 2% moved NPV by {d:+.0f}'


def test_resiliency_value_raises_npv_in_the_battery_arm():
    d = _delta({'value_of_resiliency_usd': 100.0}, en_batt=True)
    assert d > MATERIAL_USD, f'resiliency value $7.50 -> $100 moved NPV by {d:+.0f}'


# ---------------------------------------------------------------------------
# The historical bug, as an executable specification
# ---------------------------------------------------------------------------

def test_export_price_is_inert_under_net_metering():
    """
    Under net metering SAM nets exported kWh at retail and never consults the
    sell series, so the wholesale price genuinely cannot matter. This is correct
    behaviour, and it is also exactly what hid the bug: the model spent thirteen
    months computing export prices that changed nothing.

    Pinned so that anyone who makes the wholesale price bite under net metering
    has to come and justify it here.
    """
    lo = pf.npv(overrides={'ur_metering_option': 0, 'wholesale_mean_usd_per_kwh': 0.003})
    hi = pf.npv(overrides={'ur_metering_option': 0, 'wholesale_mean_usd_per_kwh': 0.300})
    assert abs(hi - lo) < 1e-6, (
        f'a 100x swing in the export price moved NPV by {hi - lo:+.0f} under net '
        f'metering; it should be exactly inert')


def test_export_price_binds_under_net_billing():
    """
    The other half, and the one that would have caught the bug. Under net
    billing the sell series must reach the answer.
    """
    lo = pf.npv(overrides={'ur_metering_option': 2, 'wholesale_mean_usd_per_kwh': 0.003})
    hi = pf.npv(overrides={'ur_metering_option': 2, 'wholesale_mean_usd_per_kwh': 0.300})
    assert hi - lo > MATERIAL_USD, (
        f'a 100x swing in the export price moved NPV by only {hi - lo:+.0f} under net '
        f'billing; the wholesale series is not reaching the bill')


# ---------------------------------------------------------------------------
# Known defects, found by these tests on their first run
# ---------------------------------------------------------------------------
# These pin CURRENT behaviour, which is wrong. They are deliberately written to
# pass today so the suite stays honest about what the model does. Fix the defect
# and the test fails, which is the prompt to update it.

def test_known_defect_rate_escalation_ignores_the_agent_value():
    """
    KNOWN DEFECT. financial_functions pins rate_escalation to 3% for every
    agent. Meanwhile apply_elec_price_multiplier_and_escalator computes a
    per-agent elec_price_escalator from the price trajectory and clips it to
    plus or minus 1%. That computed value is never read.

    So every agent escalates retail rates at 3% real, where the underlying data
    supports at most 1%. Over a 25 year horizon that is a large difference in
    lifetime savings, and it is the single most attackable assumption in the
    published numbers.

    The perturbation here confirms the hardcoded value does drive the answer,
    which is what makes the override matter.
    """
    d = _delta({'rate_escalation_pct': 6.0})
    assert d > MATERIAL_USD, (
        f'rate escalation 3% -> 6% moved NPV by {d:+.0f}; it should be a large '
        f'positive move')


def test_known_defect_elec_price_multiplier_never_reaches_the_retail_tariff():
    """
    KNOWN DEFECT. elec_price_multiplier is documented as the ratio of present
    day electricity cost to 2016, when the tariffs were curated, and its job is
    to bring those tariffs up to current price levels.

    It is only ever applied to wholesale_prices. It never touches the retail
    tariff. So under net metering it is completely inert, and under net billing
    it scales only the export price, which is not what it is for.

    Whether this matters turns on whether the current tariff set is already at
    present day levels. The agent file is named for 2026 tariffs, which would
    make the multiplier redundant rather than harmful, but that needs
    confirming rather than assuming.
    """
    under_nem = _delta({'ur_metering_option': 0, 'elec_price_multiplier': 1.5})
    assert abs(under_nem) < 1e-6, (
        f'elec_price_multiplier moved NPV by {under_nem:+.0f} under net metering. '
        f'If this now binds, the defect is fixed and this test should go.')


if __name__ == '__main__':
    import traceback
    if IMPORT_ERROR is not None:
        print(f'SKIPPED: cannot import the model modules here ({IMPORT_ERROR!r}).')
        sys.exit(0)
    fns = [(n, o) for n, o in sorted(globals().items())
           if n.startswith('test_') and callable(o)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f'  PASS  {name}')
        except Exception:
            failed += 1
            print(f'  FAIL  {name}')
            traceback.print_exc()
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
