"""
Regression tests for the run manifest's binding checks.

These are written around the errors this repo has actually shipped, not around
hypothetical ones. Each test drives a real PySAM stack into the broken state
and asserts the manifest calls it out.

The headline case is the net-metering bug: for thirteen months the model built
wholesale export prices, handed them to SAM, and ran with
``ur_metering_option = 0``, under which SAM never consults them. Exports were
credited at full retail while everyone believed they were being sold at
wholesale. Nothing raised, and the results looked entirely reasonable.

Run:  python -m pytest dgen_os/python/test_run_manifest.py -q
   or python dgen_os/python/test_run_manifest.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

# financial_functions pulls in the Cloud SQL connector and pg8000 through
# utility_functions. Those are present in the runtime environment (dg3n) but not
# in the lean CI environment, so import failure means "cannot run here", not
# "the code is broken". Skip loudly rather than failing red for the wrong reason.
try:
    import financial_functions as ff
    import run_manifest as rm
    IMPORT_ERROR = None
except Exception as _e:                                   # pragma: no cover
    ff = rm = None
    IMPORT_ERROR = _e


def _stack():
    """
    A real residential PV+battery+rate+loan stack, as the model builds it.

    The driver is returned and MUST be kept referenced by the caller. batt,
    utilityrate and loan are built with ``from_existing(driver, ...)`` and share
    the driver's underlying memory, so letting the driver be garbage collected
    turns them into dangling handles and the next ``export()`` segfaults. That
    is uncatchable from Python and takes the whole process down.

    The manifest probe is safe from this by construction: it exports from inside
    the sizing call, while the driver is still alive on the stack frame.
    """
    driver, batt, utilityrate, loan, _market = ff._init_pv_batt_stack('res')
    return driver, batt, utilityrate, loan


def _snapshot(batt, utilityrate, loan, kw=7.0):
    return {'pv_batt': {
        'utilityrate': rm._safe_export(utilityrate),
        'loan':        rm._safe_export(loan),
        'batt':        rm._safe_export(batt),
        'kw':          kw,
    }}


def _verdict(rows, key):
    for cat, k, value, _note in rows:
        if cat == 'check' and k == key:
            return value
    return None


def _explanation(rows, key):
    for cat, k, _value, note in rows:
        if cat == 'check' and k == key:
            return note
    return ''


# ---------------------------------------------------------------------------
# The net-metering bug
# ---------------------------------------------------------------------------

def test_inert_wholesale_sell_rate_is_caught():
    """Sell series enabled but net metering active: the prices do nothing."""
    _driver, batt, ur, loan = _stack()
    ur.ElectricityRates.ur_en_ts_sell_rate = 1
    ur.ElectricityRates.ur_metering_option = 0          # net metering
    rows = rm._check_rows(_snapshot(batt, ur, loan), pd.DataFrame())
    assert _verdict(rows, 'wholesale_sell_rate_binds') == 'FAIL', rows
    assert 'inert' in _explanation(rows, 'wholesale_sell_rate_binds')


def test_net_billing_sell_rate_binds():
    """Under net billing the same configuration is correct and passes."""
    _driver, batt, ur, loan = _stack()
    ur.ElectricityRates.ur_en_ts_sell_rate = 1
    ur.ElectricityRates.ur_metering_option = 2          # net billing
    rows = rm._check_rows(_snapshot(batt, ur, loan), pd.DataFrame())
    assert _verdict(rows, 'wholesale_sell_rate_binds') == 'OK', rows


# ---------------------------------------------------------------------------
# The ITC units bug: 30% entered as 0.3, a 100x understatement
# ---------------------------------------------------------------------------

def test_itc_fraction_where_percent_expected_is_caught():
    _driver, batt, ur, loan = _stack()
    loan.TaxCreditIncentives.itc_fed_percent = [0.3]    # meant 30%
    rows = rm._check_rows(_snapshot(batt, ur, loan), pd.DataFrame())
    assert _verdict(rows, 'itc_units') == 'FAIL', rows
    assert '100x' in _explanation(rows, 'itc_units')


def test_itc_in_percent_passes():
    _driver, batt, ur, loan = _stack()
    loan.TaxCreditIncentives.itc_fed_percent = [30.0]
    rows = rm._check_rows(_snapshot(batt, ur, loan), pd.DataFrame())
    assert _verdict(rows, 'itc_units') == 'OK', rows


def test_zero_itc_passes():
    """This fork removes the ITC deliberately, so zero must not be an error."""
    _driver, batt, ur, loan = _stack()
    loan.TaxCreditIncentives.itc_fed_percent = [0.0]
    rows = rm._check_rows(_snapshot(batt, ur, loan), pd.DataFrame())
    assert _verdict(rows, 'itc_units') == 'OK', rows


# ---------------------------------------------------------------------------
# Silent library defaults
# ---------------------------------------------------------------------------

def test_loan_term_is_reported_even_though_nothing_sets_it():
    """
    Nothing in this repo assigns loan_term, so the config default binds. The
    manifest has to surface the number rather than leave it invisible, because
    the published methodology says 20 years and the default is 25.
    """
    _driver, batt, ur, loan = _stack()
    rows = rm._check_rows(_snapshot(batt, ur, loan), pd.DataFrame())
    assert _verdict(rows, 'loan_term_set') in ('OK', 'FAIL')
    note = _explanation(rows, 'loan_term_set')
    assert 'loan_term' in note or 'never assigned' in note


def test_unset_fields_are_flagged_not_silently_skipped():
    """A field no code assigns must appear as NOT SET, not vanish."""
    _driver, batt, ur, loan = _stack()
    rows = rm._sam_rows(_snapshot(batt, ur, loan))
    keys = {k for _c, k, _v, _n in rows}
    assert 'loan_term' in keys and 'ur_metering_option' in keys, sorted(keys)


# ---------------------------------------------------------------------------
# Dispatch coherence
# ---------------------------------------------------------------------------

def test_peak_shaving_without_demand_charges_warns():
    """
    Peak shaving exists to cut demand charges. The model disables residential
    demand charges, so the battery optimises against a signal carrying no money.
    """
    _driver, batt, ur, loan = _stack()
    batt.BatteryDispatch.batt_dispatch_choice = 0
    prior = ff.SKIP_DEMAND_CHARGES
    try:
        ff.SKIP_DEMAND_CHARGES = True
        rows = rm._check_rows(_snapshot(batt, ur, loan), pd.DataFrame())
    finally:
        ff.SKIP_DEMAND_CHARGES = prior
    assert _verdict(rows, 'dispatch_coherent') == 'WARN', rows


def test_self_consumption_dispatch_does_not_warn():
    _driver, batt, ur, loan = _stack()
    batt.BatteryDispatch.batt_dispatch_choice = 5
    prior = ff.SKIP_DEMAND_CHARGES
    try:
        ff.SKIP_DEMAND_CHARGES = True
        rows = rm._check_rows(_snapshot(batt, ur, loan), pd.DataFrame())
    finally:
        ff.SKIP_DEMAND_CHARGES = prior
    assert _verdict(rows, 'dispatch_coherent') == 'OK', rows


# ---------------------------------------------------------------------------
# Plausibility bands -- the generic net, for errors nobody anticipated
# ---------------------------------------------------------------------------

def test_band_catches_a_value_of_the_wrong_magnitude():
    """A discount rate of 500% is not a modelling choice, it is a bug."""
    rows = rm._band_rows({'real_discount_rate': 500.0})
    assert _verdict(rows, 'range.real_discount_rate') == 'FAIL', rows


def test_band_catches_a_fraction_where_a_percent_belongs():
    """0.0774 in a percent field is the classic unit slip."""
    rows = rm._band_rows({'agent.system_capex_per_kw_combined': 3.465})
    assert _verdict(rows, 'range.agent.system_capex_per_kw_combined') == 'FAIL', rows
    assert 'unit slip' in _explanation(rows, 'range.agent.system_capex_per_kw_combined')


def test_band_passes_a_sane_value():
    rows = rm._band_rows({'real_discount_rate': 5.0,
                          'agent.system_capex_per_kw_combined': 3465.0})
    assert not [r for r in rows if r[2] == 'FAIL'], rows


def test_unbanded_values_are_counted_not_silently_passed():
    """
    Coverage has to be visible. A value with no band is unchecked, and unchecked
    is not the same as fine.
    """
    rows = rm._band_rows({'real_discount_rate': 5.0, 'some_new_field': 1.0})
    assert _verdict(rows, 'range_coverage') == 'WARN'
    assert 'some_new_field' in _explanation(rows, 'range_coverage')


def test_real_stack_values_sit_inside_their_bands():
    """
    Guards against bands so tight they cry wolf. Every scalar a real, untouched
    PySAM residential stack reports must pass its own band.
    """
    _driver, batt, ur, loan = _stack()
    numeric: dict = {}
    rm._sam_rows(_snapshot(batt, ur, loan), numeric)
    assert numeric, 'the probe recorded nothing to check'
    failures = [r for r in rm._band_rows(numeric) if r[2] == 'FAIL']
    assert not failures, f'bands are too tight for real values: {failures}'


# ---------------------------------------------------------------------------
# The manifest must never take a run down with it
# ---------------------------------------------------------------------------

def test_collect_survives_a_broken_agent_frame():
    """A guardrail that can fail a production run is worse than none."""
    df = rm.collect(con=None, agents_df=pd.DataFrame(), rate_switch_table=None,
                    year=2026, schema='unit_test')
    assert isinstance(df, pd.DataFrame) and len(df) > 0
    assert list(df.columns) == ['category', 'key', 'value', 'note']
    assert (df.category == 'error').any(), 'a failed probe should be recorded, not swallowed'


def test_check_rows_survive_an_empty_snapshot():
    rows = rm._check_rows({}, pd.DataFrame())
    assert isinstance(rows, list)


if __name__ == '__main__':
    import traceback
    if IMPORT_ERROR is not None:
        print(f'SKIPPED: cannot import the model modules here ({IMPORT_ERROR!r}).')
        print('These tests need the full runtime environment (dg3n).')
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
