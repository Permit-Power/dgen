"""
Run manifest -- record what the model ACTUALLY ran with, not what it was meant to.

Why this exists
---------------
Every modelling error found in this repo so far has had the same shape: an
assumption was silently not what anyone believed, and the model ran cleanly and
produced plausible numbers anyway. Examples:

  * wholesale export prices were computed, handed to SAM, and ignored, because
    ``ur_metering_option`` was 0 (net metering) and SAM never consults the sell
    series in that mode;
  * ``itc_fed_percent`` was set to 0.3 where SAM wanted percent, so an intended
    30% credit was 0.3%;
  * ``loan_term`` was never assigned, so runs silently took PySAM's default of
    25 years while the published methodology said 20;
  * the battery sizing ratio in code builds a pack a third smaller than the
    methodology describes.

None of those raise. None are visible in results. Reading the code more
carefully is not a control -- that control has been tested repeatedly and it
failed. What catches this class is making the resolved values visible, and
asserting that inputs the model bothers to compute actually bind.

What it does
------------
Runs ONE representative agent through the real sizing routine with a probe
installed, and snapshots the PySAM objects afterwards via ``.export()``, which
returns only assigned values. ``export()`` is used deliberately: reading an
unassigned PySAM attribute directly can segfault, which would take a production
run down with it. A field that is missing from the export is itself the signal
-- it means nothing in this repo set it and a library default is in force.

Emits a long-format table (category, key, value, note) written to the run's
output schema as ``run_manifest``, so the assumptions travel with the results
and can be read by anyone without opening Python.

The ``check`` rows are the part that earns its keep: they flag inputs that are
computed but inert, values whose magnitude implies a unit error, and internally
incoherent combinations.

Usage
-----
Called once per run from ``dgen_model``, after the agent mutations and before
sizing. Failure here must never fail a model run, so ``collect`` swallows its
own exceptions and returns whatever it managed to gather.
"""

from __future__ import annotations

import datetime as _dt
import os
import subprocess

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Probe: snapshot the SAM objects from inside the real sizing call
# ---------------------------------------------------------------------------

_SNAP: dict = {}


def _install_probe():
    """
    Wrap ``financial_functions.calc_system_performance`` so the first PV-only
    and first PV+battery evaluation each leave an exported snapshot behind.

    calc_system_size_and_performance runs the optimizer PV-only many times and
    then makes exactly one PV+battery call at the chosen size, so both arms are
    represented.
    """
    import financial_functions as ff

    if getattr(ff.calc_system_performance, "_manifest_probe", False):
        return ff

    original = ff.calc_system_performance

    def probed(kw, pv, utilityrate, loan, batt, costs, agent, rate_switch_table,
               en_batt=True, batt_dispatch='price_signal_forecast'):
        out = original(kw, pv, utilityrate, loan, batt, costs, agent,
                       rate_switch_table, en_batt, batt_dispatch)
        arm = 'pv_batt' if en_batt else 'pv_only'
        if arm not in _SNAP:
            _SNAP[arm] = {
                'utilityrate': _safe_export(utilityrate),
                'loan':        _safe_export(loan),
                'batt':        _safe_export(batt) if en_batt else {},
                'kw':          float(kw) if kw is not None else None,
            }
        return out

    probed._manifest_probe = True
    ff.calc_system_performance = probed
    return ff


def _safe_export(obj) -> dict:
    """
    ``export()`` returns only ASSIGNED values, so it never touches an unset
    attribute. Reading unset PySAM attributes directly can segfault, which is
    uncatchable from Python -- so this is the only sanctioned way to read the
    stack here. Do not replace it with getattr loops.

    Second hazard, equally fatal: batt, utilityrate and loan are built with
    ``from_existing(driver, ...)`` and share the driver's memory. Export them
    while the driver is alive. The probe satisfies this by exporting from inside
    calc_system_performance, where the driver is still held on the caller's
    frame. Exporting after the owning frame returns reads freed memory and
    segfaults. There is a regression test for this in test_run_manifest.py.
    """
    try:
        return obj.export() or {}
    except Exception:
        return {}


def _get(export: dict, group: str, key: str, default=None):
    """Fetch one field out of an export, or `default` if nothing assigned it."""
    try:
        return export.get(group, {}).get(key, default)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Facts read off the mutated agent frame
# ---------------------------------------------------------------------------

def _num(series):
    return pd.to_numeric(series, errors='coerce')


def _frame_rows(df: pd.DataFrame) -> list[tuple]:
    """Distributions of the economically material agent fields, post-mutation."""
    rows: list[tuple] = []
    n = len(df)
    rows.append(('agents', 'n_agents', n, ''))
    if 'state_abbr' in df.columns:
        rows.append(('agents', 'states', ','.join(sorted(df.state_abbr.dropna().unique())), ''))

    # Metering option carried by each agent's own tariff, before any override.
    if 'tariff_dict' in df.columns:
        opts = df.tariff_dict.apply(
            lambda t: t.get('ur_metering_option', None) if isinstance(t, dict) else None)
        counts = opts.value_counts(dropna=False).to_dict()
        pretty = ', '.join(
            f"{'absent' if (k is None or (isinstance(k, float) and np.isnan(k))) else int(k)}: {v}"
            for k, v in sorted(counts.items(), key=lambda kv: str(kv[0])))
        rows.append(('tariff', 'agent_ur_metering_option', pretty,
                     '0=net metering, 2=net billing; absent defaults to 0'))

    for col, note in (
        ('real_discount_rate',        'real, before inflation'),
        ('inflation_rate',            ''),
        ('loan_rate',                 ''),
        ('loan_term_yrs',             'agent field; SAM loan_term is what actually binds'),
        ('down_payment_fraction',     ''),
        ('tax_rate',                  ''),
        ('itc_fraction_of_capex',     'fraction; SAM wants percent'),
        ('economic_lifetime_yrs',     ''),
        ('elec_price_escalator',      ''),
        ('value_of_resiliency_usd',   'added to the battery arm only'),
        ('system_capex_per_kw_combined', 'PV portion, $/kW'),
        ('batt_capex_per_kwh_combined',  '$/kWh'),
    ):
        if col in df.columns:
            v = _num(df[col])
            if v.notna().any():
                lo, med, hi = v.min(), v.median(), v.max()
                val = f"{med:g}" if lo == hi else f"min {lo:g} / median {med:g} / max {hi:g}"
                rows.append(('financing', col, val, note))
    return rows


# ---------------------------------------------------------------------------
# Binding checks -- the part that turns silence into a signal
# ---------------------------------------------------------------------------

def _check_rows(sam: dict, df: pd.DataFrame) -> list[tuple]:
    """
    Assert that inputs the model computes actually bind, that magnitudes imply
    the right units, and that mode combinations are coherent.

    Each row is (category, key, verdict, explanation). Verdict is OK, WARN or
    FAIL so it can be grepped or filtered in SQL.
    """
    rows: list[tuple] = []
    pv_batt = sam.get('pv_batt', {})
    ur = pv_batt.get('utilityrate', {})
    ln = pv_batt.get('loan', {})
    bt = pv_batt.get('batt', {})

    mo = _get(ur, 'ElectricityRates', 'ur_metering_option')
    sell_on = _get(ur, 'ElectricityRates', 'ur_en_ts_sell_rate')

    # 1. Wholesale export prices computed but ignored. This is the net-billing bug.
    if sell_on in (1, 1.0) and mo is not None and int(mo) != 2:
        rows.append(('check', 'wholesale_sell_rate_binds', 'FAIL',
                     f'ur_en_ts_sell_rate=1 but ur_metering_option={int(mo)}. SAM only '
                     f'consults the sell series under net billing (2), so the wholesale '
                     f'export prices this model builds are inert and exports are being '
                     f'credited at full retail.'))
    elif mo is not None and int(mo) == 2:
        rows.append(('check', 'wholesale_sell_rate_binds', 'OK',
                     'net billing active; the wholesale sell series is consulted'))
    else:
        rows.append(('check', 'wholesale_sell_rate_binds', 'WARN',
                     f'could not resolve: ur_metering_option={mo}, ur_en_ts_sell_rate={sell_on}'))

    # 2. ITC units. SAM wants percent; a fraction slipped in means a 100x error.
    itc = _get(ln, 'TaxCreditIncentives', 'itc_fed_percent')
    itc_v = float(np.asarray(itc).ravel()[0]) if itc is not None and np.size(itc) else None
    if itc_v is None:
        rows.append(('check', 'itc_units', 'WARN', 'itc_fed_percent never assigned'))
    elif itc_v == 0:
        rows.append(('check', 'itc_units', 'OK', 'ITC is zero, as intended in this fork'))
    elif 0 < itc_v < 1:
        rows.append(('check', 'itc_units', 'FAIL',
                     f'itc_fed_percent={itc_v}. SAM expects PERCENT, so this is {itc_v}%, '
                     f'almost certainly a fraction passed where percent was wanted '
                     f'(a 100x understatement).'))
    else:
        rows.append(('check', 'itc_units', 'OK', f'itc_fed_percent={itc_v}%'))

    # 3. Loan term and analysis period: both silently default if nothing sets them.
    lt = _get(ln, 'FinancialParameters', 'loan_term')
    ap = _get(ln, 'FinancialParameters', 'analysis_period')
    if lt is None:
        rows.append(('check', 'loan_term_set', 'FAIL',
                     'loan_term never assigned; PySAM library default is in force'))
    else:
        rows.append(('check', 'loan_term_set', 'OK',
                     f'loan_term={lt:g} yr, analysis_period={ap if ap is None else f"{ap:g}"} yr. '
                     f'Nothing in this repo assigns loan_term, so this is the config default '
                     f'-- confirm it matches the published methodology before quoting.'))

    # 4. Dispatch coherence. Peak shaving with demand charges off optimises
    #    against a price signal the tariff does not contain.
    try:
        import financial_functions as ff
        skip_dc = bool(getattr(ff, 'SKIP_DEMAND_CHARGES', False))
    except Exception:
        skip_dc = None
    choice = _get(bt, 'BatteryDispatch', 'batt_dispatch_choice')
    if choice is not None and skip_dc and int(choice) == 0:
        rows.append(('check', 'dispatch_coherent', 'WARN',
                     'batt_dispatch_choice=0 (peak shaving) while SKIP_DEMAND_CHARGES=True. '
                     'Peak shaving exists to cut demand charges, and this tariff has none, '
                     'so the battery is optimising against a signal that carries no money.'))
    elif choice is not None:
        rows.append(('check', 'dispatch_coherent', 'OK',
                     f'batt_dispatch_choice={int(choice)}, SKIP_DEMAND_CHARGES={skip_dc}'))

    # 5. Grid charging and export permissions, which drive the storage load shape.
    gc = _get(bt, 'BatteryDispatch', 'batt_dispatch_auto_can_gridcharge')
    if gc is not None:
        rows.append(('check', 'grid_charging', 'OK' if int(gc) == 0 else 'WARN',
                     f'batt_dispatch_auto_can_gridcharge={int(gc)} '
                     f'({"no grid charging" if int(gc) == 0 else "GRID CHARGING ENABLED"})'))

    # 6. Realised battery sizing, to compare against the methodology's stated ratio.
    kw = pv_batt.get('kw')
    bk = _get(bt, 'BatterySystem', 'batt_computed_bank_capacity')
    if kw and bk:
        rows.append(('check', 'battery_sizing_ratio', 'OK',
                     f'{bk/kw:.2f} kWh of storage per kW of PV on the probe agent '
                     f'(methodology states 2.0)'))
    return rows


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _provenance_rows(schema: str, year) -> list[tuple]:
    rows = [
        ('run', 'schema', schema, ''),
        ('run', 'manifest_for_model_year', year, 'assumptions are captured once, in the first year'),
        ('run', 'captured_at_utc', _dt.datetime.utcnow().isoformat(timespec='seconds'), ''),
    ]
    try:
        sha = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                             cwd=os.path.dirname(os.path.abspath(__file__)), timeout=5)
        if sha.returncode == 0:
            rows.append(('run', 'git_commit', sha.stdout.strip(), ''))
        dirty = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True,
                               cwd=os.path.dirname(os.path.abspath(__file__)), timeout=5)
        if dirty.returncode == 0:
            rows.append(('run', 'git_dirty', 'yes' if dirty.stdout.strip() else 'no',
                         'uncommitted changes present at run time'))
    except Exception:
        pass

    # Env vars that change model behaviour. Recorded because they do not appear
    # anywhere in the code or the scenario file.
    for var in ('FORCE_NET_BILLING', 'BATT_DISPATCH_CHOICE', 'PRODUCTION_INCENTIVES',
                'FLAT_STORAGE_ATTACHMENT_RATE', 'PV_PRICE_TABLE_BASELINE',
                'PV_PRICE_TABLE_POLICY'):
        rows.append(('env', var, os.environ.get(var, '(unset)'), ''))

    try:
        import financial_functions as ff
        rows.append(('flag', 'FORCE_NET_BILLING', str(getattr(ff, 'FORCE_NET_BILLING', '?')), ''))
        rows.append(('flag', 'SKIP_DEMAND_CHARGES', str(getattr(ff, 'SKIP_DEMAND_CHARGES', '?')), ''))
    except Exception:
        pass
    return rows


# ---------------------------------------------------------------------------
# SAM rows
# ---------------------------------------------------------------------------

_SAM_FIELDS = [
    ('loan',        'FinancialParameters', 'analysis_period',      'yr'),
    ('loan',        'FinancialParameters', 'loan_term',            'yr'),
    ('loan',        'FinancialParameters', 'loan_rate',            '%'),
    ('loan',        'FinancialParameters', 'debt_fraction',        '%'),
    ('loan',        'FinancialParameters', 'mortgage',             '1 = interest is deductible'),
    ('loan',        'FinancialParameters', 'real_discount_rate',   '%'),
    ('loan',        'FinancialParameters', 'inflation_rate',       '%'),
    ('loan',        'FinancialParameters', 'federal_tax_rate',     '%'),
    ('loan',        'FinancialParameters', 'state_tax_rate',       '%'),
    ('loan',        'FinancialParameters', 'property_tax_rate',    '%'),
    ('loan',        'TaxCreditIncentives', 'itc_fed_percent',      '% -- NOT a fraction'),
    ('utilityrate', 'ElectricityRates',    'ur_metering_option',   '0 = net metering, 2 = net billing'),
    ('utilityrate', 'ElectricityRates',    'ur_en_ts_sell_rate',   '1 = time series sell rate enabled'),
    ('utilityrate', 'ElectricityRates',    'ur_nm_yearend_sell_rate', '$/kWh'),
    ('utilityrate', 'Lifetime',            'system_use_lifetime_output', ''),
    ('batt',        'BatteryDispatch',     'batt_dispatch_choice', '0=peak shaving, 4=retail rate, 5=self consumption'),
    ('batt',        'BatteryDispatch',     'batt_look_ahead_hours', 'hr'),
    ('batt',        'BatteryDispatch',     'batt_dispatch_auto_can_gridcharge', '0 = no grid charging'),
    ('batt',        'BatteryDispatch',     'batt_dispatch_charge_only_system_exceeds_load', ''),
    ('batt',        'BatteryCell',         'batt_minimum_SOC',     '%'),
    ('batt',        'BatteryCell',         'batt_initial_SOC',     '%'),
    ('batt',        'BatterySystem',       'batt_computed_bank_capacity', 'kWh, probe agent'),
    ('batt',        'BatterySystem',       'batt_power_discharge_max_kwdc', 'kW, probe agent'),
]


def _sam_rows(sam: dict) -> list[tuple]:
    rows: list[tuple] = []
    for arm in ('pv_only', 'pv_batt'):
        snap = sam.get(arm)
        if not snap:
            continue
        for obj, group, key, note in _SAM_FIELDS:
            if obj == 'batt' and arm == 'pv_only':
                continue
            exp = snap.get(obj, {})
            val = _get(exp, group, key)
            if val is None:
                rows.append((f'sam.{arm}', key, 'NOT SET',
                             (note + '; ' if note else '') + 'nothing in this repo assigns it, '
                             'so a PySAM default is in force'))
            else:
                if np.size(val) > 1:
                    a = np.asarray(val).ravel()
                    shown = f'[{a[0]:g} ... {a[-1]:g}] (n={a.size})'
                elif np.size(val) == 1:
                    shown = f'{float(np.asarray(val).ravel()[0]):g}'
                else:
                    shown = str(val)
                rows.append((f'sam.{arm}', key, shown, note))
    return rows


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def collect(con, agents_df: pd.DataFrame, rate_switch_table, year, schema: str) -> pd.DataFrame:
    """
    Build the manifest. Never raises: a guardrail that can fail a production run
    is worse than no guardrail, so everything is best effort and partial results
    are returned rather than an exception.
    """
    rows: list[tuple] = []
    try:
        rows += _provenance_rows(schema, year)
    except Exception as e:
        rows.append(('error', 'provenance', repr(e), ''))

    try:
        rows += _frame_rows(agents_df)
    except Exception as e:
        rows.append(('error', 'frame', repr(e), ''))

    # Probe one agent through the real sizing routine.
    _SNAP.clear()
    try:
        ff = _install_probe()
        probe = agents_df.iloc[0].copy()
        ff.calc_system_size_and_performance(con, probe, None, rate_switch_table)
    except Exception as e:
        rows.append(('error', 'probe', repr(e),
                     'could not size the probe agent; SAM rows and checks are missing'))

    try:
        rows += _sam_rows(_SNAP)
        rows += _check_rows(_SNAP, agents_df)
    except Exception as e:
        rows.append(('error', 'sam', repr(e), ''))

    df = pd.DataFrame(rows, columns=['category', 'key', 'value', 'note'])
    df['value'] = df['value'].astype(str)
    return df


def log(df: pd.DataFrame, logger=None) -> None:
    """Print the manifest, loudest rows first, so a failure is visible in the run log."""
    emit = logger.info if logger is not None else print
    bad = df[(df.category == 'check') & (df.value.isin(['FAIL', 'WARN']))]
    emit('---------Run manifest---------')
    for _, r in df.iterrows():
        emit(f'  {r.category:16s} {r.key:42s} {r.value}'
             + (f'   [{r.note}]' if r.note else ''))
    if len(bad):
        emit(f'  !! {len(bad)} manifest check(s) not OK -- see FAIL/WARN rows above')


def write(df: pd.DataFrame, engine, schema: str, owner: str, logger=None) -> None:
    """Persist to <schema>.run_manifest. Never raises."""
    try:
        import input_data_functions as iFuncs
        iFuncs.df_to_psql(df, engine, schema, owner, 'run_manifest', if_exists='replace')
    except Exception as e:
        msg = f'could not write run_manifest: {e!r}'
        (logger.warning if logger is not None else print)(msg)
