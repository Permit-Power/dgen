
"""
analysis_functions.py
=====================

Utilities for loading, aggregating, and visualizing **dGen** model outputs across U.S. states,
with **single-pass I/O** and **production-grade documentation**.

Key design principles
---------------------
1. **Load once, use everywhere.** The :class:`DataWarehouse` class reads all per‑state CSVs
   (baseline & policy) and hourly files (state & RTO) exactly once. All downstream analysis
   and plotting functions accept dataframes and/or a warehouse instance so they **do not
   touch the filesystem** again.
2. **Assume the schema.** These helpers are purpose-built for the schema shown in the
   example files (e.g., ``baseline.csv``, ``policy.csv``, ``*_state_hourly.csv``, ``*_rto_hourly.csv``).
   Columns are treated as present with expected dtypes; we *do not* add defensive checks,
   try/except wrappers, or fallbacks unless strictly necessary.
3. **Clear, typed docstrings.** Every function includes inputs/outputs and assumptions.

Coincident-peak methodology
---------------------------
Historically, coincident peak reduction was measured as the **difference at the single
baseline peak hour**. That metric is noisy: a one-hour spike can dominate results.
This module adds a **top‑N averaging** option (default: N=10):

- ``method="baseline_topN_avg"`` *(default)* — Find the **top N hours** in the **baseline** series.
  Return ``mean(baseline[topN]) - mean(policy[topN])``. This retains a common timestamp
  basis (true *coincidence*) while reducing noise and better representing capacity value.
- ``method="single_hour"`` — Legacy behavior: use the single **max** baseline hour.
- ``method="separate_topN_means"`` — Compare each scenario’s own top‑N mean
  (``mean(baseline.topN) - mean(policy.topN)``). This is *not* strictly coincident, but
  can be informative for sensitivity.
- ``method="percentile_mean"`` — Average across baseline hours at/above a given percentile
  (e.g., 99.9th). Set ``percentile=0.999``.

The coincident functions expose ``top_n`` and ``percentile`` parameters to switch strategies.

Example (notebook)
------------------
>>> wh = DataWarehouse.from_disk(ROOT_DIR, run_id=RUN_ID)
>>> outputs = aggregate_state_metrics(wh.agents, SavingsConfig(lifetime_years=25))
>>> peaks   = compute_peaks_by_state(wh.state_hourly)
>>> rto_co  = compute_rto_coincident_reduction(wh, method="baseline_topN_avg", top_n=10)

Author: ResStock Solar Modeling team
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import geopandas as gpd  # optional, only needed for choropleths
except Exception:  # pragma: no cover - optional at runtime
    gpd = None


# -----------------------------------------------------------------------------
# Constants (schema)
# -----------------------------------------------------------------------------

AGENT_USECOLS: Sequence[str] = (
    # identity / keys
    "state_abbr", "scenario", "year", "rto",
    # adoption cohorts & stocks
    "new_adopters", "number_of_adopters",
    "customers_in_bin", "batt_adopters_added_this_year",
    # tech sizes (per-agent & cumulative)
    "system_kw", "new_system_kw", "system_kw_cum",
    "batt_kwh", "batt_kwh_cum",
    # prices / loads
    "price_per_kwh", "load_kwh_per_customer_in_bin_initial",
    # arrays (25-year per-adopter cashflows & bills)
    "cf_energy_value_pv_only", "cf_energy_value_pv_batt",
    "utility_bill_w_sys_pv_only", "utility_bill_w_sys_pv_batt",
    "utility_bill_wo_sys_pv_only", "utility_bill_wo_sys_pv_batt",
    # market share (for plotting)
    "max_market_share",
)

STATE_HOURLY_USECOLS: Sequence[str] = ("scenario", "year", "net_sum_text")
RTO_HOURLY_USECOLS:   Sequence[str] = ("scenario", "rto", "year", "net_sum_text")


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _parse_array(s: object) -> List[float]:
    """
    Parse Postgres-like arrays (e.g. "{1,2,3}") or JSON lists ("[1,2,3]") into a list of floats.
    Returns [] if input is empty/NULL-like.
    """
    import json

    if s is None:
        return []
    if isinstance(s, (list, tuple, np.ndarray)):
        return [float(x) for x in s]

    t = str(s).strip()
    if not t or t.upper() in {"NULL", "NAN"}:
        return []

    # Fast path for Postgres-brace style: {1,2,3}
    if t.startswith("{") and t.endswith("}"):
        inner = t[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip() for p in inner.split(",")]
        out: List[float] = []
        for p in parts:
            if p:  # ignore empty tokens
                out.append(float(p))
        return out

    # JSON-like fallback: [1,2,3]
    if t.startswith("[") and t.endswith("]"):
        try:
            arr = json.loads(t)
            if isinstance(arr, list):
                return [float(x) for x in arr]
        except Exception:
            pass

    # As a last resort, try to split on commas without brackets
    try:
        parts = [p.strip() for p in t.split(",")]
        return [float(p) for p in parts if p]
    except Exception:
        return []



def _arr25(s: object) -> List[float]:
    """Parse + clip/pad to length 25 (life-year arrays)."""
    a = _parse_array(s)[:25]
    return a + [0.0] * (25 - len(a))


def _list_state_dirs(root_dir: str, states: Optional[Iterable[str]] = None) -> List[str]:
    """
    List subdirectories (two-letter state codes) under ``root_dir``.
    If ``states`` is provided, filter to that set (case-insensitive).
    """
    dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and len(d) == 2]
    if states:
        want = {s.strip().upper() for s in states if s}
        dirs = [d for d in dirs if os.path.basename(d).upper() in want]
    return sorted(dirs)


# -----------------------------------------------------------------------------
# Data warehouse (single-pass I/O)
# -----------------------------------------------------------------------------

@dataclass
class DataWarehouse:
    """
    Canonical in-memory representation of a full dGen run.

    Attributes
    ----------
    agents : pandas.DataFrame
        Per-state baseline & policy agent outputs (concatenated).
    state_hourly : pandas.DataFrame
        State-level hourly net load arrays by scenario/year.
        Columns: ``['state_abbr','scenario','year','net_load']``.
    rto_hourly : pandas.DataFrame
        RTO-level hourly net load arrays by scenario/year.
        Columns: ``['rto','scenario','year','net_load']``.
    """
    agents: pd.DataFrame
    state_hourly: pd.DataFrame
    rto_hourly: pd.DataFrame

    @classmethod
    def from_disk(
        cls,
        root_dir: str,
        run_id: Optional[str] = None,
        states: Optional[Iterable[str]] = None,
    ) -> "DataWarehouse":
        """
        Load all states once (agents + hourly) into memory.

        Expectations
        ------------
        For each state folder ``<ROOT>/<STATE>[/<RUN_ID>]`` the following files exist:

        - ``baseline.csv`` and ``policy.csv``
        - ``baseline_state_hourly.csv`` and ``policy_state_hourly.csv``
        - ``baseline_rto_hourly.csv`` and ``policy_rto_hourly.csv``

        Parameters
        ----------
        root_dir : str
            Directory whose subfolders are two-letter state codes.
        run_id : str, optional
            Optional subfolder name to select a specific run.
        states : iterable[str], optional
            Subset of states to load (e.g., ``["CA", "NY"]``).

        Returns
        -------
        DataWarehouse
        """
        agent_frames: List[pd.DataFrame] = []
        state_hourly_rows: List[Tuple[str, str, int, List[float]]] = []
        rto_hourly_rows:   List[Tuple[str, str, int, List[float]]] = []

        for state_dir in _list_state_dirs(root_dir, states=states):
            state = os.path.basename(state_dir).upper()
            basedir = os.path.join(state_dir, str(run_id)) if run_id else state_dir

            # -- Agents (baseline + policy)
            for scen in ("baseline", "policy"):
                p = os.path.join(basedir, f"{scen}.csv")
                if not os.path.exists(p):
                    continue  # keep loader simple: skip missing files
                df = pd.read_csv(p, usecols=[c for c in AGENT_USECOLS if c != "scenario"])
                df["scenario"] = scen
                df["state_abbr"] = state  # ensure uppercase
                agent_frames.append(df)

            # -- State hourly
            for scen in ("baseline", "policy"):
                p = os.path.join(basedir, f"{scen}_state_hourly.csv")
                if not os.path.exists(p):
                    continue
                h = pd.read_csv(p, usecols=STATE_HOURLY_USECOLS)
                # The file may or may not include scenario; we know which one we read
                h["scenario"] = scen
                for r in h.itertuples(index=False):
                    state_hourly_rows.append((state, scen, int(r.year), _parse_array(r.net_sum_text)))

            # -- RTO hourly
            for scen in ("baseline", "policy"):
                p = os.path.join(basedir, f"{scen}_rto_hourly.csv")
                if not os.path.exists(p):
                    continue
                h = pd.read_csv(p, usecols=RTO_HOURLY_USECOLS)
                h["scenario"] = scen
                for r in h.itertuples(index=False):
                    rto_hourly_rows.append((str(r.rto), scen, int(r.year), _parse_array(r.net_sum_text)))

        agents = pd.concat(agent_frames, ignore_index=True) if agent_frames else pd.DataFrame(columns=AGENT_USECOLS)
        state_hourly = pd.DataFrame(state_hourly_rows, columns=["state_abbr","scenario","year","net_load"])
        rto_hourly   = pd.DataFrame(rto_hourly_rows,   columns=["rto","scenario","year","net_load"])

        # Basic numeric coercions for key columns used downstream
        if not agents.empty:
            num_cols = ["year","new_adopters","number_of_adopters","customers_in_bin",
                        "batt_adopters_added_this_year","system_kw","new_system_kw",
                        "system_kw_cum","batt_kwh","batt_kwh_cum","price_per_kwh",
                        "load_kwh_per_customer_in_bin_initial","max_market_share"]
            for c in num_cols:
                if c in agents.columns:
                    agents[c] = pd.to_numeric(agents[c], errors="coerce")
            agents["state_abbr"] = agents["state_abbr"].astype(str).str.upper()
            agents["scenario"] = agents["scenario"].astype(str).str.lower()

        for df in (state_hourly, rto_hourly):
            if not df.empty:
                df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
                df["scenario"] = df["scenario"].astype(str).str.lower()

        return cls(agents=agents, state_hourly=state_hourly, rto_hourly=rto_hourly)


# -----------------------------------------------------------------------------
# Savings, totals, and market-share aggregations (agents)
# -----------------------------------------------------------------------------

@dataclass
class SavingsConfig:
    """
    Configuration for portfolio bill-savings rollups.

    Parameters
    ----------
    lifetime_years : int, default 25
        Credited lifetime of savings arrays.
    cap_to_horizon : bool, default False
        If True, only credit life-years that fall within the modeled calendar horizon
        (min(input years) .. max(input years)).
    """
    lifetime_years: int = 25
    cap_to_horizon: bool = False


def compute_portfolio_and_cumulative_savings(
    df: pd.DataFrame,
    cfg: SavingsConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build (1) annual portfolio bill savings by rolling each cohort's 25-yr array forward,
    and (2) cumulative bill savings (cumsum of annual), plus (3) lifetime totals per state/scenario.

    Uses:
      - new_adopters and batt_adopters_added_this_year to split cohorts into pv_only vs pv_batt.
      - cf_energy_value_pv_only, cf_energy_value_pv_batt arrays (per-adopter savings).
    """
    if df.empty:
        empty_annual = pd.DataFrame(columns=["state_abbr","scenario","year","portfolio_annual_savings","lifetime_savings_total"])
        empty_cum    = pd.DataFrame(columns=["state_abbr","scenario","year","cumulative_bill_savings","lifetime_savings_total"])
        return empty_annual, empty_cum

    x = df.copy()

    # cohort sizes
    x["new_adopters"] = pd.to_numeric(x.get("new_adopters", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    x["batt_adopters_added_this_year"] = pd.to_numeric(x.get("batt_adopters_added_this_year", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    x["pv_batt_n"] = np.minimum(x["batt_adopters_added_this_year"], x["new_adopters"])
    x["pv_only_n"] = (x["new_adopters"] - x["pv_batt_n"]).clip(lower=0.0)

    # arrays
    x["cf_pv_only"] = x.get("cf_energy_value_pv_only", np.nan).apply(_arr25) if "cf_energy_value_pv_only" in x.columns else [[]]*len(x)
    x["cf_pv_batt"] = x.get("cf_energy_value_pv_batt", np.nan).apply(_arr25) if "cf_energy_value_pv_batt" in x.columns else [[]]*len(x)

    # horizon
    x["year"] = pd.to_numeric(x.get("year", np.nan), errors="coerce")
    years = x["year"].dropna()
    if years.empty:
        empty_annual = pd.DataFrame(columns=["state_abbr","scenario","year","portfolio_annual_savings","lifetime_savings_total"])
        empty_cum    = pd.DataFrame(columns=["state_abbr","scenario","year","cumulative_bill_savings","lifetime_savings_total"])
        return empty_annual, empty_cum
    y_min, y_max = int(years.min()), int(years.max())
    L = int(getattr(cfg, "lifetime_years", 25) or 25)

    # annual portfolio by rolling arrays forward
    contrib = []
    for r in x.itertuples(index=False):
        state = r.state_abbr
        scen  = r.scenario
        y0    = int(r.year) if not pd.isna(r.year) else None
        if y0 is None or ((r.pv_only_n <= 0) and (r.pv_batt_n <= 0)):
            continue
        a_only, a_batt = list(r.cf_pv_only or []), list(r.cf_pv_batt or [])
        for k in range(25):
            y = y0 + k
            if cfg.cap_to_horizon and y > y_max:
                break
            if y < y_min or y > y_max:
                continue
            v = 0.0
            if k < len(a_only) and r.pv_only_n > 0: v += a_only[k] * r.pv_only_n
            if k < len(a_batt) and r.pv_batt_n > 0: v += a_batt[k] * r.pv_batt_n
            if v != 0.0:
                contrib.append((state, scen, y, v))

    annual = (
        pd.DataFrame(contrib, columns=["state_abbr","scenario","year","portfolio_annual_savings"])
          .groupby(["state_abbr","scenario","year"], as_index=False)["portfolio_annual_savings"].sum()
    ) if contrib else pd.DataFrame(columns=["state_abbr","scenario","year","portfolio_annual_savings"])

    # lifetime totals (sum credited years of each cohort × cohort size)
    life_rows = []
    for r in x.itertuples(index=False):
        y0 = int(r.year) if not pd.isna(r.year) else None
        if y0 is None:
            continue
        credited = L if not cfg.cap_to_horizon else max(0, min(L, y_max - y0 + 1))
        if credited <= 0:
            continue
        tot = 0.0
        if r.pv_only_n > 0 and r.cf_pv_only: tot += sum(r.cf_pv_only[:credited]) * r.pv_only_n
        if r.pv_batt_n > 0 and r.cf_pv_batt: tot += sum(r.cf_pv_batt[:credited]) * r.pv_batt_n
        if tot != 0.0:
            life_rows.append((r.state_abbr, r.scenario, tot))

    lifetime = (
        pd.DataFrame(life_rows, columns=["state_abbr","scenario","lifetime_savings_for_cohort"])
          .groupby(["state_abbr","scenario"], as_index=False)["lifetime_savings_for_cohort"].sum()
          .rename(columns={"lifetime_savings_for_cohort": "lifetime_savings_total"})
    ) if life_rows else pd.DataFrame(columns=["state_abbr","scenario","lifetime_savings_total"])

    # ------- ONLY CHANGE YOU NEED (2 lines) -------
    if not annual.empty:
        annual["portfolio_annual_savings"] = pd.to_numeric(annual["portfolio_annual_savings"], errors="coerce").fillna(0.0)  # NEW
    # ---------------------------------------------

    # cumulative
    if not annual.empty:
        annual = annual.sort_values(["state_abbr","scenario","year"])
        cumulative = annual.copy()
        cumulative["cumulative_bill_savings"] = cumulative.groupby(["state_abbr","scenario"], observed=True)[
            "portfolio_annual_savings"
        ].cumsum()
    else:
        cumulative = pd.DataFrame(columns=["state_abbr","scenario","year","cumulative_bill_savings"])

    # attach lifetime totals
    annual     = annual.merge(lifetime, on=["state_abbr","scenario"], how="left")
    cumulative = cumulative.merge(lifetime, on=["state_abbr","scenario"], how="left")
    return annual, cumulative, lifetime



def aggregate_state_metrics(agents: pd.DataFrame, cfg: SavingsConfig) -> Dict[str, pd.DataFrame]:
    """
    Aggregate per-state metrics for plotting and exports.

    Parameters
    ----------
    agents : pandas.DataFrame
        Agent-level state outputs (concatenated baseline & policy) as returned by
        :attr:`DataWarehouse.agents`.
    cfg : SavingsConfig
        Savings horizon/credit configuration.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keys:
          - ``median_system_kw``      — state×year×scenario median PV size (kW) (adopters-only, weighted)
          - ``median_storage_kwh``    — state×year×scenario median storage size (kWh) (adopters-only, weighted)
          - ``totals``                — state×year×scenario totals: new adopters, cumulative kW/kWh, etc.
          - ``tech_2040``             — % of technical potential (2040 only)
          - ``portfolio_annual_savings``
          - ``cumulative_bill_savings``
          - ``lifetime_totals``
          - ``avg_price_2026_model``  — adopter-weighted average price in 2026 baseline
          - ``market_share_reached``
    """
    x = agents.copy()
    x["state_abbr"] = x["state_abbr"].astype(str).str.upper()
    x["scenario"]   = x["scenario"].astype(str).str.lower()

    for c in ("year","new_adopters","number_of_adopters","customers_in_bin","max_market_share",
              "system_kw","new_system_kw","system_kw_cum","batt_kwh","batt_kwh_cum",
              "price_per_kwh","batt_adopters_added_this_year","load_kwh_per_customer_in_bin_initial"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)

    # Weighted medians (adopters-only)
    def _weighted_median(values: pd.Series, weights: pd.Series) -> float:
        v = pd.to_numeric(values, errors="coerce")
        w = pd.to_numeric(weights, errors="coerce").clip(lower=0)
        m = v.notna() & (w > 0)
        if not m.any():
            return float("nan")
        v, w = v[m].to_numpy(), w[m].to_numpy()
        order = np.argsort(v)
        v, w = v[order], w[order]
        cw = np.cumsum(w)
        return float(v[np.searchsorted(cw, 0.5 * w.sum(), side="left")])

    adopters = x[x["new_adopters"] > 0].copy()

    median_kw = (
        adopters.groupby(["state_abbr","year","scenario"], observed=True)
                .apply(lambda g: (g["system_kw_cum"].sum() / g["number_of_adopters"].sum()))
                .reset_index(name="median_system_kw")
    )

    if "batt_kwh" in adopters.columns:
        has_storage = adopters[adopters["batt_kwh"] > 0]
        median_storage = (
            has_storage.groupby(["state_abbr","year","scenario"], observed=True)
                       .apply(lambda g: _weighted_median(g["batt_kwh"], g["new_adopters"]))
                       .reset_index(name="median_batt_kwh")
        )
    else:
        median_storage = pd.DataFrame(columns=["state_abbr","year","scenario","median_batt_kwh"])

    # Totals
    totals = (
        x.groupby(["state_abbr","year","scenario"], observed=True)
         .agg(new_adopters=("new_adopters","sum"),
              new_system_kw=("new_system_kw","sum"),
              number_of_adopters=("number_of_adopters","sum"),
              system_kw_cum=("system_kw_cum","sum"),
              batt_kwh_cum=("batt_kwh_cum","sum"))
         .reset_index()
    )

    # Tech potential in 2040
    tech_2040 = x.loc[x["year"] == 2040, ["state_abbr","scenario","number_of_adopters","customers_in_bin"]].groupby(
                    ["state_abbr","scenario"], observed=True, as_index=False).sum()
    tech_2040["percent_tech_potential"] = np.where(
        tech_2040["customers_in_bin"] > 0,
        100.0 * tech_2040["number_of_adopters"] / tech_2040["customers_in_bin"],
        np.nan,
    )

    # Savings
    portfolio_annual, cumulative_savings, lifetime_totals = compute_portfolio_and_cumulative_savings(x, cfg)

    # 2026 host price (weighted by customers_in_bin, baseline only)
    price_2026 = x[(x["year"] == 2026) & (x["scenario"] == "baseline")][["state_abbr","customers_in_bin","price_per_kwh"]]
    def _wavg(g: pd.DataFrame) -> float:
        return float(np.average(g["price_per_kwh"], weights=g["customers_in_bin"])) if g["customers_in_bin"].sum() > 0 else np.nan
    avg_price_2026_model = price_2026.groupby("state_abbr", as_index=False).apply(_wavg).rename(columns={None:"price_per_kwh"})

    # Market share
    x["market_potential"] = x["customers_in_bin"] * x["max_market_share"]
    market_share = (
        x.groupby(["state_abbr","year","scenario"], observed=True)
         .agg(market_potential=("market_potential","sum"),
              market_reached=("number_of_adopters","sum"))
         .reset_index()
    )
    market_share["market_share_reached"] = np.where(
        market_share["market_potential"] > 0,
        market_share["market_reached"] / market_share["market_potential"],
        np.nan,
    )

    return {
        "median_system_kw": median_kw,
        "median_storage_kwh": median_storage,
        "totals": totals,
        "tech_2040": tech_2040,
        "portfolio_annual_savings": portfolio_annual,
        "cumulative_bill_savings": cumulative_savings,
        "lifetime_totals": lifetime_totals,
        "avg_price_2026_model": avg_price_2026_model,
        "market_share_reached": market_share,
    }


# -----------------------------------------------------------------------------
# Peaks & coincident reductions (hourly)
# -----------------------------------------------------------------------------

def compute_peaks_by_state(state_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Compute state×year×scenario peak (MW) from hourly arrays.

    Parameters
    ----------
    state_hourly : pandas.DataFrame
        Columns: ['state_abbr','scenario','year','net_load'] (list[float] per row).

    Returns
    -------
    pandas.DataFrame
        ['state_abbr','scenario','year','peak_mw']
    """
    rows: List[Tuple[str,str,int,float]] = []
    for r in state_hourly.itertuples(index=False):
        peak = float(np.max(r.net_load)) if r.net_load else np.nan
        rows.append((r.state_abbr, r.scenario, int(r.year), peak))
    return pd.DataFrame(rows, columns=["state_abbr","scenario","year","peak_mw"])


def _coincident_scalar(
    baseline: Sequence[float],
    policy: Sequence[float],
    method: str = "baseline_topN_avg",
    top_n: int = 10,
    percentile: Optional[float] = None,
) -> float:
    """
    Scalar coincident reduction from two hourly arrays.

    Parameters
    ----------
    baseline, policy : sequence of float
        Hourly net load for the same geography and year.
    method : {"baseline_topN_avg","single_hour","separate_topN_means","percentile_mean"}, default "baseline_topN_avg"
        Strategy for defining the peak slice.
    top_n : int, default 10
        Used for "baseline_topN_avg" and "separate_topN_means".
    percentile : float, optional
        Percentile for "percentile_mean" (e.g., 0.999 = top 0.1%).

    Returns
    -------
    float
        Coincident reduction in MW.
    """
    b = np.asarray(baseline, dtype=float)
    p = np.asarray(policy, dtype=float)
    if method == "single_hour":
        i = int(np.argmax(b))
        return float(b[i] - p[i])
    if method == "baseline_topN_avg":
        idx = np.argpartition(b, -top_n)[-top_n:]
        b_mean = float(b[idx].mean())
        p_mean = float(p[idx].mean())
        return b_mean - p_mean
    if method == "separate_topN_means":
        bi = np.argpartition(b, -top_n)[-top_n:]
        pi = np.argpartition(p, -top_n)[-top_n:]
        return float(b[bi].mean() - p[pi].mean())
    if method == "percentile_mean":
        assert percentile is not None and 0.0 < percentile < 1.0, "percentile must be (0,1)"
        thresh = np.quantile(b, percentile)
        mask = b >= thresh
        return float(b[mask].mean() - p[mask].mean())
    raise ValueError(f"Unknown method: {method}")


def compute_state_coincident_reduction(
    wh: DataWarehouse,
    method: str = "baseline_topN_avg",
    top_n: int = 10,
    percentile: Optional[float] = None,
) -> pd.DataFrame:
    """
    Per‑state coincident peak reduction for each year.

    Parameters
    ----------
    wh : DataWarehouse
        Loaded warehouse with ``state_hourly``.
    method : str, default "baseline_topN_avg"
        See :func:`_coincident_scalar` for options.
    top_n : int, default 10
        See :func:`_coincident_scalar`.
    percentile : float, optional
        See :func:`_coincident_scalar`.

    Returns
    -------
    pandas.DataFrame
        ['state_abbr','year','coincident_reduction_mw']
    """
    rows: List[Tuple[str,int,float]] = []
    g = wh.state_hourly.groupby(["state_abbr","year"], observed=True)
    for (state, year), sub in g:
        base = sub[sub["scenario"] == "baseline"]
        pol  = sub[sub["scenario"] == "policy"]
        if base.empty or pol.empty:
            continue
        b = base.iloc[0]["net_load"]
        p = pol.iloc[0]["net_load"]
        rows.append((state, int(year), _coincident_scalar(b, p, method=method, top_n=top_n, percentile=percentile)))
    return pd.DataFrame(rows, columns=["state_abbr","year","coincident_reduction_mw"]).sort_values(["state_abbr","year"]).reset_index(drop=True)


def compute_rto_coincident_reduction(
    wh: DataWarehouse,
    method: str = "baseline_topN_avg",
    top_n: int = 10,
    percentile: Optional[float] = None,
    return_by_rto: bool = False,
) -> pd.DataFrame:
    """
    RTO coincident peak reduction aggregated to national totals by year.

    Steps per RTO×year:
      1) Sum hourly net load across all contributing states for baseline and policy.
      2) Compute coincident reduction with the chosen method.
      3) (Optional) return each RTO series; otherwise sum across RTOs to national.

    Parameters
    ----------
    wh : DataWarehouse
        Loaded warehouse with ``rto_hourly``.
    method : str, default "baseline_topN_avg"
        See :func:`_coincident_scalar` for options.
    top_n : int, default 10
        See :func:`_coincident_scalar`.
    percentile : float, optional
        See :func:`_coincident_scalar`.
    return_by_rto : bool, default False
        If True, return a per‑RTO series; otherwise, return national totals.

    Returns
    -------
    pandas.DataFrame
        If ``return_by_rto=False``: ['year','coincident_reduction_mw']
        If ``return_by_rto=True`` : ['rto','year','coincident_reduction_mw']
    """
    # 1) Collapse all state rows → one array per (rto, scenario, year)
    def _sum_arrays(lst: List[List[float]]) -> List[float]:
        L = max(len(a) for a in lst)
        out = np.zeros(L, dtype=float)
        for a in lst:
            a = np.asarray(a, dtype=float)
            out[: len(a)] += a[:L]
        return out.tolist()

    sums = (
        wh.rto_hourly.groupby(["rto","scenario","year"], observed=True)["net_load"]
                     .apply(lambda s: _sum_arrays(list(s)))
                     .reset_index()
    )

    # 2) Compute coincident reduction per RTO×year
    rows_rto: List[Tuple[str,int,float]] = []
    for (rto, year), sub in sums.groupby(["rto","year"], observed=True):
        base = sub[sub["scenario"] == "baseline"]
        pol  = sub[sub["scenario"] == "policy"]
        if base.empty or pol.empty:
            continue
        b = base.iloc[0]["net_load"]
        p = pol.iloc[0]["net_load"]
        val = _coincident_scalar(b, p, method=method, top_n=top_n, percentile=percentile)
        rows_rto.append((str(rto), int(year), float(val)))

    df_rto = pd.DataFrame(rows_rto, columns=["rto","year","coincident_reduction_mw"]).sort_values(["rto","year"])

    if return_by_rto:
        return df_rto.reset_index(drop=True)

    # 3) National totals (sum across RTOs)
    nat = df_rto.groupby("year", as_index=False)["coincident_reduction_mw"].sum()
    return nat.sort_values("year").reset_index(drop=True)


# -----------------------------------------------------------------------------
# National totals / deltas and faceted plots
# -----------------------------------------------------------------------------

def build_national_totals(
    outputs: Dict[str, pd.DataFrame],
    peaks_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build U.S. totals per metric for plotting lines by scenario.

    Parameters
    ----------
    outputs : dict[str, pandas.DataFrame]
        Output dictionary from :func:`aggregate_state_metrics`.
    peaks_df : pandas.DataFrame, optional
        State×year×scenario peaks (MW). If provided, will include a ``metric="peak_mw"`` panel.

    Returns
    -------
    pandas.DataFrame
        ['year','scenario','metric','value']
    """
    def _sum_metric(df: pd.DataFrame, col: str, cumulative: bool) -> pd.DataFrame:
        years = pd.Index(range(int(df["year"].min()), int(df["year"].max()) + 1), name="year")
        def _fill(g: pd.DataFrame) -> pd.DataFrame:
            s = g.set_index("year")[[col]].reindex(years)
            s = (s.ffill() if cumulative else s.fillna(0.0)).fillna(0.0).reset_index()
            s["state_abbr"] = g["state_abbr"].iloc[0]
            s["scenario"]   = g["scenario"].iloc[0]
            return s
        filled = df.groupby(["state_abbr","scenario"], observed=True, as_index=False).apply(_fill).reset_index(drop=True)
        nat = filled.groupby(["year","scenario"], observed=True, as_index=False)[col].sum().rename(columns={col:"value"})
        return nat

    pieces: List[pd.DataFrame] = []
    totals = outputs["totals"]
    for col in ("number_of_adopters","system_kw_cum","batt_kwh_cum"):
        s = _sum_metric(totals[["state_abbr","year","scenario",col]].copy(), col, cumulative=True)
        s["metric"] = col
        pieces.append(s)

    pas = outputs["portfolio_annual_savings"]
    s = _sum_metric(pas[["state_abbr","year","scenario","portfolio_annual_savings"]].copy(),
                    "portfolio_annual_savings", cumulative=False)
    s["metric"] = "portfolio_annual_savings"
    pieces.append(s)

    cbs = outputs["cumulative_bill_savings"]
    s = _sum_metric(cbs[["state_abbr","year","scenario","cumulative_bill_savings"]].copy(),
                    "cumulative_bill_savings", cumulative=True)
    s["metric"] = "cumulative_bill_savings"
    pieces.append(s)

    if peaks_df is not None:
        s = _sum_metric(peaks_df[["state_abbr","year","scenario","peak_mw"]].copy(), "peak_mw", cumulative=False)
        s["metric"] = "peak_mw"
        pieces.append(s)

    return pd.concat(pieces, ignore_index=True)


def facet_lines_national_totals(
    outputs: Dict[str, pd.DataFrame],
    peaks_df: Optional[pd.DataFrame] = None,
    coincident_df: Optional[pd.DataFrame] = None,
    metrics: Optional[Iterable[str]] = ("number_of_adopters","system_kw_cum","batt_kwh_cum","cumulative_bill_savings","peak_mw","coincident_reduction_mw"),
    xticks: Iterable[int] = (2026, 2030, 2035, 2040),
    title: str = "U.S. Totals — Baseline vs Policy",
    ncols: int = 3,
) -> None:
    """
    Faceted national line charts by metric. Optionally includes a single-series panel
    for ``coincident_reduction_mw`` (delta series, not per‑scenario).

    Parameters
    ----------
    outputs : dict[str, pandas.DataFrame]
        Result of :func:`aggregate_state_metrics`.
    peaks_df : pandas.DataFrame, optional
        State peaks for inclusion (metric="peak_mw").
    coincident_df : pandas.DataFrame, optional
        National coincident reductions: ['year','coincident_reduction_mw'].
    metrics : iterable[str], optional
        Subset of metrics to plot.
    """
    nat = build_national_totals(outputs, peaks_df=peaks_df)

    if coincident_df is not None and not coincident_df.empty:
        co = coincident_df.groupby("year", as_index=False)["coincident_reduction_mw"].sum().rename(columns={"coincident_reduction_mw":"value"})
        co["scenario"] = "coincident Δ"
        co["metric"]   = "coincident_reduction_mw"
        nat = pd.concat([nat, co], ignore_index=True, sort=False)

    if metrics is not None:
        nat = nat[nat["metric"].isin(set(metrics))]

    nice = {
        "number_of_adopters": "Cumulative Adopters",
        "system_kw_cum": "Cumulative PV (kW)",
        "batt_kwh_cum": "Cumulative Storage (kWh)",
        "cumulative_bill_savings": "Cumulative Bill Savings ($)",
        "portfolio_annual_savings": "Portfolio Bill Savings ($/yr)",
        "peak_mw": "U.S. Peak Demand (MW)",
        "coincident_reduction_mw": "National Coincident Peak Reduction (MW)",
    }

    def _fmt(metric: str, v: float) -> str:
        if metric == "number_of_adopters": return f"{v/1e6:.1f}M"
        if metric == "system_kw_cum":       return f"{v/1e6:.1f} GW"
        if metric == "batt_kwh_cum":        return f"{v/1e6:.1f} GWh"
        if metric == "cumulative_bill_savings": return f"${v/1e9:.1f}B"
        if metric in {"peak_mw","coincident_reduction_mw"}: return f"{v/1e3:.1f} GW"
        return f"{v:.2g}"

    metric_list = list(nat["metric"].unique())
    n = len(metric_list)
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 3.4*nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for i, m in enumerate(metric_list):
        ax = axes[i]
        d = nat[nat["metric"] == m].sort_values("year")
        for scen, g in d.groupby("scenario", observed=True):
            ax.plot(g["year"], g["value"], marker="o", label=scen.capitalize())
            end = g[g["year"] == (2040 if 2040 in set(g["year"]) else int(g["year"].max()))]
            if not end.empty:
                y_end = float(end["value"].iloc[-1])
                x_end = float(end["year"].iloc[-1])
                ax.annotate(_fmt(m, y_end), xy=(x_end, y_end), xytext=(6, 0),
                            textcoords="offset points", ha="left", va="center", fontsize=10, fontweight="bold")
        ax.set_xticks(list(xticks))
        ax.set_xlabel("Year")
        ax.set_title(nice.get(m, m))
        ax.legend(frameon=False, loc="best")

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, y=1.02)
    plt.show()


# -----------------------------------------------------------------------------
# Exports
# -----------------------------------------------------------------------------

def export_compiled_results_to_excel(
    outputs: Dict[str, pd.DataFrame],
    run_id: str,
    out_dir: str,
    peaks_df: Optional[pd.DataFrame] = None,
    coincident_df: Optional[pd.DataFrame] = None,
) -> str:
    """
    Write a tidy Excel workbook with all aggregated tables and optional national series.

    Parameters
    ----------
    outputs : dict[str, pandas.DataFrame]
        Result dict from :func:`aggregate_state_metrics`.
    run_id : str
        Label to include in the filename.
    out_dir : str
        Destination directory (will be created if missing).
    peaks_df : pandas.DataFrame, optional
        State peaks table for a 'peaks' sheet.
    coincident_df : pandas.DataFrame, optional
        National coincident reductions for a 'coincident' sheet.

    Returns
    -------
    str
        Full path to the written .xlsx file.
    """
    import datetime as _dt
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.xlsx")

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as xw:
        meta = pd.DataFrame({
            "run_id": [run_id],
            "generated_at": [_dt.datetime.now().isoformat(timespec="seconds")],
            "tables_included": [", ".join(sorted([k for k, v in outputs.items() if isinstance(v, pd.DataFrame) and not v.empty]))]
        })
        meta.to_excel(xw, index=False, sheet_name="README")

        for key, df in outputs.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(xw, index=False, sheet_name=key[:31])

        if peaks_df is not None and not peaks_df.empty:
            peaks_df.to_excel(xw, index=False, sheet_name="peaks")

        if coincident_df is not None and not coincident_df.empty:
            coincident_df.to_excel(xw, index=False, sheet_name="coincident")

        nat_totals = build_national_totals(outputs, peaks_df=peaks_df)
        if coincident_df is not None and not coincident_df.empty:
            co = coincident_df.groupby("year", as_index=False)["coincident_reduction_mw"].sum().rename(columns={"coincident_reduction_mw":"value"})
            co["scenario"] = "coincident Δ"
            co["metric"]   = "coincident_reduction_mw"
            nat_totals = pd.concat([nat_totals, co], ignore_index=True)
        if not nat_totals.empty:
            nat_totals.to_excel(xw, index=False, sheet_name="national_totals")

    return out_path


# -----------------------------------------------------------------------------
# Thin wrappers for backward compatibility with older notebooks
# -----------------------------------------------------------------------------

def process_all_states(
    root_dir: str | None = None,
    run_id: str | None = None,
    cfg: SavingsConfig | None = None,
    n_jobs: int = 1,
    states: Optional[Iterable[str]] = None,
    warehouse: DataWarehouse | None = None,
) -> Dict[str, pd.DataFrame]:
    """
    Backward-compatible wrapper.
    If a ``warehouse`` is provided, reuse it; otherwise load from ``root_dir``/``run_id``.
    """
    if warehouse is None:
        assert root_dir is not None, "root_dir is required when warehouse is not provided"
        warehouse = DataWarehouse.from_disk(root_dir, run_id=run_id, states=states)
    return aggregate_state_metrics(warehouse.agents, cfg or SavingsConfig())


def process_all_states_peaks(
    root_dir: str | None = None,
    run_id: str | None = None,
    n_jobs: int = 1,
    states: Optional[Iterable[str]] = None,
    warehouse: DataWarehouse | None = None,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper for state peaks.
    """
    if warehouse is None:
        assert root_dir is not None, "root_dir is required when warehouse is not provided"
        warehouse = DataWarehouse.from_disk(root_dir, run_id=run_id, states=states)
    return compute_peaks_by_state(warehouse.state_hourly)


def compute_state_coincident_reduction_legacy(
    root_dir: str | None = None,
    run_id: str | None = None,
    states: Optional[Iterable[str]] = None,
    warehouse: DataWarehouse | None = None,
    method: str = "baseline_topN_avg",
    top_n: int = 10,
    percentile: Optional[float] = None,
) -> pd.DataFrame:
    """
    Wrapper that mirrors the old signature (no method in the original). The default here
    is top‑N averaging; set method="single_hour" to replicate the older behavior.
    """
    if warehouse is None:
        assert root_dir is not None, "root_dir is required when warehouse is not provided"
        warehouse = DataWarehouse.from_disk(root_dir, run_id=run_id, states=states)
    return compute_state_coincident_reduction(warehouse, method=method, top_n=top_n, percentile=percentile)


def compute_rto_coincident_reduction_legacy(
    root_dir: str | None = None,
    run_id: str | None = None,
    states: Optional[Iterable[str]] = None,
    warehouse: DataWarehouse | None = None,
    method: str = "baseline_topN_avg",
    top_n: int = 10,
    percentile: Optional[float] = None,
) -> pd.DataFrame:
    """
    Wrapper that mirrors the old signature (no method in the original). The default here
    is top‑N averaging; set method="single_hour" to replicate the older behavior.
    """
    if warehouse is None:
        assert root_dir is not None, "root_dir is required when warehouse is not provided"
        warehouse = DataWarehouse.from_disk(root_dir, run_id=run_id, states=states)
    return compute_rto_coincident_reduction(warehouse, method=method, top_n=top_n, percentile=percentile, return_by_rto=False)
