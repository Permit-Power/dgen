from __future__ import annotations

import os
import re
import glob
import math
import json
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd


# =============================================================================
# Discovery & I/O
# =============================================================================

# Columns expected in per-state CSVs. Missing columns are added as NaN/0 where appropriate.
# Updated to new savings arrays + battery cohort split; kept original metrics otherwise.
NEEDED_COLS = [
    "state_abbr",
    "scenario",
    "year",

    # Cohort/cumulative adopters:
    "new_adopters",                 # new adopters in that year (cohort size)
    "number_of_adopters",           # cumulative adopters by that year
    "customers_in_bin",
    "batt_adopters_added_this_year",# storage adopters in that cohort (may be 0/missing)

    # Savings arrays (25-year, from agent_finance_series):
    "cf_energy_value_pv_only",
    "cf_energy_value_pv_batt",

    # Utility bill arrays (25-year, from agent_finance_series):
    "utility_bill_w_sys_pv_only",
    "utility_bill_w_sys_pv_batt",
    "utility_bill_wo_sys_pv_only",
    "utility_bill_wo_sys_pv_batt",

    # Pricing:
    "price_per_kwh",
    "load_kwh_per_customer_in_bin_initial",

    # Tech potential / market share:
    "customers_in_bin",
    "max_market_share",

    # PV / storage:
    "system_kw",              # per-adopter PV size (kW) — used for medians
    "new_system_kw",          # cohort total PV capacity (kW)
    "system_kw_cum",
    "batt_kwh",               # per-agent storage size for median calc (if absent, we add NaN)
    "batt_kwh_cum",
]

# Optional columns we will read if present (for initial stock)
OPTIONAL_COLS = ["initial_batt_kwh", "initial_number_of_adopters"]

SCHEMA_RE = re.compile(r"^diffusion_results_(baseline|policy)_([a-z]{2})_", re.IGNORECASE)


# =============================================================================
# Hourly array helpers
# =============================================================================

def _parse_array_text_to_floats(s) -> List[float]:
    """
    Accept Postgres array text like '{1,2,3}' or JSON-like '[1,2,3]' or a Python list/tuple/ndarray
    and return [1.0, 2.0, 3.0]. Returns [] on any parse issue.
    """
    if s is None:
        return []
    if isinstance(s, (list, tuple, np.ndarray)):
        try:
            return [float(v) for v in s]
        except Exception:
            return []
    if not isinstance(s, str) or not s:
        return []
    t = s.strip()
    try_json = t.replace("{", "[").replace("}", "]")
    try:
        vals = json.loads(try_json)
        return [float(v) for v in vals] if isinstance(vals, list) else []
    except Exception:
        if t.startswith("{") and t.endswith("}"):
            inner = t[1:-1]
            if not inner:
                return []
            parts = inner.split(",")
            out = []
            for p in parts:
                p = p.strip()
                try:
                    out.append(float(p))
                except Exception:
                    return []
            return out
    return []


# =============================================================================
# Standalone hourly → daily/monthly plotter (Baseline vs Policy), with GW annotations
# =============================================================================

def _load_hourly_one(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "scenario" not in df.columns:
        fname = os.path.basename(path).lower()
        df["scenario"] = "policy" if "policy" in fname else ("baseline" if "baseline" in fname else "unknown")
    if "state_abbr" not in df.columns:
        df["state_abbr"] = os.path.basename(os.path.dirname(path)).upper()
    if "net_sum_text" not in df.columns:
        raise ValueError(f"'net_sum_text' column not found in {path}")
    df["net_load"] = df["net_sum_text"].apply(_parse_array_text_to_floats)
    return df


def _pick_year_row(df: pd.DataFrame, scenario: str, year: int) -> Optional[pd.Series]:
    sdf = df[(df["scenario"].str.lower() == scenario.lower()) & (df["year"] == year)].copy()
    if sdf.empty:
        return None
    exact = sdf[sdf["net_load"].apply(lambda a: isinstance(a, list) and len(a) > 0)]
    return (exact.iloc[0] if not exact.empty else None)


def _to_time_series(arr: List[float], year: int) -> pd.Series:
    n = len(arr)
    idx = pd.date_range(start=f"{year}-01-01 00:00:00", periods=n, freq="h")
    return pd.Series(arr, index=idx)


def _aggregate_series(s: pd.Series, aggregation: str = "hourly", agg_func: str = "mean") -> pd.Series:
    aggregation = aggregation.lower()
    agg_func = agg_func.lower()
    if aggregation == "hourly":
        return s
    rule = {"daily": "D", "weekly": "W", "monthly": "M"}.get(aggregation)
    if rule is None:
        raise ValueError("aggregation must be one of: 'hourly', 'daily', 'weekly', 'monthly'")
    if agg_func not in {"mean", "sum", "max", "min"}:
        raise ValueError("agg_func must be one of: 'mean', 'sum', 'max', 'min'")
    return getattr(s.resample(rule), agg_func)()


def plot_state_netload(
    state_dir: Optional[str] = None,
    baseline_csv: Optional[str] = None,
    policy_csv: Optional[str] = None,
    year: int = 2040,
    aggregation: str = "daily",
    agg_func: str = "max",
    title: Optional[str] = None,
) -> None:
    # unchanged logic
    if state_dir:
        b_matches = sorted(glob.glob(os.path.join(state_dir, "hourly_baseline*.csv")))
        p_matches = sorted(glob.glob(os.path.join(state_dir, "hourly_policy*.csv")))
        frames: List[pd.DataFrame] = []
        if b_matches: frames.append(_load_hourly_one(b_matches[0]))
        if p_matches: frames.append(_load_hourly_one(p_matches[0]))
        if not frames:
            any_matches = sorted(glob.glob(os.path.join(state_dir, "hourly*.csv")))
            if not any_matches:
                raise FileNotFoundError("No hourly_* CSVs found in state_dir.")
            frames = [_load_hourly_one(p) for p in any_matches]
        df = pd.concat(frames, ignore_index=True)
    else:
        if not (baseline_csv and policy_csv):
            raise ValueError("Provide either state_dir OR both baseline_csv and policy_csv.")
        df = pd.concat([_load_hourly_one(baseline_csv), _load_hourly_one(policy_csv)], ignore_index=True)

    if "year" not in df.columns:
        raise ValueError("'year' column is required in hourly CSVs.")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    brow = _pick_year_row(df, "baseline", year)
    prow = _pick_year_row(df, "policy", year)
    if brow is None:
        raise ValueError(f"No baseline record found for year {year}.")
    if prow is None:
        raise ValueError(f"No policy record found for year {year}.")

    s_base = _to_time_series(brow["net_load"], int(brow["year"]))
    s_poli = _to_time_series(prow["net_load"], int(prow["year"]))

    s_base_agg = _aggregate_series(s_base, aggregation=aggregation, agg_func=agg_func)
    s_poli_agg = _aggregate_series(s_poli, aggregation=aggregation, agg_func=agg_func)

    if title is None:
        nice_agg = aggregation.capitalize()
        nice_func = agg_func.upper() if aggregation != "hourly" else ""
        title = f"{nice_agg} Net Load — Baseline vs Policy ({year})" + (f" [{nice_func}]" if nice_func else "")

    plt.figure(figsize=(14, 6))
    plt.plot(s_base_agg.index, s_base_agg.values, label="Baseline", alpha=0.9)
    plt.plot(s_poli_agg.index, s_poli_agg.values, label="Policy", alpha=0.9)

    plt.xlabel("Time")
    plt.ylabel("Net Load (MW)")
    plt.title(title)
    plt.legend()

    base_max = float(s_base_agg.max())
    poli_max = float(s_poli_agg.max())
    base_date = s_base_agg.idxmax()
    poli_date = s_poli_agg.idxmax()

    plt.annotate(f"{base_max/1000:.1f} GW (baseline)",
                 xy=(base_date, base_max),
                 xytext=(0, 8),
                 textcoords="offset points",
                 ha="center", color="black", fontweight="bold")

    plt.annotate(f"{poli_max/1000:.1f} GW (policy)",
                 xy=(poli_date, poli_max),
                 xytext=(0, -12),
                 textcoords="offset points",
                 ha="center", color="black", fontweight="bold")

    plt.tight_layout()
    plt.show()


# =============================================================================
# Hourly → state/year peaks (MW) from CSVs (used by multiple plots)
# =============================================================================

def find_state_hourly_files(state_dir: str, run_id: Optional[str] = None) -> List[str]:
    paths: List[str] = []
    # NEW: prefer <STATE>/<RUN_ID>/baseline_hourly.csv & policy_hourly.csv
    if run_id:
        sub = os.path.join(state_dir, run_id)
        if os.path.isdir(sub):
            c1 = os.path.join(sub, "baseline_hourly.csv")
            c2 = os.path.join(sub, "policy_hourly.csv")
            if os.path.exists(c1): paths.append(c1)
            if os.path.exists(c2): paths.append(c2)
            if not paths:
                paths += glob.glob(os.path.join(sub, "hourly_baseline*.csv"))
                paths += glob.glob(os.path.join(sub, "hourly_policy*.csv"))
                if not paths:
                    paths += glob.glob(os.path.join(sub, "hourly*.csv"))
        # legacy file names at state root with run_id token
        if not paths:
            paths += glob.glob(os.path.join(state_dir, f"hourly_baseline_{run_id}.csv"))
            paths += glob.glob(os.path.join(state_dir, f"hourly_policy_{run_id}.csv"))
            if not paths:
                paths += glob.glob(os.path.join(state_dir, f"hourly_{run_id}.csv"))
    # legacy fallback
    if not paths:
        paths += glob.glob(os.path.join(state_dir, "hourly_baseline*.csv"))
        paths += glob.glob(os.path.join(state_dir, "hourly_policy*.csv"))
        if not paths:
            paths += glob.glob(os.path.join(state_dir, "hourly*.csv"))
    # de-dup while preserving order
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

def find_state_rto_hourly_files(state_dir: str, run_id: Optional[str] = None) -> List[str]:
    """
    Locate per-state RTO hourly CSVs written by schema_exporter:
      <STATE>/<RUN_ID>/baseline_rto_hourly.csv
      <STATE>/<RUN_ID>/policy_rto_hourly.csv
    Falls back to any '*rto_hourly*.csv' if exact names aren’t found.
    """
    paths: List[str] = []
    if run_id:
        sub = os.path.join(state_dir, str(run_id))
        if os.path.isdir(sub):
            c1 = os.path.join(sub, "baseline_rto_hourly.csv")
            c2 = os.path.join(sub, "policy_rto_hourly.csv")
            if os.path.exists(c1): paths.append(c1)
            if os.path.exists(c2): paths.append(c2)
            if not paths:
                paths += glob.glob(os.path.join(sub, "*rto_hourly*.csv"))
    if not paths:
        paths += glob.glob(os.path.join(state_dir, "*rto_hourly*.csv"))
    # de-dup while preserving order
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p); out.append(p)
    return out


def compute_rto_coincident_reduction(
    root_dir: str,
    run_id: Optional[str] = None,
    states: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    For each YEAR and each RTO:
      1) Sum hourly net load across ALL states contributing to that RTO (baseline, policy separately).
      2) Find the *baseline* peak hour index for that RTO in that year.
      3) Compute reduction at that same hour: baseline_sum[idx] - policy_sum[idx].
    Then SUM across all RTOs to a national coincident reduction series.

    Returns: ['year','coincident_reduction_mw']  (national total by summing all RTOs)
    """
    state_dirs = discover_state_dirs(root_dir)
    if states:
        wanted = {s.strip().upper() for s in states if s and s.strip()}
        state_dirs = [sd for sd in state_dirs if os.path.basename(sd).upper() in wanted]
    if not state_dirs:
        return pd.DataFrame(columns=["year","coincident_reduction_mw"])

    # Collect all rows from all states
    frames: List[pd.DataFrame] = []
    for sd in state_dirs:
        for pth in find_state_rto_hourly_files(sd, run_id=run_id):
            if not os.path.exists(pth) or os.path.getsize(pth) == 0:
                print("path not found")
                continue
            try:
                df = pd.read_csv(pth, usecols=["scenario","rto","year","net_sum_text"])
            except Exception:
                continue
            if "scenario" not in df.columns:
                fn = os.path.basename(pth).lower()
                df["scenario"] = "policy" if "policy" in fn else ("baseline" if "baseline" in fn else np.nan)
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["year","coincident_reduction_mw"])

    x = pd.concat(frames, ignore_index=True)
    x = x.dropna(subset=["scenario","rto","year","net_sum_text"]).copy()
    x["scenario"] = x["scenario"].astype(str).str.lower().str.strip()
    x["rto"] = x["rto"].astype(str)
    x["year"] = pd.to_numeric(x["year"], errors="coerce")
    x = x[x["scenario"].isin(["baseline","policy"]) & x["year"].notna()]

    # Parse arrays
    x["arr"] = x["net_sum_text"].apply(lambda s: _parse_array_text_to_floats(str(s)))
    x = x[x["arr"].apply(lambda a: isinstance(a, list) and len(a) > 0)]

    # Sum across states -> RTO × scenario × year series
    # (pad unequal lengths just in case, though they should match)
    def _sum_arrays(arrs: List[List[float]]) -> List[float]:
        L = max(len(a) for a in arrs)
        out = np.zeros(L, dtype=float)
        for a in arrs:
            if len(a) == L:
                out += np.array(a, dtype=float)
            else:
                b = np.zeros(L, dtype=float)
                b[:len(a)] = np.array(a, dtype=float)
                out += b
        return out.tolist()

    rto_sums = (
        x.groupby(["rto","scenario","year"], observed=True)["arr"]
         .apply(lambda s: _sum_arrays(list(s.values)))
         .reset_index()
    )

    # For each RTO×year, compute coincident reduction at baseline peak hour
    rows = []
    for (rto, y), g in rto_sums.groupby(["rto","year"], observed=True):
        g = {r["scenario"]: r["arr"] for _, r in g.iterrows()}
        if "baseline" not in g or "policy" not in g:
            continue
        base, pol = g["baseline"], g["policy"]
        if not base or not pol:
            continue
        idx = int(np.argmax(base))
        if idx < len(pol):
            red = float(base[idx]) - float(pol[idx])
            rows.append((str(rto), int(y), red))

    if not rows:
        return pd.DataFrame(columns=["year","coincident_reduction_mw"])

    rto_co = pd.DataFrame(rows, columns=["rto","year","coincident_reduction_mw"])

    # National = sum across RTOs
    nat = (rto_co.groupby("year", as_index=False)["coincident_reduction_mw"].sum())
    return nat.sort_values("year").reset_index(drop=True)



def load_state_peaks_df(state_dir: str, run_id: Optional[str] = None) -> pd.DataFrame:
    paths = find_state_hourly_files(state_dir, run_id)
    if not paths:
        return pd.DataFrame(columns=["state_abbr", "scenario", "year", "peak_mw"])

    frames: List[pd.DataFrame] = []
    for path in paths:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if "scenario" not in df.columns:
            fname = os.path.basename(path).lower()
            df["scenario"] = "policy" if "policy" in fname else ("baseline" if "baseline" in fname else np.nan)
        if "state_abbr" not in df.columns:
            df["state_abbr"] = os.path.basename(os.path.dirname(os.path.dirname(path) if os.path.basename(os.path.dirname(path)).lower() in {"policy", "baseline"} else os.path.dirname(path))).upper()
            # if using <STATE>/<RUN_ID>/..., dirname(path) is the run_id folder; its parent is STATE

        needed = {"state_abbr", "scenario", "year", "net_sum_text"}
        if not needed.issubset(df.columns):
            continue

        def _peak(row) -> float:
            arr = _parse_array_text_to_floats(str(row.get("net_sum_text", "")))
            return float(np.max(arr)) if arr else np.nan

        d = df.copy()
        d["peak_mw"] = d.apply(_peak, axis=1)
        d = d[["state_abbr", "scenario", "year", "peak_mw"]].dropna(subset=["peak_mw"])
        d["year"] = pd.to_numeric(d["year"], errors="coerce")
        d = d.dropna(subset=["year"]).reset_index(drop=True)
        frames.append(d)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["state_abbr", "scenario", "year", "peak_mw"])


def process_all_states_peaks(
    root_dir: str,
    run_id: Optional[str] = None,
    n_jobs: int = 1,
    states: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    state_dirs = discover_state_dirs(root_dir)
    if states:
        wanted = {s.strip().upper() for s in states if s and s.strip()}
        state_dirs = [sd for sd in state_dirs if os.path.basename(sd).upper() in wanted]
    if not state_dirs:
        return pd.DataFrame(columns=["state_abbr","scenario","year","peak_mw"])

    tasks = [(sd, run_id) for sd in state_dirs]
    results: List[pd.DataFrame] = []
    n_jobs = max(1, min(n_jobs, max(1, (cpu_count() or 2) - 1)))
    if n_jobs == 1:
        results = [load_state_peaks_df(sd, run_id) for sd, _ in tasks]
    else:
        with Pool(processes=n_jobs) as pool:
            for df in pool.starmap(load_state_peaks_df, tasks):
                results.append(df)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["state_abbr","scenario","year","peak_mw"])


# =============================================================================
# Facet helpers (unchanged visuals)
# =============================================================================

def _build_us_peak_delta(peaks_df: pd.DataFrame) -> pd.DataFrame:
    if peaks_df.empty:
        return pd.DataFrame(columns=["state_abbr","scenario","year","peak_mw"])
    nat = (peaks_df.groupby(["year","scenario"], as_index=False)["peak_mw"].sum())
    piv = nat.pivot(index="year", columns="scenario", values="peak_mw")
    if "policy" not in piv.columns or "baseline" not in piv.columns:
        return pd.DataFrame(columns=["state_abbr","scenario","year","peak_mw"])
    d = (piv["policy"] - piv["baseline"]).rename("peak_mw").reset_index()
    d["state_abbr"] = "US Δ"
    d["scenario"] = "delta"
    return d[["state_abbr","scenario","year","peak_mw"]]


def facet_peaks_by_state(
    df: pd.DataFrame,
    ylabel: str = "Peak Demand (MW)",
    title: str = "Peak Demand by Year — Baseline vs Policy",
    xticks: Iterable[int] = (2026, 2030, 2035, 2040),
    height: float = 3.5,
    col_wrap: int = 4,
    sharey: bool = False,
    include_us_delta: bool = False,
) -> None:
    if df.empty:
        return
    if include_us_delta:
        us_delta = _build_us_peak_delta(df)
        if not us_delta.empty:
            df = pd.concat([df, us_delta], ignore_index=True)

    _df = df.rename(columns={"peak_mw":"value"}).copy()
    sns.set_context("talk", rc={"lines.linewidth": 2})
    g = sns.FacetGrid(_df, col="state_abbr", col_wrap=col_wrap, height=height, sharey=sharey)
    g.map_dataframe(sns.lineplot, x="year", y="value", hue="scenario", marker="o")
    g.set_titles("{col_name}")
    g.set_axis_labels("Year", ylabel)
    g.set(xticks=list(xticks))
    g.add_legend()
    g.fig.suptitle(title, y=1.02)

    years = pd.to_numeric(_df["year"], errors="coerce").dropna().astype(int)
    preferred_year = 2040 if (len(years) and 2040 in set(years)) else (int(years.max()) if len(years) else None)
    if preferred_year is not None:
        for state, ax in g.axes_dict.items():
            sdf = _df[_df["state_abbr"] == state].dropna(subset=["value","year"])
            if sdf.empty:
                continue
            years_s = sdf["year"].astype(int).unique()
            end_year = preferred_year if preferred_year in years_s else int(sdf["year"].max())
            end_points = []
            for scen, sg in sdf.groupby("scenario"):
                sg = sg.sort_values("year")
                g_end = sg[sg["year"] == end_year]
                if g_end.empty:
                    g_end = sg[sg["year"] <= end_year].tail(1)
                if not g_end.empty:
                    end_points.append((scen, float(g_end["year"].iloc[-1]), float(g_end["value"].iloc[-1])))
            if not end_points:
                continue
            vals = np.array([p[2] for p in end_points])
            yrange = max(1.0, float(vals.max() - vals.min()) or 1.0)
            sep = yrange * 0.01
            used = []
            for scen, x_end, y_end in sorted(end_points, key=lambda t: t[2]):
                offset = 0
                for u in used:
                    if abs(y_end - u) < sep:
                        offset += 6
                used.append(y_end)
                label = f"{y_end:.1f} MW" if state != "US Δ" else f"{y_end:.1f} MW (Δ)"
                ax.annotate(label, xy=(x_end, y_end),
                            xytext=(6, offset), textcoords="offset points",
                            ha="left", va="center", fontsize=9, fontweight="bold")
    plt.show()


# =============================================================================
# Facet — state-level daily/weekly peak time series (one year)
# (unchanged interface/plot; only file discovery already updated above)
# =============================================================================

def facet_state_peak_timeseries_from_hourly(
    root_dir: str,
    run_id: Optional[str] = None,
    year: int = 2040,
    height: float = 2.8,
    col_wrap: int = 4,
    sharey: bool = False,
    title: Optional[str] = None,
    states: Optional[Iterable[str]] = None,
) -> None:
    state_dirs = discover_state_dirs(root_dir)
    if states:
        wanted = {s.strip().upper() for s in states if s and s.strip()}
        state_dirs = [sd for sd in state_dirs if os.path.basename(sd).upper() in wanted]
    if not state_dirs:
        return

    def _load_state_hourly_two(sd: str) -> Optional[pd.DataFrame]:
        paths = find_state_hourly_files(sd, run_id)
        if not paths:
            return None
        frames = []
        for pth in paths:
            try:
                df = pd.read_csv(pth)
            except Exception:
                continue
            if "scenario" not in df.columns:
                fn = os.path.basename(pth).lower()
                df["scenario"] = "policy" if "policy" in fn else ("baseline" if "baseline" in fn else np.nan)
            if "state_abbr" not in df.columns:
                # if using <STATE>/<RUN_ID>/..., dirname(dirname(path)) is the state folder
                state_guess = os.path.basename(os.path.dirname(os.path.dirname(pth)))
                df["state_abbr"] = state_guess.upper()
            frames.append(df)
        if not frames:
            return None
        df = pd.concat(frames, ignore_index=True)

        needed = {"state_abbr", "scenario", "year", "net_sum_text"}
        if not needed.issubset(df.columns):
            return None

        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df[df["year"] == year]

        def _arr_ok(x):
            arr = _parse_array_text_to_floats(str(x))
            return (len(arr) > 0, arr)

        if df.empty:
            return None
        df["_ok"], df["_arr"] = zip(*df["net_sum_text"].map(_arr_ok))
        df = df[df["_ok"]]
        if df.empty:
            return None

        out_rows = []
        for scen in ("baseline", "policy"):
            g = df[df["scenario"].str.lower() == scen]
            if not g.empty:
                r = g.iloc[0]
                out_rows.append({
                    "state_abbr": r["state_abbr"],
                    "scenario": scen,
                    "year": int(year),
                    "net_load": r["_arr"],
                })
        return pd.DataFrame(out_rows) if out_rows else None

    rows = []
    for sd in state_dirs:
        d = _load_state_hourly_two(sd)
        if d is not None:
            rows.append(d)
    if not rows:
        return
    df_pairs = pd.concat(rows, ignore_index=True)

    def _weekly_max(arr: List[float]) -> pd.Series:
        # Build an index confined strictly to the target year (handles leap years)
        idx = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:00:00", freq="h")
        # If the hourly array is longer/shorter than expected, trim/pad to the year window
        a = np.asarray(arr, dtype=float)
        if len(a) >= len(idx):
            a = a[:len(idx)]
        else:
            # pad with NaNs so .max() ignores the padding
            pad = np.full(len(idx) - len(a), np.nan)
            a = np.concatenate([a, pad])

        s = pd.Series(a, index=idx)

        # Anchor week bins to a specific weekday and keep bins that start inside the year.
        # Using W-MON avoids the ISO spillover issues at year end.
        wk = s.resample("W-MON", label="left", closed="left").max()

        # Keep only weeks whose bin start is within the year
        wk = wk[wk.index.year == int(year)]

        # Use sequential week numbers 1..N for a clean x-axis
        wk.index = pd.Index(np.arange(1, len(wk) + 1), name="week")
        return wk


    pieces = []
    for (state, scen), g in df_pairs.groupby(["state_abbr", "scenario"]):
        ts = _weekly_max(g["net_load"].iloc[0])
        pieces.append(pd.DataFrame({
            "state_abbr": state,
            "scenario": scen,
            "week": ts.index.astype(int),
            "peak_mw": ts.values,
        }))
    d = pd.concat(pieces, ignore_index=True)

    sns.set_context("talk", rc={"lines.linewidth": 1.6})
    states_order = sorted(d["state_abbr"].unique())
    n = len(states_order)
    ncols = col_wrap
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6*ncols, height*nrows), sharex=True, sharey=sharey)
    axes = np.atleast_1d(axes).ravel()

    pal = sns.color_palette()
    color_baseline = pal[0]
    color_policy   = pal[1]
    zorders = {"policy": 2, "baseline": 3}

    for idx, (ax, state) in enumerate(zip(axes, states_order)):
        sub = d[d["state_abbr"] == state]
        pol = sub[sub["scenario"] == "policy"].sort_values("week")
        bas = sub[sub["scenario"] == "baseline"].sort_values("week")

        if not pol.empty:
            ax.plot(pol["week"], pol["peak_mw"], color=color_policy, linewidth=2.0, label="policy", zorder=zorders["policy"])
        if not bas.empty:
            ax.plot(bas["week"], bas["peak_mw"], color=color_baseline, linewidth=2.0, label="baseline", zorder=zorders["baseline"])

        ax.set_title(state, fontsize=10, pad=8)
        ax.set_xlim(1, 53)
        ax.set_xticks([1, 25, 52])
        if idx % ncols == 0:
            ax.set_ylabel("Weekly Peak (MW)")
        else:
            ax.set_ylabel(None)

        if idx >= 44:
            ax.set_xlabel("Week of Year")

        ax.grid(True, axis="y", alpha=0.25)

        base_peak = bas["peak_mw"].max() if not bas.empty else np.nan
        pol_peak  = pol["peak_mw"].max() if not pol.empty else np.nan
        lines = []
        if np.isfinite(base_peak):
            lines.append(f"baseline: {base_peak/1000:.1f} GW")
        if np.isfinite(pol_peak):
            lines.append(f"policy:   {pol_peak/1000:.1f} GW")
        if lines:
            ax.text(
                0.98, 0.98, "\n".join(lines),
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8, color="black",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, linewidth=0.0),
                zorder=10,
            )

    for ax in axes[len(states_order):]:
        ax.set_visible(False)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=color_baseline, linewidth=2, label="baseline"),
        Line2D([0], [0], color=color_policy,   linewidth=2, label="policy"),
    ]
    fig.legend(handles=handles, loc="lower right", ncol=2, frameon=False)

    fig.suptitle(title or f"Weekly Peak Net Load — Baseline vs Policy ({year})", y=0.995, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.show()


# =============================================================================
# Savings & Aggregations (UPDATED to use 25-yr arrays + battery split)
# =============================================================================

@dataclass
class SavingsConfig:
    lifetime_years: int = 25
    cap_to_horizon: bool = False


def _arr25(x) -> List[float]:
    """Parse and clip/pad to length 25."""
    a = _parse_array_text_to_floats(x)
    a = [float(v) for v in a][:25]
    if len(a) < 25:
        a = a + [0.0] * (25 - len(a))
    return a


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
    return annual, cumulative


def aggregate_state_metrics(df: pd.DataFrame, cfg: SavingsConfig) -> Dict[str, pd.DataFrame]:
    """
    Aggregate per-state metrics for plotting and exports.
    """
    if df.empty:
        return {
            "median_system_kw": pd.DataFrame(),
            "median_storage_kwh": pd.DataFrame(),
            "totals": pd.DataFrame(),
            "tech_2040": pd.DataFrame(),
            "portfolio_annual_savings": pd.DataFrame(),
            "cumulative_bill_savings": pd.DataFrame(),
            "lifetime_totals": pd.DataFrame(),
            "avg_price_2026_model": pd.DataFrame(),
            "market_share_reached": pd.DataFrame(),
        }

    x = df.copy()
    x["state_abbr"] = x.get("state_abbr", "").astype(str).str.strip().str.upper()

    # --- numeric coercions / fills ---
    for c in ("year","new_adopters","number_of_adopters",
              "customers_in_bin","max_market_share","system_kw","new_system_kw",
              "system_kw_cum","batt_kwh","batt_kwh_cum","price_per_kwh",
              "initial_batt_kwh","batt_adopters_added_this_year","load_kwh_per_customer_in_bin_initial"):
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")

    defaults = {
        "new_adopters": 0.0, "number_of_adopters": 0.0,
        "customers_in_bin": 0.0, "max_market_share": 0.0, "system_kw": 0.0,
        "new_system_kw": 0.0, "system_kw_cum": 0.0, "batt_kwh_cum": 0.0,
        "initial_batt_kwh": 0.0, "batt_adopters_added_this_year": 0.0,
        "load_kwh_per_customer_in_bin_initial": 0.0,
    }
    for c, v in defaults.items():
        if c in x.columns:
            x[c] = x[c].fillna(v)

    # --------------------------
    # MEDIANS (weighted by adopters; adopters-only)
    # --------------------------
    def _weighted_median(values: pd.Series, weights: pd.Series) -> float:
        v = pd.to_numeric(values, errors="coerce")
        w = pd.to_numeric(weights, errors="coerce").fillna(0).clip(lower=0)
        mask = v.notna() & (w > 0)
        if not mask.any():
            return np.nan
        v = v[mask].to_numpy()
        w = w[mask].to_numpy()
        order = np.argsort(v)
        v = v[order]
        w = w[order]
        cw = np.cumsum(w)
        cutoff = 0.5 * w.sum()
        idx = int(np.searchsorted(cw, cutoff, side="left"))
        return float(v[min(idx, len(v) - 1)])

    adopt = x.copy()
    adopt["new_adopters"] = pd.to_numeric(adopt.get("new_adopters", 0.0), errors="coerce").fillna(0.0)
    adopt = adopt[adopt["new_adopters"] > 0]

    if "system_kw" in adopt.columns and not adopt.empty:
        median_kw = (
            adopt.groupby(["state_abbr", "year", "scenario"], observed=True)
                 .apply(lambda g: g["system_kw_cum"].sum()/g["number_of_adopters"].sum())
                 .reset_index(name="median_system_kw")
        )
    else:
        median_kw = pd.DataFrame(columns=["state_abbr", "year", "scenario", "median_system_kw"])

    if "batt_kwh" in adopt.columns:
        has_storage = adopt.copy()
        has_storage["batt_kwh"] = pd.to_numeric(has_storage["batt_kwh"], errors="coerce")
        has_storage = has_storage[has_storage["batt_kwh"] > 0]
        if not has_storage.empty:
            median_storage = (
                has_storage.groupby(["state_abbr", "year", "scenario"], observed=True)
                           .apply(lambda g: _weighted_median(g["batt_kwh"], g["new_adopters"]))
                           .reset_index(name="median_batt_kwh")
            )
        else:
            median_storage = pd.DataFrame(columns=["state_abbr", "year", "scenario", "median_batt_kwh"])
    else:
        median_storage = pd.DataFrame(columns=["state_abbr", "year", "scenario", "median_batt_kwh"])

    # --- totals ---
    totals = (
        x.groupby(["state_abbr","year","scenario"], as_index=False)
         .agg(
             new_adopters=("new_adopters","sum"),
             new_system_kw=("new_system_kw","sum"),
             number_of_adopters=("number_of_adopters","sum"),
             system_kw_cum=("system_kw_cum","sum"),
             batt_kwh_cum=("batt_kwh_cum","sum"),
         )
    )

    # --- tech potential ---
    tech_2040_src = x.loc[x["year"] == 2040, ["state_abbr","scenario","number_of_adopters","customers_in_bin"]]
    tech_2040 = tech_2040_src.groupby(["state_abbr","scenario"], as_index=False).sum()
    if not tech_2040.empty:
        tech_2040["percent_tech_potential"] = np.where(
            tech_2040["customers_in_bin"] > 0,
            100.0 * tech_2040["number_of_adopters"] / tech_2040["customers_in_bin"],
            np.nan,
        )

    # --- savings (new arrays) ---
    portfolio_annual, cumulative_savings = compute_portfolio_and_cumulative_savings(x, cfg)
    lifetime_totals = (
        portfolio_annual[["state_abbr","scenario","lifetime_savings_total"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # --- avg price 2026 (unchanged) ---
    if (
        "price_per_kwh" in x.columns
        and "customers_in_bin" in x.columns
        and x["price_per_kwh"].notna().any()
    ):
        price_2026 = x[(x["year"] == 2026) & (x["scenario"] == "baseline")].copy()
        price_2026["customers_in_bin"] = price_2026["customers_in_bin"].fillna(0.0).clip(lower=0.0)

        def _weighted_avg(g: pd.DataFrame) -> float:
            w = g["customers_in_bin"].to_numpy()
            v = g["price_per_kwh"].to_numpy()
            ws = w.sum()
            return float(np.average(v, weights=w)) if ws > 0 else np.nan

        avg_price_2026_model = (
            price_2026.groupby("state_abbr", as_index=False)
                      .apply(_weighted_avg)
                      .rename(columns={None: "price_per_kwh"})
        )
    else:
        avg_price_2026_model = pd.DataFrame(columns=["state_abbr","price_per_kwh"])

    # --- market share (unchanged) ---
    x["market_potential"] = x["customers_in_bin"] * x["max_market_share"]
    market_share = (
        x.groupby(["state_abbr","year","scenario"], as_index=False)
         .agg(
             market_potential=("market_potential","sum"),
             market_reached=("number_of_adopters","sum"),
         )
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


# =============================================================================
# Parallel processing across states (UPDATED discovery to support subfolders)
# =============================================================================

def discover_state_dirs(root_dir: str) -> List[str]:
    return sorted(
        os.path.join(root_dir, d)
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )


def find_state_files(state_dir: str, run_id: Optional[str] = None, strict_run_id: bool = True) -> Tuple[Optional[str], Optional[str]]:
    """
    Locate per-state CSVs for baseline and policy.

    New preferred layout:
        <STATE>/<RUN_ID>/baseline.csv
        <STATE>/<RUN_ID>/policy.csv

    Legacy fallbacks preserved.
    """
    if run_id:
        sub = os.path.join(state_dir, run_id)
        if os.path.isdir(sub):
            b = os.path.join(sub, "baseline.csv")
            p = os.path.join(sub, "policy.csv")
            return (b if os.path.exists(b) else None), (p if os.path.exists(p) else None)
        if strict_run_id:
            b = glob.glob(os.path.join(state_dir, f"baseline_{run_id}.csv"))
            p = glob.glob(os.path.join(state_dir, f"policy_{run_id}.csv"))
            return (b[0] if b else None), (p[0] if p else None)
        else:
            b = glob.glob(os.path.join(state_dir, f"baseline_{run_id}.csv"))
            p = glob.glob(os.path.join(state_dir, f"policy_{run_id}.csv"))
            if b and p:
                return b[0], p[0]

    # Fallbacks (legacy)
    b = glob.glob(os.path.join(state_dir, "baseline*.csv"))
    p = glob.glob(os.path.join(state_dir, "policy*.csv"))
    return (b[0] if b else None), (p[0] if p else None)


def _read_with_selected_cols(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=NEEDED_COLS)
    try:
        header = pd.read_csv(path, nrows=0)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=NEEDED_COLS)

    core = [c for c in NEEDED_COLS if c in header.columns]
    opt  = [c for c in OPTIONAL_COLS if c in header.columns]
    usecols = core + opt

    try:
        df = pd.read_csv(path, usecols=usecols) if usecols else pd.DataFrame(columns=NEEDED_COLS)
    except (pd.errors.EmptyDataError, ValueError):
        return pd.DataFrame(columns=NEEDED_COLS)

    for c in NEEDED_COLS:
        if c not in df.columns:
            df[c] = np.nan

    filename = os.path.basename(path).lower()
    if df["scenario"].isna().all():
        df["scenario"] = "baseline" if "baseline" in filename else ("policy" if "policy" in filename else np.nan)

    if df["state_abbr"].isna().any():
        # if path is <STATE>/<RUN_ID>/<file>.csv, parent of parent is the state folder
        state = os.path.basename(os.path.dirname(os.path.dirname(path))) if os.path.basename(os.path.dirname(path)) in {"policy", "baseline"} else os.path.basename(os.path.dirname(os.path.dirname(path)))
        # safer: fallback to immediate parent if above logic misfires
        if not state or len(state) != 2:
            state = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if not state or len(state) != 2:
            state = os.path.basename(os.path.dirname(path))
        df.loc[df["state_abbr"].isna(), "state_abbr"] = state.upper()

    return df


def _process_one_state(args) -> Dict[str, pd.DataFrame]:
    state_dir, run_id, cfg_payload = args

    # Rebuild SavingsConfig from a small, pickle-safe payload
    if isinstance(cfg_payload, dict):
        cfg = SavingsConfig(
            lifetime_years=int(cfg_payload.get("lifetime_years", 25)),
            cap_to_horizon=bool(cfg_payload.get("cap_to_horizon", False)),
        )
    df = pd.DataFrame(columns=NEEDED_COLS)
    b_csv, p_csv = find_state_files(state_dir, run_id)
    parts: List[pd.DataFrame] = []
    for p in (b_csv, p_csv):
        if p:
            parts.append(_read_with_selected_cols(p))
    if parts:
        df = pd.concat(parts, ignore_index=True)
        for col in ("year", "new_adopters", "number_of_adopters",
                    "system_kw", "system_kw_cum", "batt_kwh", "batt_kwh_cum",
                    "customers_in_bin", "max_market_share",
                    "avg_elec_price_cents_per_kwh",
                    "initial_batt_kwh", "initial_number_of_adopters",
                    "batt_adopters_added_this_year", "load_kwh_per_customer_in_bin_initial"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("new_adopters", "number_of_adopters",
                    "customers_in_bin", "max_market_share",
                    "system_kw_cum", "batt_kwh_cum",
                    "batt_adopters_added_this_year", "load_kwh_per_customer_in_bin_initial"):
            if col in df.columns:
                df[col] = df[col].fillna(0.0)
    return aggregate_state_metrics(df, cfg)


def process_all_states(
    root_dir: str,
    run_id: Optional[str] = None,
    cfg: Optional[SavingsConfig] = None,
    n_jobs: int = 1,
    states: Optional[Iterable[str]] = None,
) -> Dict[str, pd.DataFrame]:
    state_dirs = discover_state_dirs(root_dir)
    if states:
        wanted = {s.strip().upper() for s in states if s and s.strip()}
        state_dirs = [sd for sd in state_dirs if os.path.basename(sd).upper() in wanted]
    if not state_dirs:
        return {
            "median_system_kw": pd.DataFrame(),
            "median_storage_kwh": pd.DataFrame(),
            "totals": pd.DataFrame(),
            "tech_2040": pd.DataFrame(),
            "portfolio_annual_savings": pd.DataFrame(),
            "cumulative_bill_savings": pd.DataFrame(),
            "lifetime_totals": pd.DataFrame(),
            "avg_price_2026_model": pd.DataFrame(),
            "market_share_reached": pd.DataFrame(),
        }

    # Ensure we have a fresh instance locally (don’t default-construct in signature)
    if cfg is None:
        cfg = SavingsConfig()

    # Ship only primitives to workers (pickle-safe)
    cfg_payload = {
        "lifetime_years": int(getattr(cfg, "lifetime_years", 25) or 25),
        "cap_to_horizon": bool(getattr(cfg, "cap_to_horizon", False)),
    }

    tasks = [(sd, run_id, cfg_payload) for sd in state_dirs]
    outputs: List[Dict[str, pd.DataFrame]] = []

    n_jobs = max(1, min(n_jobs, max(1, (cpu_count() or 2) - 1)))
    if n_jobs == 1:
        outputs = [_process_one_state(t) for t in tasks]
    else:
        with Pool(processes=n_jobs) as pool:
            for result in pool.imap_unordered(_process_one_state, tasks):
                outputs.append(result)

    merged: Dict[str, List[pd.DataFrame]] = {}
    for out in outputs:
        for key, df in out.items():
            merged.setdefault(key, []).append(df)

    return {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in merged.items()}


# =============================================================================
# Generic state-level faceter (unchanged)
# =============================================================================

def facet_lines_by_state(
    df: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    xticks: Iterable[int] = (2026, 2030, 2035, 2040),
    height: float = 3.5,
    col_wrap: int = 4,
    sharey: bool = False
) -> None:
    if df.empty:
        return

    def _fmt(v: float) -> str:
        m = y_col
        if m == "number_of_adopters":
            return f"{v/1e6:.1f}M"
        if m == "system_kw_cum":
            return f"{v/1e6:.1f} GW"
        if m == "batt_kwh_cum":
            return f"{v/1e6:.1f} GWh"
        if m == "cumulative_bill_savings":
            return f"${v/1e9:.1f}B"
        if m == "portfolio_annual_savings":
            return f"${v/1e9:.1f}B/yr"
        if m in ("median_system_kw", "system_kw"):
            return f"{v:.1f} kW"
        if m == "market_share_reached":
            return f"{v*100:.1f}%"
        if m == "median_batt_kwh":
            return f"{v:.1f} kWh"
        return f"{v:.2g}"

    sns.set_context("talk", rc={"lines.linewidth": 2})
    g = sns.FacetGrid(df, col="state_abbr", col_wrap=col_wrap, height=height, sharey=sharey)

    g.map_dataframe(sns.lineplot, x="year", y=y_col, hue="scenario", marker="o")
    g.set_titles("{col_name}")
    g.set_axis_labels("Year", ylabel)
    g.set(xticks=list(xticks))
    g.add_legend()
    g.fig.suptitle(title, y=1.02)

    global_years = pd.to_numeric(df["year"], errors="coerce").dropna().astype(int)
    preferred_year = 2040 if (len(global_years) and 2040 in set(global_years)) else (int(global_years.max()) if len(global_years) else None)

    if preferred_year is not None:
        for state, ax in g.axes_dict.items():
            sdf = df[(df["state_abbr"] == state)].copy()
            if sdf.empty or y_col not in sdf.columns:
                continue
            sdf = sdf.dropna(subset=[y_col, "year"])

            years = sdf["year"].astype(int).unique()
            end_year = preferred_year if preferred_year in years else int(sdf["year"].max())

            end_points = []
            for scen, sg in sdf.groupby("scenario"):
                sg = sg.sort_values("year")
                g_end = sg[sg["year"] == end_year]
                if g_end.empty:
                    g_end = sg[sg["year"] <= end_year].tail(1)
                if not g_end.empty:
                    x_end = float(g_end["year"].iloc[-1])
                    y_end = float(g_end[y_col].iloc[-1])
                    end_points.append((scen, x_end, y_end))

            if not end_points:
                continue

            sv = sdf[y_col].to_numpy()
            vmin = float(np.nanmin(sv)) if sv.size else 0.0
            vmax = float(np.nanmax(sv)) if sv.size else 1.0
            yrange = max(1.0, vmax - vmin)
            sep_needed = yrange * 0.01  # 1%

            used_y = []
            for scen, x_end, y_end in sorted(end_points, key=lambda t: t[2]):
                offset_pts = 0
                for uy in used_y:
                    if abs(y_end - uy) < sep_needed:
                        offset_pts += 6
                used_y.append(y_end)

                ax.annotate(
                    _fmt(y_end),
                    xy=(x_end, y_end),
                    xytext=(6, offset_pts),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

    plt.show()


# =============================================================================
# National deltas & totals + peaks (unchanged)
# =============================================================================

def build_national_deltas(outputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    years_seen = []
    for k in ("totals", "portfolio_annual_savings", "cumulative_bill_savings", "market_share_reached"):
        dfk = outputs.get(k, pd.DataFrame())
        if not dfk.empty and "year" in dfk.columns:
            years_seen.append(dfk["year"])
    if not years_seen:
        return pd.DataFrame(columns=["year", "metric", "value"])

    year_min = int(pd.concat(years_seen).min())
    year_max = int(pd.concat(years_seen).max())
    all_years = pd.Index(range(year_min, year_max + 1), name="year")

    pieces: List[pd.DataFrame] = []

    def _delta_from_series(df: pd.DataFrame, value_col: str, cumulative: bool) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["year", "value"])
        def _fill(g: pd.DataFrame) -> pd.DataFrame:
            s = g.set_index("year")[[value_col]].reindex(all_years)
            s = (s.ffill() if cumulative else s.fillna(0.0)).fillna(0.0)
            s = s.reset_index()
            s["state_abbr"] = g["state_abbr"].iloc[0]
            s["scenario"] = g["scenario"].iloc[0]
            return s
        filled = (
            df.groupby(["state_abbr", "scenario"], observed=True, as_index=False)
              .apply(_fill)
              .reset_index(drop=True)
        )
        nat = filled.groupby(["year", "scenario"], as_index=False)[value_col].sum()
        piv = nat.pivot(index="year", columns="scenario", values=value_col)
        if "policy" not in piv.columns or "baseline" not in piv.columns:
            return pd.DataFrame(columns=["year", "value"])
        out = piv["policy"].sub(piv["baseline"]).rename("value").reset_index()
        return out

    totals = outputs.get("totals", pd.DataFrame())
    for col, metric in [
        ("number_of_adopters", "number_of_adopters"),
        ("system_kw_cum", "system_kw_cum"),
        ("batt_kwh_cum", "batt_kwh_cum"),
    ]:
        if not totals.empty and col in totals.columns:
            d = _delta_from_series(totals[["state_abbr","year","scenario",col]].copy(), col, cumulative=True)
            if not d.empty:
                d["metric"] = metric
                pieces.append(d)

    pas = outputs.get("portfolio_annual_savings", pd.DataFrame())
    if not pas.empty and "portfolio_annual_savings" in pas.columns:
        d = _delta_from_series(pas[["state_abbr","year","scenario","portfolio_annual_savings"]].copy(),
                               "portfolio_annual_savings", cumulative=False)
        if not d.empty:
            d["metric"] = "portfolio_annual_savings"
            pieces.append(d)

    cbs = outputs.get("cumulative_bill_savings", pd.DataFrame())
    if not cbs.empty and "cumulative_bill_savings" in cbs.columns:
        d = _delta_from_series(cbs[["state_abbr","year","scenario","cumulative_bill_savings"]].copy(),
                               "cumulative_bill_savings", cumulative=True)
        if not d.empty:
            d["metric"] = "cumulative_bill_savings"
            pieces.append(d)

    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["year","metric","value"])


def build_national_totals(
    outputs: Dict[str, pd.DataFrame],
    peaks_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    years_seen = []
    for k in ("totals", "portfolio_annual_savings", "cumulative_bill_savings"):
        dfk = outputs.get(k, pd.DataFrame())
        if not dfk.empty and "year" in dfk.columns:
            years_seen.append(dfk["year"])
    if peaks_df is not None and not peaks_df.empty:
        years_seen.append(peaks_df["year"])

    if not years_seen:
        return pd.DataFrame(columns=["year", "scenario", "metric", "value"])

    year_min = int(pd.concat(years_seen).min())
    year_max = int(pd.concat(years_seen).max())
    all_years = pd.Index(range(year_min, year_max + 1), name="year")

    pieces: List[pd.DataFrame] = []

    def _sum_series(df: pd.DataFrame, value_col: str, cumulative: bool) -> pd.DataFrame:
        if df.empty or value_col not in df.columns:
            return pd.DataFrame(columns=["year", "scenario", "value"])
        def _fill(g: pd.DataFrame) -> pd.DataFrame:
            s = g.set_index("year")[[value_col]].reindex(all_years)
            s = (s.ffill() if cumulative else s.fillna(0.0)).fillna(0.0)
            s = s.reset_index()
            s["state_abbr"] = g["state_abbr"].iloc[0]
            s["scenario"] = g["scenario"].iloc[0]
            return s
        filled = (
            df.groupby(["state_abbr", "scenario"], observed=True, as_index=False)
              .apply(_fill)
              .reset_index(drop=True)
        )
        nat = filled.groupby(["year", "scenario"], as_index=False)[value_col].sum()
        nat = nat.rename(columns={value_col: "value"})
        return nat

    totals = outputs.get("totals", pd.DataFrame())
    if not totals.empty:
        for col in ("number_of_adopters", "system_kw_cum", "batt_kwh_cum"):
            s = _sum_series(totals[["state_abbr", "year", "scenario", col]].copy(), col, cumulative=True)
            if not s.empty:
                s["metric"] = col
                pieces.append(s)

    cbs = outputs.get("cumulative_bill_savings", pd.DataFrame())
    if not cbs.empty:
        s = _sum_series(cbs[["state_abbr", "year", "scenario", "cumulative_bill_savings"]].copy(),
                        "cumulative_bill_savings", cumulative=True)
        if not s.empty:
            s["metric"] = "cumulative_bill_savings"
            pieces.append(s)

    pas = outputs.get("portfolio_annual_savings", pd.DataFrame())
    if not pas.empty:
        s = _sum_series(pas[["state_abbr", "year", "scenario", "portfolio_annual_savings"]].copy(),
                        "portfolio_annual_savings", cumulative=False)
        if not s.empty:
            s["metric"] = "portfolio_annual_savings"
            pieces.append(s)

    if peaks_df is not None and not peaks_df.empty:
        if set(["state_abbr","scenario","year","peak_mw"]).issubset(peaks_df.columns):
            s = _sum_series(peaks_df[["state_abbr","year","scenario","peak_mw"]].copy(),
                            "peak_mw", cumulative=False)
            if not s.empty:
                s["metric"] = "peak_mw"
                pieces.append(s)

    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=["year", "scenario", "metric", "value"])


def facet_lines_national_totals(
    outputs: Dict[str, pd.DataFrame],
    peaks_df: Optional[pd.DataFrame] = None,
    coincident_df: Optional[pd.DataFrame] = None, 
    metrics: Optional[Iterable[str]] = ("number_of_adopters", "system_kw_cum", "batt_kwh_cum", "cumulative_bill_savings", "peak_mw", "coincident_reduction_mw"),
    xticks: Iterable[int] = (2026, 2030, 2035, 2040),
    title: str = "U.S. Totals: Baseline vs Policy",
    ncols: int = 3,
) -> None:
    nat = build_national_totals(outputs, peaks_df=peaks_df)

    if coincident_df is not None and isinstance(coincident_df, pd.DataFrame) and not coincident_df.empty:
        need = {"year","coincident_reduction_mw"}
        if need.issubset(coincident_df.columns):
            nat_co = (
                coincident_df.groupby("year", as_index=False)["coincident_reduction_mw"].sum()
                            .rename(columns={"coincident_reduction_mw":"value"})
            )
            nat_co["scenario"] = "coincident Δ"      # single series (delta)
            nat_co["metric"]   = "coincident_reduction_mw"
            nat = pd.concat([nat, nat_co], ignore_index=True, sort=False)

    if nat.empty:
        return

    if metrics:
        nat = nat[nat["metric"].isin(set(metrics))]
        if nat.empty:
            return

    nice_titles = {
        "number_of_adopters": "Cumulative Adopters",
        "system_kw_cum": "Cumulative PV (kW)",
        "batt_kwh_cum": "Cumulative Storage (kWh)",
        "cumulative_bill_savings": "Cumulative Bill Savings ($)",
        "portfolio_annual_savings": "Portfolio Bill Savings ($/yr)",
        "peak_mw": "U.S. Peak Demand (MW)",
    }

    def _fmt(metric: str, v: float) -> str:
        if metric == "number_of_adopters":
            return f"{v/1e6:.1f}M"
        if metric == "system_kw_cum":
            return f"{v/1e6:.1f} GW"
        if metric == "batt_kwh_cum":
            return f"{v/1e6:.1f} GWh"
        if metric == "cumulative_bill_savings":
            return f"${v/1e9:.1f}B"
        if metric == "peak_mw":
            return f"{v/1e3:.1f} GW"
        return f"{v:.2g}"

    metric_list = list(nat["metric"].unique())
    n = len(metric_list)
    ncols = max(1, min(ncols, n))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 3.4*nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for i, m in enumerate(metric_list):
        ax = axes[i]
        d = nat[nat["metric"] == m].sort_values("year")

        years = d["year"].unique()
        end_year = 2040 if 2040 in years else int(d["year"].max())

        end_points = []
        for scen, g in d.groupby("scenario"):
            g_sorted = g.sort_values("year")
            ax.plot(g_sorted["year"], g_sorted["value"], marker="o", label=scen.capitalize())
            g_end = g_sorted[g_sorted["year"] == end_year]
            if g_end.empty:
                g_end = g_sorted[g_sorted["year"] <= end_year].tail(1)
            if not g_end.empty:
                end_points.append((scen, float(g_end["year"].iloc[-1]), float(g_end["value"].iloc[-1])))

        ax.set_xticks(list(xticks))
        ax.set_title(nice_titles.get(m, m))
        ax.set_xlabel("Year")
        ax.set_ylabel("U.S. Total")
        ax.legend(frameon=False, loc="best")

        if end_points:
            ys = np.array([p[2] for p in end_points])
            yrange = max(1.0, float(d["value"].max() - d["value"].min()) or 1.0)
            sep_needed = yrange * 0.01
            used_offsets = {}
            for scen, x_end, y_end in sorted(end_points, key=lambda t: t[2]):
                offset_pts = 0
                for other_y in used_offsets.keys():
                    if abs(y_end - other_y) < sep_needed:
                        offset_pts += 6
                used_offsets[y_end] = True

                ax.annotate(
                    _fmt(m, y_end),
                    xy=(x_end, y_end),
                    xytext=(6, offset_pts),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, y=1.05)
    plt.show()


# =============================================================================
# Other plots (unchanged)
# =============================================================================

def facet_median_storage_by_state(
    outputs: Dict[str, pd.DataFrame],
    xticks: Iterable[int] = (2026, 2030, 2035, 2040),
    height: float = 3.5,
    col_wrap: int = 4,
    sharey: bool = False,
    title: str = "Median Storage Size (kWh) — Baseline vs Policy",
) -> None:
    df = outputs.get("median_storage_kwh", pd.DataFrame())
    if df is None or df.empty:
        return
    d = df.rename(columns={"median_batt_kwh": "value"}).copy()
    d = d.dropna(subset=["value"])

    sns.set_context("talk", rc={"lines.linewidth": 2})
    g = sns.FacetGrid(d, col="state_abbr", col_wrap=col_wrap, height=height, sharey=sharey)
    g.map_dataframe(sns.lineplot, x="year", y="value", hue="scenario", marker="o")
    g.set_titles("{col_name}")
    g.set_axis_labels("Year", "Median Storage Size (kWh)")
    g.set(xticks=list(xticks))
    g.add_legend()
    g.fig.suptitle(title, y=1.02)

    global_years = pd.to_numeric(d["year"], errors="coerce").dropna().astype(int)
    preferred_year = 2040 if (len(global_years) and 2040 in set(global_years)) else (int(global_years.max()) if len(global_years) else None)
    if preferred_year is not None:
        for state, ax in g.axes_dict.items():
            sdf = d[(d["state_abbr"] == state)].dropna(subset=["value","year"])
            if sdf.empty:
                continue
            years = sdf["year"].astype(int).unique()
            end_year = preferred_year if preferred_year in years else int(sdf["year"].max())

            end_points = []
            for scen, sg in sdf.groupby("scenario"):
                sg = sg.sort_values("year")
                g_end = sg[sg["year"] == end_year]
                if g_end.empty:
                    g_end = sg[sg["year"] <= end_year].tail(1)
                if not g_end.empty:
                    end_points.append((scen, float(g_end["year"].iloc[-1]), float(g_end["value"].iloc[-1])))

            if not end_points:
                continue

            vals = np.array([p[2] for p in end_points])
            yrange = max(1.0, float(vals.max() - vals.min()) or 1.0)
            sep = yrange * 0.01
            used = []
            for scen, x_end, y_end in sorted(end_points, key=lambda t: t[2]):
                offset = 0
                for u in used:
                    if abs(y_end - u) < sep:
                        offset += 6
                used.append(y_end)
                ax.annotate(f"{y_end:.1f} kWh", xy=(x_end, y_end),
                            xytext=(6, offset), textcoords="offset points",
                            ha="left", va="center", fontsize=9, fontweight="bold")
    plt.show()


def bar_tech_potential_2040(tech_agg: pd.DataFrame, title: str = "Solar Technical Potential Reached in 2040") -> None:
    if tech_agg.empty:
        return
    order = (
        tech_agg[tech_agg["scenario"] == "policy"]
        .sort_values("percent_tech_potential", ascending=False)["state_abbr"]
        .tolist()
    )
    plt.figure(figsize=(14, 6), constrained_layout=True)
    ax = sns.barplot(
        data=tech_agg, x="state_abbr", y="percent_tech_potential",
        hue="scenario", order=order, errorbar=None
    )
    for container in ax.containers:
        for bar in container:
            h = bar.get_height()
            if h > 0 and not math.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h - 1.5,
                    f"{h:.1f}%",
                    ha="center",
                    va="top",
                    color="white",
                    fontsize=9,
                    fontweight="bold",
                )
    plt.title(f"{title} (Sorted by Policy %)")
    plt.ylabel("Percent of Technical Potential (%)")
    plt.xlabel("State")
    plt.xticks(rotation=45)
    plt.ylim(0, tech_agg["percent_tech_potential"].max() * 1.1)
    plt.show()

def export_compiled_results_to_excel(
    outputs: Dict[str, pd.DataFrame],
    run_id: str,
    base_dir: str = "/Volumes/Seagate Portabl/permit_power/dgen_runs",
    peaks_df: Optional[pd.DataFrame] = None,
    include_national: bool = True,
) -> str:
    import datetime
    import pandas as pd
    import os
    import re

    compiled_dir = os.path.join(base_dir, "compiled_results")
    os.makedirs(compiled_dir, exist_ok=True)
    out_path = os.path.join(compiled_dir, f"{run_id}.xlsx")

    def _sheet_name(name: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9 _\-]", "_", str(name))
        return cleaned[:31] if cleaned else "Sheet"

    nat_totals = None
    nat_deltas = None
    if include_national:
        try:
            nat_totals = build_national_totals(outputs, peaks_df=peaks_df)
        except Exception:
            nat_totals = None
        try:
            nat_deltas = build_national_deltas(outputs)
        except Exception:
            nat_deltas = None

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as xw:
        meta = pd.DataFrame({
            "run_id": [run_id],
            "generated_at": [datetime.datetime.now().isoformat(timespec="seconds")],
            "tables_included": [", ".join(sorted([k for k, v in outputs.items() if isinstance(v, pd.DataFrame) and not v.empty]))]
        })
        meta.to_excel(xw, index=False, sheet_name=_sheet_name("README"))

        for key, df in outputs.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_excel(xw, index=False, sheet_name=_sheet_name(key))

        if peaks_df is not None and isinstance(peaks_df, pd.DataFrame) and not peaks_df.empty:
            peaks_df.to_excel(xw, index=False, sheet_name=_sheet_name("peaks"))

        if include_national and isinstance(nat_totals, pd.DataFrame) and not nat_totals.empty:
            nat_totals.to_excel(xw, index=False, sheet_name=_sheet_name("national_totals"))

        if include_national and isinstance(nat_deltas, pd.DataFrame) and not nat_deltas.empty:
            nat_deltas.to_excel(xw, index=False, sheet_name=_sheet_name("national_deltas"))

    return out_path

def choropleth_pv_delta_gw_policy_vs_baseline(
    outputs: Dict[str, pd.DataFrame],
    shapefile_path: str = "../../../data/states.shp",
    year: int = 2040,
    k_bins: int = 10,  # Jenks classes
) -> pd.DataFrame:
    """
    Map absolute PV installations in millions:  Δ installations = (Policy_2040 − Baseline_2040) / 1e6.

    Uses outputs["totals"] from process_all_states(...) which already contains
    state/year/scenario aggregates.

    Returns a tidy DataFrame with:
      ['state_abbr','baseline_installations','policy_installations','delta_installations','delta_installations_millions']
    """

    # ---- grab processed totals (already built upstream) ----
    totals = outputs.get("totals", pd.DataFrame())
    if totals.empty:
        raise ValueError("outputs['totals'] is empty; run process_all_states(...) first.")

    need = {"state_abbr", "year", "scenario", "number_of_adopters"}
    if not need.issubset(totals.columns):
        missing = need - set(totals.columns)
        raise ValueError(f"outputs['totals'] missing columns: {sorted(missing)}")

    s = totals.loc[totals["year"] == year, ["state_abbr", "scenario", "number_of_adopters"]].copy()
    # pivot to [state] x {baseline, policy}
    piv = s.pivot_table(index="state_abbr", columns="scenario", values="number_of_adopters", aggfunc="sum")
    # Make sure expected columns exist
    for col in ("baseline", "policy"):
        if col not in piv.columns:
            piv[col] = np.nan

    df = piv.reset_index().rename_axis(None, axis=1)
    df["delta_installations"] = df["policy"] - df["baseline"]
    df["delta_installations_millions"] = df["delta_installations"] / 1e6
    df = df.rename(columns={"baseline": "baseline_installations", "policy": "policy_installations"})

    # ---- map join & plot (contiguous U.S.; exclude AK & HI) ----
    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:5070")
    if "STUSPS" not in gdf.columns:
        for c in ("stusps", "STATE_ABBR", "STATE", "STATEFP"):
            if c in gdf.columns:
                gdf["STUSPS"] = gdf[c].astype(str).str.upper()
                break
    if "STUSPS" not in gdf.columns:
        raise ValueError("Shapefile must include a 'STUSPS' two-letter state code.")

    gdf["STUSPS"] = gdf["STUSPS"].astype(str).str.upper()
    gdf = gdf[~gdf["STUSPS"].isin({"AK", "HI", "PR", "GU", "VI", "AS", "MP", "DC"})].copy()

    plot_df = gdf.merge(df.rename(columns={"state_abbr": "STUSPS"}), on="STUSPS", how="left")

    # Jenks (Fisher-Jenks) when available; fallback to quantiles
    try:
        import mapclassify  # noqa: F401
        scheme = "fisher_jenks"
        plot_kwargs_extra = dict(k=int(k_bins))
    except Exception:
        scheme = "quantiles"
        plot_kwargs_extra = dict(k=int(k_bins))

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    plt.rcParams["font.family"] = "Cabin"
    plot_df.plot(
        column="delta_installations_millions",
        cmap="Blues",
        linewidth=0.6,
        edgecolor="grey",
        legend=True,
        scheme=scheme,
        legend_kwds={"title": f"Installations in Millions", 
                     "ncols":2, "fmt": "{:.1f}", "loc":"lower left"},
        ax=ax,
        missing_kwds={"color": "lightgray"},
        **plot_kwargs_extra,
    )
    #ax.set_title(f"Additional Solar Installations in 2040 by State", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.show()

def policy_only_bill_price_diffs_after_adoption(
    root_dir: str,
    year: int = 2040,
    level: str = "US",   # "US" or "state"
    run_id: str | None = None,
    strict_run_id: bool = True,
) -> "pd.DataFrame":
    """
    Policy-only comparison of bills and prices among adopters in a given year.

    Weighted by adopters:
      price_with_kwh = sum(bill_with * adopters) / sum(load_kwh * adopters)
      price_wo_kwh   = sum(bill_wo   * adopters) / sum(load_kwh * adopters)

    Also returns average bill per adopter and differences.

    Returns columns:
      ['geo','year',
       'price_with_c_per_kwh','price_wo_c_per_kwh',
       'diff_price_c_per_kwh','pct_change_price',
       'avg_bill_with_per_adopter','avg_bill_wo_per_adopter',
       'diff_avg_bill_per_adopter']
    """
    import os
    import pandas as pd
    import numpy as np

    # Required columns (assumed present)
    USECOLS = [
        "state_abbr", "year", "new_adopters",
        "first_year_elec_bill_with_system",
        "first_year_elec_bill_wo_system",
        "load_kwh_per_customer_in_bin_initial",
    ]

    # Collect POLICY files only
    frames = []
    for state_dir in discover_state_dirs(root_dir):
        _b_csv, p_csv = find_state_files(state_dir, run_id=run_id, strict_run_id=strict_run_id)
        if not p_csv:
            continue
        df = pd.read_csv(p_csv, usecols=USECOLS)
        frames.append(df)

    x = pd.concat(frames, ignore_index=True)

    # Focus year + adopters only
    x = x[x["year"] == year].copy()
    x = x[x["new_adopters"] > 0].copy()

    # Adoption-weighted totals
    x["tot_bill_with"] = x["first_year_elec_bill_with_system"] * x["new_adopters"]
    x["tot_bill_wo"]   = x["first_year_elec_bill_wo_system"]   * x["new_adopters"]
    x["tot_load"]      = x["load_kwh_per_customer_in_bin_initial"] * x["new_adopters"]

    if level.lower() == "state":
        g = (x.groupby("state_abbr", as_index=False)
              [["tot_bill_with","tot_bill_wo","tot_load","new_adopters"]].sum())
        geo = g["state_abbr"]
    else:
        s = x[["tot_bill_with","tot_bill_wo","tot_load","new_adopters"]].sum()
        g = pd.DataFrame([s])
        geo = pd.Series(["US"])

    # Prices (USD/kWh) and bills ($/adopter)
    price_with = g["tot_bill_with"] / g["tot_load"]
    price_wo   = g["tot_bill_wo"]   / g["tot_load"]
    avg_with   = g["tot_bill_with"] / g["new_adopters"]
    avg_wo     = g["tot_bill_wo"]   / g["new_adopters"]

    out = pd.DataFrame({
        "geo": geo.values,
        "year": year,
        "price_with_c_per_kwh": price_with * 100.0,
        "price_wo_c_per_kwh":   price_wo   * 100.0,
    })
    out["diff_price_c_per_kwh"] = out["price_with_c_per_kwh"] - out["price_wo_c_per_kwh"]
    out["pct_change_price"] = 100.0 * (out["diff_price_c_per_kwh"] / out["price_wo_c_per_kwh"])

    out["avg_bill_with_per_adopter"] = avg_with
    out["avg_bill_wo_per_adopter"]   = avg_wo
    out["diff_avg_bill_per_adopter"] = out["avg_bill_with_per_adopter"] - out["avg_bill_wo_per_adopter"]

    return out.sort_values("geo").reset_index(drop=True)


def plot_us_cum_adopters_grouped(outputs: Dict[str, pd.DataFrame],
                                 xticks: Iterable[int] = (2026, 2030, 2035, 2040)) -> pd.DataFrame:
    """
    Grouped bar plot of *national cumulative adopters* by year,
    comparing Status Quo vs. $1/Watt.

    Returns the tidy table used for plotting:
        ['year','scenario','value'] where value = U.S. total cumulative adopters
    """

    nat = build_national_totals(outputs)
    if nat.empty:
        raise ValueError("No national totals found. Run process_all_states(...) first.")

    d = nat[nat["metric"] == "number_of_adopters"].copy()
    if d.empty:
        raise ValueError("National totals lack 'number_of_adopters' metric.")

    # Map scenario labels to custom names
    rename_map = {
        "baseline": "Business-as-usual",
        "policy": "$1/Watt"
    }
    d["scenario"] = d["scenario"].map(rename_map).fillna(d["scenario"])

    # Plot
    plt.rcParams["font.family"] = "Cabin"
    plt.figure(figsize=(12, 5), constrained_layout=True)
    ax = sns.barplot(
        data=d, x="year", y="value", hue="scenario",
        errorbar=None, palette=["#a2e0fc", "#1bb3ef"]
    )
    #ax.set_title("Solar Adoption - Business-as-usual vs. $1/Watt")
    ax.set_xlabel("")
    ax.set_ylabel("Solar Installations (millions)")

    # Format y-axis in millions
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.0f}")
    )

    # annotate bars in millions
    for c in ax.containers:
        ax.bar_label(
            c, labels=[f"{v/1e6:.1f}M" if np.isfinite(v) else "" for v in c.datavalues],
            padding=2, fontsize=9
        )

    # Legend formatting
    plt.legend(title=None, frameon=False, fontsize=12)

    # Remove grid lines
    ax.grid(False)

    plt.show()
    return d.sort_values(["year", "scenario"]).reset_index(drop=True)

def _weighted_median_simple(values: "pd.Series", weights: "pd.Series") -> float:
    """Small local helper: weighted median ignoring NaNs and nonpositive weights."""
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    mask = v.notna() & (w > 0)
    if not mask.any():
        return float("nan")
    v = v[mask].to_numpy()
    w = w[mask].to_numpy()
    order = np.argsort(v)
    v, w = v[order], w[order]
    cw = np.cumsum(w)
    cutoff = 0.5 * w.sum()
    idx = int(np.searchsorted(cw, cutoff, side="left"))
    return float(v[min(idx, len(v) - 1)])


def build_payback_timeseries(
    root_dir: str,
    run_id: str | None = None,
    strict_run_id: bool = True,
    level: str = "state"  # "state" or "US"
) -> "pd.DataFrame":
    """
    Build an adoption-weighted *average (mean)* payback series by year for baseline vs policy.

    NOTE: For backward compatibility, the output column is named
          'payback_weighted_median' even though it is a weighted *mean*.

    Expects per-state CSVs with columns:
      - state_abbr, year, scenario, new_adopters, payback_period

    Returns:
      ['geo','scenario','year','payback_weighted_median']
        where geo = state_abbr (level="state") or 'US' (level="US")
    """
    frames = []
    for state_dir in discover_state_dirs(root_dir):
        b_csv, p_csv = find_state_files(state_dir, run_id=run_id, strict_run_id=strict_run_id)
        for path in (b_csv, p_csv):
            if not path:
                continue
            try:
                df = pd.read_csv(path, usecols=["state_abbr", "year", "scenario", "new_adopters", "payback_period"])
            except Exception:
                continue
            if "scenario" not in df.columns:
                fn = os.path.basename(path).lower()
                df["scenario"] = "policy" if "policy" in fn else ("baseline" if "baseline" in fn else np.nan)
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["geo","scenario","year","payback_weighted_median"])

    x = pd.concat(frames, ignore_index=True)

    # Coerce numerics
    x["year"] = pd.to_numeric(x.get("year"), errors="coerce")
    x["new_adopters"] = pd.to_numeric(x.get("new_adopters"), errors="coerce").fillna(0.0).clip(lower=0.0)
    x["payback_period"] = pd.to_numeric(x.get("payback_period"), errors="coerce")

    # Keep rows with adopters, valid year, and valid payback
    x = x[(x["new_adopters"] > 0) & x["year"].notna() & x["payback_period"].notna()].copy()

    if level.lower() == "state":
        grp_keys = ["state_abbr", "scenario", "year"]
        geo_col = "state_abbr"
    else:
        # National: pool all states before averaging
        x["geo"] = "US"
        grp_keys = ["geo", "scenario", "year"]
        geo_col = "geo"

    # Vectorized weighted mean: sum(w * v) / sum(w)
    x["__w"] = x["new_adopters"]
    x["__wv"] = x["payback_period"] * x["__w"]

    agg = (
        x.groupby(grp_keys, observed=True)[["__w", "__wv"]]
         .sum()
         .reset_index()
    )
    agg["payback_weighted_median"] = np.where(
        agg["__w"] > 0, agg["__wv"] / agg["__w"], np.nan
    )

    # Tidy output
    out_cols = [geo_col, "scenario", "year", "payback_weighted_median"]
    out = agg[out_cols].rename(columns={geo_col: "geo"}).sort_values(["geo","scenario","year"])
    return out.reset_index(drop=True)



def summarize_affordability_milestones(
    payback_ts: "pd.DataFrame",
    thresholds: tuple[float, ...] = (15.0, 10.0, 5.0)
) -> "pd.DataFrame":
    """
    From a payback time series (output of build_payback_timeseries), compute:
      - First year baseline reaches each threshold
      - First year policy reaches each threshold
      - Advantage (years earlier under policy; positive = policy sooner)
      - Crossover flag (baseline never reaches but policy does)

    Returns columns:
      ['geo','threshold_years','baseline_year','policy_year','advantage_years','crossover']
    """
    if payback_ts.empty:
        return pd.DataFrame(columns=["geo","threshold_years","baseline_year","policy_year","advantage_years","crossover"])

    # Ensure sorting for "first year" logic
    d = payback_ts.dropna(subset=["payback_weighted_median","year"]).sort_values(["geo","scenario","year"]).copy()

    # Helper: first year ≤ threshold for a single series
    def _first_year_le(g: "pd.DataFrame", thr: float) -> float | None:
        m = g[g["payback_weighted_median"] <= thr]
        if m.empty:
            return None
        return float(m["year"].iloc[0])

    rows = []
    for geo, g_geo in d.groupby("geo", observed=True):
        b = g_geo[g_geo["scenario"].str.lower() == "baseline"]
        p = g_geo[g_geo["scenario"].str.lower() == "policy"]
        for thr in thresholds:
            yb = _first_year_le(b, thr)
            yp = _first_year_le(p, thr)
            # advantage: years baseline - policy (positive => policy earlier)
            adv = (yb - yp) if (yb is not None and yp is not None) else (None if yp is None else float("inf"))
            crossover = (yb is None and yp is not None)
            rows.append({
                "geo": geo,
                "threshold_years": float(thr),
                "baseline_year": yb,
                "policy_year": yp,
                "advantage_years": adv,
                "crossover": crossover,
            })
    out = pd.DataFrame(rows)
    return out.sort_values(["geo","threshold_years"]).reset_index(drop=True)


def table_crossover_states(
    milestones: "pd.DataFrame",
    threshold: float = 10.0,
    top_n: int = 10
) -> "pd.DataFrame":
    """
    Convenience view: states where policy achieves ≤threshold and baseline does not,
    or policy achieves it earlier. Sorted by largest advantage (years).

    Returns columns:
      ['geo','baseline_year','policy_year','advantage_years','crossover']
    """
    if milestones.empty:
        return pd.DataFrame(columns=["geo","baseline_year","policy_year","advantage_years","crossover"])
    m = milestones[milestones["threshold_years"] == float(threshold)].copy()
    m["adv_sort"] = m["advantage_years"].replace({np.inf: 1e9})
    m = m[(m["crossover"] == True) | (m["adv_sort"] > 0)].sort_values("adv_sort", ascending=False)
    cols = ["geo","baseline_year","policy_year","advantage_years","crossover"]
    return m[cols].head(top_n).reset_index(drop=True)

def payback_threshold_flags(
    root_dir: str,
    run_id: str | None = None,
    strict_run_id: bool = True,
    threshold_years: float = 5.0,
    year: int = 2040,
    long_format: bool = True,
) -> "pd.DataFrame":
    """
    Build flags indicating whether each state meets an adoption-weighted payback threshold
    in a given YEAR, for both scenarios.

    If long_format=True (default), returns:
      ['state_abbr','scenario','flag']  (one row per state×scenario)

    If long_format=False, returns the previous wide shape:
      ['state_abbr','baseline_flag','policy_flag']
    """
    pb = build_payback_timeseries(
        root_dir=root_dir, run_id=run_id, strict_run_id=strict_run_id, level="state"
    )
    if pb.empty:
        return pd.DataFrame(columns=(["state_abbr","scenario","flag"] if long_format
                                     else ["state_abbr","baseline_flag","policy_flag"]))

    d = pb.copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d = d[d["year"] == year].copy()
    if d.empty:
        return pd.DataFrame(columns=(["state_abbr","scenario","flag"] if long_format
                                     else ["state_abbr","baseline_flag","policy_flag"]))

    d["scenario"] = d["scenario"].astype(str).str.lower().str.strip()
    d["geo"] = d["geo"].astype(str).str.upper()
    d["flag"] = pd.to_numeric(d["payback_weighted_median"], errors="coerce") <= float(threshold_years)

    # Dedup by OR if multiples
    agg = (d.groupby(["geo","scenario"], as_index=False)["flag"].max()
             .rename(columns={"geo":"state_abbr"}))
    agg["state_abbr"] = agg["state_abbr"].astype(str).str.upper()

    if long_format:
        return agg[["state_abbr","scenario","flag"]].reset_index(drop=True)

    # wide (back-compat)
    piv = agg.pivot(index="state_abbr", columns="scenario", values="flag").reset_index()
    for col in ("baseline","policy"):
        if col not in piv.columns:
            piv[col] = False
    piv = piv.rename(columns={"baseline":"baseline_flag","policy":"policy_flag"})
    return piv[["state_abbr","baseline_flag","policy_flag"]].reset_index(drop=True)

def facet_choropleth_payback_continuous(
    root_dir: str,
    shapefile_path: str,
    run_id: str | None = None,
    strict_run_id: bool = True,
    year: int = 2040,
    cmap: str = "Blues_r",   # reversed so lower payback = darker (more affordable)
) -> "pd.DataFrame":
    """
    Vertically faceted choropleth of adoption-weighted *average payback (years)* in `year`.
      - Top: Baseline
      - Bottom: Policy
    Uses a shared color scale so the two maps are directly comparable.

    Returns the tidy table used to plot: ['state_abbr','scenario','payback_years']
    """
    import matplotlib.pyplot as plt

    # 1) Build state-level payback (you already switched this function to weighted *mean*)
    pb = build_payback_timeseries(root_dir=root_dir, run_id=run_id,
                                  strict_run_id=strict_run_id, level="state")
    if pb.empty:
        raise ValueError("No payback data found. Make sure baseline/policy CSVs include 'payback_period' and 'new_adopters'.")

    d = pb.copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d = d[(d["year"] == year) & d["geo"].notna() & d["scenario"].notna()].copy()
    d = d.rename(columns={"geo": "state_abbr", "payback_weighted_median": "payback_years"})
    d["state_abbr"] = d["state_abbr"].astype(str).str.upper()
    d["scenario"] = d["scenario"].astype(str).str.lower().str.strip()

    # Keep only baseline/policy
    d = d[d["scenario"].isin(["baseline", "policy"])]
    if d.empty:
        raise ValueError(f"No baseline/policy payback rows for year {year}.")

    # 2) Load shapefile and prep join key
    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:5070")
    if "STUSPS" not in gdf.columns:
        for c in ("stusps","STATE_ABBR","STATE","STATEFP","STATEFP20"):
            if c in gdf.columns:
                gdf["STUSPS"] = gdf[c].astype(str).str.upper()
                break
    if "STUSPS" not in gdf.columns:
        raise ValueError("Shapefile must include a two-letter state code (e.g., STUSPS).")

    gdf["STUSPS"] = gdf["STUSPS"].astype(str).str.upper()
    # Contiguous U.S. map
    gdf = gdf[~gdf["STUSPS"].isin({"AK","HI","PR","GU","VI","AS","MP","DC"})].copy()

    # 3) Shared color scale across both scenarios (min/max over both)
    vmin = float(np.nanmin(d["payback_years"].values))
    vmax = float(np.nanmax(d["payback_years"].values))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise ValueError("Payback values are not finite; check inputs.")

    # 4) Plot: two panels, legend only on the Policy panel
    fig, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)
    plt.rcParams["font.family"] = "Cabin"

    for ax, scen, title in [
        (axes[0], "baseline", f"Business-as-usual: Payback (Years) in {year}"),
        (axes[1], "policy",   f"$1 per Watt: Payback (Years) in {year}"),
    ]:
        sub = d[d["scenario"] == scen][["state_abbr","payback_years"]]
        m = gdf.merge(sub.rename(columns={"state_abbr":"STUSPS"}), on="STUSPS", how="left")

        m.plot(
            column="payback_years",
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            linewidth=0.6, edgecolor="grey",
            legend=(scen == "policy"),  # legend only on bottom facet
            legend_kwds={
                "label": "Payback (years)",
                "orientation": "horizontal",
                "shrink": 0.8,
                "pad": 0.02,
            },
            ax=ax,
            missing_kwds={"color": "#f5f5f5"},
        )
        #ax.set_title(title, fontsize=13)
        ax.axis("off")

        # Simple join sanity check
        missing = sorted(set(sub["state_abbr"]) - set(m["STUSPS"]))
        if missing:
            print(f"[Warning] Missing states in join for '{scen}': {missing}")

    plt.show()
    # Return the tidy values used to paint the map
    return d[["state_abbr","scenario","payback_years"]].reset_index(drop=True)


def build_eabs_calendar_timeseries(
    root_dir: str,
    run_id: str | None = None,
    strict_run_id: bool = True,
    level: str = "state",  # "state" or "US"
) -> "pd.DataFrame":
    """
    Calendar-year, adoption-weighted average bill metrics from 25-yr cohorts that adopt in prior years.
    Assumes these columns exist in each per-state CSV (new schema):
      - utility_bill_w_sys_pv_only, utility_bill_w_sys_pv_batt
      - utility_bill_wo_sys_pv_only, utility_bill_wo_sys_pv_batt
    Also uses: state_abbr, year, scenario, new_adopters, batt_adopters_added_this_year

    Outputs per geo×scenario×year:
      - adopters_active: total adopters present in that calendar year
      - avg_bill_with:   adoption-weighted avg bill WITH system (USD/yr per adopter)
      - avg_bill_wo:     adoption-weighted avg bill WITHOUT system (USD/yr per adopter)
      - eabs:            avg annual bill savings = avg_bill_wo - avg_bill_with
      - pct_savings:     eabs / avg_bill_wo
    """

    USECOLS = [
        "state_abbr","year","scenario","new_adopters","batt_adopters_added_this_year",
        "utility_bill_w_sys_pv_only","utility_bill_w_sys_pv_batt",
        "utility_bill_wo_sys_pv_only","utility_bill_wo_sys_pv_batt",
    ]

    frames = []
    for state_dir in discover_state_dirs(root_dir):
        b_csv, p_csv = find_state_files(state_dir, run_id=run_id, strict_run_id=strict_run_id)
        for path in (b_csv, p_csv):
            if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
                continue
            try:
                df = pd.read_csv(path, usecols=USECOLS)
            except Exception:
                continue
            if "scenario" not in df.columns:
                fn = os.path.basename(path).lower()
                df["scenario"] = "policy" if "policy" in fn else ("baseline" if "baseline" in fn else np.nan)
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["geo","scenario","year","adopters_active","avg_bill_with","avg_bill_wo","eabs","pct_savings"])

    x = pd.concat(frames, ignore_index=True)

    # Clean + derive cohort sizes
    x["year"] = pd.to_numeric(x["year"], errors="coerce")
    x["new_adopters"] = pd.to_numeric(x["new_adopters"], errors="coerce").fillna(0.0).clip(lower=0.0)
    x["batt_adopters_added_this_year"] = pd.to_numeric(x["batt_adopters_added_this_year"], errors="coerce").fillna(0.0).clip(lower=0.0)
    x["pv_batt_n"] = np.minimum(x["batt_adopters_added_this_year"], x["new_adopters"])
    x["pv_only_n"] = (x["new_adopters"] - x["pv_batt_n"]).clip(lower=0.0)

    # Parse arrays (length-25) — assumes _arr25 exists in your module
    x["bw_only"] = x["utility_bill_w_sys_pv_only"].apply(_arr25)
    x["bw_batt"] = x["utility_bill_w_sys_pv_batt"].apply(_arr25)
    x["bo_only"] = x["utility_bill_wo_sys_pv_only"].apply(_arr25)
    x["bo_batt"] = x["utility_bill_wo_sys_pv_batt"].apply(_arr25)

    # Calendar-year bounds (we only report within the modeled horizon)
    years = x["year"].dropna()
    if years.empty:
        return pd.DataFrame(columns=["geo","scenario","year","adopters_active","avg_bill_with","avg_bill_wo","eabs","pct_savings"])
    y_min, y_max = int(years.min()), int(years.max())

    bucket: dict[tuple[str,str,int], tuple[float,float,float]] = {}
    for r in x.itertuples(index=False):
        if pd.isna(r.year) or not r.scenario:
            continue
        y0 = int(r.year)
        geos = [r.state_abbr] if level.lower() == "state" else ["US"]

        pw_only, pw_batt = float(r.pv_only_n or 0.0), float(r.pv_batt_n or 0.0)
        a_bw_only, a_bw_batt = list(r.bw_only or []), list(r.bw_batt or [])
        a_bo_only, a_bo_batt = list(r.bo_only or []), list(r.bo_batt or [])

        # Roll forward each life-year k into calendar year y0+k
        for k in range(25):
            y = y0 + k
            if y < y_min or y > y_max:
                continue

            sum_with = (a_bw_only[k]*pw_only if k < len(a_bw_only) and pw_only>0 else 0.0) + \
                       (a_bw_batt[k]*pw_batt if k < len(a_bw_batt) and pw_batt>0 else 0.0)
            sum_wo   = (a_bo_only[k]*pw_only if k < len(a_bo_only) and pw_only>0 else 0.0) + \
                       (a_bo_batt[k]*pw_batt if k < len(a_bo_batt) and pw_batt>0 else 0.0)
            adopters = (pw_only if (k < len(a_bo_only) or k < len(a_bw_only)) else 0.0) + \
                       (pw_batt if (k < len(a_bo_batt) or k < len(a_bw_batt)) else 0.0)

            if (sum_with != 0.0) or (sum_wo != 0.0) or (adopters > 0.0):
                for geo in geos:
                    key = (geo, r.scenario, y)
                    sw, so, na = bucket.get(key, (0.0, 0.0, 0.0))
                    bucket[key] = (sw + sum_with, so + sum_wo, na + adopters)

    if not bucket:
        return pd.DataFrame(columns=["geo","scenario","year","adopters_active","avg_bill_with","avg_bill_wo","eabs","pct_savings"])

    rows = []
    for (geo, scen, y), (sum_with, sum_wo, adopters) in bucket.items():
        if adopters > 0:
            avg_with = sum_with / adopters
            avg_wo   = sum_wo   / adopters
            eabs     = avg_wo - avg_with
            pct      = (eabs / avg_wo) if avg_wo > 0 else 0.0
        else:
            # If no adopters active that year in this geo×scenario, define savings as 0 by convention.
            avg_with = 0.0
            avg_wo   = 0.0
            eabs     = 0.0
            pct      = 0.0
        rows.append((geo, scen, int(y), float(adopters), float(avg_with), float(avg_wo), float(eabs), float(pct)))

    out = pd.DataFrame(rows, columns=["geo","scenario","year","adopters_active","avg_bill_with","avg_bill_wo","eabs","pct_savings"])
    out = out.sort_values(["geo","scenario","year"]).reset_index(drop=True)
    return out


def summarize_us_eabs_for_year(eabs_ts: "pd.DataFrame", year: int = 2040) -> "pd.DataFrame":
    """
    Return a compact table for US only in `year`:
      ['scenario','year','adopters_active','avg_bill_wo','avg_bill_with','eabs','pct_savings']
    """
    d = eabs_ts[(eabs_ts["geo"] == "US") & (eabs_ts["year"] == int(year))].copy()
    keep = ["scenario","year","adopters_active","avg_bill_wo","avg_bill_with","eabs","pct_savings"]
    return d[keep].sort_values("scenario").reset_index(drop=True)

def table_top_states_by_eabs(
    eabs_ts: "pd.DataFrame",
    year: int = 2040,
    scenario: str = "policy",
    top_n: int = 5
) -> "pd.DataFrame":
    """
    Return top-N states by EABS in `year` for the chosen `scenario`,
    including avg bills with/without system.
      ['state_abbr','eabs','pct_savings','avg_bill_wo','avg_bill_with','adopters_active']
    """
    d = eabs_ts[(eabs_ts["geo"] != "US") & (eabs_ts["year"] == int(year)) & (eabs_ts["scenario"].str.lower() == scenario.lower())].copy()
    if d.empty:
        return pd.DataFrame(columns=["state_abbr","eabs","pct_savings","avg_bill_wo","avg_bill_with","adopters_active"])
    d = d.rename(columns={"geo":"state_abbr"})
    keep = ["state_abbr","eabs","pct_savings","avg_bill_wo","avg_bill_with","adopters_active"]
    return (d[keep].sort_values("pct_savings", ascending=False).head(int(top_n)).reset_index(drop=True))

def build_population_denominator_from_agents(
    root_dir: str,
    run_id: str | None = None,
    strict_run_id: bool = True,
) -> "pd.DataFrame":
    """
    Denominator from per-agent BASELINE CSVs.
    Uses ONLY `customers_in_bin` as the population weight and k=0 of the without-system bill array.

    Returns: ['state_abbr','year','households','denominator_usd']
    """

    pieces = []
    for state_dir in discover_state_dirs(root_dir):
        b_csv, _ = find_state_files(state_dir, run_id=run_id, strict_run_id=strict_run_id)
        if not b_csv:
            print("Csv not found")
            continue

        df = pd.read_csv(b_csv, low_memory=False)

        # If there's a scenario col, keep baseline rows; otherwise assume this is the baseline file.
        if "scenario" in df.columns:
            df = df[df["scenario"].str.lower() == "baseline"]

        # Pick wo_sys array column (prefer pv_only)
        wo_col = "utility_bill_wo_sys_pv_only" if "utility_bill_wo_sys_pv_only" in df.columns else "utility_bill_wo_sys_pv_batt"

        # Minimal subset
        d = df[["state_abbr", "year", "customers_in_bin", wo_col]].copy()

        # First life-year (k=0) of the without-system bill array
        d["wo0"] = d[wo_col].apply(_arr25).str[1]

        # Product and aggregate
        d["prod"] = d["customers_in_bin"] * d["wo0"]
        g = (d.groupby(["state_abbr","year"], as_index=False)
               .agg(households=("customers_in_bin","sum"),
                    denominator_usd=("prod","sum")))
        pieces.append(g)

    if not pieces:
        return pd.DataFrame(columns=["state_abbr","year","households","denominator_usd"])

    out = pd.concat(pieces, ignore_index=True)
    out["state_abbr"] = out["state_abbr"].astype(str).str.upper()
    return out.sort_values(["state_abbr","year"]).reset_index(drop=True)

def compute_us_percent_savings_internal(
    eabs_state_ts: "pd.DataFrame",
    denom_state_year: "pd.DataFrame",
) -> "pd.DataFrame":
    """
    Numerator from EABS (state × year × scenario):
      total_savings = eabs * adopters_active

    Denominator from per-agent baseline:
      denom_state_year: ['state_abbr','year','households','denominator_usd']

    Returns US series:
      ['scenario','year','percent_savings','total_savings_usd','denominator_usd']
    """

    s = eabs_state_ts.copy()
    s["state_abbr"] = s["geo"].astype(str).str.upper()
    s["total_savings_usd"] = (s["eabs"].fillna(0.0) * s["adopters_active"].fillna(0.0)).astype(float)

    D = denom_state_year.copy()
    D["state_abbr"] = D["state_abbr"].astype(str).str.upper()
    D["year"] = pd.to_numeric(D["year"], errors="coerce")

    st = s.merge(D[["state_abbr","year","denominator_usd"]], on=["state_abbr","year"], how="left")

    us = (
        st.groupby(["scenario","year"], as_index=False)
          .agg(total_savings_usd=("total_savings_usd","sum"),
               denominator_usd=("denominator_usd","sum"))
    )
    us["percent_savings"] = np.where(us["denominator_usd"] > 0,
                                     us["total_savings_usd"] / us["denominator_usd"],
                                     np.nan)
    us["scenario"] = us["scenario"].astype(str).str.lower()
    return us.sort_values(["scenario","year"]).reset_index(drop=True)

def plot_us_percent_savings(
    us_pct_ts: "pd.DataFrame",
    title: str = "Aggregate Annual % Bill Savings Across All Households"
) -> None:

    # Map scenarios → display labels
    label_map = {"baseline": "Business-as-usual", "policy": "$1 per watt"}
    hue_order = ["$1 per watt", "Business-as-usual"]
    palette = {"Business-as-usual": "#a2e0fc", "$1 per watt": "#1bb3ef"}

    d = us_pct_ts.copy()
    d["scenario"] = d["scenario"].astype(str).str.lower()
    d = d[d["scenario"].isin(label_map.keys())].copy()
    d["scenario_label"] = d["scenario"].map(label_map)

    # Style
    plt.rcParams["font.family"] = "Cabin"
    sns.set_context("talk")

    # Plot
    plt.figure(figsize=(10, 5), constrained_layout=True)
    ax = sns.lineplot(
        data=d, x="year", y="percent_savings",
        hue="scenario_label", hue_order=hue_order,
        palette=palette, marker="o", linewidth=2
    )

    #ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Percent of Residential Spend Avoided")
    ax.yaxis.set_major_formatter(lambda y, pos: f"{y*100:.0f}%")
    ax.legend(title=None, frameon=False, loc="best")
    ax.grid(True, axis="y", alpha=0.2)
    plt.show()


def table_top_states_by_percent_savings_internal(
    eabs_state_ts: "pd.DataFrame",
    denom_state_year: "pd.DataFrame",
    year: int = 2040,
    scenario: str = "policy",
    top_n: int = 5,
) -> "pd.DataFrame":
    """
    Top-N by economy-wide % savings in `year` for `scenario`.
    Columns:
      ['state_abbr','percent_savings','total_savings_usd','denominator_usd',
       'adopters_active','avg_bill_wo_adopters','avg_bill_with_adopters']
    """
    import pandas as pd
    import numpy as np

    s = eabs_state_ts.copy()
    s = s[(s["scenario"].str.lower() == scenario.lower()) & (s["year"] == int(year))].copy()
    s["state_abbr"] = s["geo"].astype(str).str.upper()
    s["total_savings_usd"] = (s["eabs"].fillna(0.0) * s["adopters_active"].fillna(0.0)).astype(float)

    D = denom_state_year.copy()
    D["state_abbr"] = D["state_abbr"].astype(str).str.upper()
    D = D[D["year"] == int(year)][["state_abbr","denominator_usd"]]

    out = s.merge(D, on="state_abbr", how="left")
    out["percent_savings"] = np.where(out["denominator_usd"] > 0,
                                      out["total_savings_usd"] / out["denominator_usd"],
                                      np.nan)
    out = out.rename(columns={"adopters_active":"adopters_active",
                              "avg_bill_wo":"avg_bill_wo_adopters",
                              "avg_bill_with":"avg_bill_with_adopters"})
    cols = ["state_abbr","percent_savings","total_savings_usd","denominator_usd",
            "adopters_active","avg_bill_wo_adopters","avg_bill_with_adopters"]
    return out[cols].sort_values("percent_savings", ascending=False).head(int(top_n)).reset_index(drop=True)

def compute_state_coincident_reduction(
    root_dir: str,
    run_id: Optional[str] = None,
    states: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Coincident peak reduction at the *baseline* peak hour (STATE-LEVEL ONLY):
      reduction = baseline_net_load(t_peak_baseline) - policy_net_load(t_peak_baseline)

    Returns per-state, per-year:
      ['state_abbr','year','coincident_reduction_mw']
    """
    rows = []

    # Discover state dirs
    state_dirs = discover_state_dirs(root_dir)
    if states:
        want = {s.strip().upper() for s in states if s and s.strip()}
        state_dirs = [sd for sd in state_dirs if os.path.basename(sd).upper() in want]
    if not state_dirs:
        return pd.DataFrame(columns=["state_abbr","year","coincident_reduction_mw"])

    for sd in state_dirs:
        state_abbr = os.path.basename(sd).upper()

        # --- STRICT: read only *_state_hourly.csv files, ignore legacy or RTO hourly ---
        if run_id:
            basedir = os.path.join(sd, str(run_id))
        else:
            basedir = sd
        f_base = os.path.join(basedir, "baseline_state_hourly.csv")
        f_pol  = os.path.join(basedir, "policy_state_hourly.csv")

        if not os.path.exists(f_base) or not os.path.exists(f_pol):
            # fall back: look for files that contain "state_hourly"
            cand = [p for p in glob.glob(os.path.join(basedir, "*state_hourly*.csv")) if os.path.isfile(p)]
            # try to pick baseline/policy among candidates
            f_base = next((p for p in cand if "baseline" in os.path.basename(p).lower()), None) or f_base
            f_pol  = next((p for p in cand if "policy"   in os.path.basename(p).lower()), None) or f_pol
        if not (os.path.exists(f_base) and os.path.exists(f_pol)):
            # No state-level hourly files; skip this state
            continue

        # Read both files; keep per-year arrays
        try:
            dfb = pd.read_csv(f_base, usecols=["year", "net_sum_text"])
            dfp = pd.read_csv(f_pol,  usecols=["year", "net_sum_text"])
        except Exception:
            continue

        # coerce year and drop bad
        for df in (dfb, dfp):
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df.dropna(subset=["year", "net_sum_text"], inplace=True)
            df["year"] = df["year"].astype(int)

        # Common years only (avoid mismatched sets)
        years = sorted(set(dfb["year"]).intersection(set(dfp["year"])))
        if not years:
            continue

        for y in years:
            try:
                arr_b_txt = dfb.loc[dfb["year"] == y].iloc[0]["net_sum_text"]
                arr_p_txt = dfp.loc[dfp["year"] == y].iloc[0]["net_sum_text"]
            except Exception:
                continue

            base_arr = _parse_array_text_to_floats(str(arr_b_txt))
            pol_arr  = _parse_array_text_to_floats(str(arr_p_txt))
            if not base_arr or not pol_arr:
                continue

            # Strict Jan1..Dec31 hourly index for the year; trim/pad arrays to match
            idx = pd.date_range(f"{y}-01-01 00:00:00", f"{y}-12-31 23:00:00", freq="h")

            def _to_series(a):
                a = np.asarray(a, dtype=float)
                if len(a) >= len(idx):
                    a = a[:len(idx)]
                else:
                    a = np.concatenate([a, np.full(len(idx) - len(a), np.nan)])
                return pd.Series(a, index=idx)

            s_base = _to_series(base_arr)
            s_pol  = _to_series(pol_arr).reindex(s_base.index)

            if not s_base.notna().any():
                continue

            # Baseline coincident-peak hour; reduction measured at the SAME timestamp
            peak_idx = int(np.nanargmax(s_base.values))
            reduction = float(s_base.iat[peak_idx]) - float(s_pol.iat[peak_idx])

            rows.append((state_abbr, y, reduction))

    out = pd.DataFrame(rows, columns=["state_abbr", "year", "coincident_reduction_mw"])
    return out.sort_values(["state_abbr", "year"]).reset_index(drop=True)

def choropleth_state_coincident_reduction(
    root_dir: str,
    shapefile_path: str,
    run_id: str | None = None,
    year: int = 2040,
    k_bins: int = 6,  # Jenks classes (similar to your installations map)
    states: Optional[Iterable[str]] = None,
) -> "pd.DataFrame":
    """
    Choropleth of state-level coincident peak reduction (GW) in `year`.

    Uses compute_state_coincident_reduction(...) which returns:
        ['state_abbr','year','coincident_reduction_mw']  (MW)
    We convert to GW for mapping.
    """
    # 1) Build per-state coincident reductions (MW)
    co = compute_state_coincident_reduction(
        root_dir=root_dir,
        run_id=run_id,
        states=states
    )
    if co.empty:
        raise ValueError("No coincident reduction rows found; ensure hourly CSVs exist for baseline/policy.")

    d = co.copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d = d[(d["year"] == int(year)) & d["state_abbr"].notna()].copy()
    if d.empty:
        raise ValueError(f"No coincident reduction rows found for year {year}.")

    d["state_abbr"] = d["state_abbr"].astype(str).str.upper()

    # Convert MW -> GW for plotting; keep clear column name
    d["coincident_reduction_gw"] = pd.to_numeric(d["coincident_reduction_mw"], errors="coerce") / 1000.0
    d = d[["state_abbr", "coincident_reduction_gw"]].copy()

    # 2) Load shapefile and prep join key
    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:5070")
    if "STUSPS" not in gdf.columns:
        for c in ("stusps", "STATE_ABBR", "STATE", "STATEFP", "STATEFP20"):
            if c in gdf.columns:
                gdf["STUSPS"] = gdf[c].astype(str).str.upper()
                break
    if "STUSPS" not in gdf.columns:
        raise ValueError("Shapefile must include a two-letter state code (e.g., STUSPS).")

    gdf["STUSPS"] = gdf["STUSPS"].astype(str).str.upper()
    # Contiguous U.S. only (match other maps)
    gdf = gdf[~gdf["STUSPS"].isin({"AK","HI","PR","GU","VI","AS","MP","DC"})].copy()

    plot_df = gdf.merge(d.rename(columns={"state_abbr": "STUSPS"}), on="STUSPS", how="left")

    # 3) Jenks (Fisher–Jenks) if available; quantiles fallback
    try:
        import mapclassify  # noqa: F401
        scheme = "fisher_jenks"
        plot_kwargs_extra = dict(k=int(k_bins))
    except Exception:
        scheme = "quantiles"
        plot_kwargs_extra = dict(k=int(k_bins))

    # 4) Plot
    plt.rcParams["font.family"] = "Cabin"

    # give a bit more width so the legend fits on the right
    fig, ax = plt.subplots(1, 1, figsize=(13.5, 7))

    plot_df.plot(
        column="coincident_reduction_gw",
        cmap="Blues",
        linewidth=0.6,
        edgecolor="grey",
        legend=True,
        scheme=scheme,
        legend_kwds={
            "title": "Coincident Peak Reduction (GW)",
            "ncols": 2,               # one column reads better vertically
            "fmt": "{:.1f}",
            "loc": "center left",     # place relative to bbox_to_anchor
            "bbox_to_anchor": (0.05, 0.15),  # just outside the right edge, vertically centered
            "borderaxespad": 0.0,
            "frameon": False,
        },
        ax=ax,
        missing_kwds={"color": "lightgray"},
        **plot_kwargs_extra,
    )

    # make room on the right so the legend doesn't overlap or get clipped
    plt.tight_layout(rect=(0, 0, 0.86, 1))  # leave ~14% margin on the right

    # --- Post-process legend to replace top label with ">" ---
    leg = ax.get_legend()
    if leg and hasattr(leg, "texts"):
        texts = leg.get_texts()
        if texts:
            last = texts[-1]
            last.set_text(f" > {last.get_text().split(',')[0].strip()}")

    ax.axis("off")
    plt.show()

def plot_state_weekly_peak_reduction(
    root_dir: str,
    state: str,
    run_id: Optional[str] = None,
    year: int = 2040,
    title: Optional[str] = None,
):
    """
    Plot weekly peak net load for baseline vs policy for a single state,
    shading the reduction (gap) between them, and mark the week containing
    the baseline coincident-peak hour with a dashed v-line.

    IMPORTANT: Reads ONLY baseline_state_hourly.csv / policy_state_hourly.csv
    so results match table_state_compare_coincident_reduction(...).
    """
    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # ---- 1) Locate state-level hourly files only ----
    state_dir = os.path.join(root_dir, state.upper())
    if run_id:
        state_dir = os.path.join(state_dir, str(run_id))
    f_base = os.path.join(state_dir, "baseline_state_hourly.csv")
    f_pol  = os.path.join(state_dir, "policy_state_hourly.csv")
    if not (os.path.exists(f_base) and os.path.exists(f_pol)):
        raise FileNotFoundError(f"Missing state-level hourly CSVs in {state_dir}")

    # ---- 2) Read arrays for the requested year ----
    def _read_arr(path: str) -> list[float]:
        df = pd.read_csv(path, usecols=["year", "net_sum_text"])
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        row = df.loc[df["year"] == int(year)]
        if row.empty:
            raise ValueError(f"{os.path.basename(path)} has no rows for year {year}")
        return _parse_array_text_to_floats(str(row.iloc[0]["net_sum_text"]))

    base_arr = _read_arr(f_base)
    pol_arr  = _read_arr(f_pol)

    # ---- 3) Build strict Jan1..Dec31 hourly index; trim/pad arrays to fit ----
    idx_year = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:00:00", freq="H")

    def _to_year_series(arr: list[float]) -> pd.Series:
        a = np.asarray(arr, dtype=float)
        if len(a) >= len(idx_year):
            a = a[:len(idx_year)]
        else:
            a = np.concatenate([a, np.full(len(idx_year) - len(a), np.nan)])
        return pd.Series(a, index=idx_year)

    base_h = _to_year_series(base_arr)
    pol_h  = _to_year_series(pol_arr).reindex(base_h.index)

    # ---- 4) Baseline coincident-peak hour (yearly); reduction at that same timestamp ----
    peak_idx = int(np.nanargmax(base_h.values))
    peak_ts  = base_h.index[peak_idx]
    reduction_at_peak = float(base_h.iat[peak_idx] - pol_h.iat[peak_idx])  # should match table (~3.0 GW for CA)

    # Map that timestamp to our week bins (W-MON, left-labeled)
    week_starts = base_h.resample("W-MON", label="left", closed="left").max()
    week_starts = week_starts[week_starts.index.year == int(year)].index  # left edges
    peak_week = int(np.searchsorted(week_starts.values, peak_ts.to_datetime64(), side="right"))
    peak_week = max(1, min(len(week_starts), peak_week))

    # ---- 5) Weekly maxima for both scenarios using identical bins; relabel as 1..N ----
    def _weekly_max_strict(s: pd.Series) -> pd.Series:
        wk = s.resample("W-MON", label="left", closed="left").max()
        wk = wk[wk.index.year == int(year)]
        wk.index = pd.Index(np.arange(1, len(wk) + 1), name="week")
        return wk

    wk_base = _weekly_max_strict(base_h)
    wk_pol  = _weekly_max_strict(pol_h)

    wk = pd.DataFrame({
        "baseline_mw": wk_base,
        "policy_mw":   wk_pol.reindex(wk_base.index),
    }).dropna()
    wk["reduction_mw"] = (wk["baseline_mw"] - wk["policy_mw"]).clip(lower=0.0)

    # ---- 6) Plot ----
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    x = wk.index.astype(int)
    ax.plot(x, wk["baseline_mw"].values, label="Baseline", linewidth=2.2)
    ax.plot(x, wk["policy_mw"].values,   label="Policy",   linewidth=2.2)
    ax.fill_between(x, wk["policy_mw"].values, wk["baseline_mw"].values,
                    where=wk["baseline_mw"].values > wk["policy_mw"].values,
                    alpha=0.3, label="Reduction")

    # Vertical line & annotation at the baseline coincident-peak week
    ax.axvline(peak_week, linestyle="--", alpha=0.7)
    ax.text(peak_week + 0.3, np.nanmax(wk["baseline_mw"].values),
            f"Coincident-peak week\n(reduction ≈ {reduction_at_peak/1000:.1f} GW)",
            va="top", fontsize=9)

    ax.set_xlim(1, int(x.max()))
    ax.set_xticks([1, 10, 20, 30, 40, int(x.max())])
    ax.set_xlabel("Week of Year")
    ax.set_ylabel("Weekly Peak Net Load (MW)")
    ax.set_title(title or f"{state.lower()} — Weekly Peak Net Load {year}")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()


def table_state_compare_coincident_reduction(
    root_dir: str,
    states: Optional[Iterable[str]] = None,
    run_id: Optional[str] = None,
    year: int = 2040,
    to_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Per-state coincident peak reduction (same logic as compare_states_peak_reduction)
    for the requested year.

    Columns:
      state, baseline_peak_mw, policy_at_baseline_peak_mw, reduction_mw,
      modeled_households, baseline_peak_mw_per_1000_hh, policy_at_baseline_peak_mw_per_1000_hh,
      reduction_mw_per_1000_hh, baseline_peak_ts, baseline_peak_week
    """
    import os, numpy as np, pandas as pd

    def _state_dir(st: str) -> str:
        d = os.path.join(root_dir, st.upper())
        return os.path.join(d, str(run_id)) if run_id else d

    def _read_hourly(state_dir: str, scen: str) -> list[float]:
        p = os.path.join(state_dir, f"{scen}_state_hourly.csv")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {scen}_state_hourly.csv in {state_dir}")
        df = pd.read_csv(p, usecols=["scenario","year","net_sum_text"])
        df["scenario"] = df["scenario"].astype(str).str.lower()
        df = df[(df["scenario"]==scen) & (pd.to_numeric(df["year"], errors="coerce")==int(year))]
        if df.empty:
            raise ValueError(f"No {scen} hourly rows for year {year} in {state_dir}")
        return _parse_array_text_to_floats(str(df.iloc[0]["net_sum_text"]))

    def _series_for_year(arr: list[float]) -> pd.Series:
        idx = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:00:00", freq="h")
        a = np.asarray(arr, dtype=float)
        if len(a) >= len(idx):
            a = a[:len(idx)]
        else:
            a = np.concatenate([a, np.full(len(idx)-len(a), np.nan)])
        return pd.Series(a, index=idx)

    def _week_index_from_strict_bins(ts: pd.Timestamp) -> int:
        week_starts = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="W-MON")
        week_starts = week_starts[week_starts.year==int(year)]
        pos = int(np.searchsorted(week_starts.values, ts.to_datetime64(), side="right")-1)
        return max(0, min(len(week_starts)-1, pos))+1

    def _sum_modeled_households(state_dir: str) -> float:
        # Prefer baseline.csv; fallback to policy.csv
        for fname in ("baseline.csv","policy.csv"):
            p = os.path.join(state_dir, fname)
            if os.path.exists(p) and os.path.getsize(p)>0:
                try:
                    df = pd.read_csv(p, low_memory=False)
                    if "year" in df.columns:
                        df["year"]=pd.to_numeric(df["year"], errors="coerce")
                        df = df[df["year"]==int(year)]
                    for col in ("customers_in_bin_initial","customers_in_bin","customers"):
                        if col in df.columns:
                            s = pd.to_numeric(df[col], errors="coerce").sum()
                            if pd.notna(s) and s>0:
                                return float(s)
                except Exception:
                    continue
        return float("nan")

    rows=[]
    if states is None:
        states = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))]

    for st in states:
        sdir=_state_dir(st)
        try:
            base_arr=_read_hourly(sdir,"baseline")
            pol_arr=_read_hourly(sdir,"policy")
        except Exception as e:
            rows.append({"state":st.upper(),"error":str(e)})
            continue

        s_base=_series_for_year(base_arr)
        s_pol=_series_for_year(pol_arr).reindex(s_base.index)

        # Baseline coincident-peak hour
        peak_idx=int(np.nanargmax(s_base.values))
        peak_ts=s_base.index[peak_idx]
        base_mw=float(s_base.iat[peak_idx])
        pol_mw=float(s_pol.iat[peak_idx])
        red_mw=base_mw - pol_mw
        peak_wk=_week_index_from_strict_bins(peak_ts)

        hh=_sum_modeled_households(sdir)

        # Per-1k intensity metrics
        if hh and hh > 0:
            denom = (hh / 1000.0)
            base_per_1k = base_mw / denom
            pol_per_1k  = pol_mw  / denom
            red_per_1k  = red_mw  / denom
        else:
            base_per_1k = pol_per_1k = red_per_1k = np.nan

        rows.append({
            "state": st.upper(),
            "baseline_peak_mw": base_mw,
            "policy_at_baseline_peak_mw": pol_mw,
            "reduction_mw": red_mw,
            "modeled_households": hh,
            "baseline_peak_mw_per_1000_hh": base_per_1k,
            "policy_at_baseline_peak_mw_per_1000_hh": pol_per_1k,
            "reduction_mw_per_1000_hh": red_per_1k,
            "baseline_peak_ts": peak_ts,
            "baseline_peak_week": peak_wk,
        })

    out = pd.DataFrame(rows).sort_values("reduction_mw", ascending=False).reset_index(drop=True)
    if to_csv:
        out.to_csv(to_csv, index=False)
    return out


def compute_adopters_vs_allhouseholds_prices_2040(
    root_dir: str,
    run_id: Optional[str] = None,
    states: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Compute state-level and U.S. average prices ($/kWh) in 2040 for:
      - adopters (actual, WITH system)
      - all households (counterfactual, WO system)

    All-households(wo) combines:
      • adopters' counterfactual 2040 per-customer bills (rolled forward from cohorts)
      • non-adopters' 2040 per-customer bill from the 2040 row (array index 0)
    and weights by counts so adopters + non-adopters = customers_in_bin (2040).

    Returns tidy DataFrame:
      ['state_abbr','metric','value'] with metrics:
        - 'adopters_price_with_sys_2040'
        - 'all_households_price_wo_sys_2040'
        - 'adopters_minus_all_households_delta_2040'
    """
    # ---------- Load + stitch per-state CSVs (baseline & policy) ----------
    state_dirs = discover_state_dirs(root_dir)  # existing helper  # :contentReference[oaicite:2]{index=2}
    if states:
        wanted = {s.strip().upper() for s in states if s and s.strip()}
        state_dirs = [sd for sd in state_dirs if os.path.basename(sd).upper() in wanted]
    parts: List[pd.DataFrame] = []
    for sd in state_dirs:
        b_csv, p_csv = find_state_files(sd, run_id=run_id, strict_run_id=False)  # existing helper  # :contentReference[oaicite:3]{index=3}
        for path in (b_csv, p_csv):
            if path:
                parts.append(_read_with_selected_cols(path))  # existing helper  # :contentReference[oaicite:4]{index=4}
    if not parts:
        return pd.DataFrame(columns=["state_abbr","metric","value"])
    df = pd.concat(parts, ignore_index=True)

    # ---------- Policy filter + basic coercions ----------
    if "scenario" in df.columns:
        df = df[df["scenario"].astype(str).str.lower() == "policy"].copy()
    if df.empty:
        return pd.DataFrame(columns=["state_abbr","metric","value"])

    for c in ("year","new_adopters","batt_adopters_added_this_year","number_of_adopters",
              "customers_in_bin","load_kwh_per_customer_in_bin_initial"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["state_abbr"] = df.get("state_abbr","").astype(str).str.strip().str.upper()

    # ---------- Cohort split (matches your savings logic) ----------
    # pv_batt_n = min(batt_adopters_added_this_year, new_adopters); pv_only_n = new_adopters - pv_batt_n  # :contentReference[oaicite:5]{index=5}
    df["pv_batt_n"] = np.minimum(df.get("batt_adopters_added_this_year", 0.0), df.get("new_adopters", 0.0)).clip(lower=0.0)
    df["pv_only_n"] = (df.get("new_adopters", 0.0) - df["pv_batt_n"]).clip(lower=0.0)

    # ---------- Parse 25-yr per-customer bill arrays ----------
    # Uses your 25-length parser wrapper _arr25  # :contentReference[oaicite:6]{index=6}
    df["w_only"] = df.get("utility_bill_w_sys_pv_only", np.nan).apply(_arr25)
    df["w_batt"] = df.get("utility_bill_w_sys_pv_batt", np.nan).apply(_arr25)
    df["wo_only"] = df.get("utility_bill_wo_sys_pv_only", np.nan).apply(_arr25)
    df["wo_batt"] = df.get("utility_bill_wo_sys_pv_batt", np.nan).apply(_arr25)

    TARGET_YEAR = 2040
    from collections import defaultdict
    acc = {
        "adopt_num": defaultdict(float),  # Σ(price_with_sys * adopters_count)
        "adopt_den": defaultdict(float),  # Σ(adopters_count)
        "allwo_num": defaultdict(float),  # Σ(price_wo_sys * all_households_piece_count)
        "allwo_den": defaultdict(float),  # Σ(all_households_piece_count)
    }

    # ---------- Adopters (WITH system) & adopters' WO-system contribution to ALL-households ----------
    for r in df.itertuples(index=False):
        state = getattr(r, "state_abbr", None)
        if not state:
            continue
        y0 = int(getattr(r, "year", 0) or 0)
        k = TARGET_YEAR - y0
        load = float(getattr(r, "load_kwh_per_customer_in_bin_initial", 0.0) or 0.0)
        if load <= 0 or k < 0 or k >= 25:
            continue

        # PV-only adopters in this cohort
        n_only = float(getattr(r, "pv_only_n", 0.0) or 0.0)
        if n_only > 0 and r.w_only and r.wo_only and k < len(r.w_only) and k < len(r.wo_only):
            price_with = float(r.w_only[k]) / load
            price_wo   = float(r.wo_only[k]) / load
            acc["adopt_num"][state] += price_with * n_only
            acc["adopt_den"][state] += n_only
            acc["allwo_num"][state] += price_wo * n_only     # adopters’ counterfactual slice for ALL-households
            acc["allwo_den"][state] += n_only

        # PV+batt adopters in this cohort
        n_batt = float(getattr(r, "pv_batt_n", 0.0) or 0.0)
        if n_batt > 0 and r.w_batt and r.wo_batt and k < len(r.w_batt) and k < len(r.wo_batt):
            price_with = float(r.w_batt[k]) / load
            price_wo   = float(r.wo_batt[k]) / load
            acc["adopt_num"][state] += price_with * n_batt
            acc["adopt_den"][state] += n_batt
            acc["allwo_num"][state] += price_wo * n_batt     # adopters’ counterfactual slice for ALL-households
            acc["allwo_den"][state] += n_batt

    # ---------- Non-adopters (2040 row) contribution to ALL-households ----------
    y2040 = df[df["year"] == TARGET_YEAR]
    for r in y2040.itertuples(index=False):
        state = getattr(r, "state_abbr", None)
        if not state:
            continue
        customers = float(getattr(r, "customers_in_bin", 0.0) or 0.0)
        adopters_cum = float(getattr(r, "number_of_adopters", 0.0) or 0.0)
        non_count = max(0.0, customers - adopters_cum)
        load = float(getattr(r, "load_kwh_per_customer_in_bin_initial", 0.0) or 0.0)
        if load <= 0 or non_count <= 0:
            continue

        # Prefer wo_sys pv_only array; fall back to pv_batt if needed.
        arr = getattr(r, "wo_only", None)
        if not arr or len(arr) == 0:
            arr = getattr(r, "wo_batt", None)
        if not arr or len(arr) == 0:
            continue

        price_non = float(arr[0]) / load  # 2040 row → index 0  # :contentReference[oaicite:7]{index=7}
        acc["allwo_num"][state] += price_non * non_count
        acc["allwo_den"][state] += non_count

    # ---------- Assemble state rows ----------
    rows = []
    states_done = sorted(set(list(acc["adopt_den"].keys()) + list(acc["allwo_den"].keys())))
    for s in states_done:
        adopters_p = (acc["adopt_num"][s] / acc["adopt_den"][s]) if acc["adopt_den"][s] > 0 else np.nan
        allwo_p    = (acc["allwo_num"][s] / acc["allwo_den"][s]) if acc["allwo_den"][s] > 0 else np.nan
        rows.extend([
            (s, "adopters_price_with_sys_2040", adopters_p),
            (s, "all_households_price_wo_sys_2040", allwo_p),
            (s, "adopters_minus_all_households_delta_2040",
                 adopters_p - allwo_p if np.isfinite(adopters_p) and np.isfinite(allwo_p) else np.nan),
        ])
    out_state = pd.DataFrame(rows, columns=["state_abbr","metric","value"])

    # ---------- U.S. aggregate ----------
    def _safe_div(a, b): return (a / b) if b and np.isfinite(b) and b > 0 else np.nan
    us_adopt = _safe_div(sum(acc["adopt_num"].values()), sum(acc["adopt_den"].values()))
    us_allwo = _safe_div(sum(acc["allwo_num"].values()), sum(acc["allwo_den"].values()))
    out_us = pd.DataFrame([
        ("US", "adopters_price_with_sys_2040", us_adopt),
        ("US", "all_households_price_wo_sys_2040", us_allwo),
        ("US", "adopters_minus_all_households_delta_2040",
            us_adopt - us_allwo if np.isfinite(us_adopt) and np.isfinite(us_allwo) else np.nan),
    ], columns=["state_abbr","metric","value"])

    return pd.concat([out_state, out_us], ignore_index=True)

def choropleth_pct_households_with_solar(
    outputs: Dict[str, pd.DataFrame],
    shapefile_path: str = "../../../data/states.shp",
    year: int = 2040,
    scenario: str = "policy",
    k_bins: int = 6,  # Jenks classes (good for % spreads)
) -> pd.DataFrame:
    """
    Choropleth of the % of households with rooftop solar by state.

    Uses outputs["totals"] from process_all_states(...) which contains state/year/scenario aggregates.

    For each state:
      pct_households_with_solar = number_of_adopters(year, scenario) / hh_from_shapefile

    Returns a tidy DataFrame:
      ['state_abbr','number_of_adopters','hh','pct_households_with_solar']
    """
    # ---- 1) Grab processed totals ----
    totals = outputs.get("totals", pd.DataFrame())
    if totals.empty:
        raise ValueError("outputs['totals'] is empty; run process_all_states(...) first.")

    need = {"state_abbr", "year", "scenario", "number_of_adopters"}
    missing = need - set(totals.columns)
    if missing:
        raise ValueError(f"outputs['totals'] missing columns: {sorted(missing)}")

    # Filter to requested year + scenario; aggregate adopters by state
    s = totals.loc[
        (totals["year"] == int(year)) &
        (totals["scenario"].astype(str).str.lower() == str(scenario).lower()),
        ["state_abbr", "number_of_adopters"]
    ].copy()

    if s.empty:
        raise ValueError(f"No rows for year={year}, scenario={scenario!r} in outputs['totals'].")

    s["state_abbr"] = s["state_abbr"].astype(str).str.upper()
    adopters = s.groupby("state_abbr", as_index=False)["number_of_adopters"].sum()

    # ---- 2) Load shapefile and prep join key + households (hh) ----
    gdf = gpd.read_file(shapefile_path).to_crs("EPSG:5070")
    # Find state code column as STUSPS
    if "STUSPS" not in gdf.columns:
        for c in ("stusps", "STATE_ABBR", "STATE", "STATEFP", "STATEFP20"):
            if c in gdf.columns:
                gdf["STUSPS"] = gdf[c].astype(str).str.upper()
                break
    if "STUSPS" not in gdf.columns:
        raise ValueError("Shapefile must include a two-letter state code (e.g., STUSPS).")

    # Ensure hh exists
    if "hh" not in gdf.columns:
        raise ValueError("Shapefile must include a 'hh' column for total households.")

    gdf["STUSPS"] = gdf["STUSPS"].astype(str).str.upper()
    # Contiguous U.S. only (matches your other maps)
    gdf = gdf[~gdf["STUSPS"].isin({"AK","HI","PR","GU","VI","AS","MP","DC"})].copy()

    # Join adopters to shapefile
    plot_df = gdf.merge(
        adopters.rename(columns={"state_abbr": "STUSPS"}),
        on="STUSPS",
        how="left"
    )

    # ---- 3) Compute percentage ----
    plot_df["hh"] = pd.to_numeric(plot_df["hh"], errors="coerce")
    plot_df["number_of_adopters"] = pd.to_numeric(plot_df["number_of_adopters"], errors="coerce")
    plot_df["pct_households_with_solar"] = np.where(
        (plot_df["hh"] > 0) & plot_df["number_of_adopters"].notna(),
        plot_df["number_of_adopters"] / plot_df["hh"],
        np.nan
    )

    # Prepare a tidy return table
    out = (
        plot_df[["STUSPS", "number_of_adopters", "hh", "pct_households_with_solar"]]
        .rename(columns={"STUSPS": "state_abbr"})
        .copy()
    )

    # ---- 4) Classing & Plot ----
    try:
        import mapclassify  # noqa: F401
        scheme = "fisher_jenks"
        plot_kwargs_extra = dict(k=int(k_bins))
    except Exception:
        scheme = "quantiles"
        plot_kwargs_extra = dict(k=int(k_bins))

    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Cabin"
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # We plot as percentage *100 to get easy legend labels like "12%"
    plot_df["pct_times_100"] = plot_df["pct_households_with_solar"] * 100.0

    plot_df.plot(
        column="pct_times_100",
        cmap="Blues",
        linewidth=0.6,
        edgecolor="grey",
        legend=True,
        scheme=scheme,
        legend_kwds={
            "title": f"Households with Solar ({year}, $1 per watt)",
            "ncols": 2,
            "fmt": "{:.0f}%",   # show as percent
            "loc": "lower left",
        },
        ax=ax,
        missing_kwds={"color": "lightgray"},
        **plot_kwargs_extra,
    )

    # Optional: make the top bin open-ended label (e.g., "> 18%")
    # Uncomment if you want that behavior here too:
    # leg = ax.get_legend()
    # if leg and hasattr(leg, "texts") and leg.get_texts():
    #     last = leg.get_texts()[-1]
    #     parts = last.get_text().split("–")
    #     if parts:
    #         lower = parts[0].strip()
    #         if lower.endswith("%"):
    #             last.set_text(f">{lower}")

    # ax.set_title(f"Percent of Households with Rooftop Solar in {year} ({scenario.title()})", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.show()
