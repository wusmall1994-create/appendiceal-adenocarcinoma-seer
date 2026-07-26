from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs"
OUT.mkdir(exist_ok=True)

INC_STAGE = ROOT / "adenocarcinoma_stage_age_incidence_seer17_2000_2023.csv"
HIST_ALL = ROOT / "adenocarcinoma_histology_year_seer17_2000_2023.csv"
HIST_YOUNG = ROOT / "adenocarcinoma_histology_year_age20_49_seer17_2000_2023.csv"
IBM = ROOT / "adenocarcinoma_ibm_age_mortality_seer17_2000_2023_with_ci.csv"


PER_100K = 100_000
BASE_PERIOD = (2004, 2008)
RECENT_PERIOD = (2019, 2023)
TREND_PERIOD = (2004, 2023)


def is_year(value: object) -> bool:
    return bool(re.fullmatch(r"\d{4}", str(value)))


def period_label(year: int) -> str | None:
    if BASE_PERIOD[0] <= year <= BASE_PERIOD[1]:
        return f"{BASE_PERIOD[0]}-{BASE_PERIOD[1]}"
    if RECENT_PERIOD[0] <= year <= RECENT_PERIOD[1]:
        return f"{RECENT_PERIOD[0]}-{RECENT_PERIOD[1]}"
    return None


def age_group(label: str) -> str:
    label = str(label)
    if label.startswith("00"):
        lo, hi = 0, 0
    elif label.startswith("90"):
        lo, hi = 90, 120
    else:
        nums = [int(x) for x in re.findall(r"\d+", label)]
        if len(nums) >= 2:
            lo, hi = nums[0], nums[1]
        elif len(nums) == 1:
            lo = hi = nums[0]
        else:
            return "Unknown"
    if hi < 20:
        return "<20"
    if lo < 50:
        return "20-49"
    if lo < 65:
        return "50-64"
    return "65+"


def stage_group(label: str) -> str:
    label = str(label)
    if label == "Localized_only":
        return "Localized"
    if label.startswith("Regional_"):
        return "Regional"
    if label == "Distant_sitesnodes_involved":
        return "Distant"
    if label in {"UnknownunstagedunspecifiedDCO", "Blanks"}:
        return "Unknown/blank"
    if label == "In_situ":
        return "In situ"
    return "Other"


def hist_code(label: str) -> str:
    return str(label).split(":")[0].strip()


def hist_group_from_code(code: str) -> str:
    mucinous = {"8470/3", "8471/3", "8472/3", "8480/3", "8481/3"}
    signet = {"8490/3"}
    conventional = {
        "8140/3",
        "8141/3",
        "8142/3",
        "8143/3",
        "8144/3",
        "8145/3",
        "8146/3",
        "8147/3",
        "8210/3",
        "8211/3",
        "8260/3",
        "8261/3",
        "8262/3",
        "8263/3",
    }
    if code in mucinous:
        return "Mucinous-type adenocarcinoma"
    if code in signet:
        return "Signet-ring cell carcinoma"
    if code in conventional:
        return "Conventional/intestinal adenocarcinoma"
    return "Other"


def rate(cases: float, pop: float) -> float:
    return cases / pop * PER_100K if pop else math.nan


def rate_ratio_lcl_ucl(c1: float, p1: float, c0: float, p0: float) -> tuple[float, float, float]:
    """Approximate rate ratio CI using log-rate SE from Poisson counts."""
    r1 = rate(c1, p1)
    r0 = rate(c0, p0)
    if c1 <= 0 or c0 <= 0 or not np.isfinite(r1) or not np.isfinite(r0):
        return math.nan, math.nan, math.nan
    rr = r1 / r0
    se = math.sqrt(1 / c1 + 1 / c0)
    return rr, math.exp(math.log(rr) - 1.96 * se), math.exp(math.log(rr) + 1.96 * se)


def weighted_loglinear_apc(years: np.ndarray, cases: np.ndarray, pops: np.ndarray) -> tuple[float, float, float, float]:
    """Preliminary APC from WLS log(rate) ~ year, weighted by cases.

    This is not a substitute for Joinpoint permutation models; it is a transparent
    screening model for the first analysis pass.
    """
    mask = (cases > 0) & (pops > 0)
    years = years[mask].astype(float)
    cases = cases[mask].astype(float)
    pops = pops[mask].astype(float)
    if len(years) < 3:
        return math.nan, math.nan, math.nan, math.nan
    y = np.log(cases / pops)
    x = years - years.mean()
    X = np.column_stack([np.ones_like(x), x])
    w = np.maximum(cases, 1.0)
    sw = np.sqrt(w)
    Xw = X * sw[:, None]
    yw = y * sw
    beta = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    resid = y - X @ beta
    dof = max(len(y) - 2, 1)
    sigma2 = float(np.sum(w * resid**2) / dof)
    cov = sigma2 * np.linalg.inv(X.T @ (w[:, None] * X))
    slope = float(beta[1])
    se = math.sqrt(float(cov[1, 1]))
    apc = (math.exp(slope) - 1) * 100
    lcl = (math.exp(slope - 1.96 * se) - 1) * 100
    ucl = (math.exp(slope + 1.96 * se) - 1) * 100
    return apc, lcl, ucl, se


def norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def poisson_irls(
    y: np.ndarray,
    X: np.ndarray,
    offset: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, float, int, float]:
    """Poisson GLM with log link and offset using IRLS.

    Returns beta, model-based covariance matrix, log-likelihood, and iterations.
    This deliberately avoids statsmodels/scipy so the analysis remains runnable in
    the bundled Python environment.
    """
    y = y.astype(float)
    X = X.astype(float)
    offset = offset.astype(float)
    beta = np.zeros(X.shape[1], dtype=float)
    beta[0] = math.log((y.sum() + 0.5) / np.exp(offset).sum())
    for it in range(1, max_iter + 1):
        eta = np.clip(offset + X @ beta, -30, 30)
        mu = np.exp(eta)
        z = eta - offset + (y - mu) / np.maximum(mu, 1e-12)
        w = np.maximum(mu, 1e-12)
        XtW = X.T * w
        XtWX = XtW @ X
        XtWz = XtW @ z
        beta_new = np.linalg.pinv(XtWX) @ XtWz
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = np.clip(offset + X @ beta, -30, 30)
    mu = np.exp(eta)
    XtWX = (X.T * np.maximum(mu, 1e-12)) @ X
    cov = np.linalg.pinv(XtWX)
    loglik = float(np.sum(y * np.log(np.maximum(mu, 1e-12)) - mu))
    pearson = np.sum((y - mu) ** 2 / np.maximum(mu, 1e-12))
    dispersion = float(max(1.0, pearson / max(len(y) - X.shape[1], 1)))
    return beta, cov, loglik, it, dispersion


def poisson_age_year_interaction(
    df: pd.DataFrame,
    case_col: str,
    pop_col: str,
    outcome_label: str,
    strata_cols: list[str],
    exclude_years: set[int] | None = None,
) -> pd.DataFrame:
    """Fit log-rate ~ centered year * age group within each stratum.

    Reference age group is 20-49. Output gives APC by age group and interaction
    p-values for whether 50-64 or 65+ slopes differ from 20-49.
    """
    d = df[(df["year"] >= TREND_PERIOD[0]) & (df["year"] <= TREND_PERIOD[1])].copy()
    if exclude_years:
        d = d[~d["year"].isin(exclude_years)].copy()
    d = d[d["age_group"].isin(["20-49", "50-64", "65+"])]
    rows = []
    grouped = [((), d)] if not strata_cols else d.groupby(strata_cols, dropna=False)
    for keys, g in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        g = g.sort_values(["year", "age_group"]).copy()
        g = g[g[pop_col] > 0]
        if g[case_col].sum() <= 0 or g.empty:
            continue
        year_c = g["year"].to_numpy(dtype=float) - TREND_PERIOD[0]
        age_5064 = (g["age_group"] == "50-64").astype(float).to_numpy()
        age_65 = (g["age_group"] == "65+").astype(float).to_numpy()
        X = np.column_stack(
            [
                np.ones(len(g)),
                year_c,
                age_5064,
                age_65,
                year_c * age_5064,
                year_c * age_65,
            ]
        )
        beta, cov, loglik, it, dispersion = poisson_irls(
            g[case_col].to_numpy(dtype=float),
            X,
            np.log(g[pop_col].to_numpy(dtype=float)),
        )

        contrasts = {
            "20-49": np.array([0, 1, 0, 0, 0, 0], dtype=float),
            "50-64": np.array([0, 1, 0, 0, 1, 0], dtype=float),
            "65+": np.array([0, 1, 0, 0, 0, 1], dtype=float),
        }
        for age, cvec in contrasts.items():
            slope = float(cvec @ beta)
            se = math.sqrt(float(cvec @ cov @ cvec))
            se_q = se * math.sqrt(dispersion)
            apc = (math.exp(slope) - 1) * 100
            lcl = (math.exp(slope - 1.96 * se) - 1) * 100
            ucl = (math.exp(slope + 1.96 * se) - 1) * 100
            lcl_q = (math.exp(slope - 1.96 * se_q) - 1) * 100
            ucl_q = (math.exp(slope + 1.96 * se_q) - 1) * 100
            z = slope / se if se > 0 else math.nan
            z_q = slope / se_q if se_q > 0 else math.nan
            p = 2 * (1 - norm_cdf(abs(z))) if np.isfinite(z) else math.nan
            p_q = 2 * (1 - norm_cdf(abs(z_q))) if np.isfinite(z_q) else math.nan
            row = {col: val for col, val in zip(strata_cols, keys)}
            row.update(
                {
                    "outcome": outcome_label,
                    "age_group": age,
                    "apc_percent_poisson": apc,
                    "apc_lcl": lcl,
                    "apc_ucl": ucl,
                    "apc_p": p,
                    "apc_lcl_quasi": lcl_q,
                    "apc_ucl_quasi": ucl_q,
                    "apc_p_quasi": p_q,
                    "pearson_dispersion": dispersion,
                    "cases_total": float(g.loc[g["age_group"] == age, case_col].sum()),
                    "model_iterations": it,
                }
            )
            if age == "50-64":
                diff_c = np.array([0, 0, 0, 0, 1, 0], dtype=float)
                diff = float(diff_c @ beta)
                diff_se = math.sqrt(float(diff_c @ cov @ diff_c))
                diff_se_q = diff_se * math.sqrt(dispersion)
                row["slope_ratio_vs_20_49"] = math.exp(diff)
                row["interaction_p_vs_20_49"] = 2 * (1 - norm_cdf(abs(diff / diff_se))) if diff_se > 0 else math.nan
                row["interaction_p_vs_20_49_quasi"] = 2 * (1 - norm_cdf(abs(diff / diff_se_q))) if diff_se_q > 0 else math.nan
            elif age == "65+":
                diff_c = np.array([0, 0, 0, 0, 0, 1], dtype=float)
                diff = float(diff_c @ beta)
                diff_se = math.sqrt(float(diff_c @ cov @ diff_c))
                diff_se_q = diff_se * math.sqrt(dispersion)
                row["slope_ratio_vs_20_49"] = math.exp(diff)
                row["interaction_p_vs_20_49"] = 2 * (1 - norm_cdf(abs(diff / diff_se))) if diff_se > 0 else math.nan
                row["interaction_p_vs_20_49_quasi"] = 2 * (1 - norm_cdf(abs(diff / diff_se_q))) if diff_se_q > 0 else math.nan
            else:
                row["slope_ratio_vs_20_49"] = 1.0
                row["interaction_p_vs_20_49"] = math.nan
                row["interaction_p_vs_20_49_quasi"] = math.nan
            rows.append(row)
    return pd.DataFrame(rows)


def poisson_simple_trend(
    df: pd.DataFrame,
    group_cols: list[str],
    case_col: str,
    pop_col: str,
    outcome_label: str,
    exclude_years: set[int] | None = None,
) -> pd.DataFrame:
    """Poisson trend model log-rate ~ centered year within each group."""
    d = df[(df["year"] >= TREND_PERIOD[0]) & (df["year"] <= TREND_PERIOD[1])].copy()
    if exclude_years:
        d = d[~d["year"].isin(exclude_years)].copy()
    rows = []
    for keys, g in d.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        g = g.sort_values("year")
        g = g[g[pop_col] > 0]
        if g[case_col].sum() <= 0 or len(g) < 3:
            continue
        year_c = g["year"].to_numpy(dtype=float) - TREND_PERIOD[0]
        X = np.column_stack([np.ones(len(g)), year_c])
        beta, cov, loglik, it, dispersion = poisson_irls(
            g[case_col].to_numpy(dtype=float),
            X,
            np.log(g[pop_col].to_numpy(dtype=float)),
        )
        slope = float(beta[1])
        se = math.sqrt(float(cov[1, 1]))
        se_q = se * math.sqrt(dispersion)
        z = slope / se if se > 0 else math.nan
        z_q = slope / se_q if se_q > 0 else math.nan
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "outcome": outcome_label,
                "apc_percent_poisson": (math.exp(slope) - 1) * 100,
                "apc_lcl": (math.exp(slope - 1.96 * se) - 1) * 100,
                "apc_ucl": (math.exp(slope + 1.96 * se) - 1) * 100,
                "apc_p": 2 * (1 - norm_cdf(abs(z))) if np.isfinite(z) else math.nan,
                "apc_lcl_quasi": (math.exp(slope - 1.96 * se_q) - 1) * 100,
                "apc_ucl_quasi": (math.exp(slope + 1.96 * se_q) - 1) * 100,
                "apc_p_quasi": 2 * (1 - norm_cdf(abs(z_q))) if np.isfinite(z_q) else math.nan,
                "pearson_dispersion": dispersion,
                "cases_total": float(g[case_col].sum()),
                "model_iterations": it,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def load_stage_incidence() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(INC_STAGE, na_values=["~", " "])
    df = df[df["Year of diagnosis"].map(is_year)].copy()
    df["year"] = df["Year of diagnosis"].astype(int)
    df["age_group"] = df["Age recode with <1 year olds and 90+"].map(age_group)
    df["stage_group"] = df["Combined Summary Stage with Expanded Regional Codes (2004+)"].map(stage_group)
    df["cases"] = pd.to_numeric(df["Count"], errors="coerce").fillna(0)
    df["population"] = pd.to_numeric(df["Pop"], errors="coerce").fillna(0)

    analytic_stages = ["Localized", "Regional", "Distant", "Unknown/blank"]
    dfa = df[df["stage_group"].isin(analytic_stages)].copy()

    # stage-specific rows: counts are additive; population must be counted once per year-age-stage
    cell = (
        dfa.groupby(["year", "age_group", "stage_group", "Age recode with <1 year olds and 90+"], as_index=False)
        .agg(cases=("cases", "sum"), population=("population", "max"))
    )
    yearly_stage = (
        cell.groupby(["year", "age_group", "stage_group"], as_index=False)
        .agg(cases=("cases", "sum"), population=("population", "sum"))
    )
    yearly_stage["rate_per_100k"] = yearly_stage.apply(lambda r: rate(r.cases, r.population), axis=1)

    # overall invasive/unstaged burden: stage counts summed, population once per year-age
    overall_cell = (
        dfa.groupby(["year", "age_group", "Age recode with <1 year olds and 90+"], as_index=False)
        .agg(cases=("cases", "sum"), population=("population", "max"))
    )
    yearly_overall = (
        overall_cell.groupby(["year", "age_group"], as_index=False)
        .agg(cases=("cases", "sum"), population=("population", "sum"))
    )
    yearly_overall["stage_group"] = "Overall"
    yearly_overall["rate_per_100k"] = yearly_overall.apply(lambda r: rate(r.cases, r.population), axis=1)
    return yearly_stage, yearly_overall


def load_ibm() -> pd.DataFrame:
    df = pd.read_csv(IBM, na_values=["~", " "])
    df = df[df["Year of death recode"].map(is_year)].copy()
    df["year"] = df["Year of death recode"].astype(int)
    df["age_group"] = df["Age at death with <1 yr olds and 90+"].map(age_group)
    df["cases"] = pd.to_numeric(df["Count"], errors="coerce").fillna(0)
    df["population"] = pd.to_numeric(df["Population"], errors="coerce").fillna(0)
    yearly = (
        df.groupby(["year", "age_group"], as_index=False)
        .agg(deaths=("cases", "sum"), population=("population", "sum"))
    )
    yearly["ibm_rate_per_100k"] = yearly.apply(lambda r: rate(r.deaths, r.population), axis=1)
    return yearly


def load_histology(path: Path, cohort_label: str) -> pd.DataFrame:
    df = pd.read_csv(path, na_values=["~", " "])
    df = df[df["Year of diagnosis"].map(is_year)].copy()
    df["year"] = df["Year of diagnosis"].astype(int)
    df["hist_code"] = df["ICD-O-3 Hist/behav, malignant"].map(hist_code)
    df["hist_group"] = df["hist_code"].map(hist_group_from_code)
    df["cases"] = pd.to_numeric(df["Count"], errors="coerce").fillna(0)
    df["population"] = pd.to_numeric(df["Population"], errors="coerce").fillna(0)
    df = df[df["hist_group"] != "Other"].copy()
    # Sum counts by histology group; denominator once per year from maximum population.
    counts = df.groupby(["year", "hist_group"], as_index=False).agg(cases=("cases", "sum"))
    pops = df.groupby(["year"], as_index=False).agg(population=("population", "max"))
    out = counts.merge(pops, on="year", how="left")
    out["rate_per_100k"] = out.apply(lambda r: rate(r.cases, r.population), axis=1)
    out["cohort"] = cohort_label
    return out


def period_compare(df: pd.DataFrame, group_cols: list[str], case_col: str, pop_col: str) -> pd.DataFrame:
    d = df.copy()
    d["period"] = d["year"].map(period_label)
    d = d[d["period"].notna()].copy()
    agg = d.groupby(["period"] + group_cols, as_index=False).agg(cases=(case_col, "sum"), population=(pop_col, "sum"))
    agg["rate_per_100k"] = agg.apply(lambda r: rate(r.cases, r.population), axis=1)
    base = agg[agg["period"] == f"{BASE_PERIOD[0]}-{BASE_PERIOD[1]}"].drop(columns="period")
    recent = agg[agg["period"] == f"{RECENT_PERIOD[0]}-{RECENT_PERIOD[1]}"].drop(columns="period")
    m = recent.merge(base, on=group_cols, suffixes=("_recent", "_base"), how="outer")
    rr = m.apply(
        lambda r: rate_ratio_lcl_ucl(r.cases_recent, r.population_recent, r.cases_base, r.population_base),
        axis=1,
        result_type="expand",
    )
    m["rate_ratio"] = rr[0]
    m["rate_ratio_lcl"] = rr[1]
    m["rate_ratio_ucl"] = rr[2]
    return m


def trend_table(df: pd.DataFrame, group_cols: list[str], case_col: str, pop_col: str, label: str) -> pd.DataFrame:
    d = df[(df["year"] >= TREND_PERIOD[0]) & (df["year"] <= TREND_PERIOD[1])].copy()
    rows = []
    for keys, g in d.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        g = g.sort_values("year")
        apc, lcl, ucl, se = weighted_loglinear_apc(
            g["year"].to_numpy(),
            g[case_col].to_numpy(),
            g[pop_col].to_numpy(),
        )
        row = {col: val for col, val in zip(group_cols, keys)}
        row.update(
            {
                "analysis": label,
                "years": f"{TREND_PERIOD[0]}-{TREND_PERIOD[1]}",
                "cases_total": float(g[case_col].sum()),
                "apc_percent_wls": apc,
                "apc_lcl": lcl,
                "apc_ucl": ucl,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def simple_svg_line(
    data: pd.DataFrame,
    x: str,
    y: str,
    series: str,
    title: str,
    output: Path,
    width: int = 900,
    height: int = 520,
) -> None:
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]
    margin = dict(left=80, right=180, top=60, bottom=70)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    xs = data[x].astype(float)
    ys = data[y].astype(float)
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = 0.0, float(ys.max() * 1.12 if ys.max() > 0 else 1)

    def sx(v: float) -> float:
        return margin["left"] + (v - xmin) / (xmax - xmin) * plot_w

    def sy(v: float) -> float:
        return margin["top"] + (ymax - v) / (ymax - ymin) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="30" text-anchor="middle" font-family="Arial" font-size="20">{title}</text>',
        f'<line x1="{margin["left"]}" y1="{margin["top"]+plot_h}" x2="{margin["left"]+plot_w}" y2="{margin["top"]+plot_h}" stroke="#333"/>',
        f'<line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"]+plot_h}" stroke="#333"/>',
    ]
    for yr in range(int(xmin), int(xmax) + 1, 5):
        px = sx(yr)
        lines.append(f'<line x1="{px:.1f}" y1="{margin["top"]+plot_h}" x2="{px:.1f}" y2="{margin["top"]+plot_h+5}" stroke="#333"/>')
        lines.append(f'<text x="{px:.1f}" y="{height-28}" text-anchor="middle" font-family="Arial" font-size="12">{yr}</text>')
    for frac in np.linspace(0, 1, 6):
        val = ymin + frac * (ymax - ymin)
        py = sy(val)
        lines.append(f'<line x1="{margin["left"]-5}" y1="{py:.1f}" x2="{margin["left"]}" y2="{py:.1f}" stroke="#333"/>')
        lines.append(f'<line x1="{margin["left"]}" y1="{py:.1f}" x2="{margin["left"]+plot_w}" y2="{py:.1f}" stroke="#ddd"/>')
        lines.append(f'<text x="{margin["left"]-10}" y="{py+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{val:.2f}</text>')
    lines.append(f'<text x="{margin["left"]+plot_w/2}" y="{height-8}" text-anchor="middle" font-family="Arial" font-size="13">Year</text>')
    lines.append(f'<text x="18" y="{margin["top"]+plot_h/2}" text-anchor="middle" transform="rotate(-90 18 {margin["top"]+plot_h/2})" font-family="Arial" font-size="13">Rate per 100,000</text>')

    for i, (name, g) in enumerate(data.sort_values(x).groupby(series)):
        color = colors[i % len(colors)]
        pts = " ".join(f'{sx(float(r[x])):.1f},{sy(float(r[y])):.1f}' for _, r in g.iterrows())
        lines.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for _, r in g.iterrows():
            lines.append(f'<circle cx="{sx(float(r[x])):.1f}" cy="{sy(float(r[y])):.1f}" r="2.5" fill="{color}"/>')
        ly = margin["top"] + 25 + i * 22
        lx = margin["left"] + plot_w + 25
        lines.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+28}" y2="{ly}" stroke="{color}" stroke-width="2.5"/>')
        lines.append(f'<text x="{lx+36}" y="{ly+4}" font-family="Arial" font-size="12">{name}</text>')
    lines.append("</svg>")
    output.write_text("\n".join(lines), encoding="utf-8")


def forest_svg(
    data: pd.DataFrame,
    label_col: str,
    estimate_col: str,
    lcl_col: str,
    ucl_col: str,
    title: str,
    output: Path,
    x_label: str = "Annual percent change (%)",
    width: int = 980,
) -> None:
    d = data.copy().reset_index(drop=True)
    height = max(280, 80 + len(d) * 34)
    left = 360
    right = 50
    top = 58
    bottom = 50
    plot_w = width - left - right
    vals = pd.concat([d[lcl_col], d[ucl_col], pd.Series([0.0])]).astype(float)
    xmin = math.floor(vals.min() - 1)
    xmax = math.ceil(vals.max() + 1)
    if xmin == xmax:
        xmax = xmin + 1
    row_gap = (height - top - bottom) / max(len(d), 1)

    def sx(v: float) -> float:
        return left + (v - xmin) / (xmax - xmin) * plot_w

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{title}</text>',
        f'<line x1="{left}" y1="{top-10}" x2="{left+plot_w}" y2="{top-10}" stroke="#333"/>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{left+plot_w}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{sx(0):.1f}" y1="{top-18}" x2="{sx(0):.1f}" y2="{height-bottom}" stroke="#999" stroke-dasharray="4 3"/>',
    ]
    for tick in range(int(xmin), int(xmax) + 1):
        if tick % 2 == 0 or tick == 0:
            x = sx(tick)
            lines.append(f'<line x1="{x:.1f}" y1="{height-bottom}" x2="{x:.1f}" y2="{height-bottom+5}" stroke="#333"/>')
            lines.append(f'<text x="{x:.1f}" y="{height-bottom+22}" text-anchor="middle" font-family="Arial" font-size="11">{tick}</text>')
    colors = {"incidence": "#1f77b4", "ibm": "#d62728", "histology": "#2ca02c"}
    for i, r in d.iterrows():
        y = top + i * row_gap + row_gap / 2
        est = float(r[estimate_col])
        lcl = float(r[lcl_col])
        ucl = float(r[ucl_col])
        outcome = str(r.get("outcome_family", "incidence"))
        color = colors.get(outcome, "#333")
        lines.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="12">{r[label_col]}</text>')
        lines.append(f'<line x1="{sx(lcl):.1f}" y1="{y:.1f}" x2="{sx(ucl):.1f}" y2="{y:.1f}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<circle cx="{sx(est):.1f}" cy="{y:.1f}" r="4.5" fill="{color}"/>')
        lines.append(
            f'<text x="{left+plot_w+8}" y="{y+4:.1f}" font-family="Arial" font-size="11">'
            f'{est:.2f} ({lcl:.2f}, {ucl:.2f})</text>'
        )
    lines.append(f'<text x="{left+plot_w/2}" y="{height-8}" text-anchor="middle" font-family="Arial" font-size="12">{x_label}</text>')
    lines.append("</svg>")
    output.write_text("\n".join(lines), encoding="utf-8")


def write_joinpoint_inputs(yearly_incidence: pd.DataFrame, ibm: pd.DataFrame) -> None:
    jp = yearly_incidence[
        (yearly_incidence["year"] >= 2004)
        & yearly_incidence["age_group"].isin(["20-49", "50-64", "65+"])
        & yearly_incidence["stage_group"].isin(["Overall", "Distant", "Regional", "Localized"])
    ].copy()
    jp = jp.rename(
        columns={
            "year": "Year",
            "age_group": "AgeGroup",
            "stage_group": "Stage",
            "cases": "Count",
            "population": "Population",
            "rate_per_100k": "Rate",
        }
    )[["Year", "AgeGroup", "Stage", "Count", "Population", "Rate"]]
    jp.to_csv(OUT / "joinpoint_input_incidence_stage_age_2004_2023.csv", index=False, encoding="utf-8-sig")

    jp_ibm = ibm[(ibm["year"] >= 2004) & ibm["age_group"].isin(["20-49", "50-64", "65+"])].copy()
    jp_ibm = jp_ibm.rename(
        columns={
            "year": "Year",
            "age_group": "AgeGroup",
            "deaths": "Count",
            "population": "Population",
            "ibm_rate_per_100k": "Rate",
        }
    )[["Year", "AgeGroup", "Count", "Population", "Rate"]]
    jp_ibm.to_csv(OUT / "joinpoint_input_ibm_age_2004_2023.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    audit_rows = []
    for p in [INC_STAGE, HIST_ALL, HIST_YOUNG, IBM]:
        df = pd.read_csv(p)
        audit_rows.append({"file": p.name, "rows": len(df), "columns": len(df.columns), "column_names": " | ".join(df.columns)})
    pd.DataFrame(audit_rows).to_csv(OUT / "data_audit.csv", index=False, encoding="utf-8-sig")

    yearly_stage, yearly_overall = load_stage_incidence()
    yearly_incidence = pd.concat([yearly_stage, yearly_overall], ignore_index=True)
    yearly_incidence.to_csv(OUT / "yearly_incidence_stage_age_rates.csv", index=False, encoding="utf-8-sig")

    inc_period = period_compare(
        yearly_incidence[yearly_incidence["age_group"].isin(["20-49", "50-64", "65+"])],
        ["age_group", "stage_group"],
        "cases",
        "population",
    )
    inc_period.to_csv(OUT / "incidence_period_comparison_2004_2008_vs_2019_2023.csv", index=False, encoding="utf-8-sig")

    inc_trends = trend_table(
        yearly_incidence[
            yearly_incidence["age_group"].isin(["20-49", "50-64", "65+"])
            & yearly_incidence["stage_group"].isin(["Overall", "Localized", "Regional", "Distant"])
        ],
        ["age_group", "stage_group"],
        "cases",
        "population",
        "incidence",
    )
    inc_trends.to_csv(OUT / "incidence_loglinear_apc_2004_2023.csv", index=False, encoding="utf-8-sig")
    inc_poisson = poisson_age_year_interaction(
        yearly_incidence[
            yearly_incidence["age_group"].isin(["20-49", "50-64", "65+"])
            & yearly_incidence["stage_group"].isin(["Overall", "Localized", "Regional", "Distant"])
        ],
        "cases",
        "population",
        "incidence",
        ["stage_group"],
    )
    inc_poisson.to_csv(OUT / "poisson_incidence_age_year_interaction_2004_2023.csv", index=False, encoding="utf-8-sig")
    inc_poisson_no2020 = poisson_age_year_interaction(
        yearly_incidence[
            yearly_incidence["age_group"].isin(["20-49", "50-64", "65+"])
            & yearly_incidence["stage_group"].isin(["Overall", "Localized", "Regional", "Distant"])
        ],
        "cases",
        "population",
        "incidence_no_2020",
        ["stage_group"],
        exclude_years={2020},
    )
    inc_poisson_no2020.to_csv(
        OUT / "sensitivity_no2020_poisson_incidence_age_year_interaction_2004_2023.csv",
        index=False,
        encoding="utf-8-sig",
    )

    hist_all = load_histology(HIST_ALL, "all_ages")
    hist_young = load_histology(HIST_YOUNG, "age20_49")
    hist = pd.concat([hist_all, hist_young], ignore_index=True)
    hist.to_csv(OUT / "yearly_histology_rates.csv", index=False, encoding="utf-8-sig")
    hist_period = period_compare(hist, ["cohort", "hist_group"], "cases", "population")
    hist_period.to_csv(OUT / "histology_period_comparison_2004_2008_vs_2019_2023.csv", index=False, encoding="utf-8-sig")
    hist_trends = trend_table(hist, ["cohort", "hist_group"], "cases", "population", "histology")
    hist_trends.to_csv(OUT / "histology_loglinear_apc_2004_2023.csv", index=False, encoding="utf-8-sig")
    hist_poisson = poisson_simple_trend(hist, ["cohort", "hist_group"], "cases", "population", "histology")
    hist_poisson.to_csv(OUT / "poisson_histology_trends_2004_2023.csv", index=False, encoding="utf-8-sig")
    hist_poisson_no2020 = poisson_simple_trend(
        hist,
        ["cohort", "hist_group"],
        "cases",
        "population",
        "histology_no_2020",
        exclude_years={2020},
    )
    hist_poisson_no2020.to_csv(OUT / "sensitivity_no2020_poisson_histology_trends_2004_2023.csv", index=False, encoding="utf-8-sig")

    ibm = load_ibm()
    ibm.to_csv(OUT / "yearly_ibm_age_rates.csv", index=False, encoding="utf-8-sig")
    ibm_period = period_compare(ibm[ibm["age_group"].isin(["20-49", "50-64", "65+"])], ["age_group"], "deaths", "population")
    ibm_period.to_csv(OUT / "ibm_period_comparison_2004_2008_vs_2019_2023.csv", index=False, encoding="utf-8-sig")
    ibm_trends = trend_table(ibm[ibm["age_group"].isin(["20-49", "50-64", "65+"])], ["age_group"], "deaths", "population", "ibm")
    ibm_trends.to_csv(OUT / "ibm_loglinear_apc_2004_2023.csv", index=False, encoding="utf-8-sig")
    ibm_poisson = poisson_age_year_interaction(
        ibm[ibm["age_group"].isin(["20-49", "50-64", "65+"])],
        "deaths",
        "population",
        "ibm",
        [],
    )
    ibm_poisson.to_csv(OUT / "poisson_ibm_age_year_interaction_2004_2023.csv", index=False, encoding="utf-8-sig")
    ibm_poisson_no2020 = poisson_age_year_interaction(
        ibm[ibm["age_group"].isin(["20-49", "50-64", "65+"])],
        "deaths",
        "population",
        "ibm_no_2020",
        [],
        exclude_years={2020},
    )
    ibm_poisson_no2020.to_csv(OUT / "sensitivity_no2020_poisson_ibm_age_year_interaction_2004_2023.csv", index=False, encoding="utf-8-sig")

    # Figures
    figdata = yearly_incidence[
        (yearly_incidence["age_group"] == "20-49")
        & (yearly_incidence["stage_group"].isin(["Localized", "Regional", "Distant", "Overall"]))
        & (yearly_incidence["year"] >= 2004)
    ].copy()
    simple_svg_line(figdata, "year", "rate_per_100k", "stage_group", "Age 20-49 incidence by stage", OUT / "figure_1_age20_49_stage_incidence.svg")

    figdata2 = ibm[(ibm["age_group"].isin(["20-49", "50-64", "65+"])) & (ibm["year"] >= 2004)].copy()
    simple_svg_line(figdata2, "year", "ibm_rate_per_100k", "age_group", "Incidence-based mortality by age group", OUT / "figure_2_ibm_by_age.svg")

    figdata3 = hist[(hist["cohort"] == "age20_49") & (hist["year"] >= 2004)].copy()
    simple_svg_line(figdata3, "year", "rate_per_100k", "hist_group", "Age 20-49 incidence by histology", OUT / "figure_3_age20_49_histology.svg")

    # Submission-oriented figure panels and source-data files.
    write_joinpoint_inputs(yearly_incidence, ibm)

    forest_rows = []
    inc_focus = inc_poisson[
        (inc_poisson["age_group"] == "20-49")
        & (inc_poisson["stage_group"].isin(["Overall", "Distant", "Regional"]))
    ].copy()
    for _, r in inc_focus.sort_values("stage_group").iterrows():
        forest_rows.append(
            {
                "label": f"Incidence, age 20-49, {r.stage_group}",
                "estimate": r.apc_percent_poisson,
                "lcl": r.apc_lcl_quasi,
                "ucl": r.apc_ucl_quasi,
                "outcome_family": "incidence",
            }
        )
    for _, r in ibm_poisson.sort_values("age_group").iterrows():
        forest_rows.append(
            {
                "label": f"IBM mortality, age {r.age_group}",
                "estimate": r.apc_percent_poisson,
                "lcl": r.apc_lcl_quasi,
                "ucl": r.apc_ucl_quasi,
                "outcome_family": "ibm",
            }
        )
    hp = hist_poisson[hist_poisson["cohort"] == "age20_49"].copy()
    for _, r in hp.sort_values("hist_group").iterrows():
        forest_rows.append(
            {
                "label": f"Histology 20-49, {r.hist_group}",
                "estimate": r.apc_percent_poisson,
                "lcl": r.apc_lcl_quasi,
                "ucl": r.apc_ucl_quasi,
                "outcome_family": "histology",
            }
        )
    forest_df = pd.DataFrame(forest_rows)
    forest_df.to_csv(OUT / "source_data_figure_4_apc_forest.csv", index=False, encoding="utf-8-sig")
    forest_svg(
        forest_df,
        "label",
        "estimate",
        "lcl",
        "ucl",
        "Quasi-Poisson APC estimates, 2004-2023",
        OUT / "figure_4_apc_forest_quasipoisson.svg",
    )

    # Short Markdown report
    def fmt(x, nd=2):
        return "" if pd.isna(x) else f"{x:.{nd}f}"

    report = [
        "# SEER appendiceal adenocarcinoma formal analysis pass 1",
        "",
        "Data source: SEER Research Data 17 registries, Nov 2025 submission, 2000-2023; IBM SEER 17 database for incidence-based mortality.",
        "",
        "Main analytic window for stage-specific analyses: 2004-2023, because the selected Combined Summary Stage variable is defined for 2004+.",
        "",
        "## Key period comparisons",
        "",
        "### Incidence, 2004-2008 vs 2019-2023",
    ]
    inc_focus = inc_period[
        (inc_period["age_group"].isin(["20-49", "50-64", "65+"]))
        & (inc_period["stage_group"].isin(["Overall", "Distant", "Regional", "Localized"]))
    ].copy()
    for _, r in inc_focus.sort_values(["age_group", "stage_group"]).iterrows():
        report.append(
            f"- {r.age_group}, {r.stage_group}: {int(r.cases_base)} to {int(r.cases_recent)} cases; "
            f"rate ratio {fmt(r.rate_ratio)} ({fmt(r.rate_ratio_lcl)}-{fmt(r.rate_ratio_ucl)})."
        )
    report += ["", "### IBM mortality, 2004-2008 vs 2019-2023"]
    for _, r in ibm_period.sort_values("age_group").iterrows():
        report.append(
            f"- {r.age_group}: {int(r.cases_base)} to {int(r.cases_recent)} deaths; "
            f"rate ratio {fmt(r.rate_ratio)} ({fmt(r.rate_ratio_lcl)}-{fmt(r.rate_ratio_ucl)})."
        )
    report += ["", "### Histology, age 20-49"]
    hy = hist_period[hist_period["cohort"] == "age20_49"].copy()
    for _, r in hy.sort_values("hist_group").iterrows():
        report.append(
            f"- {r.hist_group}: {int(r.cases_base)} to {int(r.cases_recent)} cases; "
            f"rate ratio {fmt(r.rate_ratio)} ({fmt(r.rate_ratio_lcl)}-{fmt(r.rate_ratio_ucl)})."
        )
    report += [
        "",
        "## Poisson offset model trend tests",
        "",
        "Model: log(cases or deaths) = log(population) + centered year + age group + centered year × age group. The reference age group is 20-49. Confidence intervals below use Pearson-dispersion quasi-Poisson scaling.",
        "",
        "### Incidence APC from Poisson models",
    ]
    for _, r in inc_poisson.sort_values(["stage_group", "age_group"]).iterrows():
        report.append(
            f"- {r.stage_group}, {r.age_group}: APC {fmt(r.apc_percent_poisson)}% "
            f"({fmt(r.apc_lcl_quasi)} to {fmt(r.apc_ucl_quasi)}), quasi-P={fmt(r.apc_p_quasi, 3)}; "
            f"interaction vs 20-49 quasi-P={fmt(r.interaction_p_vs_20_49_quasi, 3)}."
        )
    report += ["", "### IBM APC from Poisson model"]
    for _, r in ibm_poisson.sort_values("age_group").iterrows():
        report.append(
            f"- {r.age_group}: APC {fmt(r.apc_percent_poisson)}% "
            f"({fmt(r.apc_lcl_quasi)} to {fmt(r.apc_ucl_quasi)}), quasi-P={fmt(r.apc_p_quasi, 3)}; "
            f"interaction vs 20-49 quasi-P={fmt(r.interaction_p_vs_20_49_quasi, 3)}."
        )
    report += ["", "### Histology APC from Poisson models"]
    hp = hist_poisson[hist_poisson["cohort"] == "age20_49"].copy()
    for _, r in hp.sort_values("hist_group").iterrows():
        report.append(
            f"- Age 20-49, {r.hist_group}: APC {fmt(r.apc_percent_poisson)}% "
            f"({fmt(r.apc_lcl_quasi)} to {fmt(r.apc_ucl_quasi)}), quasi-P={fmt(r.apc_p_quasi, 3)}."
        )
    report += [
        "",
        "### Sensitivity excluding 2020",
        "",
    ]
    inc_no2020_focus = inc_poisson_no2020[
        (inc_poisson_no2020["age_group"] == "20-49")
        & (inc_poisson_no2020["stage_group"].isin(["Overall", "Distant"]))
    ]
    for _, r in inc_no2020_focus.sort_values("stage_group").iterrows():
        report.append(
            f"- Excluding 2020, incidence {r.stage_group}, age 20-49: APC {fmt(r.apc_percent_poisson)}% "
            f"({fmt(r.apc_lcl_quasi)} to {fmt(r.apc_ucl_quasi)}), quasi-P={fmt(r.apc_p_quasi, 3)}."
        )
    for _, r in ibm_poisson_no2020.sort_values("age_group").iterrows():
        report.append(
            f"- Excluding 2020, IBM {r.age_group}: APC {fmt(r.apc_percent_poisson)}% "
            f"({fmt(r.apc_lcl_quasi)} to {fmt(r.apc_ucl_quasi)}), quasi-P={fmt(r.apc_p_quasi, 3)}."
        )
    report += [
        "",
        "## Interpretation guardrails",
        "",
        "- These are first-pass descriptive and log-linear screening analyses, not final Joinpoint/APC permutation-model outputs.",
        "- The current stage-specific incidence table is age-group crude after aggregating SEER 5-year age strata, not directly age-standardized within the broad age groups.",
        "- Final manuscript should use Joinpoint or equivalent Poisson/negative-binomial age-stratified models for inferential trend testing.",
    ]
    (OUT / "formal_analysis_pass1_report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
