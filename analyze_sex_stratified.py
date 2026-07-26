"""Sex-stratified analysis of appendiceal adenocarcinoma incidence, SEER 17, 2004-2023.

Replicates the methods of analyze_appendiceal_adenocarcinoma_seer.py:
- Period comparison 2004-2008 vs 2019-2023 (Poisson rate ratios, normal-approx CI on log scale)
- Log-linear Poisson offset models fit by IRLS, quasi-Poisson (Pearson dispersion) SEs
- Sex-by-year interaction test within each age group
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "adenocarcinoma_sex_age_year_incidence_seer17_2000_2023.csv"
OUT = ROOT / "analysis_outputs"
OUT.mkdir(exist_ok=True)

BASE = (2004, 2008)
RECENT = (2019, 2023)
T0, T1 = 2004, 2023


def age_group(label: str) -> str:
    label = str(label)
    nums = [int(x) for x in re.findall(r"\d+", label)]
    if not nums:
        return "Unknown"
    if label.startswith("00"):
        lo, hi = 0, 0
    elif label.startswith("90"):
        lo, hi = 90, 120
    else:
        lo, hi = (nums[0], nums[1]) if len(nums) >= 2 else (nums[0], nums[0])
    if hi < 20:
        return "<20"
    if lo < 50:
        return "20-49"
    if lo < 65:
        return "50-64"
    return "65+"


def load() -> pd.DataFrame:
    df = pd.read_csv(SRC)
    df["year"] = pd.to_numeric(df["Year of diagnosis"], errors="coerce")
    df = df[df["Sex"].isin(["Male", "Female"])].copy()
    df["ag"] = df["Age recode with <1 year olds and 90+"].map(age_group)
    df = df[(df.year >= T0) & (df.year <= T1) & (df.ag != "Unknown")]
    return df


def period_compare(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["period"] = np.where(
        (BASE[0] <= d.year) & (d.year <= BASE[1]), "base",
        np.where((RECENT[0] <= d.year) & (d.year <= RECENT[1]), "recent", None),
    )
    d = d[d["period"].notna()]
    agg = d.groupby(["ag", "Sex", "period"], as_index=False).agg(cases=("Count", "sum"), pop=("Population", "sum"))
    b = agg[agg.period == "base"].drop(columns="period")
    r = agg[agg.period == "recent"].drop(columns="period")
    m = r.merge(b, on=["ag", "Sex"], suffixes=("_r", "_b"))
    rows = []
    for _, x in m.iterrows():
        rr = (x.cases_r / x.pop_r) / (x.cases_b / x.pop_b)
        se = math.sqrt(1 / x.cases_r + 1 / x.cases_b)
        lcl, ucl = math.exp(math.log(rr) - 1.96 * se), math.exp(math.log(rr) + 1.96 * se)
        rows.append({
            "age_group": x.ag, "sex": x.Sex,
            "cases_base": int(x.cases_b), "rate_base_per_100k": x.cases_b / x.pop_b * 1e5,
            "cases_recent": int(x.cases_r), "rate_recent_per_100k": x.cases_r / x.pop_r * 1e5,
            "rr": rr, "rr_lcl": lcl, "rr_ucl": ucl,
        })
    return pd.DataFrame(rows)


def poisson_irls(X: np.ndarray, y: np.ndarray, offset: np.ndarray):
    beta = np.zeros(X.shape[1])
    for it in range(200):
        eta = X @ beta + offset
        mu = np.exp(eta)
        w = mu
        z = eta + (y - mu) / mu - offset
        xtwx = (X * w[:, None]).T @ X
        xtwz = (X * w[:, None]).T @ z
        new = np.linalg.solve(xtwx, xtwz)
        if np.max(np.abs(new - beta)) < 1e-10:
            beta = new
            break
        beta = new
    mu = np.exp(X @ beta + offset)
    resid = (y - mu) ** 2 / mu
    df_res = len(y) - X.shape[1]
    dispersion = resid.sum() / df_res
    cov = np.linalg.inv((X * mu[:, None]).T @ X)
    return beta, cov, dispersion


def apc_models(d: pd.DataFrame) -> pd.DataFrame:
    """Per-sex APC and sex-by-year interaction within each age group (quasi-Poisson)."""
    rows = []
    for ag in ["20-49", "50-64", "65+"]:
        g = d[d.ag == ag]
        yrs = g.groupby(["year", "Sex"], as_index=False).agg(cases=("Count", "sum"), pop=("Population", "sum"))
        # Joint model: cases ~ sex + year + sex:year, offset log pop
        yv = yrs["year"].to_numpy(float) - T0
        sx = (yrs["Sex"] == "Female").to_numpy(float)
        X = np.column_stack([np.ones(len(yrs)), sx, yv, sx * yv])
        y = yrs["cases"].to_numpy(float)
        off = np.log(yrs["pop"].to_numpy(float))
        beta, cov, disp = poisson_irls(X, y, off)
        se = np.sqrt(np.diag(cov) * disp)
        def est(b, s):
            z = b / s
            p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
            return (math.exp(b) - 1) * 100, (math.exp(b - 1.96 * s) - 1) * 100, (math.exp(b + 1.96 * s) - 1) * 100, p
        m_apc, m_l, m_u, _ = est(beta[2], se[2])               # male slope
        f_apc, f_l, f_u, _ = est(beta[2] + beta[3], math.sqrt(cov[2, 2] + cov[3, 3] + 2 * cov[2, 3]) * math.sqrt(disp))  # female slope
        z_int = beta[3] / se[3]
        p_int = 2 * (1 - 0.5 * (1 + math.erf(abs(z_int) / math.sqrt(2))))
        rows.append({"age_group": ag, "sex": "Male", "apc": m_apc, "apc_lcl": m_l, "apc_ucl": m_u,
                     "pearson_dispersion": disp, "cases_total": int(yrs[yrs.Sex == "Male"].cases.sum())})
        rows.append({"age_group": ag, "sex": "Female", "apc": f_apc, "apc_lcl": f_l, "apc_ucl": f_u,
                     "pearson_dispersion": disp, "cases_total": int(yrs[yrs.Sex == "Female"].cases.sum())})
        rows.append({"age_group": ag, "sex": "interaction_p", "apc": p_int, "apc_lcl": np.nan, "apc_ucl": np.nan,
                     "pearson_dispersion": disp, "cases_total": int(y.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    d = load()
    pc = period_compare(d)
    ap = apc_models(d)
    pc.to_csv(OUT / "sex_stratified_period_comparison_2004_2008_vs_2019_2023.csv", index=False, encoding="utf-8-sig")
    ap.to_csv(OUT / "sex_stratified_quasipoisson_apc_2004_2023.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("=== Period comparison (2004-2008 vs 2019-2023) ===")
    print(pc.to_string(index=False))
    print("\n=== Quasi-Poisson APC 2004-2023 by sex ===")
    print(ap.to_string(index=False))


if __name__ == "__main__":
    main()
