from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs"
OUT.mkdir(exist_ok=True)

INPUT_CANDIDATES = [
    ROOT / "adenocarcinoma_age_year_incidence_seer8_1975_2023.csv",
    ROOT / "appendix_adenocarcinoma_age_year_incidence_seer8_1975_2023.csv",
    ROOT / "seer8_birth_cohort.csv",
]


def find_input() -> Path:
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    matches = sorted(ROOT.glob("*seer8*1975*2023*.csv"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "No SEER 8 birth-cohort input file found. Expected one of: "
        + ", ".join(str(p.name) for p in INPUT_CANDIDATES)
        + ". Please follow SEERStat_SEER8_birth_cohort_export_guide.md."
    )


def norm_col(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def infer_col(cols: list[str], patterns: list[str]) -> str:
    ncols = {c: norm_col(c) for c in cols}
    for pat in patterns:
        rx = re.compile(pat)
        hits = [c for c, nc in ncols.items() if rx.search(nc)]
        if hits:
            return hits[0]
    raise KeyError(f"Could not infer column from patterns {patterns}. Available columns: {cols}")


def parse_year(x) -> int:
    m = re.search(r"(19|20)\d{2}", str(x))
    if not m:
        raise ValueError(f"Cannot parse year from {x!r}")
    return int(m.group(0))


def parse_age_mid(label) -> float:
    s = str(label)
    if "unknown" in s.lower():
        return np.nan
    nums = [int(x) for x in re.findall(r"\d+", s)]
    if not nums:
        return np.nan
    if "+" in s or "plus" in s.lower():
        return float(nums[0] + 2.5)
    if len(nums) >= 2:
        return (nums[0] + nums[1]) / 2.0
    return float(nums[0])


def age_band_from_mid(mid: float) -> str | None:
    if 20 <= mid < 35:
        return "20-34"
    if 35 <= mid < 50:
        return "35-49"
    if 50 <= mid < 65:
        return "50-64"
    if 65 <= mid < 85:
        return "65-84"
    return None


def broad_age_group(mid: float) -> str | None:
    if 20 <= mid < 50:
        return "20-49"
    if 50 <= mid < 65:
        return "50-64"
    if 65 <= mid < 85:
        return "65-84"
    return None


def birth_decade(year: int, age_mid: float) -> str:
    by = int(math.floor(year - age_mid))
    if by < 1930:
        return "<1930"
    if by >= 2000:
        return "2000+"
    lo = (by // 10) * 10
    return f"{lo}-{lo+9}"


def period_group(year: int) -> str:
    if year < 1980:
        return "1975-1979"
    lo = (year // 5) * 5
    hi = min(lo + 4, 2023)
    return f"{lo}-{hi}"


def clean_numeric(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", "").replace("~", "")
    if s == "":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def poisson_rr(a_cases, a_pop, b_cases, b_pop):
    rr = (a_cases / a_pop) / (b_cases / b_pop)
    se = math.sqrt(1 / a_cases + 1 / b_cases) if a_cases > 0 and b_cases > 0 else np.nan
    return rr, math.exp(math.log(rr) - 1.96 * se), math.exp(math.log(rr) + 1.96 * se)


def rr_from_groups(df: pd.DataFrame, age_group: str, late_cohorts: list[str], ref_cohorts: list[str]):
    d = df[df["age_group"] == age_group]
    late = d[d["birth_cohort"].isin(late_cohorts)]
    ref = d[d["birth_cohort"].isin(ref_cohorts)]
    rr, lcl, ucl = poisson_rr(late["cases"].sum(), late["population"].sum(), ref["cases"].sum(), ref["population"].sum())
    return {
        "age_group": age_group,
        "reference_cohorts": "+".join(ref_cohorts),
        "later_cohorts": "+".join(late_cohorts),
        "reference_cases": int(ref["cases"].sum()),
        "reference_population": int(ref["population"].sum()),
        "reference_rate_per_100k": ref["cases"].sum() / ref["population"].sum() * 100_000,
        "later_cases": int(late["cases"].sum()),
        "later_population": int(late["population"].sum()),
        "later_rate_per_100k": late["cases"].sum() / late["population"].sum() * 100_000,
        "rate_ratio": rr,
        "rr_lcl": lcl,
        "rr_ucl": ucl,
    }


def poisson_irls(y, pop, X):
    offset = np.log(pop.astype(float))
    beta = np.zeros(X.shape[1])
    beta[0] = math.log(max(y.sum(), 0.5) / pop.sum())
    for _ in range(100):
        eta = offset + X @ beta
        mu = np.exp(np.clip(eta, -30, 30))
        W = np.maximum(mu, 1e-9)
        z = eta + (y - mu) / W - offset
        XtW = X.T * W
        beta_new = np.linalg.pinv(XtW @ X) @ (XtW @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-10:
            beta = beta_new
            break
        beta = beta_new
    mu = np.exp(np.clip(offset + X @ beta, -30, 30))
    df = max(len(y) - X.shape[1], 1)
    dispersion = max(1.0, float(np.sum((y - mu) ** 2 / np.maximum(mu, 1e-12))) / df)
    cov = dispersion * np.linalg.pinv(X.T @ (X * mu[:, None]))
    return beta, cov, dispersion


def fit_cohort_model(d: pd.DataFrame):
    # A pragmatic sensitivity model: age-band adjusted cohort trend among adults 20-49.
    # We avoid a full age-period-cohort decomposition because age, period, and cohort are linearly dependent.
    d = d[d["age_group"] == "20-49"].copy()
    d = d[d["birth_year"] >= 1930]
    d["cohort_decade_num"] = ((d["birth_year"] // 10) * 10 - 1950) / 10
    age_dummies = pd.get_dummies(d["age_band"], prefix="age", drop_first=True, dtype=float)
    X = np.column_stack([np.ones(len(d)), d["cohort_decade_num"].to_numpy(float), age_dummies.to_numpy(float)])
    y = d["cases"].to_numpy(float)
    pop = d["population"].to_numpy(float)
    beta, cov, dispersion = poisson_irls(y, pop, X)
    slope = beta[1]
    se = math.sqrt(max(cov[1, 1], 0))
    rr_per_decade = math.exp(slope)
    return {
        "rr_per_birth_decade": rr_per_decade,
        "lcl": math.exp(slope - 1.96 * se),
        "ucl": math.exp(slope + 1.96 * se),
        "dispersion": dispersion,
        "n_cells": len(d),
        "cases": int(y.sum()),
    }


def make_svg(cohort_rates: pd.DataFrame, path: Path):
    d = cohort_rates[cohort_rates["age_group"].isin(["20-34", "35-49", "20-49"])].copy()
    order = ["<1930", "1930-1939", "1940-1949", "1950-1959", "1960-1969", "1970-1979", "1980-1989", "1990-1999", "2000+"]
    d["cohort_order"] = d["birth_cohort"].map({v: i for i, v in enumerate(order)})
    d = d.dropna(subset=["cohort_order"]).sort_values(["age_group", "cohort_order"])
    W, H = 920, 520
    x0, y0, pw, ph = 100, 70, 740, 330
    ymax = max(d["rate_per_100k"].max() * 1.25, 0.01)
    colors = {"20-34": "#1f77b4", "35-49": "#ff7f0e", "20-49": "#2ca02c"}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append('<style>text{font-family:Arial,Helvetica,sans-serif;font-size:14px}.title{font-size:18px;font-weight:bold}.axis{stroke:#333}.grid{stroke:#ddd}</style>')
    svg.append(f'<text class="title" x="{W/2}" y="35" text-anchor="middle">SEER 8 birth-cohort incidence sensitivity analysis</text>')
    for frac in [0, .25, .5, .75, 1]:
        yy = y0 + ph - frac * ph
        svg.append(f'<line class="grid" x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}"/>')
        svg.append(f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end">{ymax*frac:.2f}</text>')
    svg.append(f'<line class="axis" x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}"/>')
    svg.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ph}"/>')
    def sx(i): return x0 + i / (len(order) - 1) * pw
    def sy(v): return y0 + ph - v / ymax * ph
    for i, lab in enumerate(order):
        svg.append(f'<text x="{sx(i):.1f}" y="{y0+ph+24}" text-anchor="middle" transform="rotate(35 {sx(i):.1f},{y0+ph+24})">{lab}</text>')
    for ag in ["20-34", "35-49", "20-49"]:
        g = d[d["age_group"] == ag]
        if g.empty:
            continue
        pts = " ".join(f'{sx(int(r.cohort_order)):.1f},{sy(float(r.rate_per_100k)):.1f}' for r in g.itertuples())
        svg.append(f'<polyline points="{pts}" fill="none" stroke="{colors[ag]}" stroke-width="2.5"/>')
        for r in g.itertuples():
            svg.append(f'<circle cx="{sx(int(r.cohort_order)):.1f}" cy="{sy(float(r.rate_per_100k)):.1f}" r="4" fill="{colors[ag]}"/>')
    for j, ag in enumerate(["20-34", "35-49", "20-49"]):
        svg.append(f'<text x="690" y="{95 + 22*j}" fill="{colors[ag]}">● {ag}</text>')
    svg.append('<text x="460" y="500" text-anchor="middle">Birth cohort estimated as diagnosis year minus age-group midpoint; rates per 100,000 person-years.</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def main():
    inp = find_input()
    df = pd.read_csv(inp, na_values=["~", " "])
    cols = list(df.columns)
    year_col = infer_col(cols, [r"year.*diagnosis", r"year"])
    age_col = infer_col(cols, [r"age.*recode", r"age"])
    count_col = infer_col(cols, [r"^count$", r"cases"])
    pop_col = infer_col(cols, [r"^pop$", r"population"])

    d = df[[year_col, age_col, count_col, pop_col]].copy()
    d.columns = ["year_raw", "age_raw", "cases", "population"]
    d["year"] = d["year_raw"].map(parse_year)
    d["age_mid"] = d["age_raw"].map(parse_age_mid)
    d["cases"] = d["cases"].map(clean_numeric).fillna(0)
    d["population"] = d["population"].map(clean_numeric)
    d = d[(d["population"] > 0) & (d["age_mid"].notna()) & (d["year"].between(1975, 2023))].copy()
    d["age_band"] = d["age_mid"].map(age_band_from_mid)
    d["age_group"] = d["age_mid"].map(broad_age_group)
    d = d[d["age_group"].notna()].copy()
    d["birth_year"] = np.floor(d["year"] - d["age_mid"]).astype(int)
    d["birth_cohort"] = d.apply(lambda r: birth_decade(int(r["year"]), float(r["age_mid"])), axis=1)
    d["period_group"] = d["year"].map(period_group)

    lexis = d.rename(columns={"cases": "Count", "population": "Population"})
    lexis.to_csv(OUT / "seer8_birth_cohort_lexis_cells.csv", index=False, encoding="utf-8-sig")

    agg = (
        d.groupby(["age_group", "birth_cohort"], as_index=False)
        .agg(cases=("cases", "sum"), population=("population", "sum"), min_year=("year", "min"), max_year=("year", "max"))
    )
    agg["rate_per_100k"] = agg["cases"] / agg["population"] * 100_000
    agg.to_csv(OUT / "seer8_birth_cohort_rates.csv", index=False, encoding="utf-8-sig")

    # More detailed young-adult bands for plotting.
    young_band = (
        d[d["age_band"].isin(["20-34", "35-49"])]
        .groupby(["age_band", "birth_cohort"], as_index=False)
        .agg(cases=("cases", "sum"), population=("population", "sum"))
    )
    young_band["age_group"] = young_band["age_band"]
    young_band["rate_per_100k"] = young_band["cases"] / young_band["population"] * 100_000
    combined_young = agg[agg["age_group"] == "20-49"][["age_group", "birth_cohort", "cases", "population", "rate_per_100k"]]
    plot_rates = pd.concat([
        young_band[["age_group", "birth_cohort", "cases", "population", "rate_per_100k"]],
        combined_young,
    ], ignore_index=True)
    plot_rates.to_csv(OUT / "seer8_birth_cohort_young_rates_for_plot.csv", index=False, encoding="utf-8-sig")
    make_svg(plot_rates, OUT / "figure_s2_seer8_birth_cohort_young_incidence.svg")

    # Period comparison within young adults: pre-1960 vs 1970+ cohorts when observed at ages 20-49.
    young = agg[agg["age_group"] == "20-49"].copy()
    early = young[young["birth_cohort"].isin(["1930-1939", "1940-1949", "1950-1959"])]
    late = young[young["birth_cohort"].isin(["1970-1979", "1980-1989", "1990-1999", "2000+"])]
    rr, lcl, ucl = poisson_rr(late["cases"].sum(), late["population"].sum(), early["cases"].sum(), early["population"].sum())
    model = fit_cohort_model(d)

    rr_table = pd.DataFrame(
        [
            rr_from_groups(plot_rates, "20-49", ["1970-1979", "1980-1989", "1990-1999", "2000+"], ["1930-1939", "1940-1949", "1950-1959"]),
            rr_from_groups(plot_rates, "35-49", ["1970-1979", "1980-1989"], ["1950-1959", "1960-1969"]),
            rr_from_groups(plot_rates, "20-34", ["1980-1989", "1990-1999"], ["1940-1949", "1950-1959", "1960-1969"]),
        ]
    )
    rr_table.to_csv(OUT / "seer8_birth_cohort_rr_sensitivity.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# SEER 8 birth-cohort sensitivity analysis",
        "",
        f"Input file: `{inp.name}`",
        "",
        "Definition: birth year was approximated as calendar year of diagnosis minus the midpoint of the 5-year age group. Birth cohorts were grouped by decade. Rates were calculated by summing cases and person-years across Lexis cells.",
        "",
        "## Key young-adult comparison",
        "",
        f"Among adults aged 20-49 years, later birth cohorts (1970+) had a rate ratio of {rr:.2f} (95% CI, {lcl:.2f}-{ucl:.2f}) compared with earlier birth cohorts born during 1930-1959.",
        "",
        "A pragmatic age-band-adjusted Poisson model among adults aged 20-49 years estimated the rate ratio per successive birth decade as "
        f"{model['rr_per_birth_decade']:.2f} (95% CI, {model['lcl']:.2f}-{model['ucl']:.2f}; dispersion={model['dispersion']:.2f}).",
        "",
        "Age-band sensitivity comparisons:",
        "",
        "| Age group | Later cohorts | Reference cohorts | Later rate | Reference rate | RR (95% CI) |",
        "|---|---|---|---:|---:|---|",
    ]
    for r in rr_table.itertuples():
        lines.append(
            f"| {r.age_group} | {r.later_cohorts} | {r.reference_cohorts} | "
            f"{r.later_rate_per_100k:.3f} | {r.reference_rate_per_100k:.3f} | "
            f"{r.rate_ratio:.2f} ({r.rr_lcl:.2f}-{r.rr_ucl:.2f}) |"
        )
    lines += [
        "",
        "The 35-49-year band provides the cleanest young-adult cohort contrast because it has more cases than the 20-34-year band and less extreme truncation of the youngest birth cohorts.",
        "",
        "Interpretation: this is a descriptive birth-cohort sensitivity analysis, not a full age-period-cohort decomposition. It tests whether the main SEER 17 finding is compatible with higher incidence in later-born cohorts within the longer-standing SEER 8 registries.",
        "",
        "## Output files",
        "",
        "- `seer8_birth_cohort_lexis_cells.csv`",
        "- `seer8_birth_cohort_rates.csv`",
        "- `seer8_birth_cohort_young_rates_for_plot.csv`",
        "- `seer8_birth_cohort_rr_sensitivity.csv`",
        "- `figure_s2_seer8_birth_cohort_young_incidence.svg`",
    ]
    (OUT / "seer8_birth_cohort_results_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
