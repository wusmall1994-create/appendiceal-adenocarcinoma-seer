from __future__ import annotations

from pathlib import Path

import pandas as pd

from joinpoint_style_analysis import fit_series


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs"


def run_joinpoint_style(inc: pd.DataFrame, ibm: pd.DataFrame, suffix: str):
    inc_focus = inc[
        inc["AgeGroup"].isin(["20-49", "50-64", "65+"])
        & inc["Stage"].isin(["Overall", "Localized", "Regional", "Distant"])
    ].copy()
    ibm_focus = ibm[ibm["AgeGroup"].isin(["20-49", "50-64", "65+"])].copy()
    m1, s1, p1 = fit_series(inc_focus, ["AgeGroup", "Stage"], "Incidence")
    m2, s2, p2 = fit_series(ibm_focus, ["AgeGroup"], "IBM")
    models = pd.concat([m1, m2], ignore_index=True)
    segments = pd.concat([s1, s2], ignore_index=True)
    pred = pd.concat([p1, p2], ignore_index=True)
    models.to_csv(OUT / f"joinpoint_style_model_selection_2004_2023_{suffix}.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(OUT / f"joinpoint_style_segments_apc_2004_2023_{suffix}.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(OUT / f"joinpoint_style_fitted_rates_2004_2023_{suffix}.csv", index=False, encoding="utf-8-sig")
    return segments


def format_segments(segs: pd.DataFrame, outcome: str, age: str, stage: str | None):
    d = segs[(segs["outcome"] == outcome) & (segs["AgeGroup"] == age)].copy()
    if stage is not None:
        d = d[d["Stage"] == stage]
    if d.empty:
        return None, ""
    jp_value = d.iloc[0]["joinpoints"]
    jp = "None" if pd.isna(jp_value) else str(jp_value)
    pieces = []
    for r in d.sort_values("segment").itertuples():
        pieces.append(f"{int(r.start_year)}-{int(r.end_year)}: {r.apc_percent:.2f} ({r.apc_lcl:.2f} to {r.apc_ucl:.2f})")
    return jp, "; ".join(pieces)


def main():
    inc = pd.read_csv(OUT / "joinpoint_input_incidence_stage_age_2004_2023.csv")
    ibm = pd.read_csv(OUT / "joinpoint_input_ibm_age_2004_2023.csv")
    inc.assign(StandardError=(inc["Count"].clip(lower=0) ** 0.5) / inc["Population"] * 100_000).to_csv(
        OUT / "nci_joinpoint_input_incidence_stage_age_2004_2023_with_se.csv",
        index=False,
        encoding="utf-8-sig",
    )
    ibm.assign(StandardError=(ibm["Count"].clip(lower=0) ** 0.5) / ibm["Population"] * 100_000).to_csv(
        OUT / "nci_joinpoint_input_ibm_age_2004_2023_with_se.csv",
        index=False,
        encoding="utf-8-sig",
    )
    primary = pd.read_csv(OUT / "joinpoint_style_segments_apc_2004_2023.csv")
    no2020 = run_joinpoint_style(inc[inc["Year"] != 2020], ibm[ibm["Year"] != 2020], "excluding_2020")

    keys = [
        ("Incidence", "20-49", "Overall"),
        ("Incidence", "20-49", "Distant"),
        ("Incidence", "20-49", "Regional"),
        ("Incidence", "20-49", "Localized"),
        ("IBM", "20-49", None),
        ("IBM", "50-64", None),
        ("IBM", "65+", None),
    ]
    lines = [
        "# Joinpoint-style primary vs excluding-2020 sensitivity summary",
        "",
        "Models are continuous segmented Poisson log-linear regressions selected by BIC. APCs are percent change per year with Pearson-dispersion-adjusted 95% CIs.",
        "",
        "| Outcome | Age group | Stage | Analysis | Joinpoints | Segment APCs, %/year (95% CI) |",
        "|---|---|---|---|---|---|",
    ]
    for outcome, age, stage in keys:
        for label, segs in [("Primary", primary), ("Excluding 2020", no2020)]:
            jp, text = format_segments(segs, outcome, age, stage)
            if jp is None:
                continue
            lines.append(f"| {outcome} | {age} | {stage or ''} | {label} | {jp} | {text} |")
    lines += [
        "",
        "Interpretive use in manuscript:",
        "",
        "- The joinpoint-style analysis suggests non-linear calendar-time patterns for several endpoints, with some terminal acceleration around 2019.",
        "- Because terminal segments are short, these results should be treated as exploratory/supplementary unless confirmed with the NCI Joinpoint Regression Program permutation test.",
        "- The main conclusion is unchanged: in adults aged 20-49 years, overall and distant-stage appendiceal adenocarcinoma incidence increased, whereas incidence-based mortality did not show a sustained parallel increase across the entire period.",
    ]
    (OUT / "joinpoint_style_primary_vs_no2020_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
