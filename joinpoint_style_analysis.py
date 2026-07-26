from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs"
INC = OUT / "joinpoint_input_incidence_stage_age_2004_2023.csv"
IBM = OUT / "joinpoint_input_ibm_age_2004_2023.csv"


def poisson_fit(y: np.ndarray, pop: np.ndarray, year: np.ndarray, knots: tuple[int, ...]):
    """Poisson log-linear segmented regression with continuous joinpoints.

    log(E[y]) = log(pop) + b0 + b1*t + sum b_h * max(0, t-k)
    where t is calendar year centered at first year.
    """
    t = year.astype(float) - year.min()
    xcols = [np.ones_like(t), t]
    for k in knots:
        xcols.append(np.maximum(0, year.astype(float) - k))
    X = np.column_stack(xcols)
    offset = np.log(pop.astype(float))
    beta = np.zeros(X.shape[1])
    beta[0] = math.log(max(y.sum(), 0.5) / pop.sum())

    for _ in range(100):
        eta = offset + X @ beta
        mu = np.exp(np.clip(eta, -30, 30))
        W = np.maximum(mu, 1e-9)
        z = eta + (y - mu) / W - offset
        XtW = X.T * W
        try:
            beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.pinv(XtW @ X) @ (XtW @ z)
        if np.max(np.abs(beta_new - beta)) < 1e-10:
            beta = beta_new
            break
        beta = beta_new

    eta = offset + X @ beta
    mu = np.exp(np.clip(eta, -30, 30))
    # Poisson log likelihood, constants included for comparability only within same y.
    loglik = float(np.sum(y * np.log(np.maximum(mu, 1e-12)) - mu - np.array([math.lgamma(v + 1) for v in y])))
    p = X.shape[1]
    n = len(y)
    aic = -2 * loglik + 2 * p
    bic = -2 * loglik + math.log(n) * p
    df = max(n - p, 1)
    pearson = float(np.sum((y - mu) ** 2 / np.maximum(mu, 1e-12)))
    dispersion = max(1.0, pearson / df)
    try:
        cov = dispersion * np.linalg.inv(X.T @ (X * mu[:, None]))
    except np.linalg.LinAlgError:
        cov = dispersion * np.linalg.pinv(X.T @ (X * mu[:, None]))
    return {
        "beta": beta,
        "cov": cov,
        "mu": mu,
        "loglik": loglik,
        "aic": aic,
        "bic": bic,
        "dispersion": dispersion,
        "df": df,
    }


def candidate_knots(years: np.ndarray, max_joinpoints: int = 2, min_segment: int = 4):
    min_y, max_y = int(years.min()), int(years.max())
    candidates = list(range(min_y + min_segment, max_y - min_segment + 1))
    yield tuple()
    for m in range(1, max_joinpoints + 1):
        for ks in itertools.combinations(candidates, m):
            all_bounds = [min_y] + list(ks) + [max_y]
            if all(all_bounds[i + 1] - all_bounds[i] >= min_segment for i in range(len(all_bounds) - 1)):
                yield ks


def slopes_for_model(fit, knots: tuple[int, ...], start_year: int, end_year: int):
    beta = fit["beta"]
    cov = fit["cov"]
    bounds = [start_year] + list(knots) + [end_year]
    rows = []
    for seg_i in range(len(bounds) - 1):
        lo, hi = bounds[seg_i], bounds[seg_i + 1]
        c = np.zeros_like(beta)
        c[1] = 1
        if seg_i > 0:
            for j in range(seg_i):
                c[2 + j] = 1
        slope = float(c @ beta)
        se = float(math.sqrt(max(c @ cov @ c, 0)))
        apc = (math.exp(slope) - 1) * 100
        lcl = (math.exp(slope - 1.96 * se) - 1) * 100
        ucl = (math.exp(slope + 1.96 * se) - 1) * 100
        rows.append(
            {
                "segment": seg_i + 1,
                "start_year": lo,
                "end_year": hi,
                "apc_percent": apc,
                "apc_lcl": lcl,
                "apc_ucl": ucl,
            }
        )
    return rows


def fit_series(df: pd.DataFrame, label_cols: list[str], outcome_name: str):
    result_models = []
    result_segments = []
    result_pred = []
    for keys, g in df.groupby(label_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        label = dict(zip(label_cols, keys))
        g = g.sort_values("Year").copy()
        g = g[(g["Year"] >= 2004) & (g["Year"] <= 2023) & (g["Population"] > 0)]
        if len(g) < 12 or g["Count"].sum() <= 0:
            continue
        y = g["Count"].to_numpy(float)
        pop = g["Population"].to_numpy(float)
        years = g["Year"].to_numpy(int)
        fits = []
        for ks in candidate_knots(years, max_joinpoints=2, min_segment=4):
            fit = poisson_fit(y, pop, years, ks)
            fits.append((ks, fit))
        # BIC as primary parsimonious criterion; AIC retained for transparency.
        best_ks, best_fit = min(fits, key=lambda item: item[1]["bic"])
        model_row = {"outcome": outcome_name, **label, "joinpoints": ";".join(map(str, best_ks)) or "None"}
        model_row.update(
            {
                "n_years": len(g),
                "total_count": int(y.sum()),
                "best_aic": best_fit["aic"],
                "best_bic": best_fit["bic"],
                "dispersion": best_fit["dispersion"],
            }
        )
        for ks, fit in fits:
            result_models.append(
                {
                    "outcome": outcome_name,
                    **label,
                    "candidate_joinpoints": ";".join(map(str, ks)) or "None",
                    "n_joinpoints": len(ks),
                    "aic": fit["aic"],
                    "bic": fit["bic"],
                    "selected_by_bic": ks == best_ks,
                    "dispersion": fit["dispersion"],
                }
            )
        for row in slopes_for_model(best_fit, best_ks, int(years.min()), int(years.max())):
            result_segments.append({**model_row, **row})
        pred = best_fit["mu"] / pop * 100_000
        for yr, obs, mu_rate, count, pp in zip(years, g["Rate"], pred, y, pop):
            result_pred.append({"outcome": outcome_name, **label, "Year": int(yr), "observed_rate": obs, "fitted_rate": mu_rate, "Count": int(count), "Population": int(pp)})
    return pd.DataFrame(result_models), pd.DataFrame(result_segments), pd.DataFrame(result_pred)


def make_svg(pred: pd.DataFrame, segments: pd.DataFrame, path: Path):
    # Minimal dependency-free SVG: selected key panels.
    panels = [
        ("Incidence", {"AgeGroup": "20-49", "Stage": "Overall"}, "A. Age 20-49 overall incidence"),
        ("Incidence", {"AgeGroup": "20-49", "Stage": "Distant"}, "B. Age 20-49 distant-stage incidence"),
        ("IBM", {"AgeGroup": "20-49"}, "C. Age 20-49 IBM"),
        ("IBM", {"AgeGroup": "65+"}, "D. Age 65+ IBM"),
    ]
    W, H = 980, 720
    margin = 70
    panel_w, panel_h = 410, 250
    xs0 = [70, 550, 70, 550]
    ys0 = [70, 70, 410, 410]
    colors = {"obs": "#1f77b4", "fit": "#d62728"}
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    svg.append('<rect width="100%" height="100%" fill="white"/>')
    svg.append('<style>text{font-family:Arial,Helvetica,sans-serif;font-size:14px}.title{font-size:16px;font-weight:bold}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.obs{fill:#1f77b4}.fit{fill:none;stroke:#d62728;stroke-width:2.5}</style>')

    for idx, (outcome, filt, title) in enumerate(panels):
        d = pred[pred["outcome"].eq(outcome)].copy()
        for k, v in filt.items():
            d = d[d[k].astype(str).eq(str(v))]
        if d.empty:
            continue
        x0, y0 = xs0[idx], ys0[idx]
        years = d["Year"].to_numpy()
        vals = np.concatenate([d["observed_rate"].to_numpy(float), d["fitted_rate"].to_numpy(float)])
        y_max = max(vals.max() * 1.15, 0.01)
        def sx(year):
            return x0 + (year - 2004) / (2023 - 2004) * panel_w
        def sy(rate):
            return y0 + panel_h - rate / y_max * panel_h
        # axes and horizontal grid
        for frac in [0, .25, .5, .75, 1.0]:
            yy = y0 + panel_h - frac * panel_h
            svg.append(f'<line class="grid" x1="{x0}" y1="{yy:.1f}" x2="{x0+panel_w}" y2="{yy:.1f}"/>')
            svg.append(f'<text x="{x0-8}" y="{yy+4:.1f}" text-anchor="end">{y_max*frac:.2f}</text>')
        svg.append(f'<line class="axis" x1="{x0}" y1="{y0+panel_h}" x2="{x0+panel_w}" y2="{y0+panel_h}"/>')
        svg.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+panel_h}"/>')
        for yr in [2004, 2009, 2014, 2019, 2023]:
            svg.append(f'<text x="{sx(yr):.1f}" y="{y0+panel_h+22}" text-anchor="middle">{yr}</text>')
        svg.append(f'<text class="title" x="{x0}" y="{y0-25}">{title}</text>')
        # fitted line
        pts = " ".join(f'{sx(int(r.Year)):.1f},{sy(float(r.fitted_rate)):.1f}' for r in d.itertuples())
        svg.append(f'<polyline class="fit" points="{pts}"/>')
        for r in d.itertuples():
            svg.append(f'<circle class="obs" cx="{sx(int(r.Year)):.1f}" cy="{sy(float(r.observed_rate)):.1f}" r="3"/>')
        # joinpoint markers
        seg = segments[segments["outcome"].eq(outcome)].copy()
        for k, v in filt.items():
            seg = seg[seg[k].astype(str).eq(str(v))]
        if not seg.empty:
            jps = str(seg.iloc[0]["joinpoints"])
            if jps != "None":
                for jp in [int(x) for x in jps.split(";") if x]:
                    xx = sx(jp)
                    svg.append(f'<line x1="{xx:.1f}" y1="{y0}" x2="{xx:.1f}" y2="{y0+panel_h}" stroke="#777" stroke-dasharray="4 3"/>')
                    svg.append(f'<text x="{xx+4:.1f}" y="{y0+14}" fill="#555">JP {jp}</text>')
        svg.append(f'<text x="{x0+panel_w-120}" y="{y0+20}" fill="{colors["obs"]}">● observed</text>')
        svg.append(f'<text x="{x0+panel_w-120}" y="{y0+40}" fill="{colors["fit"]}">— fitted</text>')
    svg.append('<text x="490" y="695" text-anchor="middle">Rates are per 100,000; segmented log-linear Poisson models selected by BIC.</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def main():
    inc = pd.read_csv(INC)
    ibm = pd.read_csv(IBM)
    # Keep the manuscript-focused incidence series.
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
    models.to_csv(OUT / "joinpoint_style_model_selection_2004_2023.csv", index=False, encoding="utf-8-sig")
    segments.to_csv(OUT / "joinpoint_style_segments_apc_2004_2023.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(OUT / "joinpoint_style_fitted_rates_2004_2023.csv", index=False, encoding="utf-8-sig")
    make_svg(pred, segments, OUT / "figure_s1_joinpoint_style_trends.svg")

    # Concise Chinese report
    key = segments[
        ((segments["outcome"] == "Incidence") & (segments["AgeGroup"] == "20-49") & (segments["Stage"].isin(["Overall", "Distant", "Regional", "Localized"])))
        | ((segments["outcome"] == "IBM") & (segments["AgeGroup"].isin(["20-49", "50-64", "65+"])))
    ].copy()
    key["apc_fmt"] = key.apply(lambda r: f"{r.apc_percent:.2f}% ({r.apc_lcl:.2f} to {r.apc_ucl:.2f})", axis=1)
    lines = [
        "# Joinpoint-style 分段趋势分析补充结果",
        "",
        "说明：本分析使用连续分段 Poisson log-linear 模型，因变量为年度病例数/死亡数，offset 为 log(人口数)，候选模型为 0、1 或 2 个 joinpoints；主选择标准为 BIC。APC 的置信区间采用 Pearson dispersion 校正后的协方差矩阵。该结果用于当前稿件的可复现补充分析；若正式投稿需严格使用 NCI Joinpoint Regression Program，可用已生成的 `joinpoint_input_*.csv` 复核。",
        "",
        "## 关键序列",
        "",
        "| Outcome | Age group | Stage | Joinpoints | Segment | Years | APC, %/year (95% CI) |",
        "|---|---|---|---|---:|---|---|",
    ]
    for r in key.sort_values(["outcome", "AgeGroup", "Stage" if "Stage" in key.columns else "segment", "segment"]).itertuples():
        stage = getattr(r, "Stage", "")
        lines.append(f"| {r.outcome} | {r.AgeGroup} | {stage if pd.notna(stage) else ''} | {r.joinpoints} | {r.segment} | {r.start_year}-{r.end_year} | {r.apc_fmt} |")
    lines += [
        "",
        "## 对论文结果的影响",
        "",
        "- 若 BIC 选择 `None`，说明在 2004–2023 年间没有足够证据支持加入拐点；趋势可用单一 APC 描述。",
        "- 若出现 joinpoint，应在 Results 中报告“趋势并非全程线性”，并分段描述 APC。",
        "- 这一步主要回答审稿人可能提出的疑问：年轻发病率上升是否被某一短时期或疫情年份驱动。",
    ]
    (OUT / "joinpoint_style_results_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
