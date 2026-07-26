# appendiceal-adenocarcinoma-seer

Analysis code and derived aggregate tables for the manuscript:

**Rising Young-Adult Appendiceal Adenocarcinoma Incidence in SEER: Stage, Histology, Mortality, and Birth-Cohort Patterns**

Submitted to *JNCI Cancer Spectrum* (2026).

Corresponding author: Fengxiang Zheng, Department of Gastroenterology, Longyan First Affiliated Hospital of Fujian Medical University, Longyan, Fujian, China. Email: ruq520@163.com

## Contents

### Code (repository root)

| Script | Purpose |
|---|---|
| `analyze_appendiceal_adenocarcinoma_seer.py` | Main SEER 17 analysis: period comparison (2004-2008 vs 2019-2023), Poisson rate ratios, log-linear Poisson offset APC models with quasi-Poisson (Pearson dispersion) correction, age-group interaction tests, histology-specific trends, 2020-exclusion sensitivity, yearly rate tables |
| `joinpoint_style_analysis.py` | Exploratory segmented log-linear analyses (0/1/2 change points, BIC model selection); **not** the official NCI Joinpoint Regression Program |
| `joinpoint_style_sensitivity_no2020.py` | Segmented analyses repeated after excluding 2020 |
| `analyze_seer8_birth_cohort.py` | SEER 8 (1975-2023) descriptive birth-cohort sensitivity analysis (birth year approximated as diagnosis year minus age-group midpoint; age-band-adjusted Poisson RR per birth decade) |
| `analyze_sex_stratified.py` | Sex-stratified young-adult incidence trends with sex-by-year interaction tests |

Requirements: Python 3.12, NumPy, pandas. Poisson models are fit by iteratively reweighted least squares.

### Derived aggregate tables

- Repository root: SEER*Stat aggregate exports used as script inputs (rate-session output: counts, populations, and rates by year/age/sex).
- `analysis_outputs/`: model result tables underlying each reported figure and table (period comparisons, quasi-Poisson APCs, interaction tests, segmented-model selection grids and segment APCs, SEER 8 birth-cohort rates and Lexis cells, sex-stratified estimates, yearly rate tables).

## Data access

The SEER incidence and incidence-based mortality data are available from the National Cancer Institute SEER Program after execution of the SEER Research Data Agreement (https://seer.cancer.gov/). Raw SEER data files are not redistributable and are therefore not hosted here.

Databases used (November 2025 submission, exported with SEER*Stat 9.0.43.0):

- `Incidence - SEER Research Data, 17 Registries, Nov 2025 Sub (2000-2023)`
- `Incidence-Based Mortality - SEER Research Data, 17 Registries, Nov 2025 Sub (2000-2023)`
- `Incidence - SEER Research Data, 8 Registries, Nov 2025 Sub (1975-2023)`

Case definition: primary site C18.1 (appendix), malignant behavior (ICD-O-3 behavior code 3), adenocarcinoma-related histology codes grouped as conventional/intestinal, mucinous-type, and signet-ring cell carcinoma (full code list in the manuscript's Supplementary Table 1). Neuroendocrine neoplasms and goblet cell adenocarcinoma were excluded; borderline/uncertain-behavior mucinous neoplasms (eg, 8480/1) were not part of the case definition.

Large intermediate SEER*Stat extracts (>300 KB; stage-age and histology-year exports) are not hosted in this repository. They can be regenerated exactly from the databases above using rate sessions with the case definition and row/column variables evident from each script's input parsing, or obtained from the corresponding author on reasonable request.

## Reproducing the analyses

1. Obtain the SEER data and export the aggregate rate tables as described above (three small exports are already included in this repository root).
2. Place the export CSVs and the scripts in the same directory.
3. Run each script with Python 3.12; outputs are written to `analysis_outputs/`.

## Notes on interpretation

- Segmented log-linear results are exploratory (BIC-selected change points), not official NCI Joinpoint Regression Program output.
- The birth-cohort analysis is descriptive and does not constitute a formal age-period-cohort decomposition.
