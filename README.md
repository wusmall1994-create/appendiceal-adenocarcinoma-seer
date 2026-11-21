# Appendiceal adenocarcinoma incidence and mortality trends across adult age groups in SEER

Analysis and figure code for the Discover Oncology manuscript.

## Run

The analysis uses base R and was verified with R 4.6.1. Figure generation uses ggplot2 and ragg. Put the requisite, authorized SEER*Stat CSV exports in one local directory.

~~~sh
Rscript analyze.R "/path/to/seer_exports" "/path/to/results"
Rscript make_figures.R "/path/to/results" "/path/to/figures"
~~~

`prepare_mortality.R` is sourced by `analyze.R`. File paths are passed on the command line. Generated figures are PNGs.

## Data source and exports

Use the November 2025 submission of the [SEER Research Data](https://seer.cancer.gov/data/) under the NCI data use agreement. The input files are SEER*Stat Rate session aggregate exports. Export as comma-delimited files with column names and value labels. Keep the paired `.dic` files with the local inputs.

| Filename | Database and grouping |
| --- | --- |
| `adenocarcinoma_stage_age_incidence_seer17_2000_2023.csv` | SEER 17 incidence, diagnosis year, age at diagnosis and Combined Summary Stage with Expanded Regional Codes (2004+). Include Count and Pop. |
| `adenocarcinoma_histology_year_age20_49_seer17_2000_2023.csv` | SEER 17 incidence, diagnosis year and ICD-O-3 Hist/behav, malignant. Restrict diagnosis ages to 20-49. |
| `adenocarcinoma_histology_year_seer17_2000_2023.csv` | Same histology-by-diagnosis-year table, all ages. |
| `adenocarcinoma_sex_age_year_incidence_seer17_2000_2023.csv` | SEER 17 incidence, diagnosis year, age at diagnosis and sex. |
| `adenocarcinoma_age_year_incidence_seer8_1975_2023.csv` | SEER 8 incidence, annual diagnosis year and age at diagnosis. The analysis uses ages 20-49. |
| `F03_Appendix_8480_Behavior_Year_SEER17_Nov2025_v01.csv` | Separate SEER 17 incidence query for C18.1, histology 8480 and diagnosis ages 20-49. Export diagnosis year, ICD-O-3 Hist/behav, Behavior recode for analysis and Count across available behaviors. |
| `R01_AppendixCOD_DeathYear_Age_SEER17_v01.csv` | SEER 17 IBM, year and age at death. Include Count and Population. |
| `R01_AppendixCOD_DxYear_DeathYear_Age_SEER17_v01.csv` | Same IBM selection, with diagnosis year and age added as table dimensions. Include Count. |

The case-selection criteria are primary site C18.1 and malignant histology codes 8140-8147, 8210-8211, 8260-8263, 8470-8472, 8480-8481, and 8490. Use the standard age recodes with the 90+ terminal group. Both IBM exports use death years 2000-2023, first matching record per person, and `Appendix` for the underlying cause of death in the 2023 expanded (1999+) cause-of-death recode.

The primary IBM analysis includes deaths at ages 20 or older from 2004 through 2023 linked to a diagnosis in the death year or preceding four calendar years. It uses population person-years indexed by death year and age. The script reconciles the counts in the two IBM exports before analysis. For the SEER 8 cohort analysis, the code uses annual age-by-diagnosis-year cells.

## Outputs

`incidence_comparison`, `incidence_standardized`, `inc_apc`, and `mort_*` contain estimates for the principal incidence and mortality tables. `inc_period`, `hist_period`, `ibm_period`, and `cohort_rates` support the figures. Other outputs cover subgroup and sensitivity analyses, diagnosis-to-death intervals, and the composition of the analysis sets. Rates are per 100,000 person-years.

SEER exports and generated cell-level summaries are retained in the controlled working directory. See the [SEER data use agreement and sharing guidance](https://seer.cancer.gov/data/agreements.html).
