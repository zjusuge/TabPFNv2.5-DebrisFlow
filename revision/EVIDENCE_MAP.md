# Manuscript evidence map

| Manuscript item | Source files under revision |
|---|---|
| Table 1 and S11 | `data/debris_flow_longmenshan.xlsx`, `source_audit/Table_S11_60_catchments.csv` |
| Tables 2 and 3, Figures 5 and 7 | `archived/FSA_all_folds.csv`, `archived/FSB_all_folds.csv`, `results/paired_statistics_audit.json` |
| Table 4 and Figure 8 | `archived/comparison_A_vs_B.csv`, matched primary folds and `results/paired_statistics_audit.json` |
| Figure 6 | `archived/predictions_setA_cv5fold.csv` |
| Figure 9 | `archived/FS{A,B}_learning_curve_raw.csv` and corresponding summaries |
| Figure 10 | `archived/UQ_XConf_samples_FSA.csv` |
| Figure 11 | `archived/Korea_all_folds.csv` |
| Figures 12 and 13 and Table S10 | `archived/all_results_tabpfn_exact_shap.npz`, `results/SHAP_sample_attributions.csv`, `results/SHAP_region_mean_absolute.csv` |
| Tables S2 and S3 | Fixed-configuration rows in `archived/FS{A,B}_all_folds.csv` |
| Table S4 and Figure S1 | `results/transform_summary_folds.csv`, `results/transform_summary.csv`, `results/TabPFN_*_predictions.csv` |
| Table S5 | `results/budget_summary_folds.csv`, `results/budget_summary.csv`, trial and prediction CSVs |
| Tables S6 and S7 and Figure S2 | `results/Grouped_*_predictions.csv`, `results/foundation_summary.csv`, `results/results_summary.json` |
| Table S8 and Figure S3 | `results/TabPFN_*_split_conformal.csv`, `results/*_split_conformal_original_units.csv` |
| Table S9 | `results/longmenshan_with_source_region.csv` |
| Figure S4 | `results/SHAP_region_mean_absolute.csv` |
| Model configuration | `results/runtime.json`, `source_audit/TabPFN_regression_architecture_audit.json` |

`idx` is the zero-based input row position; `No` is the original one-based source number. Historical `zero-shot` labels identify TabPFN in-context inference.

Run `python revision/verify_results.py` to recompute added prediction metrics, coverage and principal claims. This verifies consistency of saved outputs. It does not regenerate training histories or establish causal conclusions.
