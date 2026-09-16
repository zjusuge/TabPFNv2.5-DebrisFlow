# Revision analyses and saved evidence

This directory preserves the additional analyses and archived results underlying the revised manuscript, separately from rerun outputs.

## Contents

- `data` contains the processed modelling workbooks.
- `archived` contains historical primary folds, learning curves, auxiliary predictions, residual intervals and SHAP outputs. Some historical summary tables contain superseded statistics. Revised inference uses underlying folds and `results/paired_statistics_audit.json`.
- `results` contains additional predictions, trial histories, selected configurations, warnings and derived summaries.
- `source_audit` contains the complete source table and attribute checks.
- `figures` contains delivered Fig01 to Fig13 and FigS1 to FigS4 in 900 dpi JPEG format.
- `plotting` contains adapted original plotting cells and editable original-design Figures 3 and 4.
- `run_revision.py` runs the additional model analyses.
- `plot_figures.py` redraws quantitative Figures 5 to 11 and S1 to S4 through the original plotting functions and adapted cells.
- `verify_results.py` recomputes numerical summaries without model fitting.
- `SHA256.json` records evidence and figure checksums.

## Model reproduction

Install `requirements.txt` from the repository root. The additional experiments use seed 20260915, CPU inference and eight ensemble members. The recorded checkpoint is `tabpfn-v2.5-regressor-v2.5_default.ckpt`, SHA256 `6e0308b7e5f3a8325b798b96d986b2b282c3608b1cda6e2991425a2a261d40b8`.

```bash
python revision/run_revision.py --part transform
python revision/run_revision.py --part budget
python revision/run_revision.py --part foundation --model-path /path/to/tabpfn-v2.5-regressor-v2.5_default.ckpt
```

`TABPFN_MODEL_PATH` can supply the checkpoint instead of the command-line argument. `--output-dir` chooses a separate rerun directory; writing into delivered `results` is rejected.

Raw/log predictor comparisons use training-fold scaling and 30-trial inner searches. Both MLP arms use L-BFGS, one or two layers and 8 to 64 units per layer. This MLP configuration differs from the primary protocol. Budget comparisons evaluate the best first-30 and full-100 trial configurations on matched outer tests, with separate inner R² and RMSE selection.

The foundation analysis includes a random fivefold reference, split conformal and region holdouts. Split conformal reserves 15 calibration cases within each outer training pool; the nominal 90% threshold is the largest calibration residual. Its marginal-coverage interpretation requires exchangeability. Region holdouts train within each country's own data system.

Completed job JSON files support resumption. Changed seeds, checkpoints or configurations require a **new output directory**. The resume mechanism does not validate changed settings against completed jobs. Resume interrupted runs with exactly the same settings.

## Figure reproduction

```bash
python revision/plot_figures.py --part all
```

The runner uses saved results and adapted original plotting designs. Arial is used where available, with DejaVu Sans fallback. Fonts can change layout slightly across platforms. Figures 3 and 4 are PowerPoint schematics with editable source and exported JPEGs. Unchanged original Figures 1, 2, 12 and 13 are outside this figure update.

## Interpretation and provenance

Primary point comparisons use 128 ensemble members; primary residual intervals and additional sensitivities use eight. Result tracks have distinct seeds and designs. Fold-average and pooled R² also use different reference means. Compare values within the specified protocol.

All 420 Longmen Shan source attributes matched both the modelling workbook and input CSV. Original numbers and six source regions are retained. No coordinates were inferred from map symbols.

The historical primary interval method pools internal cross-validation residuals and uses an interpolated quantile. Its coverage is empirical. Added split-conformal results are identified separately. Original-unit interval files use a 10⁴ m³ multiplier for China and 1 m³ for Korea.

## Confirmed figure layout update

The delivered JPEG figures use the confirmed original-style layout with compact panel spacing, internal legends and no subplot titles. Numerical inputs and experimental protocols are unchanged. Figures 1, 3 and 4 retain their existing designs.

Run `python revision/plot_current_figures.py all` to regenerate Figures 2, 5 to 13 and S1 to S4 from saved inputs. Outputs go to `revision/rerun_current_figures`, including internal font-audit PDFs and previews. Only the 900 dpi JPG files are publication figures. The earlier `plot_figures.py` remains available for historical layout reproduction.
