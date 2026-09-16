# Sediment-source information and small-sample debris-flow volume prediction

Code and saved evidence accompanying **Sediment-source information improves small-sample prediction of maximum recorded debris-flow volume with a pretrained tabular model**.

The study evaluates coseismic deposit inventories using 60 Longmen Shan catchments and a separate benchmark of 63 Korean events. Matched feature removal, model comparisons, regional holdouts and prediction intervals support the environmental interpretation. Model performance depends on predictor information, preprocessing and validation protocol.

## Start here

- [Reproduction instructions](revision/README.md)
- [Manuscript evidence map](revision/EVIDENCE_MAP.md)
- [Additional experiment results](revision/results/)
- [Archived primary results](revision/archived/)
- [Updated figures in 900 dpi JPEG format](revision/figures/)
- [Release changes](CHANGELOG.md)

## Evaluation protocols

| Analysis | Design | Manuscript evidence |
|---|---|---|
| Primary comparison | Ten repetitions of fivefold outer validation; threefold inner validation; 30 TPE trials; raw predictors; 128-member TabPFN | Tables 2 to 4; Figures 5, 7 and 8 |
| Learning curves | Fixed configurations and outer tests; training sizes 10 to 48 | Figure 9 |
| Auxiliary diagnostics | Separate seed-42 fivefold evaluation with fixed RF and 128-member TabPFN | Figure 6 |
| Primary residual intervals | Eight-member TabPFN; internal cross-validation residuals; repeated outer tests | Figure 10 |
| Korean benchmark | Training and testing within Korean data; fixed conventional configurations | Figure 11 |
| Transformation and tuning | One fivefold repetition with seed 20260915; threefold inner selection; eight-member TabPFN reference | Tables S4 and S5; Figure S1 |
| Region holdouts | Six Chinese regions or three Korean districts; RF tuning within training regions | Tables S6 and S7; Figure S2 |
| Split conformal | Fifteen calibration observations inside each outer training pool; separate fivefold evaluation | Table S8; Figure S3 |
| Attribution | Full-data SHAP and regional aggregation | Figures 12 and 13; Table S10; Figure S4 |

Archived results establish the reported values. Reruns can change with package versions, model weights, device and numerical implementation. No numerical tolerance is guaranteed. Historical CSV labels such as `zero-shot` remain as archival identifiers; the manuscript uses **in-context inference**.

## Principal results and scope

- With deposit information, primary TabPFN mean R² is 0.9190; without it, 0.6692. The mean gain across eleven models is 0.2582.
- In Longmen Shan region holdouts, TabPFN pooled R² is 0.9216 with deposits and 0.6284 without them. RF gives 0.8783 and 0.6873.
- Korean district holdout R² is 0.7425 for TabPFN and 0.6185 for RF. The primary Korean comparison is close, and RF has lower MAE.
- Transformed Lasso and Elastic Net match or slightly exceed the eight-member TabPFN reference on the separate sensitivity partitions.
- Primary residual intervals cover 553/600 repeated test instances at nominal 90%, or 92.2%. Separate split-conformal coverage is 55/60 for China and 61/63 for Korea. The 600 repeated instances reuse 60 catchments.

These are observational prediction results. SHAP describes the fitted model. Direct China-to-Korea transfer was not evaluated. Regional membership is documented; verified individual coordinates, catchment boundaries and time-matched soil and land-use layers are unavailable in the compiled attribute tables.

## Installation and use

Install the inspected versions in `requirements.txt`. TabPFN package version **6.4.1** and model checkpoint version **2.5** are separate identifiers. Obtain the regression checkpoint from the official provider under its applicable terms. Weights are excluded from this repository.

```bash
pip install -r requirements.txt
python revision/verify_results.py
python revision/plot_figures.py --part all
```

New plots go to `revision/rerun_figures`; delivered figures remain in `revision/figures`.

```bash
python revision/run_revision.py --part transform
python revision/run_revision.py --part budget
python revision/run_revision.py --part foundation --model-path /path/to/tabpfn-v2.5-regressor-v2.5_default.ckpt
```

Model reruns write to `revision/rerun_results`. The original primary workflow remains in `main.py`. Set `TABPFN_MODEL_PATH` before running it. `FAST_MODE = True` runs a reduced diagnostic, while `False` runs the full design. This entry point requires explicit weights and preserves the requested seed and ensemble size. New reruns remain separate from the archived evidence.

## Data and attribution

The datasets are from [Huang et al. (2020)](https://doi.org/10.1016/j.geomorph.2020.107333) and [Lee et al. (2021)](https://doi.org/10.1016/j.enggeo.2020.105979). Their measurement protocols and third-party data rights remain attributable to those sources. Processed records preserve original row identifiers and available regional membership.

Chinese volumes are recorded in 10⁴ m³ and Korean volumes in m³; targets are log₁₀ of the respective numeric volume columns. The Chinese response is the maximum recorded episode volume during 2008 to 2018. Korean responses describe historical individual events during 2011 to 2013.

For TabPFN, cite [Hollmann et al. (2025)](https://doi.org/10.1038/s41586-024-08328-6) and [Grinsztajn et al. (2025)](https://doi.org/10.48550/arXiv.2511.08667).

## Licence

The MIT licence applies to the authors' repository code. It grants no additional rights to third-party datasets, model weights or provider software. See [LICENSE](LICENSE) and the cited sources.
