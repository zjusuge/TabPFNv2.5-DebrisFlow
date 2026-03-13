# Sediment-Source Information Controls Small-Sample Debris-Flow Volume Prediction

> Benchmarking a pretrained tabular foundation model (TabPFN-2.5) against
> Bayesian-optimised machine-learning baselines for catchment-scale
> maximum single-event debris-flow volume prediction.

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/licence-MIT-brightgreen)
![Seed: 42](https://img.shields.io/badge/seed-42-orange)

---

## Scope

This repository releases the **primary comparative benchmark code**
corresponding to Sections 3--4.3 of the paper.

### Implemented in `main.py`

| Component | Paper section |
|:----------|:-------------|
| 10 x 5-fold repeated nested cross-validation | S3.4 |
| TabPFN-2.5 zero-shot inference | S3.2 |
| 10 baselines -- default **and** Bayesian-optimised (Optuna TPE) | S3.3 |
| Paired feature-set ablation (Set A vs Set B) | S4.2 |
| Learning-curve analysis across training sizes | S4.3 |
| Repeat-level Wilcoxon signed-rank tests + Holm--Bonferroni correction | S3.4 |

### Not included in this release

The following analyses reported in the paper were produced with separate
scripts and are **not** part of this repository:

- Independent Korean benchmark (S4.5)
- Cross-validated conformal prediction-interval calibration (S4.4)
- SHAP-based interpretability analysis (S5)
- All figures and visualisations

---

## Repository Structure

```text
.
├── data/
│   └── longmenshan_60.csv          # Primary dataset (n = 60)
├── results/                        # Auto-created on first run
│   ├── FS{A,B}*.csv                # Per-feature-set outputs
│   ├── comparison_A_vs_B*.csv      # Feature-ablation comparison
│   ├── grand_summary.csv           # Cross-set combined summary
│   ├── experiment_config.json      # Run configuration snapshot
│   └── experiment_log_*.txt        # Full console log
├── main.py                         # Benchmark script (this release)
├── requirements.txt                # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

---

## Dataset

The primary dataset is the post-earthquake debris-flow compilation by
[Huang et al. (2020)](https://doi.org/10.1016/j.geomorph.2020.107333),
covering 60 catchments in the Longmen Shan fault zone (2008--2018).

The CSV file `data/longmenshan_60.csv` must contain the following columns:

| Column | Symbol | Unit | Description |
|:-------|:-------|:-----|:------------|
| `A_km2` | *A* | km2 | Catchment area |
| `H_m` | *H* | m | Relative relief |
| `L_km` | *L* | km | Main channel length |
| `D_km` | *D* | km | Fault distance |
| `J_permille` | *J* | permil | Mean longitudinal channel gradient |
| `V_landslide_1e4m3` | *V_landslide* | x10^4 m3 | Coseismic landslide-deposit volume |
| `V0_1e4m3` | *V0* | x10^4 m3 | Max single-event debris-flow volume (**response**) |

An optional `No` column (row index) is dropped automatically if present.
The modelling target is log10(*V0*).

---

## Models

| Model | Family | Tuning mode | Inner loop |
|:------|:-------|:------------|:-----------|
| **TabPFN-2.5** | Pretrained tabular foundation model | Zero-shot | None |
| Ridge | Regularised linear | Default + BO | 3-fold CV, 30 TPE trials |
| Lasso | Regularised linear | Default + BO | 3-fold CV, 30 TPE trials |
| Elastic Net | Regularised linear | Default + BO | 3-fold CV, 30 TPE trials |
| SVR | Kernel | Default + BO | 3-fold CV, 30 TPE trials |
| KNN | Neighbour-based | Default + BO | 3-fold CV, 30 TPE trials |
| Random Forest | Tree ensemble | Default + BO | 3-fold CV, 30 TPE trials |
| GBR | Tree ensemble | Default + BO | 3-fold CV, 30 TPE trials |
| XGBoost | Tree ensemble | Default + BO | 3-fold CV, 30 TPE trials |
| AdaBoost | Tree ensemble | Default + BO | 3-fold CV, 30 TPE trials |
| MLP | Neural network | Default + BO | 3-fold CV, 30 TPE trials |

Feature Set A includes all six predictors; Feature Set B excludes
*V_landslide*. Both sets share identical outer splits, random seeds,
and evaluation logic to enable a clean paired comparison.

---

## Installation

### Prerequisites

- Python >= 3.9
- pip (or conda)
- A Hugging Face account is required for the initial TabPFN-2.5 model
  weight download. Run `huggingface-cli login` before first use.
- A CUDA-enabled GPU is **recommended** but not required. TabPFN-2.5
  runs on CPU if no GPU is detected.

### Install dependencies

```bash
pip install -r requirements.txt
```

### Verify TabPFN installation

```python
from tabpfn import TabPFNRegressor
print("TabPFN OK")
```

---

## Usage

### Quick sanity check (approx. 5--10 min)

Open `main.py` and ensure:

```python
FAST_MODE = True
```

Then run:

```bash
python main.py
```

This executes a reduced experiment (2 repeats x 2-fold, 8 BO trials)
to verify that all dependencies, data paths, and model calls work
correctly.

### Full experiment (paper results)

```python
FAST_MODE = False
```

```bash
python main.py
```

| Setting | Value |
|:--------|:------|
| Outer CV | 10 repeats x 5-fold = 50 folds |
| Inner BO | 3-fold CV, 30 TPE trials per baseline per fold |
| Learning-curve sizes | 10, 15, 20, 25, 30, 35, 40, 45, 48 |
| Estimated wall time | 2--6 h (hardware-dependent) |

All outputs are written to `results/`.

---

## Output Files

After a full run, `results/` contains approximately 27 files:

| Category | File pattern | Description |
|:---------|:-------------|:------------|
| Config | `experiment_config.json` | Run parameters + package versions (JSON) |
| CV plan | `outer_cv_plan.csv` | Train/test indices for all 50 folds |
| Raw results | `FS{A,B}_all_folds.csv` | Per-fold x model x mode long-format records |
| Fold-level R2 | `FS{A,B}_per_fold_R2.csv` | Wide-format R2 per fold |
| Repeat-level R2 | `FS{A,B}_per_repeat_R2_{BO,Default}.csv` | Mean R2 per repeat (inferential unit) |
| Summaries | `FS{A,B}_summary_vs_{BO,Default}.csv` | Descriptive statistics across 50 folds |
| Statistical tests | `FS{A,B}_wilcoxon_vs_{BO,Default}.csv` | One-sided Wilcoxon + Holm--Bonferroni |
| Learning curves | `FS{A,B}_learning_curve_{raw,summary}.csv` | R2 across training sizes |
| Feature ablation | `comparison_A_vs_B_{BO,Default}.csv` | Two-sided paired test, Set A vs Set B |
| Grand summary | `grand_summary.csv` | Combined cross-set overview |
| Log | `experiment_log_{FULL,FAST_MODE}.txt` | Full console log |

---

## Reproducibility Notes

| Item | Value |
|:-----|:------|
| Global random seed | 42 |
| Outer splits | Deterministic via `RandomState(42)` |
| BO sampler seed | 42 (Optuna TPE) |
| Inner CV seed | 42 |

Known sources of minor non-determinism:

- Floating-point ordering may vary across CPU architectures.
- TabPFN inference on GPU may introduce non-deterministic reductions.

Results should be reproducible to within +/-0.002 R2 on the same hardware.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE)
for details.

---

## Citation
Please cite the dataset source and the TabPFN model:

```bibtex
@article{huang2020hybrid,
  author  = {Huang, J. and Hales, T. C. and Huang, R. and Ju, N.
             and Li, Q. and Huang, Y.},
  title   = {A hybrid machine-learning model to estimate potential
             debris-flow volumes},
  journal = {Geomorphology},
  volume  = {367},
  pages   = {107333},
  year    = {2020},
  doi     = {10.1016/j.geomorph.2020.107333}
}

@article{hollmann2025tabpfn,
  author  = {Hollmann, N. and M\"{u}ller, S. and Purucker, L. and
             Krishnakumar, A. and K\"{o}rfer, M. and Hoo, S. B. and
             Schirrmeister, R. T. and Hutter, F.},
  title   = {Accurate predictions on small data with a tabular
             foundation model},
  journal = {Nature},
  volume  = {637},
  pages   = {319--326},
  year    = {2025},
  doi     = {10.1038/s41586-024-08328-6}
}
```

---

## Contact

Corresponding author information is available in the published version
of the paper.

<!--
Uncomment after acceptance:
Tianlong Wang -- tianlong_wang@zju.edu.cn
Ocean College, Zhejiang University
No. 1 Zheda Road, Dinghai District, Zhoushan 316000, Zhejiang, China
-->
