#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===========================================================================
TabPFN (In-context inference) vs 10 Baselines (Default + Bayesian Optimised)
Debris-Flow Volume Prediction -- Longmen Shan Fault Zone
===========================================================================

Core design
-----------
  (1) Statistical tests are performed on repeat-level mean R2
      (full mode: 10 repeats), NOT on 50 fold-level R2 values.
  (2) Outer CV splits are generated explicitly with repeat / split IDs.
  (3) Per-repeat R2 tables are exported for BO and Default modes.
  (4) No plotting / visualisation -- only CSV / JSON / TXT outputs.
  (5) Keeps FAST_MODE for quick sanity checking.

Experiment design
-----------------
  - Outer CV  : 10 repeats x 5-fold  (FAST_MODE: 2 repeats x 2-fold)
  - Inner BO  : 3-fold CV with Optuna TPE
  - TabPFN    : in-context inference, no inner tuning
  - Baselines : default and BO-tuned
  - Learning curves: tabular output only (no plots)

Target
------
  log10(V0_1e4m3)

Metrics
-------
  R2, RMSE, MAE   (all in log10 space)

Requirements
------------
  pip install -r requirements.txt
===========================================================================
"""

# =================================================================
# QUICK-TEST SWITCHES
#   FAST_MODE = True  -> rapid sanity check
#   FAST_MODE = False -> full benchmark rerun; inspect the archived results for reported values
# =================================================================
FAST_MODE = False

# Whether to run learning-curve analysis (CSV export only, no plots)
RUN_LEARNING_CURVES = True


# =================================================================
# 0. Imports
# =================================================================
import json
import os
import hashlib
import sys
import time
import warnings
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError as e:
    raise ImportError("xgboost not installed. Run: pip install xgboost") from e

try:
    import optuna
except ImportError as e:
    raise ImportError("optuna not installed. Run: pip install optuna") from e

try:
    from tabpfn import TabPFNRegressor
    HAS_TABPFN = True
except Exception:
    HAS_TABPFN = False
    print("[WARNING] tabpfn not installed or failed to import. "
          "Run: pip install tabpfn")

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =================================================================
# 1. Configuration
# =================================================================
SEED = 42
np.random.seed(SEED)

# ---- Paths (relative to this script, portable across machines) ----
DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_FILE = DATA_DIR / "longmenshan_60.csv"
OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Column definitions ----
TARGET = "V0_1e4m3"
FEAT_A = ["A_km2", "H_m", "L_km", "D_km", "J_permille", "V_landslide_1e4m3"]
FEAT_B = ["A_km2", "H_m", "L_km", "D_km", "J_permille"]

# ---- CV / BO parameters ----
if FAST_MODE:
    N_SPLITS = 2
    N_REPEATS = 2
    BO_TRIALS = 8
    LC_SIZES = [10, 20, 30]
    TABPFN_N_EST = 8
else:
    N_SPLITS = 5
    N_REPEATS = 10
    BO_TRIALS = 30
    LC_SIZES = [10, 15, 20, 25, 30, 35, 40, 45, 48]
    TABPFN_N_EST = 128

N_FOLDS = N_SPLITS * N_REPEATS
INNER_SPLITS = 3

# Minimum number of repeats required for inferential testing
MIN_REPEATS_FOR_TEST = 5

# ---- Model taxonomy ----
NEEDS_SCALING = frozenset(["Ridge", "Lasso", "ElasticNet", "SVR", "KNN", "MLP"])
BASELINES = [
    "Ridge", "Lasso", "ElasticNet", "SVR", "KNN",
    "RF", "GBR", "XGBoost", "AdaBoost", "MLP",
]
LC_MODELS = (["TabPFN"] if HAS_TABPFN else []) + [
    "RF", "XGBoost", "SVR", "Ridge", "GBR", "MLP",
]
ALL_MODELS = (["TabPFN"] if HAS_TABPFN else []) + BASELINES


# =================================================================
# 2. Logging (console + in-memory buffer)
# =================================================================
_LOG = StringIO()


def log(msg="", end="\n"):
    """Print to console and write to buffer simultaneously."""
    print(msg, end=end, flush=True)
    _LOG.write(str(msg) + end)


def banner(title):
    """Print a section banner."""
    border = "=" * 72
    log(f"\n{border}")
    log(f"  {title}")
    log(f"{border}")


# =================================================================
# 3. Small utility helpers
# =================================================================
def _safe_nanmean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.any(~np.isnan(x)) else np.nan


def _safe_nanstd(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanstd(x)) if np.any(~np.isnan(x)) else np.nan


def _safe_nanmedian(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmedian(x)) if np.any(~np.isnan(x)) else np.nan


def _safe_nansum(x):
    x = np.asarray(x, dtype=float)
    return float(np.nansum(x)) if x.size > 0 else np.nan


def _pkg_version(name):
    """Return installed version string for a package, or 'N/A'."""
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return "N/A"


# =================================================================
# 4. Data loading
# =================================================================
def load_data():
    """Read CSV, apply log10 transform to the target, and build
    dual feature matrices.

    Returns
    -------
    feat_sets : dict
        {"A": (X_A, col_names_A), "B": (X_B, col_names_B)}
    y : ndarray
        log10-transformed target vector.
    """
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    log(f"  Loaded: {df.shape[0]} samples x {df.shape[1]} columns")
    log(f"  Columns: {list(df.columns)}")

    if "No" in df.columns:
        df = df.drop(columns=["No"])
        log("  Dropped 'No' column (row index, not a feature)")

    required_cols = sorted(set(FEAT_A + FEAT_B + [TARGET]))
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in data file: {missing_cols}")

    n_miss = int(df[required_cols].isnull().sum().sum())
    log(f"  Missing values in required columns: {n_miss}")
    if n_miss > 0:
        raise ValueError(
            "Data contains missing values in required columns. "
            "Please clean or impute the dataset before running."
        )

    y_raw = df[TARGET].values.astype(np.float64)
    if np.any(y_raw <= 0):
        raise ValueError(
            f"Target '{TARGET}' contains non-positive values; "
            f"log10 transform is invalid."
        )
    y = np.log10(y_raw)

    log(f"\n  Target: {TARGET}")
    log(f"  Raw range:   [{y_raw.min():.2f}, {y_raw.max():.2f}] (x10^4 m3)")
    log(f"  log10 range: [{y.min():.4f}, {y.max():.4f}]")

    X_A = df[FEAT_A].values.astype(np.float64)
    X_B = df[FEAT_B].values.astype(np.float64)

    log(f"\n  Feature Set A ({len(FEAT_A)} features): {FEAT_A}")
    log(f"  Feature Set B ({len(FEAT_B)} features): {FEAT_B}")

    log(f"\n  Descriptive statistics:")
    desc = df[FEAT_A + [TARGET]].describe().T[["mean", "std", "min", "max"]]
    log(desc.to_string())

    return {"A": (X_A, FEAT_A), "B": (X_B, FEAT_B)}, y


# =================================================================
# 5. Outer split generator
# =================================================================
def make_outer_splits(n_samples):
    """Generate explicit repeated K-fold splits with repeat and split
    indices, enabling repeat-level aggregation and paired testing."""
    idx = np.arange(n_samples)
    rng = np.random.RandomState(SEED)
    splits = []

    for rep in range(N_REPEATS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=rng)
        for split_id, (tri, tei) in enumerate(kf.split(idx)):
            splits.append(
                dict(
                    fold=rep * N_SPLITS + split_id,
                    repeat=rep,
                    split=split_id,
                    train_idx=tri,
                    test_idx=tei,
                )
            )
    return splits


# =================================================================
# 6. Model factory
# =================================================================
def _defaults(name):
    """Return carefully chosen default hyperparameters (no tuning)."""
    return {
        "Ridge": dict(alpha=1.0),
        "Lasso": dict(alpha=0.01, max_iter=5000),
        "ElasticNet": dict(alpha=0.01, l1_ratio=0.5, max_iter=5000),
        "SVR": dict(C=10.0, epsilon=0.1, kernel="rbf"),
        "KNN": dict(n_neighbors=5, weights="distance"),
        "RF": dict(n_estimators=200, random_state=SEED, n_jobs=1),
        "GBR": dict(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=SEED,
        ),
        "XGBoost": dict(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=SEED,
            verbosity=0,
            objective="reg:squarederror",
            n_jobs=1,
        ),
        "AdaBoost": dict(
            n_estimators=200,
            learning_rate=0.1,
            random_state=SEED,
        ),
        "MLP": dict(
            hidden_layer_sizes=(64, 32),
            max_iter=2000,
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        ),
    }.get(name, {})


_CLS = {
    "Ridge": Ridge,
    "Lasso": Lasso,
    "ElasticNet": ElasticNet,
    "SVR": SVR,
    "KNN": KNeighborsRegressor,
    "RF": RandomForestRegressor,
    "GBR": GradientBoostingRegressor,
    "XGBoost": XGBRegressor,
    "AdaBoost": AdaBoostRegressor,
    "MLP": MLPRegressor,
}


def make_model(name, params=None):
    """Instantiate a scikit-learn or XGBoost regression model."""
    p = params if params is not None else _defaults(name)
    return _CLS[name](**p)


def make_tabpfn():
    """Use an explicitly selected checkpoint and preserve all inference settings."""
    raw_path = os.environ.get("TABPFN_MODEL_PATH")
    if not raw_path:
        raise ValueError("Set TABPFN_MODEL_PATH to the TabPFN-2.5 regression checkpoint.")
    checkpoint = Path(raw_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return TabPFNRegressor(model_path=checkpoint, n_estimators=TABPFN_N_EST,
                          random_state=SEED)


# =================================================================
# 7. Bayesian optimisation -- search spaces
# =================================================================
def _suggest(trial, name):
    """Define per-model Optuna search space; returns a param dict."""

    if name == "Ridge":
        return dict(
            alpha=trial.suggest_float("alpha", 1e-3, 1e3, log=True),
        )

    if name == "Lasso":
        return dict(
            alpha=trial.suggest_float("alpha", 1e-5, 1e1, log=True),
            max_iter=5000,
        )

    if name == "ElasticNet":
        return dict(
            alpha=trial.suggest_float("alpha", 1e-5, 1e1, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.05, 0.95),
            max_iter=5000,
        )

    if name == "SVR":
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
        p = dict(
            C=trial.suggest_float("C", 1e-1, 1e3, log=True),
            epsilon=trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
            kernel=kernel,
        )
        if kernel == "rbf":
            p["gamma"] = trial.suggest_float("gamma", 1e-4, 1e1, log=True)
        return p

    if name == "KNN":
        return dict(
            n_neighbors=trial.suggest_int("n_neighbors", 1, 20),
            weights=trial.suggest_categorical("weights",
                                              ["uniform", "distance"]),
            p=trial.suggest_int("p", 1, 2),
        )

    if name == "RF":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            random_state=SEED,
            n_jobs=1,
        )

    if name == "GBR":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3,
                                              log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            random_state=SEED,
        )

    if name == "XGBoost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3,
                                              log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5,
                                                  1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            random_state=SEED,
            verbosity=0,
            objective="reg:squarederror",
            n_jobs=1,
        )

    if name == "AdaBoost":
        return dict(
            n_estimators=trial.suggest_int("n_estimators", 50, 500),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 2.0,
                                              log=True),
            random_state=SEED,
        )

    if name == "MLP":
        n_layers = trial.suggest_int("n_layers", 1, 3)
        layers = tuple(
            trial.suggest_int(f"u{i}", 8, 128) for i in range(n_layers)
        )
        return dict(
            hidden_layer_sizes=layers,
            alpha=trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            learning_rate_init=trial.suggest_float(
                "learning_rate_init", 1e-4, 1e-2, log=True
            ),
            max_iter=2000,
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
        )

    return {}


# =================================================================
# 8. Bayesian optimisation engine
# =================================================================
def run_bo(name, X_tr, y_tr):
    """Inner-loop Bayesian optimisation with INNER_SPLITS-fold CV.

    Returns the best hyperparameter dict found by TPE.
    """
    need_scale = name in NEEDS_SCALING
    inner_cv = KFold(n_splits=INNER_SPLITS, shuffle=True,
                     random_state=SEED)
    trial_params = {}

    def objective(trial):
        params = _suggest(trial, name)
        trial_params[trial.number] = params
        scores = []

        for itr, ival in inner_cv.split(X_tr):
            try:
                Xi_tr, Xi_va = X_tr[itr], X_tr[ival]
                yi_tr, yi_va = y_tr[itr], y_tr[ival]

                if need_scale:
                    sc = StandardScaler()
                    Xi_tr = sc.fit_transform(Xi_tr)
                    Xi_va = sc.transform(Xi_va)

                model = make_model(name, params)
                model.fit(Xi_tr, yi_tr)
                pred = np.asarray(model.predict(Xi_va),
                                  dtype=float).ravel()
                scores.append(r2_score(yi_va, pred))

            except Exception:
                return -999.0

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=BO_TRIALS, show_progress_bar=False)

    if len(trial_params) == 0 or study.best_value < -100:
        return _defaults(name)

    return trial_params.get(study.best_trial.number, _defaults(name))


# =================================================================
# 9. Evaluation metrics
# =================================================================
def calc_metrics(y_true, y_pred):
    """Compute R2, RMSE, and MAE in log10 space."""
    return dict(
        R2=float(r2_score(y_true, y_pred)),
        RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        MAE=float(mean_absolute_error(y_true, y_pred)),
    )


# =================================================================
# 10. Fit / predict helper
# =================================================================
def _fit_predict(name, params, X_tr, y_tr, X_te):
    """Fit a baseline model (with optional scaling) and predict."""
    if name in NEEDS_SCALING:
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)

    model = make_model(name, params)
    model.fit(X_tr, y_tr)
    return np.asarray(model.predict(X_te), dtype=float).ravel()


# =================================================================
# 11. Nested cross-validation
# =================================================================
def run_nested_cv(X, y, fs_label, outer_splits):
    """Outer loop: explicit repeated K-fold for unbiased evaluation.
    Inner loop: K-fold for BO hyperparameter selection (baselines).
    TabPFN: zero-shot inference (no inner loop).

    Returns
    -------
    records_df : DataFrame
        Long-format per-fold x model x mode results.
    """
    records = []

    for info in outer_splits:
        fold = info["fold"]
        repeat = info["repeat"]
        split = info["split"]
        tri = info["train_idx"]
        tei = info["test_idx"]

        Xtr, Xte = X[tri], X[tei]
        ytr, yte = y[tri], y[tei]
        t_fold = time.time()

        tabpfn_r2 = np.nan

        # ---- TabPFN (zero-shot) ----
        if HAS_TABPFN:
            t0 = time.time()
            try:
                mdl = make_tabpfn()
                mdl.fit(Xtr, ytr)
                pred = np.asarray(mdl.predict(Xte),
                                  dtype=float).ravel()
                m = calc_metrics(yte, pred)
                tabpfn_r2 = m["R2"]
            except Exception as exc:
                m = dict(R2=np.nan, RMSE=np.nan, MAE=np.nan)
                log(f"    [TabPFN ERROR fold {fold}] {exc}")
            dt = time.time() - t0

            records.append(
                dict(
                    feature_set=fs_label,
                    fold=fold,
                    repeat=repeat,
                    split=split,
                    n_train=len(tri),
                    n_test=len(tei),
                    model="TabPFN",
                    mode="zero-shot",
                    **m,
                    time_s=dt,
                )
            )

        # ---- Baselines ----
        for mn in BASELINES:
            # (a) Default parameters
            t0 = time.time()
            try:
                pred_d = _fit_predict(mn, None, Xtr, ytr, Xte)
                m_d = calc_metrics(yte, pred_d)
            except Exception:
                m_d = dict(R2=np.nan, RMSE=np.nan, MAE=np.nan)
            dt_d = time.time() - t0

            records.append(
                dict(
                    feature_set=fs_label,
                    fold=fold,
                    repeat=repeat,
                    split=split,
                    n_train=len(tri),
                    n_test=len(tei),
                    model=mn,
                    mode="default",
                    **m_d,
                    time_s=dt_d,
                )
            )

            # (b) Bayesian-optimised
            t0 = time.time()
            try:
                best_p = run_bo(mn, Xtr, ytr)
                pred_b = _fit_predict(mn, best_p, Xtr, ytr, Xte)
                m_b = calc_metrics(yte, pred_b)
            except Exception:
                m_b = dict(R2=np.nan, RMSE=np.nan, MAE=np.nan)
            dt_b = time.time() - t0

            records.append(
                dict(
                    feature_set=fs_label,
                    fold=fold,
                    repeat=repeat,
                    split=split,
                    n_train=len(tri),
                    n_test=len(tei),
                    model=mn,
                    mode="bo",
                    **m_b,
                    time_s=dt_b,
                )
            )

        elapsed = time.time() - t_fold
        log(
            f"  [{fs_label}] Fold {fold + 1:>2}/{N_FOLDS} "
            f"(repeat={repeat + 1}/{N_REPEATS}, "
            f"split={split + 1}/{N_SPLITS}) "
            f"TabPFN R2={tabpfn_r2:.4f}  ({elapsed:.1f}s)"
        )

    return pd.DataFrame(records)


# =================================================================
# 12. Fold-level and repeat-level table builders
# =================================================================
def build_fold_level_r2(records_df):
    """Pivot to wide table: fold | repeat | split | TabPFN | ... """
    df = records_df.copy()
    df["col_name"] = np.where(
        df["model"] == "TabPFN",
        "TabPFN",
        df["model"] + "_" + df["mode"],
    )

    wide = (
        df.pivot_table(
            index=["fold", "repeat", "split"],
            columns="col_name",
            values="R2",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["fold", "repeat", "split"])
        .reset_index(drop=True)
    )

    col_order = ["fold", "repeat", "split"]
    if "TabPFN" in wide.columns:
        col_order.append("TabPFN")
    for mn in BASELINES:
        for md in ("default", "bo"):
            c = f"{mn}_{md}"
            if c in wide.columns:
                col_order.append(c)

    remaining = [c for c in wide.columns if c not in col_order]
    return wide[col_order + remaining]


def build_repeat_level_r2(records_df, mode_filter):
    """Wide repeat-level table: each value is the mean R2 across
    folds within a single repeat.

    Parameters
    ----------
    mode_filter : str
        Either ``"bo"`` or ``"default"``.
    """
    mask = ((records_df["model"] == "TabPFN") |
            (records_df["mode"] == mode_filter))
    sub = records_df.loc[mask, ["repeat", "model", "R2"]].copy()

    rep = (
        sub.groupby(["repeat", "model"], as_index=False)["R2"]
        .mean()
        .sort_values(["repeat", "model"])
    )

    wide = (
        rep.pivot(index="repeat", columns="model", values="R2")
        .reset_index()
        .sort_values("repeat")
        .reset_index(drop=True)
    )

    col_order = ["repeat"] + [m for m in ALL_MODELS if m in wide.columns]
    remaining = [c for c in wide.columns if c not in col_order]
    return wide[col_order + remaining]


# =================================================================
# 13. Learning curves (tabular only)
# =================================================================
def run_learning_curves(X, y, fs_label, outer_splits):
    """For each outer fold, sub-sample n in LC_SIZES from the training
    pool while keeping the test set fixed.  Default params only."""
    records = []

    for i, info in enumerate(outer_splits):
        fold = info["fold"]
        repeat = info["repeat"]
        split = info["split"]
        tri = info["train_idx"]
        tei = info["test_idx"]

        Xtr_full, Xte = X[tri], X[tei]
        ytr_full, yte = y[tri], y[tei]

        for n in LC_SIZES:
            if n > len(Xtr_full):
                continue

            rng = np.random.RandomState(SEED + fold * 1000 + n)
            idx = rng.choice(len(Xtr_full), size=n, replace=False)
            Xsub, ysub = Xtr_full[idx], ytr_full[idx]

            for mn in LC_MODELS:
                try:
                    if mn == "TabPFN":
                        mdl = make_tabpfn()
                        mdl.fit(Xsub, ysub)
                        pred = np.asarray(mdl.predict(Xte),
                                          dtype=float).ravel()
                    else:
                        Xs, Xt = Xsub, Xte
                        if mn in NEEDS_SCALING:
                            sc = StandardScaler()
                            Xs = sc.fit_transform(Xs)
                            Xt = sc.transform(Xt)
                        mdl = make_model(mn)
                        mdl.fit(Xs, ysub)
                        pred = np.asarray(mdl.predict(Xt),
                                          dtype=float).ravel()

                    r2 = float(r2_score(yte, pred))
                except Exception:
                    r2 = np.nan

                records.append(
                    dict(
                        feature_set=fs_label,
                        fold=fold,
                        repeat=repeat,
                        split=split,
                        model=mn,
                        n_train=n,
                        R2=r2,
                    )
                )

        if (i + 1) % max(N_FOLDS // 5, 1) == 0:
            log(f"  [{fs_label}] LC fold {i + 1}/{N_FOLDS}")

    return pd.DataFrame(records)


def lc_summary(lc_df):
    """Aggregate learning-curve records into a tidy summary."""
    if lc_df.empty:
        return pd.DataFrame(
            columns=["Model", "n_train", "R2_mean", "R2_std",
                      "R2_median", "n_folds"]
        )

    rows = []
    for mn in LC_MODELS:
        for n in LC_SIZES:
            sub = lc_df[(lc_df["model"] == mn) & (lc_df["n_train"] == n)]
            v = sub["R2"].dropna()

            if len(v) > 0:
                rows.append(
                    dict(
                        Model=mn,
                        n_train=n,
                        R2_mean=float(v.mean()),
                        R2_std=float(v.std(ddof=0)),
                        R2_median=float(v.median()),
                        n_folds=int(len(v)),
                    )
                )

    return pd.DataFrame(rows)


# =================================================================
# 14. Statistical testing helpers
# =================================================================
def holm_bonferroni(pvals):
    """Apply Holm-Bonferroni step-down correction."""
    arr = np.asarray(pvals, dtype=float)
    n = len(arr)
    order = np.argsort(arr)
    sorted_p = arr[order]

    corr = sorted_p * np.arange(n, 0, -1)
    for i in range(1, n):
        corr[i] = max(corr[i], corr[i - 1])
    corr = np.minimum(corr, 1.0)

    out = np.empty(n)
    out[order] = corr
    return out


def rank_biserial_from_diff(diff):
    """Compute rank-biserial correlation for paired signed-rank data.

    Returns
    -------
    W_plus, W_minus, r_rb : float
    """
    diff = np.asarray(diff, dtype=float)
    diff = diff[~np.isnan(diff)]
    diff = diff[diff != 0]

    if len(diff) == 0:
        return np.nan, np.nan, np.nan

    ranks = stats.rankdata(np.abs(diff))
    W_plus = float(ranks[diff > 0].sum())
    W_minus = float(ranks[diff < 0].sum())
    denom = W_plus + W_minus
    r_rb = (W_plus - W_minus) / denom if denom > 0 else np.nan
    return W_plus, W_minus, float(r_rb)


def wilcoxon_tests_from_repeat_wide(repeat_wide):
    """One-sided Wilcoxon signed-rank test (H1: TabPFN > baseline)
    on repeat-level mean R2 with Holm-Bonferroni correction."""
    if (not HAS_TABPFN) or ("TabPFN" not in repeat_wide.columns):
        return pd.DataFrame()

    ref = repeat_wide["TabPFN"].to_numpy(dtype=float)
    rows = []
    raw_ps = []

    for mn in BASELINES:
        other = (
            repeat_wide[mn].to_numpy(dtype=float)
            if mn in repeat_wide.columns
            else np.full_like(ref, np.nan)
        )

        valid = ~(np.isnan(ref) | np.isnan(other))
        diff = ref[valid] - other[valid]

        n_valid = int(valid.sum())
        n_nonzero = int(np.sum(diff != 0))
        mean_d = float(np.mean(diff)) if n_valid > 0 else np.nan
        win_pct = float(np.mean(diff > 0) * 100) if n_valid > 0 else np.nan

        if n_valid < MIN_REPEATS_FOR_TEST or n_nonzero == 0:
            rows.append(
                dict(
                    Model=mn,
                    n_repeat=n_valid,
                    n_nonzero=n_nonzero,
                    mean_dR2=mean_d,
                    win_pct=win_pct,
                    W=np.nan,
                    p_raw=np.nan,
                    effect_r=np.nan,
                )
            )
            raw_ps.append(1.0)
            continue

        try:
            _, p = stats.wilcoxon(diff, alternative="greater",
                                  zero_method="wilcox")
            W_plus, W_minus, r_eff = rank_biserial_from_diff(diff)
        except Exception:
            p, W_plus, r_eff = np.nan, np.nan, np.nan

        rows.append(
            dict(
                Model=mn,
                n_repeat=n_valid,
                n_nonzero=n_nonzero,
                mean_dR2=mean_d,
                win_pct=win_pct,
                W=W_plus,
                p_raw=float(p) if not np.isnan(p) else np.nan,
                effect_r=float(r_eff) if not np.isnan(r_eff) else np.nan,
            )
        )
        raw_ps.append(p if not np.isnan(p) else 1.0)

    corrected = holm_bonferroni(np.array(raw_ps, dtype=float))

    for i, row in enumerate(rows):
        row["p_holm"] = float(corrected[i])

        if np.isnan(row["p_raw"]):
            row["sig"] = "N/A"
        elif row["p_holm"] < 0.001:
            row["sig"] = "***"
        elif row["p_holm"] < 0.01:
            row["sig"] = "**"
        elif row["p_holm"] < 0.05:
            row["sig"] = "*"
        else:
            row["sig"] = "n.s."

    return pd.DataFrame(rows)


def compare_feature_sets(repeat_A, repeat_B, mode_label="bo"):
    """Compare Feature Set A vs B for each model using paired
    repeat-level mean R2 (two-sided Wilcoxon + Holm correction)."""
    common_models = [
        m for m in ALL_MODELS
        if (m in repeat_A.columns and m in repeat_B.columns)
    ]

    rows = []
    raw_ps = []

    for mn in common_models:
        merged = pd.merge(
            repeat_A[["repeat", mn]],
            repeat_B[["repeat", mn]],
            on="repeat",
            how="inner",
            suffixes=("_A", "_B"),
        )

        a = merged[f"{mn}_A"].to_numpy(dtype=float)
        b = merged[f"{mn}_B"].to_numpy(dtype=float)

        valid = ~(np.isnan(a) | np.isnan(b))
        a = a[valid]
        b = b[valid]
        diff = a - b

        n_valid = int(len(diff))
        n_nonzero = int(np.sum(diff != 0))

        r2_a = _safe_nanmean(a)
        r2_b = _safe_nanmean(b)
        delta = float(np.mean(diff)) if n_valid > 0 else np.nan

        if n_valid < MIN_REPEATS_FOR_TEST or n_nonzero == 0:
            p, W_plus, r_eff = np.nan, np.nan, np.nan
        else:
            try:
                _, p = stats.wilcoxon(diff, alternative="two-sided",
                                      zero_method="wilcox")
                W_plus, W_minus, r_eff = rank_biserial_from_diff(diff)
            except Exception:
                p, W_plus, r_eff = np.nan, np.nan, np.nan

        rows.append(
            dict(
                Mode=mode_label,
                Model=mn,
                n_repeat=n_valid,
                n_nonzero=n_nonzero,
                R2_A=r2_a,
                R2_B=r2_b,
                Delta_AB=delta,
                W=W_plus,
                p_raw=float(p) if not np.isnan(p) else np.nan,
                effect_r=float(r_eff) if not np.isnan(r_eff) else np.nan,
            )
        )
        raw_ps.append(p if not np.isnan(p) else 1.0)

    corrected = holm_bonferroni(np.array(raw_ps, dtype=float))
    for i, row in enumerate(rows):
        row["p_holm"] = float(corrected[i])

        if np.isnan(row["p_raw"]):
            row["sig"] = "N/A"
        elif row["p_holm"] < 0.001:
            row["sig"] = "***"
        elif row["p_holm"] < 0.01:
            row["sig"] = "**"
        elif row["p_holm"] < 0.05:
            row["sig"] = "*"
        else:
            row["sig"] = "n.s."

    comp_df = pd.DataFrame(rows)
    if not comp_df.empty:
        comp_df = (comp_df.sort_values("R2_A", ascending=False)
                   .reset_index(drop=True))
    return comp_df


# =================================================================
# 15. Reporting helpers
# =================================================================
def make_summary(records_df, mode_filter):
    """Aggregate per-fold records into a summary table.

    Parameters
    ----------
    mode_filter : str
        Either ``"bo"`` or ``"default"``.
    """
    df = records_df.copy()
    mask = (df["mode"] == mode_filter) | (df["model"] == "TabPFN")
    df = df[mask].copy()

    rows = []
    for mn in ALL_MODELS:
        sub = df[df["model"] == mn]
        if len(sub) == 0:
            continue

        rows.append(
            dict(
                Model=mn,
                Mode="zero-shot" if mn == "TabPFN" else mode_filter,
                n_folds_valid=int(sub["R2"].notna().sum()),
                R2_mean=_safe_nanmean(sub["R2"]),
                R2_std=_safe_nanstd(sub["R2"]),
                R2_median=_safe_nanmedian(sub["R2"]),
                RMSE_mean=_safe_nanmean(sub["RMSE"]),
                RMSE_std=_safe_nanstd(sub["RMSE"]),
                MAE_mean=_safe_nanmean(sub["MAE"]),
                MAE_std=_safe_nanstd(sub["MAE"]),
                Time_total_s=_safe_nansum(sub["time_s"]),
                Time_per_fold_s=_safe_nanmean(sub["time_s"]),
            )
        )

    sdf = pd.DataFrame(rows)
    if not sdf.empty:
        sdf = (sdf.sort_values("R2_mean", ascending=False)
               .reset_index(drop=True))
    return sdf


def print_summary(sdf, title):
    """Pretty-print a summary DataFrame to the log."""
    log(f"\n  {title}")
    log(
        f"  {'Rank':<5s} {'Model':<12s} {'Mode':<10s} "
        f"{'R2(mean+/-std)':>18s} {'RMSE(mean+/-std)':>18s} "
        f"{'MAE(mean+/-std)':>18s} {'Time/fold':>10s}"
    )
    log(
        f"  {'-' * 5} {'-' * 12} {'-' * 10} "
        f"{'-' * 18} {'-' * 18} {'-' * 18} {'-' * 10}"
    )

    for i, row in sdf.iterrows():
        tag = " *" if row["Model"] == "TabPFN" else ""
        log(
            f"  {i + 1:<5d} {row['Model']:<12s} {row['Mode']:<10s} "
            f"{row['R2_mean']:>7.4f}+/-{row['R2_std']:<7.4f}  "
            f"{row['RMSE_mean']:>7.4f}+/-{row['RMSE_std']:<7.4f}  "
            f"{row['MAE_mean']:>7.4f}+/-{row['MAE_std']:<7.4f}  "
            f"{row['Time_per_fold_s']:>8.2f}s{tag}"
        )


# =================================================================
# 16. Grand summary builder
# =================================================================
def build_grand_summary(all_summaries):
    """Combine summaries from all feature sets into one table for a
    quick cross-set overview.  Exported as grand_summary.csv."""
    frames = []
    for fs_name, modes in all_summaries.items():
        for mode_label, sdf in modes.items():
            tmp = sdf.copy()
            tmp.insert(0, "FeatureSet", fs_name)
            tmp.insert(3, "Comparison", mode_label.upper())
            frames.append(tmp)

    if frames:
        grand = pd.concat(frames, ignore_index=True)
        grand = grand.sort_values(
            ["FeatureSet", "R2_mean"], ascending=[True, False]
        ).reset_index(drop=True)
        return grand
    return pd.DataFrame()


# =================================================================
# 17. Main entry point
# =================================================================
def main():
    checkpoint = os.environ.get("TABPFN_MODEL_PATH")
    if not checkpoint or not Path(checkpoint).expanduser().is_file():
        raise ValueError("Set TABPFN_MODEL_PATH to an existing TabPFN-2.5 regression checkpoint before running the benchmark.")
    t_global = datetime.now()

    # ---- Configuration banner ----
    banner("EXPERIMENT CONFIGURATION")
    tag_mode = "FAST MODE (sanity check)" if FAST_MODE else "FULL EXPERIMENT"
    log(f"  Mode:       {tag_mode}")
    log(f"  Time:       {t_global.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Python:     {sys.version.split()[0]}")
    log(f"  Data:       {DATA_FILE}")
    log(f"  Output:     {OUT_DIR}")
    log(f"  CV:         {N_REPEATS} repeats x {N_SPLITS}-fold "
        f"= {N_FOLDS} folds")
    log(f"  BO trials:  {BO_TRIALS} (TPE, inner {INNER_SPLITS}-fold)")
    log(f"  LC sizes:   "
        f"{LC_SIZES if RUN_LEARNING_CURVES else 'SKIPPED'}")
    log(
        f"  TabPFN:     "
        f"{'Available | n_est=' + str(TABPFN_N_EST) if HAS_TABPFN else 'NOT AVAILABLE'}"
    )
    log(f"  Seed:       {SEED}")
    log(f"  Baselines:  {BASELINES}")
    log(
        f"  Stats unit: repeat-level mean R2 "
        f"({'N=' + str(N_REPEATS) + ' repeats' if N_REPEATS >= MIN_REPEATS_FOR_TEST else 'FAST_MODE: inferential tests likely N/A'})"
    )

    # ---- Package versions (for reproducibility) ----
    pkg_versions = {
        pkg: _pkg_version(pkg)
        for pkg in [
            "numpy", "pandas", "scipy", "scikit-learn",
            "xgboost", "optuna", "tabpfn",
        ]
    }
    log(f"\n  Package versions:")
    for pkg, ver in pkg_versions.items():
        log(f"    {pkg:<16s} {ver}")

    # ---- Save config ----
    cfg = dict(
        FAST_MODE=FAST_MODE,
        RUN_LEARNING_CURVES=RUN_LEARNING_CURVES,
        SEED=SEED,
        N_SPLITS=N_SPLITS,
        N_REPEATS=N_REPEATS,
        N_FOLDS=N_FOLDS,
        INNER_SPLITS=INNER_SPLITS,
        MIN_REPEATS_FOR_TEST=MIN_REPEATS_FOR_TEST,
        BO_TRIALS=BO_TRIALS,
        LC_SIZES=LC_SIZES,
        TABPFN_N_EST=TABPFN_N_EST,
        TABPFN_MODEL_PATH=os.environ.get("TABPFN_MODEL_PATH"),
        checkpoint_sha256=(hashlib.sha256(Path(os.environ["TABPFN_MODEL_PATH"]).expanduser().read_bytes()).hexdigest()
                           if os.environ.get("TABPFN_MODEL_PATH") else None),
        TARGET=TARGET,
        FEAT_A=FEAT_A,
        FEAT_B=FEAT_B,
        BASELINES=BASELINES,
        DATA_FILE=str(DATA_FILE),
        OUT_DIR=str(OUT_DIR),
        packages=pkg_versions,
        timestamp=t_global.isoformat(),
    )
    (OUT_DIR / "experiment_config.json").write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ---- Load data ----
    banner("DATA LOADING")
    feat_sets, y = load_data()

    # ---- Explicit outer splits ----
    banner("OUTER CV PLAN")
    outer_splits = make_outer_splits(len(y))
    assert len(outer_splits) == N_FOLDS
    log("  Outer splits generated with explicit fold / repeat / split IDs.")
    log("  This enables repeat-level aggregation and paired A-vs-B testing.")
    plan_df = pd.DataFrame(
        [
            dict(
                fold=s["fold"],
                repeat=s["repeat"],
                split=s["split"],
                n_train=len(s["train_idx"]),
                n_test=len(s["test_idx"]),
            )
            for s in outer_splits
        ]
    )
    log(f"\n  Split overview:")
    log(plan_df.head(min(10, len(plan_df))).to_string(index=False))
    plan_df.to_csv(OUT_DIR / "outer_cv_plan.csv", index=False)

    # ---- Per feature-set experiments ----
    all_summaries = {}
    repeat_tables = {}

    for fs_name, (X, col_names) in feat_sets.items():
        tag = f"FS{fs_name}"

        # ===== Nested CV =====
        banner(
            f"NESTED CV -- Feature Set {fs_name} "
            f"({len(col_names)} features x {X.shape[0]} samples)"
        )
        log(f"  Features: {col_names}")
        log(f"  Outer: {N_REPEATS} repeats x {N_SPLITS}-fold "
            f"= {N_FOLDS} folds")
        log(f"  Inner: {INNER_SPLITS}-fold, {BO_TRIALS} TPE trials "
            f"per baseline")
        log(f"  Statistical tests: repeat-level mean R2\n")

        records_df = run_nested_cv(X, y, fs_name, outer_splits)

        # Save long-format raw per-fold records
        records_df.to_csv(OUT_DIR / f"{tag}_all_folds.csv", index=False)

        # Save fold-level wide R2
        fold_r2_wide = build_fold_level_r2(records_df)
        fold_r2_wide.to_csv(OUT_DIR / f"{tag}_per_fold_R2.csv", index=False)

        # Save repeat-level wide R2
        rep_bo = build_repeat_level_r2(records_df, mode_filter="bo")
        rep_def = build_repeat_level_r2(records_df, mode_filter="default")
        rep_bo.to_csv(OUT_DIR / f"{tag}_per_repeat_R2_BO.csv", index=False)
        rep_def.to_csv(OUT_DIR / f"{tag}_per_repeat_R2_Default.csv",
                       index=False)

        repeat_tables[fs_name] = dict(bo=rep_bo, default=rep_def)

        # ===== Summaries =====
        banner(f"RESULTS -- Feature Set {fs_name}")

        sum_bo = make_summary(records_df, mode_filter="bo")
        print_summary(
            sum_bo,
            f"TabPFN(zero-shot) vs Baselines(BO-tuned) [{fs_name}]",
        )
        sum_bo.to_csv(OUT_DIR / f"{tag}_summary_vs_BO.csv", index=False)

        sum_def = make_summary(records_df, mode_filter="default")
        print_summary(
            sum_def,
            f"TabPFN(zero-shot) vs Baselines(Default) [{fs_name}]",
        )
        sum_def.to_csv(OUT_DIR / f"{tag}_summary_vs_Default.csv",
                       index=False)

        all_summaries[fs_name] = dict(bo=sum_bo, default=sum_def)

        # ===== Statistical tests =====
        if HAS_TABPFN:
            banner(f"STATISTICAL TESTS -- Feature Set {fs_name}")

            wdf_bo = wilcoxon_tests_from_repeat_wide(rep_bo)
            log(
                f"\n  Wilcoxon (one-sided, Holm-corrected): "
                f"TabPFN vs BO-tuned [{fs_name}]"
            )
            if wdf_bo.empty:
                log("  No Wilcoxon results available.")
            else:
                log("  Note: tests based on repeat-level mean R2, "
                    "not fold-level R2.")
                log(wdf_bo.to_string(
                    index=False,
                    float_format="{:.4f}".format))
            wdf_bo.to_csv(OUT_DIR / f"{tag}_wilcoxon_vs_BO.csv",
                          index=False)

            wdf_def = wilcoxon_tests_from_repeat_wide(rep_def)
            log(
                f"\n  Wilcoxon (one-sided, Holm-corrected): "
                f"TabPFN vs Default [{fs_name}]"
            )
            if wdf_def.empty:
                log("  No Wilcoxon results available.")
            else:
                log("  Note: tests based on repeat-level mean R2, "
                    "not fold-level R2.")
                log(wdf_def.to_string(
                    index=False,
                    float_format="{:.4f}".format))
            wdf_def.to_csv(OUT_DIR / f"{tag}_wilcoxon_vs_Default.csv",
                           index=False)

            if N_REPEATS < MIN_REPEATS_FOR_TEST:
                log(
                    f"\n  [WARNING] N_REPEATS={N_REPEATS} < "
                    f"{MIN_REPEATS_FOR_TEST}; inferential tests are "
                    f"expected to be N/A in FAST_MODE."
                )

        # ===== Learning curves =====
        if RUN_LEARNING_CURVES:
            banner(f"LEARNING CURVES -- Feature Set {fs_name} "
                   f"(tabular only)")
            log(f"  Sizes:  {LC_SIZES}")
            log(f"  Models: {LC_MODELS}\n")

            lc_raw = run_learning_curves(X, y, fs_name, outer_splits)
            lc_sum = lc_summary(lc_raw)

            if not lc_sum.empty:
                pivot = lc_sum.pivot_table(
                    index="n_train", columns="Model", values="R2_mean",
                )
                log(f"\n  Learning Curve R2 (mean over available folds):")
                log(pivot.to_string(float_format="{:.4f}".format))

                log(f"\n  Best model per training size:")
                for n in LC_SIZES:
                    sub = lc_sum[lc_sum["n_train"] == n]
                    if len(sub) > 0:
                        best = sub.loc[sub["R2_mean"].idxmax()]
                        mark = (" [TabPFN]"
                                if best["Model"] == "TabPFN" else "")
                        log(
                            f"    n={n:>3d}: {best['Model']:<10s} "
                            f"R2={best['R2_mean']:.4f}{mark}"
                        )
            else:
                log("  Learning-curve summary is empty.")

            lc_raw.to_csv(
                OUT_DIR / f"{tag}_learning_curve_raw.csv", index=False,
            )
            lc_sum.to_csv(
                OUT_DIR / f"{tag}_learning_curve_summary.csv", index=False,
            )
        else:
            log(f"\n  Learning curves skipped for Feature Set {fs_name}.")

    # ---- Cross feature-set comparison ----
    if set(["A", "B"]).issubset(repeat_tables.keys()):
        banner("COMPARISON: Feature Set A vs B")

        # Primary comparison: BO mode
        comp_bo = compare_feature_sets(
            repeat_tables["A"]["bo"],
            repeat_tables["B"]["bo"],
            mode_label="bo",
        )
        if not comp_bo.empty:
            log("\n  BO-tuned baselines + TabPFN(zero-shot) -- "
                "paired repeat-level comparison")
            log("  Note: two-sided Wilcoxon on repeat-level mean R2 "
                "(A vs B).")
            show_cols = ["Model", "R2_A", "R2_B", "Delta_AB",
                         "p_holm", "sig"]
            log(comp_bo[show_cols].to_string(
                index=False, float_format="{:.4f}".format))

            comp_bo.to_csv(OUT_DIR / "comparison_A_vs_B.csv", index=False)
            comp_bo.to_csv(OUT_DIR / "comparison_A_vs_B_BO.csv",
                           index=False)

            log(f"\n  Overall ranking (avg R2 across A & B, "
                f"BO-tuned baselines):")
            comp_bo["R2_avg"] = (comp_bo["R2_A"] + comp_bo["R2_B"]) / 2
            comp_bo_rank = (comp_bo.sort_values("R2_avg", ascending=False)
                            .reset_index(drop=True))
            for i, row in comp_bo_rank.iterrows():
                tag_best = " << BEST" if i == 0 else ""
                log(
                    f"    #{i + 1} {row['Model']:<12s} "
                    f"avg_R2 = {row['R2_avg']:.4f}{tag_best}"
                )
        else:
            log("  BO comparison A vs B is empty.")

        # Default-mode comparison
        comp_def = compare_feature_sets(
            repeat_tables["A"]["default"],
            repeat_tables["B"]["default"],
            mode_label="default",
        )
        comp_def.to_csv(OUT_DIR / "comparison_A_vs_B_Default.csv",
                        index=False)
        log("\n  Saved default-mode A vs B comparison: "
            "comparison_A_vs_B_Default.csv")

    # ---- Grand summary ----
    banner("GRAND SUMMARY")
    grand = build_grand_summary(all_summaries)
    if not grand.empty:
        grand.to_csv(OUT_DIR / "grand_summary.csv", index=False)
        log("  Combined cross-set summary saved: grand_summary.csv")

        for fs_name in ["A", "B"]:
            sub = grand[
                (grand["FeatureSet"] == fs_name)
                & (grand["Comparison"] == "BO")
            ]
            if not sub.empty:
                best_row = sub.loc[sub["R2_mean"].idxmax()]
                log(
                    f"\n  Feature Set {fs_name} -- best model (vs BO): "
                    f"{best_row['Model']}  "
                    f"R2 = {best_row['R2_mean']:.4f} +/- "
                    f"{best_row['R2_std']:.4f}"
                )
    else:
        log("  Grand summary is empty.")

    # ---- Output file list ----
    banner("OUTPUT FILES")
    log(f"  Directory: {OUT_DIR}\n")
    fc = 0
    for f in sorted(OUT_DIR.iterdir()):
        if f.is_file():
            sz = f.stat().st_size
            sz_str = (f"{sz / 1024:.1f} KB" if sz > 1024
                      else f"{sz} B")
            log(f"    {f.name:<60s} {sz_str:>10s}")
            fc += 1
    log(f"\n  Total: {fc} files")

    # ---- Done ----
    t_end = datetime.now()
    elapsed = (t_end - t_global).total_seconds()

    banner("EXPERIMENT COMPLETE")
    mode_tag = "FAST_MODE" if FAST_MODE else "FULL"
    log(f"  Mode:     {mode_tag}")
    log(f"  Duration: {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    log(f"  Output:   {OUT_DIR}")
    log(f"  End time: {t_end.strftime('%Y-%m-%d %H:%M:%S')}")

    if FAST_MODE:
        log("\n  Fast test complete. Verify outputs, then set "
            "FAST_MODE = False for the full experiment.")

    # Save full log
    log_path = OUT_DIR / f"experiment_log_{mode_tag}.txt"
    log_path.write_text(_LOG.getvalue(), encoding="utf-8")
    print(f"\n>>> Full log saved: {log_path}")


# =================================================================
# Entry point
# =================================================================
if __name__ == "__main__":
    main()
