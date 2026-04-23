"""
metrics.py
----------
Cross-validated PLS regression metrics, all implemented with robustness
fixes for near-constant features and SVD instability.

Every ``pls_cv_*`` function follows the same call signature::

    metric_value = pls_cv_*(X_feats, y, n_components=5, cv=5)

Higher-level helper
-------------------
``compute_all_metrics(y_true, y_pred)`` computes a full metric dictionary
from arrays of actual and predicted values.
"""

import math
import warnings
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _pls_cv_fold_predict(X_feats, y, n_components, cv):
    """
    Yield (y_test, y_pred) pairs from K-Fold CV with robustness guards.

    Filters near-constant columns per fold and clips ``n_components`` to
    valid bounds. Raises ``StopIteration`` early on catastrophic failure.
    """
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    for train_idx, test_idx in kf.split(X_feats):
        Xtr, Xte = X_feats[train_idx], X_feats[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        keep = Xtr.std(axis=0) > 1e-8
        if not np.any(keep):
            return  # signal caller to return worst-case

        Xtr2, Xte2 = Xtr[:, keep], Xte[:, keep]
        n_comp = max(1, min(n_components, Xtr2.shape[1], Xtr2.shape[0] - 1))

        pls = PLSRegression(n_components=n_comp)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pls.fit(Xtr2, ytr)
                yp = pls.predict(Xte2).ravel()
        except Exception:
            return

        yield yte, yp


# ---------------------------------------------------------------------------
# Per-metric CV functions
# ---------------------------------------------------------------------------

def pls_cv_rmse_robust(X_feats: np.ndarray, y: np.ndarray,
                       n_components: int = 5, cv: int = 5) -> float:
    """Cross-validated RMSE (lower is better). Returns ``np.inf`` on failure."""
    if X_feats.shape[1] == 0:
        return np.inf
    vals = [math.sqrt(mean_squared_error(yte, yp))
            for yte, yp in _pls_cv_fold_predict(X_feats, y, n_components, cv)]
    return np.mean(vals) if vals else np.inf


def pls_cv_r2_robust(X_feats: np.ndarray, y: np.ndarray,
                     n_components: int = 5, cv: int = 5) -> float:
    """Cross-validated R² (higher is better). Returns ``-np.inf`` on failure."""
    if X_feats.shape[1] == 0:
        return -np.inf
    vals = []
    for yte, yp in _pls_cv_fold_predict(X_feats, y, n_components, cv):
        ss_res = np.sum((yte - yp) ** 2)
        ss_tot = np.sum((yte - np.mean(yte)) ** 2)
        if ss_tot < 1e-12:
            return -np.inf
        vals.append(1 - ss_res / ss_tot)
    return np.mean(vals) if vals else -np.inf


def pls_cv_rpd_robust(X_feats: np.ndarray, y: np.ndarray,
                      n_components: int = 5, cv: int = 5) -> float:
    """Cross-validated RPD = std(y) / RMSE (higher is better)."""
    if X_feats.shape[1] == 0:
        return -np.inf
    vals = []
    for yte, yp in _pls_cv_fold_predict(X_feats, y, n_components, cv):
        rmse = np.sqrt(mean_squared_error(yte, yp))
        std_y = np.std(yte)
        if rmse < 1e-12 or std_y < 1e-12:
            return -np.inf
        vals.append(std_y / rmse)
    return np.mean(vals) if vals else -np.inf


def pls_cv_mae_robust(X_feats: np.ndarray, y: np.ndarray,
                      n_components: int = 5, cv: int = 5) -> float:
    """Cross-validated MAE (lower is better)."""
    if X_feats.shape[1] == 0:
        return np.inf
    vals = [np.mean(np.abs(yte - yp))
            for yte, yp in _pls_cv_fold_predict(X_feats, y, n_components, cv)]
    return np.mean(vals) if vals else np.inf


def pls_cv_bias_robust(X_feats: np.ndarray, y: np.ndarray,
                       n_components: int = 5, cv: int = 5) -> float:
    """Cross-validated absolute bias (lower is better)."""
    if X_feats.shape[1] == 0:
        return np.inf
    vals = [abs(np.mean(yp - yte))
            for yte, yp in _pls_cv_fold_predict(X_feats, y, n_components, cv)]
    return np.mean(vals) if vals else np.inf


def pls_cv_sep_robust(X_feats: np.ndarray, y: np.ndarray,
                      n_components: int = 5, cv: int = 5) -> float:
    """Cross-validated Standard Error of Prediction (bias-corrected RMSE)."""
    if X_feats.shape[1] == 0:
        return np.inf
    vals = []
    for yte, yp in _pls_cv_fold_predict(X_feats, y, n_components, cv):
        bias = np.mean(yp - yte)
        sep = np.sqrt(np.mean((yte - yp - bias) ** 2))
        vals.append(sep)
    return np.mean(vals) if vals else np.inf


def pls_cv_rpiq_robust(X_feats: np.ndarray, y: np.ndarray,
                       n_components: int = 5, cv: int = 5) -> float:
    """Cross-validated RPIQ = IQR(y) / RMSE (higher is better)."""
    if X_feats.shape[1] == 0:
        return -np.inf
    vals = []
    for yte, yp in _pls_cv_fold_predict(X_feats, y, n_components, cv):
        rmse = np.sqrt(mean_squared_error(yte, yp))
        iqr = np.percentile(yte, 75) - np.percentile(yte, 25)
        if rmse < 1e-12 or iqr < 1e-12:
            return -np.inf
        vals.append(iqr / rmse)
    return np.mean(vals) if vals else -np.inf


def pls_cv_ccc_robust(X_feats: np.ndarray, y: np.ndarray,
                      n_components: int = 5, cv: int = 5) -> float:
    """Cross-validated Concordance Correlation Coefficient (higher is better)."""
    if X_feats.shape[1] == 0:
        return -np.inf
    vals = []
    for yte, yp in _pls_cv_fold_predict(X_feats, y, n_components, cv):
        mean_y, mean_p = np.mean(yte), np.mean(yp)
        var_y, var_p = np.var(yte), np.var(yp)
        cov = np.mean((yte - mean_y) * (yp - mean_p))
        denom = var_y + var_p + (mean_y - mean_p) ** 2
        if abs(denom) < 1e-12:
            return -np.inf
        vals.append(2 * cov / denom)
    return np.mean(vals) if vals else -np.inf


def pls_cv_rrmse_robust(X_feats: np.ndarray, y: np.ndarray,
                        n_components: int = 5, cv: int = 5) -> float:
    """Cross-validated Relative RMSE as % of mean(y) (lower is better)."""
    if X_feats.shape[1] == 0:
        return np.inf
    vals = []
    for yte, yp in _pls_cv_fold_predict(X_feats, y, n_components, cv):
        rmse = np.sqrt(mean_squared_error(yte, yp))
        mean_y = np.mean(yte)
        if abs(mean_y) < 1e-12:
            return np.inf
        vals.append((rmse / mean_y) * 100)
    return np.mean(vals) if vals else np.inf


def pls_cv_slope_robust(X_feats: np.ndarray, y: np.ndarray,
                        n_components: int = 5, cv: int = 5) -> float:
    """Cross-validated regression slope of predicted vs. actual."""
    if X_feats.shape[1] == 0:
        return -np.inf
    vals = []
    for yte, yp in _pls_cv_fold_predict(X_feats, y, n_components, cv):
        try:
            slope, _ = np.polyfit(yte, yp, 1)
            vals.append(slope)
        except Exception:
            return -np.inf
    return np.mean(vals) if vals else -np.inf


# ---------------------------------------------------------------------------
# Convenience: compute all metrics at once from predictions
# ---------------------------------------------------------------------------

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute a comprehensive set of regression metrics from predictions.

    Parameters
    ----------
    y_true : array-like, shape (n,)
    y_pred : array-like, shape (n,)

    Returns
    -------
    dict with keys: R2, RMSE, MAE, Bias, SEP, RPD, RPIQ, Slope, Intercept, CCC
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    bias = np.mean(y_pred - y_true)
    sep = np.sqrt(np.mean((y_true - (y_pred - bias)) ** 2))
    std_ref = np.std(y_true, ddof=1)
    iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
    slope, intercept = np.polyfit(y_true, y_pred, 1)

    mean_t, mean_p = np.mean(y_true), np.mean(y_pred)
    var_t, var_p = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mean_t) * (y_pred - mean_p))
    denom = var_t + var_p + (mean_t - mean_p) ** 2

    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": rmse,
        "MAE": mean_absolute_error(y_true, y_pred),
        "Bias": bias,
        "SEP": sep,
        "RPD": std_ref / rmse if rmse > 1e-12 else np.nan,
        "RPIQ": iqr / rmse if rmse > 1e-12 else np.nan,
        "Slope": slope,
        "Intercept": intercept,
        "CCC": (2 * cov / denom) if abs(denom) > 1e-12 else np.nan,
    }
