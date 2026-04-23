"""
preprocessing.py
----------------
Common NIR spectral preprocessing utilities.

These are thin wrappers / helpers to make the most frequently used
transformations easy to apply and document.
"""

import numpy as np
from typing import Optional, Tuple


def mean_center_scale(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean-centre and unit-variance scale a spectral matrix.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_wl)
    mean : np.ndarray, optional
        Pre-computed column mean (e.g. from training set). If None, computed
        from ``X``.
    std : np.ndarray, optional
        Pre-computed column std. If None, computed from ``X``.

    Returns
    -------
    X_scaled : np.ndarray
    mean : np.ndarray
    std : np.ndarray
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)  # avoid division by zero
    return (X - mean) / std, mean, std


def savitzky_golay(
    X: np.ndarray,
    window_length: int = 7,
    polyorder: int = 2,
    deriv: int = 1,
) -> np.ndarray:
    """
    Apply a Savitzky-Golay filter (optionally with derivative) to spectra.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_wl)
    window_length : int — must be odd. Default 7.
    polyorder : int — polynomial order, must be < window_length. Default 2.
    deriv : int — derivative order (0 = smoothing only). Default 1.

    Returns
    -------
    np.ndarray, shape (n_samples, n_wl)
    """
    from scipy.signal import savgol_filter
    return savgol_filter(X, window_length=window_length,
                         polyorder=polyorder, deriv=deriv)


def select_wavelength_range(
    X: np.ndarray,
    wl: np.ndarray,
    wl_min: float,
    wl_max: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subset spectra to a wavelength range.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_wl)
    wl : np.ndarray — wavelength axis
    wl_min, wl_max : float — inclusive bounds (nm)

    Returns
    -------
    X_sel : np.ndarray
    wl_sel : np.ndarray
    """
    wl = np.asarray(wl)
    mask = (wl >= wl_min) & (wl <= wl_max)
    return X[:, mask], wl[mask]
