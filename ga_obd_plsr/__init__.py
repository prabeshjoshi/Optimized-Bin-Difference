"""
GA-OBD-PLSR: Genetic Algorithm Optical Band Difference Feature Selection for PLS Regression
============================================================================================

A Python package for NIR spectral feature selection using a Genetic Algorithm that evolves
Optical Band Difference (OBD) features — contrast features defined as the difference
between mean absorbance of two spectral regions — optimised for PLS regression.

Quickstart
----------
>>> from ga_obd_plsr import GAOBD
>>> model = GAOBD(pop_size=50, generations=100, P=8)
>>> model.fit(X, y, wl)
>>> features = model.transform(X)
>>> print(model.summary())
"""

from .model import GAOBD
from .ga import run_ga, run_multiple_ga
from .features import make_feature_matrix_from_pairs, chrom_to_wl_ranges
from .metrics import (
    pls_cv_rmse_robust,
    pls_cv_r2_robust,
    pls_cv_rpd_robust,
    pls_cv_mae_robust,
    pls_cv_rpiq_robust,
    pls_cv_ccc_robust,
    pls_cv_sep_robust,
    pls_cv_bias_robust,
    pls_cv_rrmse_robust,
    compute_all_metrics,
)
from .plotting import plot_ga_history, plot_spectra_with_features, plot_wavelength_frequency

__version__ = "0.1.0"
__author__ = "GA-OBD-PLSR Contributors"

__all__ = [
    "GAOBD",
    "run_ga",
    "run_multiple_ga",
    "make_feature_matrix_from_pairs",
    "chrom_to_wl_ranges",
    "pls_cv_rmse_robust",
    "pls_cv_r2_robust",
    "pls_cv_rpd_robust",
    "pls_cv_mae_robust",
    "pls_cv_rpiq_robust",
    "pls_cv_ccc_robust",
    "pls_cv_sep_robust",
    "pls_cv_bias_robust",
    "pls_cv_rrmse_robust",
    "compute_all_metrics",
    "plot_ga_history",
    "plot_spectra_with_features",
    "plot_wavelength_frequency",
]
