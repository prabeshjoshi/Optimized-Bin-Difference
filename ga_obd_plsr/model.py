"""
model.py
--------
High-level sklearn-compatible estimator wrapping the GA-OBD feature
selection pipeline.

Usage
-----
>>> from ga_obd_plsr import GAOBD
>>> model = GAOBD(pop_size=50, generations=100, P=8, n_components=6)
>>> model.fit(X_train, y_train, wl)
>>> X_feats = model.transform(X_train)
>>> print(model.summary())
>>> model.plot_history()
>>> model.plot_spectra(X_train)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from typing import Optional, List

from .ga import run_ga
from .features import (
    make_feature_matrix_from_pairs,
    chromosome_to_dataframe,
    build_feature_names,
)
from .metrics import pls_cv_rmse_robust, pls_cv_r2_robust, pls_cv_rpd_robust, compute_all_metrics
from .plotting import plot_ga_history, plot_spectra_with_features


class GAOBD(BaseEstimator, TransformerMixin):
    """
    Genetic Algorithm — Optical Band Difference feature selector for PLSR.

    Learns a set of spectral contrast features (OBDs) that minimise
    cross-validated RMSE of a PLS regression model. Implements the
    ``fit`` / ``transform`` sklearn interface.

    Parameters
    ----------
    pop_size : int
        GA population size. Default 40.
    generations : int
        Number of GA generations. Default 100.
    P : int
        Number of OBD genes per chromosome. Default 8.
    max_width : int
        Maximum spectral region width (channels). Default 20.
    cx_prob : float
        Crossover probability. Default 0.7.
    mut_prob : float
        Per-gene mutation probability. Default 0.25.
    n_components : int
        Maximum PLS latent variables used during fitness evaluation. Default 6.
    cv : int
        Number of K-Fold CV folds. Default 5.
    penalty_per_feat : float
        Sparsity penalty per active gene added to RMSE. Default 0.01.
    random_state : int or None
        Seed for reproducibility. Default 0.
    verbose : bool
        Print GA progress. Default True.
    """

    def __init__(
        self,
        pop_size: int = 40,
        generations: int = 100,
        P: int = 8,
        max_width: int = 20,
        cx_prob: float = 0.7,
        mut_prob: float = 0.25,
        n_components: int = 6,
        cv: int = 5,
        penalty_per_feat: float = 0.01,
        random_state: Optional[int] = 0,
        verbose: bool = True,
    ):
        self.pop_size = pop_size
        self.generations = generations
        self.P = P
        self.max_width = max_width
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.n_components = n_components
        self.cv = cv
        self.penalty_per_feat = penalty_per_feat
        self.random_state = random_state
        self.verbose = verbose

        # Set after fit
        self.best_chrom_: Optional[np.ndarray] = None
        self.best_fitness_: Optional[float] = None
        self.history_: Optional[List[float]] = None
        self.wl_: Optional[np.ndarray] = None
        self.n_wl_: Optional[int] = None
        self.feature_names_: Optional[List[str]] = None
        self.selected_df_: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, wl=None):
        """
        Run the GA to select OBD features on (X, y).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_wl)
            Spectral data. Should be pre-processed (smoothed, derivative,
            standardised, etc.) before calling fit.
        y : np.ndarray, shape (n_samples,)
            Target variable.
        wl : array-like, optional
            Wavelength axis. If not provided, integer indices are used.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n_wl = X.shape[1]
        self.n_wl_ = n_wl
        self.wl_ = np.asarray(wl) if wl is not None else np.arange(n_wl)

        self.best_chrom_, self.best_fitness_, self.history_ = run_ga(
            X, y, n_wl,
            pop_size=self.pop_size,
            generations=self.generations,
            P=self.P,
            max_width=self.max_width,
            cx_prob=self.cx_prob,
            mut_prob=self.mut_prob,
            n_components=self.n_components,
            cv=self.cv,
            penalty_per_feat=self.penalty_per_feat,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        _, active_idx = make_feature_matrix_from_pairs(X, self.best_chrom_)
        self.feature_names_ = build_feature_names(self.best_chrom_, self.wl_)
        self.selected_df_ = chromosome_to_dataframe(
            self.best_chrom_, self.wl_, self.n_wl_
        )
        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the selected OBD features to new spectral data.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_wl)

        Returns
        -------
        np.ndarray, shape (n_samples, n_active_features)
        """
        self._check_is_fitted()
        X_feats, _ = make_feature_matrix_from_pairs(X, self.best_chrom_)
        return X_feats

    def fit_transform(self, X: np.ndarray, y: np.ndarray, wl=None) -> np.ndarray:
        """Fit then transform in one call."""
        return self.fit(X, y, wl).transform(X)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def score_cv(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Compute cross-validated metrics on the selected features.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_wl)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        dict with keys: RMSE, R2, RPD
        """
        self._check_is_fitted()
        X_feats = self.transform(X)
        return {
            "RMSE": pls_cv_rmse_robust(X_feats, y, self.n_components, self.cv),
            "R2":   pls_cv_r2_robust(X_feats, y, self.n_components, self.cv),
            "RPD":  pls_cv_rpd_robust(X_feats, y, self.n_components, self.cv),
        }

    def predict(self, X_train: np.ndarray, y_train: np.ndarray,
                X_test: np.ndarray) -> np.ndarray:
        """
        Fit a PLS model on the training OBD features and predict on test data.

        Parameters
        ----------
        X_train, y_train : training data (pre-processed spectra + target)
        X_test : np.ndarray — test spectra (pre-processed)

        Returns
        -------
        np.ndarray, shape (n_test_samples,)
        """
        self._check_is_fitted()
        X_tr_feats = self.transform(X_train)
        X_te_feats = self.transform(X_test)
        pls = PLSRegression(n_components=self.n_components)
        pls.fit(X_tr_feats, y_train)
        return pls.predict(X_te_feats).ravel()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """
        Return a human-readable summary of the best solution.

        Returns
        -------
        str
        """
        self._check_is_fitted()
        n_active = int(np.sum(self.best_chrom_[:, 4]))
        lines = [
            "=" * 55,
            "  GA-OBD-PLSR Result Summary",
            "=" * 55,
            f"  Active features : {n_active}",
            f"  Best fitness    : {self.best_fitness_:.4f}",
            "",
            "  Selected OBD Features:",
        ]
        for name in self.feature_names_:
            lines.append(f"    • {name}")
        lines.append("=" * 55)
        return "\n".join(lines)

    def get_selected_features_df(self) -> pd.DataFrame:
        """
        Return the selected genes as a DataFrame.

        Columns: gene_index, start1_idx, width1, start2_idx, width2,
                 wl_start1_nm, wl_end1_nm, wl_start2_nm, wl_end2_nm
        """
        self._check_is_fitted()
        return self.selected_df_.copy()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_history(self, **kwargs):
        """Plot GA fitness history. Returns a Figure."""
        self._check_is_fitted()
        return plot_ga_history(self.history_, **kwargs)

    def plot_spectra(self, X: np.ndarray, **kwargs):
        """Plot spectra with selected feature regions shaded. Returns a Figure."""
        self._check_is_fitted()
        return plot_spectra_with_features(X, self.wl_, self.best_chrom_,
                                          n_wl=self.n_wl_, **kwargs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_is_fitted(self):
        if self.best_chrom_ is None:
            raise RuntimeError("Call fit() before using this method.")
