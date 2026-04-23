"""
plotting.py
-----------
Visualisation utilities for GA results and spectral feature analysis.

All functions return a ``matplotlib.figure.Figure`` object so they can
be used both interactively and in scripts (just call ``fig.savefig(...)``).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional


def plot_ga_history(
    history: List[float],
    title: str = "GA Fitness Over Generations",
    figsize: tuple = (7, 3),
) -> plt.Figure:
    """
    Plot the best fitness value per generation.

    Parameters
    ----------
    history : list of float
        Best fitness value recorded at each generation.
    title : str
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(1, len(history) + 1), history, lw=1.5, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness (RMSE + penalty)")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_spectra_with_features(
    X: np.ndarray,
    wl,
    best_chrom: np.ndarray,
    n_wl: Optional[int] = None,
    title: str = "Spectra with Selected OBD Features",
    ylabel: str = "Absorbance",
    figsize: tuple = (10, 6),
    spectra_alpha: float = 0.25,
    cmap: str = "viridis",
) -> plt.Figure:
    """
    Plot all spectra and shade the wavelength regions selected by the chromosome.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_wl)
        Raw (or pre-processed) spectral data to display.
    wl : array-like
        Wavelength axis, length n_wl.
    best_chrom : np.ndarray, shape (P, 5)
        Best GA chromosome.
    n_wl : int, optional
        Number of wavelengths. Inferred from ``wl`` if not given.
    title : str
    ylabel : str
    figsize : tuple
    spectra_alpha : float — transparency for individual spectra lines
    cmap : str — matplotlib colourmap for shaded feature regions

    Returns
    -------
    matplotlib.figure.Figure
    """
    wl = np.asarray(wl)
    if n_wl is None:
        n_wl = len(wl)

    active_genes = best_chrom[best_chrom[:, 4] == 1]
    n_active = len(active_genes)
    colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, max(n_active, 1)))

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(X.shape[0]):
        ax.plot(wl, X[i, :], color="grey", alpha=spectra_alpha, lw=0.7)

    for i, (s1, w1, s2, w2, _) in enumerate(active_genes.astype(int)):
        color = colors[i]
        e1 = min(s1 + w1, n_wl) - 1
        e2 = min(s2 + w2, n_wl) - 1
        ax.axvspan(wl[s1], wl[e1], color=color, alpha=0.3,
                   label=f"Feature {i + 1} R1")
        ax.axvspan(wl[s2], wl[e2], color=color, alpha=0.3,
                   label=f"Feature {i + 1} R2")

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_wavelength_frequency(
    freq_df,
    title: str = "Wavelength Selection Frequency Across GA Runs",
    figsize: tuple = (9, 4),
    color: str = "steelblue",
) -> plt.Figure:
    """
    Plot the wavelength selection frequency from multiple GA runs.

    Parameters
    ----------
    freq_df : pd.DataFrame
        Output of ``run_multiple_ga``. Must have columns
        ``wavelength_nm`` and ``selection_count``.
    title : str
    figsize : tuple
    color : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(freq_df["wavelength_nm"], freq_df["selection_count"],
            lw=1.5, color=color)
    ax.fill_between(freq_df["wavelength_nm"], freq_df["selection_count"],
                    alpha=0.2, color=color)
    ax.set_title(title)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Selection Count")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    return fig


def plot_predicted_vs_actual(
    y_true,
    y_pred,
    title: str = "Predicted vs Actual",
    units: str = "",
    figsize: tuple = (5, 5),
) -> plt.Figure:
    """
    Scatter plot of predicted vs actual values with a 1:1 reference line.

    Parameters
    ----------
    y_true, y_pred : array-like
    title : str
    units : str — label suffix for axes, e.g. "% w/w"
    figsize : tuple

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    pad = (lims[1] - lims[0]) * 0.05
    lims = [lims[0] - pad, lims[1] + pad]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidths=0.5)
    ax.plot(lims, lims, "r--", lw=1, label="1:1")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"Actual{' (' + units + ')' if units else ''}")
    ax.set_ylabel(f"Predicted{' (' + units + ')' if units else ''}")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig
