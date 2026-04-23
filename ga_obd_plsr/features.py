"""
features.py
-----------
Utilities for building Optical Band Difference (OBD) feature matrices
from GA chromosomes.

A chromosome is an (P, 5) integer array where each row encodes one OBD gene:
    [start1, width1, start2, width2, active]

The feature value for a gene is:
    mean(X[:, start1:start1+width1]) - mean(X[:, start2:start2+width2])
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional


def make_feature_matrix_from_pairs(
    X: np.ndarray,
    chromosome: np.ndarray,
) -> Tuple[np.ndarray, List[int]]:
    """
    Build an OBD feature matrix from a GA chromosome.

    Each active gene produces one feature column:
        feature = mean(region_1) - mean(region_2)

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_wavelengths)
        Spectral data matrix.
    chromosome : np.ndarray, shape (P, 5)
        GA chromosome. Each row is [start1, width1, start2, width2, active].

    Returns
    -------
    X_feats : np.ndarray, shape (n_samples, n_active_genes)
        OBD feature matrix. Empty array with 0 columns if no active genes.
    active_idx : list of int
        Row indices of the chromosome that are active (active == 1).
    """
    n_samples = X.shape[0]
    features = []
    active_idx = []

    for i, gene in enumerate(chromosome):
        s1, w1, s2, w2, active = gene.astype(int)
        if active:
            s1 = max(0, min(s1, X.shape[1] - 1))
            w1 = max(1, min(w1, X.shape[1] - s1))
            s2 = max(0, min(s2, X.shape[1] - 1))
            w2 = max(1, min(w2, X.shape[1] - s2))

            mean1 = X[:, s1:s1 + w1].mean(axis=1)
            mean2 = X[:, s2:s2 + w2].mean(axis=1)
            features.append((mean1 - mean2).reshape(-1, 1))
            active_idx.append(i)

    if len(features) == 0:
        return np.zeros((n_samples, 0)), active_idx

    return np.hstack(features), active_idx


def chrom_to_wl_ranges(
    chromosome: np.ndarray,
    wl: np.ndarray,
    n_wl: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Extract the wavelength values covered by each active gene.

    Parameters
    ----------
    chromosome : np.ndarray, shape (P, 5)
        GA chromosome.
    wl : array-like
        Wavelength axis array, length n_wavelengths.
    n_wl : int, optional
        Total number of wavelengths. Inferred from ``wl`` if not given.

    Returns
    -------
    selected_wl : list of np.ndarray
        One entry per active *region* (two per active gene).
        Each entry is the wavelength values covered by that region.
    """
    wl = np.asarray(wl)
    if n_wl is None:
        n_wl = len(wl)

    selected_wl = []
    for s1, w1, s2, w2, active in chromosome.astype(int):
        if active:
            selected_wl.append(wl[max(0, s1):min(n_wl, s1 + w1)])
            selected_wl.append(wl[max(0, s2):min(n_wl, s2 + w2)])
    return selected_wl


def chrom_to_variable_indices(
    chromosome: np.ndarray,
    n_wl: int,
) -> List[int]:
    """
    Return the sorted list of all wavelength indices selected by a chromosome.

    Parameters
    ----------
    chromosome : np.ndarray, shape (P, 5)
    n_wl : int
        Total number of wavelengths.

    Returns
    -------
    list of int
    """
    selected = set()
    for s1, w1, s2, w2, active in chromosome.astype(int):
        if active:
            selected.update(range(max(0, s1), min(n_wl, s1 + w1)))
            selected.update(range(max(0, s2), min(n_wl, s2 + w2)))
    return sorted(selected)


def chromosome_to_dataframe(
    chromosome: np.ndarray,
    wl: np.ndarray,
    n_wl: Optional[int] = None,
) -> pd.DataFrame:
    """
    Summarise the active genes of a chromosome as a DataFrame.

    Parameters
    ----------
    chromosome : np.ndarray, shape (P, 5)
    wl : array-like
    n_wl : int, optional

    Returns
    -------
    pd.DataFrame with columns:
        gene_index, start1_idx, width1, start2_idx, width2,
        wl_start1_nm, wl_end1_nm, wl_start2_nm, wl_end2_nm
    """
    wl = np.asarray(wl)
    if n_wl is None:
        n_wl = len(wl)

    rows = []
    for i, (s1, w1, s2, w2, active) in enumerate(chromosome.astype(int)):
        if active:
            rows.append({
                "gene_index": i,
                "start1_idx": s1,
                "width1": w1,
                "start2_idx": s2,
                "width2": w2,
                "wl_start1_nm": float(wl[s1]),
                "wl_end1_nm": float(wl[min(s1 + w1 - 1, n_wl - 1)]),
                "wl_start2_nm": float(wl[s2]),
                "wl_end2_nm": float(wl[min(s2 + w2 - 1, n_wl - 1)]),
            })
    return pd.DataFrame(rows)


def build_feature_names(chromosome: np.ndarray, wl: np.ndarray) -> List[str]:
    """
    Generate human-readable OBD feature names from an active chromosome.

    Example output: ``"[850 nm : 870 nm] - [920 nm : 940 nm]"``

    Parameters
    ----------
    chromosome : np.ndarray, shape (P, 5)
    wl : array-like

    Returns
    -------
    list of str, one per active gene
    """
    wl = np.asarray(wl)
    names = []
    for s1, w1, s2, w2, active in chromosome.astype(int):
        if active:
            names.append(
                f"[{wl[s1]:.0f} nm : {wl[s1 + w1 - 1]:.0f} nm]"
                f" - "
                f"[{wl[s2]:.0f} nm : {wl[s2 + w2 - 1]:.0f} nm]"
            )
    return names
