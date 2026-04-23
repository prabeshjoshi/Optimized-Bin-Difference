"""
ga.py
-----
Core Genetic Algorithm (GA) for OBD feature selection.

The GA evolves a population of chromosomes, where each chromosome
defines a set of Optical Band Difference (OBD) features.

Chromosome encoding
-------------------
Each chromosome is an (P, 5) integer array. Each row (gene) encodes:
    [start1, width1, start2, width2, active]

    - start1, start2 : start index of each spectral region
    - width1, width2 : number of wavelengths in each region
    - active         : 0 or 1 — whether this gene contributes a feature

Fitness
-------
fitness = CV_RMSE(OBD_features) + penalty_per_feat * n_active_features

The sparsity penalty encourages the GA to use fewer features.
"""

import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict

from .features import make_feature_matrix_from_pairs
from .metrics import pls_cv_rmse_robust


# ---------------------------------------------------------------------------
# Population initialisation
# ---------------------------------------------------------------------------

def initialize_population(
    pop_size: int,
    P: int,
    n_wl: int,
    max_width: int,
) -> List[np.ndarray]:
    """
    Create a random initial population.

    Parameters
    ----------
    pop_size : int
        Number of individuals.
    P : int
        Number of genes (OBD pairs) per chromosome.
    n_wl : int
        Total number of wavelengths in the spectrum.
    max_width : int
        Maximum allowed width of a spectral region.

    Returns
    -------
    list of np.ndarray, each shape (P, 5)
    """
    population = []
    for _ in range(pop_size):
        chrom = np.zeros((P, 5), dtype=int)
        for i in range(P):
            s1 = random.randrange(0, n_wl)
            w1 = random.randint(1, max_width)
            s2 = random.randrange(0, n_wl)
            w2 = random.randint(1, max_width)
            active = random.choice([0, 1])
            chrom[i] = [s1, w1, s2, w2, active]
        population.append(chrom)
    return population


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def mutate(
    chrom: np.ndarray,
    n_wl: int,
    max_width: int,
    mut_prob: float = 0.2,
) -> np.ndarray:
    """
    Apply random mutation to one chromosome.

    Each gene is independently mutated with probability ``mut_prob``.
    A random field (start1/width1/start2/width2/active) is perturbed.

    Parameters
    ----------
    chrom : np.ndarray, shape (P, 5)
    n_wl : int
    max_width : int
    mut_prob : float

    Returns
    -------
    np.ndarray, shape (P, 5)  — mutated copy
    """
    new = chrom.copy()
    for i in range(new.shape[0]):
        if random.random() < mut_prob:
            field = random.randrange(0, 5)
            if field == 0:
                new[i, 0] = random.randrange(0, n_wl)
            elif field == 1:
                new[i, 1] = random.randint(1, max(1, min(max_width, n_wl - new[i, 0])))
            elif field == 2:
                new[i, 2] = random.randrange(0, n_wl)
            elif field == 3:
                new[i, 3] = random.randint(1, max(1, min(max_width, n_wl - new[i, 2])))
            else:
                new[i, 4] = 1 - new[i, 4]
    return new


def crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniform gene-level crossover between two parent chromosomes.

    Each gene row is swapped with probability 0.5.

    Parameters
    ----------
    parent1, parent2 : np.ndarray, shape (P, 5)

    Returns
    -------
    child1, child2 : np.ndarray, shape (P, 5)
    """
    child1, child2 = parent1.copy(), parent2.copy()
    for i in range(parent1.shape[0]):
        if random.random() < 0.5:
            child1[i, :], child2[i, :] = child2[i, :].copy(), child1[i, :].copy()
    return child1, child2


def tournament_selection(
    population: List[np.ndarray],
    fitnesses: List[float],
    k: int = 3,
) -> np.ndarray:
    """
    Select one individual via tournament selection.

    Parameters
    ----------
    population : list of chromosomes
    fitnesses : list of fitness values (lower is better)
    k : int — tournament size

    Returns
    -------
    np.ndarray — copy of the winning chromosome
    """
    candidates = random.sample(range(len(population)), k)
    winner = min(candidates, key=lambda idx: fitnesses[idx])
    return population[winner].copy()


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

def evaluate_fitness(
    chrom: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 5,
    cv: int = 5,
    penalty_per_feat: float = 0.01,
) -> float:
    """
    Evaluate the fitness of a single chromosome.

    fitness = CV_RMSE + penalty_per_feat * n_active_genes

    Parameters
    ----------
    chrom : np.ndarray, shape (P, 5)
    X : np.ndarray, shape (n_samples, n_wl)
    y : np.ndarray, shape (n_samples,)
    n_components : int — max PLS components
    cv : int — number of CV folds
    penalty_per_feat : float — sparsity regularisation weight

    Returns
    -------
    float — fitness score (lower is better)
    """
    X_feats, active_idx = make_feature_matrix_from_pairs(X, chrom)
    rmse = pls_cv_rmse_robust(X_feats, y, n_components=n_components, cv=cv)
    penalty = penalty_per_feat * len(active_idx)
    return rmse + penalty


# ---------------------------------------------------------------------------
# Main GA loop
# ---------------------------------------------------------------------------

def run_ga(
    X: np.ndarray,
    y: np.ndarray,
    n_wl: int,
    pop_size: int = 40,
    generations: int = 30,
    P: int = 6,
    max_width: int = 20,
    cx_prob: float = 0.6,
    mut_prob: float = 0.2,
    n_components: int = 5,
    cv: int = 5,
    penalty_per_feat: float = 0.01,
    random_state: Optional[int] = 0,
    verbose: bool = True,
) -> Tuple[np.ndarray, float, List[float]]:
    """
    Run the Genetic Algorithm to optimise OBD feature selection for PLSR.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_wl)
        Spectral data (pre-processed).
    y : np.ndarray, shape (n_samples,)
        Target variable.
    n_wl : int
        Number of wavelengths. Must equal ``X.shape[1]``.
    pop_size : int
        Population size. Default 40.
    generations : int
        Number of generations. Default 30.
    P : int
        Genes per chromosome (number of OBD pairs). Default 6.
    max_width : int
        Maximum spectral region width in channels. Default 20.
    cx_prob : float
        Crossover probability. Default 0.6.
    mut_prob : float
        Per-gene mutation probability. Default 0.2.
    n_components : int
        Maximum PLS latent variables for fitness evaluation. Default 5.
    cv : int
        Number of CV folds. Default 5.
    penalty_per_feat : float
        Sparsity penalty per active gene. Default 0.01.
    random_state : int or None
        Seed for reproducibility. Default 0.
    verbose : bool
        Print progress every ~10% of generations. Default True.

    Returns
    -------
    best_chrom : np.ndarray, shape (P, 5)
    best_fitness : float
    history : list of float — best fitness per generation
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    population = initialize_population(pop_size, P, n_wl, max_width)
    fitnesses = [
        evaluate_fitness(ind, X, y, n_components, cv, penalty_per_feat)
        for ind in population
    ]
    history = []

    log_every = max(1, generations // 10)

    for gen in range(generations):
        sorted_idx = sorted(range(len(population)), key=lambda i: fitnesses[i])
        # Elitism: keep top 2
        new_pop = [population[sorted_idx[0]].copy(), population[sorted_idx[1]].copy()]

        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fitnesses, k=3)
            p2 = tournament_selection(population, fitnesses, k=3)
            if random.random() < cx_prob:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1, p2
            new_pop.extend([mutate(c1, n_wl, max_width, mut_prob),
                             mutate(c2, n_wl, max_width, mut_prob)])

        population = new_pop[:pop_size]
        fitnesses = [
            evaluate_fitness(ind, X, y, n_components, cv, penalty_per_feat)
            for ind in population
        ]

        best_idx = int(np.argmin(fitnesses))
        best_fit = fitnesses[best_idx]
        history.append(best_fit)

        if verbose and (gen % log_every == 0 or gen == generations - 1):
            n_active = int(np.sum(population[best_idx][:, 4]))
            print(f"Gen {gen + 1:>{len(str(generations))}}/{generations}  "
                  f"fitness (RMSE+pen) = {best_fit:.4f}  "
                  f"active_features = {n_active}")

    best_idx = int(np.argmin(fitnesses))
    return population[best_idx], fitnesses[best_idx], history


# ---------------------------------------------------------------------------
# Multiple GA runs — wavelength stability analysis
# ---------------------------------------------------------------------------

def run_multiple_ga(
    X: np.ndarray,
    y: np.ndarray,
    wl,
    n_runs: int = 10,
    random_state_base: int = 0,
    verbose: bool = False,
    **ga_kwargs,
) -> Tuple[pd.DataFrame, List[Tuple[int, int]]]:
    """
    Run the GA ``n_runs`` times and aggregate wavelength selection frequency.

    This is useful for assessing which spectral regions are consistently
    identified across independent runs (wavelength stability analysis).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_wl)
    y : np.ndarray, shape (n_samples,)
    wl : array-like — wavelength axis, length n_wl
    n_runs : int — number of independent GA runs
    random_state_base : int — seeds are base, base+1, ..., base+n_runs-1
    verbose : bool — passed to ``run_ga``
    **ga_kwargs : extra keyword arguments forwarded to ``run_ga``

    Returns
    -------
    freq_df : pd.DataFrame
        Columns: ``wavelength_nm``, ``selection_count``
    all_regions : list of (start, end) index tuples from every run
    """
    wl = np.asarray(wl)
    n_wl = X.shape[1]
    freq_counts = np.zeros(n_wl, dtype=int)
    all_regions = []

    for run in range(n_runs):
        print(f"\n=== GA Run {run + 1}/{n_runs} ===")
        best_chrom, _, _ = run_ga(
            X, y, n_wl,
            random_state=random_state_base + run,
            verbose=verbose,
            **ga_kwargs,
        )
        _, active_idx = make_feature_matrix_from_pairs(X, best_chrom)

        for idx in active_idx:
            s1, w1, s2, w2, _ = best_chrom[idx].astype(int)
            for s, w in [(s1, w1), (s2, w2)]:
                start = max(0, s)
                end = min(n_wl, s + w)
                freq_counts[start:end] += 1
                all_regions.append((start, end))

    freq_df = pd.DataFrame({
        "wavelength_nm": wl,
        "selection_count": freq_counts,
    })

    print(f"\n=== Summary over {n_runs} GA runs ===")
    print("Top 10 most frequently selected wavelengths:")
    print(freq_df.sort_values("selection_count", ascending=False).head(10).to_string(index=False))

    return freq_df, all_regions
