"""
tests/test_ga_obd_plsr.py
--------------------------
Basic unit tests for ga_obd_plsr. Run with:
    pytest tests/
"""

import numpy as np
import pytest
from ga_obd_plsr import (
    GAOBD,
    make_feature_matrix_from_pairs,
    pls_cv_rmse_robust,
    pls_cv_r2_robust,
    compute_all_metrics,
)
from ga_obd_plsr.ga import (
    initialize_population,
    mutate,
    crossover,
    tournament_selection,
    evaluate_fitness,
)
from ga_obd_plsr.features import chromosome_to_dataframe, build_feature_names


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(0)
    n, p = 50, 100
    wl = np.linspace(1000, 2000, p)
    X = rng.standard_normal((n, p))
    y = X[:, 10] - X[:, 50] + rng.normal(0, 0.1, n)
    return X, y, wl


@pytest.fixture
def simple_chrom():
    return np.array([[5, 3, 20, 4, 1],
                     [10, 2, 30, 3, 0],
                     [40, 5, 60, 5, 1]], dtype=int)


# -----------------------------------------------------------------------
# Feature tests
# -----------------------------------------------------------------------

class TestFeatures:
    def test_make_feature_matrix_shape(self, synthetic_data, simple_chrom):
        X, y, wl = synthetic_data
        X_feats, active_idx = make_feature_matrix_from_pairs(X, simple_chrom)
        # 2 active genes
        assert X_feats.shape == (X.shape[0], 2)
        assert active_idx == [0, 2]

    def test_make_feature_matrix_no_active(self, synthetic_data):
        X, y, wl = synthetic_data
        chrom = np.zeros((3, 5), dtype=int)  # all inactive
        X_feats, active_idx = make_feature_matrix_from_pairs(X, chrom)
        assert X_feats.shape[1] == 0
        assert active_idx == []

    def test_chromosome_to_dataframe(self, simple_chrom):
        wl = np.linspace(1000, 2000, 100)
        df = chromosome_to_dataframe(simple_chrom, wl)
        assert len(df) == 2
        assert "wl_start1_nm" in df.columns

    def test_build_feature_names(self, simple_chrom):
        wl = np.linspace(1000, 2000, 100)
        names = build_feature_names(simple_chrom, wl)
        assert len(names) == 2
        assert "nm" in names[0]


# -----------------------------------------------------------------------
# Metrics tests
# -----------------------------------------------------------------------

class TestMetrics:
    def test_rmse_returns_float(self, synthetic_data, simple_chrom):
        X, y, wl = synthetic_data
        X_feats, _ = make_feature_matrix_from_pairs(X, simple_chrom)
        rmse = pls_cv_rmse_robust(X_feats, y)
        assert isinstance(rmse, float)
        assert rmse > 0

    def test_rmse_empty_features(self, synthetic_data):
        X, y, wl = synthetic_data
        X_empty = np.zeros((X.shape[0], 0))
        assert pls_cv_rmse_robust(X_empty, y) == np.inf

    def test_r2_empty_features(self, synthetic_data):
        X, y, wl = synthetic_data
        X_empty = np.zeros((X.shape[0], 0))
        assert pls_cv_r2_robust(X_empty, y) == -np.inf

    def test_compute_all_metrics_keys(self):
        rng = np.random.default_rng(1)
        y_true = rng.uniform(0, 10, 30)
        y_pred = y_true + rng.normal(0, 0.5, 30)
        m = compute_all_metrics(y_true, y_pred)
        for key in ["R2", "RMSE", "MAE", "Bias", "SEP", "RPD", "RPIQ", "CCC"]:
            assert key in m


# -----------------------------------------------------------------------
# GA operator tests
# -----------------------------------------------------------------------

class TestGAOperators:
    def test_initialize_population_shape(self):
        pop = initialize_population(pop_size=10, P=5, n_wl=100, max_width=15)
        assert len(pop) == 10
        assert pop[0].shape == (5, 5)

    def test_mutate_returns_copy(self):
        chrom = np.zeros((5, 5), dtype=int)
        mutated = mutate(chrom, n_wl=100, max_width=10, mut_prob=1.0)
        assert mutated is not chrom  # new object

    def test_crossover_preserves_shape(self):
        p1 = np.ones((5, 5), dtype=int)
        p2 = np.zeros((5, 5), dtype=int)
        c1, c2 = crossover(p1, p2)
        assert c1.shape == (5, 5)
        assert c2.shape == (5, 5)

    def test_tournament_selection_returns_chrom(self):
        pop = [np.eye(3, dtype=int)] * 5
        fitnesses = [0.5, 0.1, 0.9, 0.3, 0.7]
        winner = tournament_selection(pop, fitnesses, k=3)
        assert winner.shape == (3, 3)

    def test_evaluate_fitness_scalar(self, synthetic_data):
        X, y, wl = synthetic_data
        chrom = initialize_population(1, P=4, n_wl=X.shape[1], max_width=10)[0]
        fit = evaluate_fitness(chrom, X, y)
        assert isinstance(fit, float)


# -----------------------------------------------------------------------
# GAOBD model tests
# -----------------------------------------------------------------------

class TestGAOBD:
    def test_fit_transform(self, synthetic_data):
        X, y, wl = synthetic_data
        model = GAOBD(pop_size=10, generations=5, P=4, verbose=False)
        X_feats = model.fit_transform(X, y, wl)
        assert X_feats.shape[0] == X.shape[0]

    def test_summary_string(self, synthetic_data):
        X, y, wl = synthetic_data
        model = GAOBD(pop_size=10, generations=5, P=4, verbose=False)
        model.fit(X, y, wl)
        s = model.summary()
        assert "Active features" in s

    def test_score_cv_keys(self, synthetic_data):
        X, y, wl = synthetic_data
        model = GAOBD(pop_size=10, generations=5, P=4, verbose=False)
        model.fit(X, y, wl)
        m = model.score_cv(X, y)
        assert "RMSE" in m and "R2" in m and "RPD" in m

    def test_not_fitted_raises(self):
        model = GAOBD()
        with pytest.raises(RuntimeError):
            model.transform(np.zeros((10, 50)))

    def test_get_selected_features_df(self, synthetic_data):
        X, y, wl = synthetic_data
        model = GAOBD(pop_size=10, generations=5, P=4, verbose=False)
        model.fit(X, y, wl)
        df = model.get_selected_features_df()
        assert "gene_index" in df.columns
