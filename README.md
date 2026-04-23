<<<<<<< HEAD
# GA-OBD-PLSR

**Genetic Algorithm — Optical Band Difference feature selection for PLS Regression**

A Python package for selecting interpretable spectral features from NIR data. The GA evolves **Optical Band Difference (OBD)** features — contrast signals defined as the difference between the mean absorbance of two spectral regions — and optimises them for PLS regression via cross-validated RMSE.

---

## What is an OBD feature?

An OBD feature is a simple, physically interpretable contrast:

```
feature = mean(X[:, start1 : start1+width1]) - mean(X[:, start2 : start2+width2])
```

The GA searches for the region pairs (bands) that produce the most predictive and sparse set of features for your target variable.

---

## Installation

```bash
git clone https://github.com/your-username/ga-obd-plsr.git
cd ga-obd-plsr
pip install -e .
```

Or install directly (not published to PyPI yet):

```bash
pip install ga-obd-plsr
```

---

## Quickstart

```python
import numpy as np
from ga_obd_plsr import GAOBD

# X : (n_samples, n_wavelengths) pre-processed spectral matrix
# y : (n_samples,) target variable
# wl : (n_wavelengths,) wavelength axis in nm

model = GAOBD(
    pop_size=50,
    generations=100,
    P=8,              # number of OBD genes per chromosome
    n_components=6,   # max PLS components
    penalty_per_feat=0.01,
)
model.fit(X, y, wl)

print(model.summary())

# Transform new spectra to OBD features
X_feats = model.transform(X_new)

# Cross-validated metrics
metrics = model.score_cv(X, y)
# {'RMSE': ..., 'R2': ..., 'RPD': ...}

# Plots
model.plot_history()
model.plot_spectra(X)
```

---

## Key API

### `GAOBD` (sklearn-compatible)

| Method | Description |
|---|---|
| `fit(X, y, wl)` | Run the GA to select OBD features |
| `transform(X)` | Apply selected features to new spectra |
| `fit_transform(X, y, wl)` | Fit then transform in one call |
| `score_cv(X, y)` | Returns `{RMSE, R2, RPD}` from CV on selected features |
| `predict(X_train, y_train, X_test)` | Fit PLS on OBD features and predict |
| `summary()` | Human-readable summary of selected bands |
| `get_selected_features_df()` | DataFrame of selected gene details |
| `plot_history()` | GA fitness convergence plot |
| `plot_spectra(X)` | Spectra with selected regions shaded |

### GA parameters

| Parameter | Default | Description |
|---|---|---|
| `pop_size` | 40 | Population size |
| `generations` | 100 | Number of generations |
| `P` | 8 | OBD genes per chromosome |
| `max_width` | 20 | Max region width (channels) |
| `cx_prob` | 0.7 | Crossover probability |
| `mut_prob` | 0.25 | Per-gene mutation probability |
| `n_components` | 6 | Max PLS components in fitness eval |
| `cv` | 5 | K-Fold CV folds |
| `penalty_per_feat` | 0.01 | Sparsity penalty per active gene |
| `random_state` | 0 | Seed for reproducibility |

---

## Module structure

```
ga_obd_plsr/
├── __init__.py          # Public API
├── model.py             # GAOBD sklearn-compatible class
├── ga.py                # GA loop, operators, run_multiple_ga
├── features.py          # OBD feature construction
├── metrics.py           # CV metric functions (RMSE, R2, RPD, ...)
├── preprocessing.py     # Savitzky-Golay, scaling, wavelength range
└── plotting.py          # All visualisation functions

examples/
├── quickstart.py        # Synthetic data demo (no CSV needed)
└── coffee_barley_example.py  # Full pipeline on real NIR data

tests/
└── test_ga_obd_plsr.py  # pytest unit tests
```

---

## Preprocessing your data

`GAOBD.fit()` expects pre-processed spectra. Common steps are available in `ga_obd_plsr.preprocessing`:

```python
from ga_obd_plsr.preprocessing import savitzky_golay, select_wavelength_range, mean_center_scale

# 1. Restrict wavelength range
X_sel, wl_sel = select_wavelength_range(X_raw, wl, wl_min=4000, wl_max=9000)

# 2. Savitzky-Golay 1st derivative
X_sg = savitzky_golay(X_sel, window_length=7, polyorder=2, deriv=1)

# 3. Mean-centre and scale (optional — not needed after derivative)
X_proc, mu, sd = mean_center_scale(X_sg)
```

---

## Wavelength stability analysis

Run the GA multiple times and count how often each wavelength is selected:

```python
from ga_obd_plsr import run_multiple_ga

freq_df, regions = run_multiple_ga(
    X, y, wl,
    n_runs=20,
    pop_size=40, generations=100, P=8,
    n_components=6, cv=5, penalty_per_feat=0.005,
)
# freq_df: DataFrame with columns [wavelength_nm, selection_count]
```

---

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Citation

If you use this package in published research, please cite:

```
@software{ga_obd_plsr,
  title  = {GA-OBD-PLSR: Genetic Algorithm Optical Band Difference feature selection for PLSR},
  year   = {2024},
  url    = {https://github.com/your-username/ga-obd-plsr}
}
```

---

## License

MIT
=======
# Optimized-Bin-Difference-for-Chemometric-modeling
This repository contains python implementation of Optimized Bin Difference method for chemometric modeling with spectral data. The core idea is that Genetic Algorithm selects bins, their width such that the difference of the average of bins (contiguous features) becomes new features for PLSR model.
>>>>>>> 02f94069047942d47a8c7b4afcc04aabfd621fbb
