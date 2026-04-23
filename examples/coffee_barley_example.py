"""
examples/coffee_barley_example.py
----------------------------------
Demonstrates GA-OBD-PLSR on a NIR coffee/barley adulteration dataset.

Dataset
-------
CSV file: ``coffee_barley_nir.csv``
    - Column 0 : sample ID (ignored)
    - Column 1 : % (w/w) of Barley  ← target variable
    - Columns 2–1502 : NIR absorbance at each wavelength

Preprocessing
-------------
    1. Wavelength range restriction to 4094.5–9029.7 nm
    2. Savitzky-Golay first derivative (window=7, poly=2)

Run
---
    python examples/coffee_barley_example.py --csv path/to/coffee_barley_nir.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ga_obd_plsr import GAOBD
from ga_obd_plsr.preprocessing import savitzky_golay, select_wavelength_range


def load_and_preprocess(csv_path: str):
    df = pd.read_csv(csv_path)

    y = np.array(df.iloc[:, 1], dtype=float)
    X_raw = np.array(df.iloc[:, 2:1503], dtype=float)
    wl = np.array([int(c) for c in df.columns[2:1503]])

    # --- Wavelength range restriction ---
    X_sel, wl_sel = select_wavelength_range(X_raw, wl, wl_min=4094.5, wl_max=9029.7)

    # --- Savitzky-Golay 1st derivative ---
    X_proc = savitzky_golay(X_sel, window_length=7, polyorder=2, deriv=1)

    print(f"Samples: {X_proc.shape[0]}   Wavelengths: {X_proc.shape[1]}")
    return X_proc, y, wl_sel


def main(csv_path: str):
    X, y, wl = load_and_preprocess(csv_path)

    # ---------------------------------------------------------------
    # Fit GA-OBD-PLSR
    # ---------------------------------------------------------------
    model = GAOBD(
        pop_size=100,
        generations=250,
        P=8,
        max_width=20,
        cx_prob=0.7,
        mut_prob=0.25,
        n_components=6,
        cv=5,
        penalty_per_feat=0.01,
        random_state=0,
        verbose=True,
    )
    model.fit(X, y, wl)

    print(model.summary())

    # ---------------------------------------------------------------
    # Cross-validated metrics
    # ---------------------------------------------------------------
    metrics = model.score_cv(X, y)
    print("\nCross-validated metrics on selected OBD features:")
    for k, v in metrics.items():
        print(f"  {k:6s}: {v:.4f}")

    # ---------------------------------------------------------------
    # Compare GA-OBD vs full-spectrum PLS
    # ---------------------------------------------------------------
    from ga_obd_plsr.metrics import pls_cv_rmse_robust, pls_cv_r2_robust, pls_cv_rpd_robust

    print("\nComparison: GA-OBD-PLS vs Full-Spectrum PLS (10 components)")
    full_rmse = pls_cv_rmse_robust(X, y, n_components=10, cv=5)
    full_r2   = pls_cv_r2_robust(X, y, n_components=10, cv=5)
    full_rpd  = pls_cv_rpd_robust(X, y, n_components=10, cv=5)

    comparison = pd.DataFrame({
        "GA-OBD-PLS": metrics,
        "Full-Spectrum PLS": {"RMSE": full_rmse, "R2": full_r2, "RPD": full_rpd},
    }).T
    print(comparison.to_string())

    # ---------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------
    fig_hist = model.plot_history()
    fig_hist.savefig("ga_history.png", dpi=150)
    print("\nSaved: ga_history.png")

    fig_spec = model.plot_spectra(X, ylabel="1st Derivative Absorbance",
                                  title="Coffee/Barley NIR with GA-OBD Features")
    fig_spec.savefig("spectra_features.png", dpi=150)
    print("Saved: spectra_features.png")

    plt.show()

    # ---------------------------------------------------------------
    # Save selected features
    # ---------------------------------------------------------------
    df_sel = model.get_selected_features_df()
    df_sel.to_csv("selected_obd_features.csv", index=False)
    print("Saved: selected_obd_features.csv")

    # Save OBD feature matrix
    X_feats = model.transform(X)
    feat_df = pd.DataFrame(X_feats, columns=model.feature_names_)
    feat_df["target"] = y
    feat_df.to_csv("obd_feature_matrix.csv", index=False)
    print("Saved: obd_feature_matrix.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GA-OBD-PLSR: Coffee/Barley NIR example")
    parser.add_argument("--csv", required=True, help="Path to coffee_barley_nir.csv")
    args = parser.parse_args()
    main(args.csv)
