"""
examples/quickstart.py
-----------------------
Minimal working example using synthetic data — no real dataset needed.

Run:
    python examples/quickstart.py
"""

import numpy as np
from ga_obd_plsr import GAOBD

# -----------------------------------------------------------------------
# Generate synthetic spectral data
# -----------------------------------------------------------------------
rng = np.random.default_rng(42)
n_samples = 80
n_wl = 500
wl = np.linspace(1000, 2500, n_wl)   # 1000–2500 nm, 500 channels

# True signal lives at ~1200 nm and ~1800 nm
signal = (np.exp(-((wl - 1200) ** 2) / (2 * 30 ** 2))
          - np.exp(-((wl - 1800) ** 2) / (2 * 40 ** 2)))

X = rng.standard_normal((n_samples, n_wl)) * 0.05
true_weights = rng.uniform(0, 10, n_samples)
X += np.outer(true_weights, signal)
y = true_weights + rng.normal(0, 0.5, n_samples)

# -----------------------------------------------------------------------
# Fit GA-OBD-PLSR
# -----------------------------------------------------------------------
model = GAOBD(
    pop_size=30,      # small for speed in this demo
    generations=20,
    P=6,
    n_components=4,
    penalty_per_feat=0.05,
    verbose=True,
)
model.fit(X, y, wl)

print(model.summary())

# Cross-validated metrics
metrics = model.score_cv(X, y)
print("\nCV Metrics:", metrics)

# Plot results
fig_h = model.plot_history()
fig_h.savefig("quickstart_history.png", dpi=120)

fig_s = model.plot_spectra(X)
fig_s.savefig("quickstart_spectra.png", dpi=120)

print("\nDone. Figures saved to quickstart_history.png and quickstart_spectra.png")
