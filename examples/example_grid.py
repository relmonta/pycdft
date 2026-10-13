#!/usr/bin/env python
"""Example 3: a small grid ((T, N) array) with per-cell statistics.

The CDF-t is applied independently to every cell of a (T, N) field in one
vectorized call (the trailing dimensions are flattened internally; here a
4x4 grid is used).  A spatially varying mean is added to the model series
to mimic a real grid.  Prints per-cell summaries and saves quantile-
quantile plots (downscaled vs observations) for a few cells plus the
before/after CDFs when matplotlib is available.
"""

import os

import numpy as np

from pycdft import CDFt

RNG = np.random.default_rng(42)
T, N = 2000, 16                      # 2000 time steps, 4x4 grid = 16 cells
gy, gx = 4, 4
grid = np.arange(N).reshape(gy, gx)

# spatially varying baseline (a smooth field, e.g. orography/continentality)
field = np.sin(grid / 2.0) * 1.5 + grid / 4.0
field = field.ravel()

obs = RNG.normal(0.0, 1.0, (T, N)) + field            # observations
cm_hist = RNG.normal(3.0, 1.0, (T, N)) + field        # model, calibration
cm_future = RNG.normal(4.0, 1.2, (T, N)) + field      # model, target (+ drift
                                                      # and spread change)

model = CDFt(n_quantiles=1001, tail="linear", delta_shift="additive")
pred = model.fit_transform(obs, cm_hist, cm_future)

print("=== small grid ((T, N) = (%d, %d), per-cell CDF-t) ===" % (T, N))
print("cell | obs mean/sd | cm_future mean/sd | pred mean/sd | "
      "delta shift")
for j in range(6):
    print(f"{j:4d} | {np.mean(obs[:, j]):+.2f}/{np.std(obs[:, j]):.2f} "
          f"| {np.mean(cm_future[:, j]):+.2f}/"
          f"{np.std(cm_future[:, j]):.2f} "
          f"| {np.mean(pred[:, j]):+.2f}/{np.std(pred[:, j]):.2f} "
          f"| {model.calibration_.delta[j]:+.3f}")

# the per-cell mean bias is removed, the model change is preserved
bias_removed = np.mean(pred, axis=0) - (
    np.mean(obs, axis=0) + (np.mean(cm_future, axis=0)
                            - np.mean(cm_hist, axis=0)))
print(f"\nmax |mean(pred) - [mean(obs) + model change]| = "
      f"{np.abs(bias_removed).max():.4f} (should be ~0)")

# ---- plots --------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

if plt is not None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))

    def cdf(x):
        x = np.sort(x)
        return x, np.arange(1, x.size + 1) / x.size

    # panel 0: before/after CDFs for cell 0
    ax = axes[0, 0]
    v, f = cdf(cm_future[:, 0])
    ax.plot(v, f, label="cm_future (before)", color="C2")
    v, f = cdf(pred[:, 0])
    ax.plot(v, f, label="pred (after)", color="C0")
    v, f = cdf(obs[:, 0])
    ax.plot(v, f, label="obs (cal)", color="C1", ls="--")
    ax.set_title("cell 0: target CDF before/after")
    ax.legend(loc="lower right", fontsize=8)

    # panel 1: QQ plot pred vs obs for cells 0-3
    ax = axes[0, 1]
    qs = np.linspace(0.01, 0.99, 99)
    for j in range(4):
        ax.plot(np.quantile(obs[:, j], qs), np.quantile(pred[:, j], qs),
                label=f"cell {j}")
    lim = [np.quantile(obs, 0.01), np.quantile(obs, 0.99)]
    ax.plot(lim, lim, "k--", lw=0.8, label="1:1")
    ax.set_title("QQ: downscaled vs obs (cells 0-3)")
    ax.set_xlabel("obs quantile")
    ax.set_ylabel("pred quantile")
    ax.legend(fontsize=8)

    # panel 2: before (cm_future) vs obs QQ -- shows the raw bias
    ax = axes[1, 0]
    for j in range(4):
        ax.plot(np.quantile(obs[:, j], qs), np.quantile(cm_future[:, j], qs),
                label=f"cell {j}", ls=":")
    ax.plot(lim, lim, "k--", lw=0.8, label="1:1")
    ax.set_title("QQ: raw cm_future vs obs (before)")
    ax.set_xlabel("obs quantile")
    ax.set_ylabel("cm_future quantile")
    ax.legend(fontsize=8)

    # panel 3: mean bias before / after per cell
    ax = axes[1, 1]
    jj = np.arange(N)
    before = np.mean(cm_future, axis=0) - np.mean(obs, axis=0)
    after = np.mean(pred, axis=0) - np.mean(obs, axis=0)
    ax.bar(jj - 0.2, before, 0.4, label="before CDF-t", color="C3")
    ax.bar(jj + 0.2, after, 0.4, label="after CDF-t", color="C0")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("per-cell mean bias")
    ax.set_xlabel("cell")
    ax.set_ylabel("mean bias")
    ax.legend(fontsize=8)

    out = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "example_grid.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nfigure saved to {path}")
