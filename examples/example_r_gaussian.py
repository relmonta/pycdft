#!/usr/bin/env python
"""Example 1: the R ``CDFt`` package's own Gaussian example.

obs ~ N(0, 1), cm_hist ~ N(3, 1), cm_future ~ N(4, 1), n = 1000
(the same synthetic setup as the CDFt R package documentation; the R
package's published output is reproduced in ``tests/test_r_cdft.py``).

Prints the basic statistics of the downscaled series and, when matplotlib
is available, saves a figure with the calibration CDFs and the
before/after target CDFs to ``examples/figures/``.
"""

import os

import numpy as np

from pycdft import CDFt

RNG = np.random.default_rng(7)
N = 1000

obs = RNG.normal(0.0, 1.0, N)        # observations, calibration period
cm_hist = RNG.normal(3.0, 1.0, N)    # model, calibration period
cm_future = RNG.normal(4.0, 1.0, N)  # model, target period

# CDF-t with the standard options (additive mean shift, linear tail,
# 1001-point probability grid)
model = CDFt(n_quantiles=1001, tail="linear", delta_shift="additive")
pred = model.fit_transform(obs, cm_hist, cm_future)

print("=== R-package Gaussian example (obs N(0,1), cm_hist N(3,1), "
      "cm_future N(4,1)) ===")
print(f"obs        mean {np.mean(obs):+.3f}  sd {np.std(obs):.3f}")
print(f"cm_hist    mean {np.mean(cm_hist):+.3f}  sd {np.std(cm_hist):.3f}")
print(f"cm_future  mean {np.mean(cm_future):+.3f}  sd {np.std(cm_future):.3f}")
print(f"pred       mean {np.mean(pred):+.3f}  sd {np.std(pred):.3f}")
print(f"delta shift (additive) = {model.calibration_.delta[0]:+.3f} "
      "(= mean(obs) - mean(cm_hist))")
print(f"mean bias removed: {np.mean(obs):.3f} vs "
      f"{np.mean(cm_future) + model.calibration_.delta[0]:.3f} (shifted target)")
print(f"pred vs obs quantiles (p = 0.01, 0.5, 0.99): "
      f"{np.quantile(pred, [0.01, 0.5, 0.99]).round(3)}")

# ---- plot ---------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:                       # matplotlib is optional
    plt = None

if plt is not None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    def cdf(x):
        x = np.sort(x)
        return x, np.arange(1, x.size + 1) / x.size

    ax = axes[0]
    for arr, label in ((obs, "obs (cal)"), (cm_hist, "cm_hist"),
                       (cm_future, "cm_future")):
        v, f = cdf(arr)
        ax.plot(v, f, label=label)
    ax.set_title("Calibration / target ECDFs")
    ax.set_xlabel("value")
    ax.set_ylabel("F")
    ax.legend(loc="lower right")

    ax = axes[1]
    v, f = cdf(cm_future)
    ax.plot(v, f, label="cm_future (before)", color="C2")
    v, f = cdf(pred)
    ax.plot(v, f, label="pred (after CDF-t)", color="C0")
    v, f = cdf(obs)
    ax.plot(v, f, label="obs (cal)", color="C1", ls="--")
    ax.set_title("Target CDF before / after CDF-t")
    ax.set_xlabel("value")
    ax.legend(loc="lower right")

    out = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "example_r_gaussian.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"figure saved to {path}")
