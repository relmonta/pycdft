#!/usr/bin/env python
"""Example 2: non-Gaussian (skewed, gamma-distributed) case.

Shows the effect of the delta-shift choice and of the tail option on a
strongly skewed variable with both a mean drift and a variance change
between the calibration and target periods.  Prints summary statistics and
saves a before/after CDF figure when matplotlib is available.
"""

import os

import numpy as np

from pycdft import CDFt

RNG = np.random.default_rng(11)
N = 1000

obs = RNG.gamma(2.0, 1.5, N)             # skewed (gamma) observations
cm_hist = RNG.gamma(2.0, 1.5, N) + 3.0   # model cal: shifted right
cm_future = RNG.gamma(2.0, 1.8, N) + 4.5  # model target: more drift + spread

configs = {
    "additive / linear (default)": dict(delta_shift="additive",
                                        tail="linear"),
    "additive / clip": dict(delta_shift="additive", tail="clip"),
    "none / linear": dict(delta_shift="none", tail="linear"),
    "multiplicative / linear": dict(delta_shift="multiplicative",
                                    tail="linear"),
}

print("=== skewed (gamma) case ===")
print(f"obs        mean {np.mean(obs):.3f}  sd {np.std(obs):.3f}  "
      f"skew {np.mean((obs - obs.mean()) ** 3) / obs.std() ** 3:.2f}")
print(f"cm_hist    mean {np.mean(cm_hist):.3f}  sd {np.std(cm_hist):.3f}")
print(f"cm_future  mean {np.mean(cm_future):.3f}  sd {np.std(cm_future):.3f}")
print()
for label, kw in configs.items():
    pred = CDFt(**kw).fit_transform(obs, cm_hist, cm_future)
    print(f"{label:28s} mean {np.mean(pred):+.3f}  sd {np.std(pred):.3f}  "
          f"min {np.min(pred):.3f}  max {np.max(pred):.3f}")

# ---- plot ---------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
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
    ax.set_title("Skewed calibration / target ECDFs")
    ax.set_xlabel("value")
    ax.legend(loc="lower right")

    ax = axes[1]
    colors = iter(["C0", "C3", "C4", "C5"])
    for label, kw in configs.items():
        pred = CDFt(**kw).fit_transform(obs, cm_hist, cm_future)
        v, f = cdf(pred)
        ax.plot(v, f, label=label, color=next(colors))
    v, f = cdf(obs)
    ax.plot(v, f, label="obs (cal)", color="C1", ls="--")
    ax.set_title("Downscaled target CDFs (shift / tail options)")
    ax.set_xlabel("value")
    ax.legend(loc="lower right", fontsize=8)

    out = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "example_skewed.png")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"\nfigure saved to {path}")
