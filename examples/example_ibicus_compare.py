#!/usr/bin/env python
"""Example 4: comparison against ibicus (ECMWF) on synthetic data.

ibicus is an independent implementation of CDF-t (Apache-2.0).  With
``ecdf_method="step_function"`` and ``iecdf_method="linear"`` it uses the
same step-ECDF / type-7-quantile conventions as PyCDFt, but it evaluates
the composition per value (no probability grid) and has no tail model
(its mapping clamps at the observed/model extremes, like ``tail="clip"``).

The residual difference is the discretization of the transfer curve on the
Q-point grid: interior (2%-98% ranks) RMSE ~0.005-0.01 on Gaussian data,
~0.01-0.02 on skewed data; the extremes differ more (~0.06) because of the
tail models.

Run only if ibicus is installed: ``pip install ibicus``.
"""

import numpy as np

from pycdft import CDFt

try:
    from ibicus.debias._cdft import CDFt as IbicusCDFt
except ImportError:                       # pragma: no cover
    raise SystemExit("ibicus is not installed; pip install ibicus to run "
                     "this example")

RNG = np.random.default_rng(7)
N = 1000

obs = RNG.normal(0.0, 1.0, N)
cm_hist = RNG.normal(3.0, 1.0, N)
cm_future = RNG.normal(4.0, 1.0, N)

# PyCDFt: grid-based transfer, clipped tails (comparable to ibicus)
ours = CDFt(tail="clip").fit_transform(obs, cm_hist, cm_future)
ours_linear = CDFt(tail="linear").fit_transform(obs, cm_hist, cm_future)

# ibicus: exact per-value chain, no running windows, no SSR
ib = IbicusCDFt(delta_shift="additive", running_window_mode=False,
                running_window_mode_over_years_of_cm_future=False,
                ecdf_method="step_function", iecdf_method="linear",
                SSR=False)
theirs = ib.apply(obs[:, None, None], cm_hist[:, None, None],
                  cm_future[:, None, None], progressbar=False)[:, 0, 0]


def metrics(a, b):
    srt = np.sort(cm_future)
    interior = (cm_future > srt[20]) & (cm_future < srt[980])
    return {
        "RMSE": np.sqrt(np.mean((a - b) ** 2)),
        "max|d|": np.abs(a - b).max(),
        "interior RMSE": np.sqrt(np.mean((a[interior] - b[interior]) ** 2)),
        "interior max|d|": np.abs(a[interior] - b[interior]).max(),
    }


print("=== PyCDFt vs ibicus (step ECDF / linear iecdf, no windows) ===")
for label, a in (("ours tail=clip", ours), ("ours tail=linear", ours_linear)):
    m = metrics(a, theirs)
    print(f"{label:17s} vs ibicus: RMSE {m['RMSE']:.4f} "
          f"max|d| {m['max|d|']:.4f} | interior RMSE "
          f"{m['interior RMSE']:.4f} interior max|d| "
          f"{m['interior max|d|']:.4f}")
print("\nmeans:", f"ours {np.mean(ours):+.3f}  ibicus {np.mean(theirs):+.3f} "
      f"(both in the shifted frame; mean bias removed by both)")
