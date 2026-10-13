"""CDF-t on a torch device (GPU backend) -- runs on whatever is available.

This example mirrors ``examples/example_r_gaussian.py`` with the optional
torch backend (:mod:`pycdft.gpu`): it reports the torch build and the
detected device, runs the CDF-t transform on the best available device
(CUDA / ROCm / MPS / CPU), and checks the agreement with the numpy backend.

Run from the repository root:

    python examples/gpu_example.py
"""

import numpy as np

from pycdft import CDFt as NumpyCDFt
from pycdft.gpu import (CDFt as GpuCDFt, available_devices, describe,
                        resolve_device)

rng = np.random.default_rng(7)

print("torch / devices:")
for key, value in describe().items():
    print(f"  {key}: {value}")
print(f"  chosen device: {resolve_device()}")

# ---------------------------------------------------------------------------
# 1-D series on the best available device
# ---------------------------------------------------------------------------
obs = rng.normal(0.0, 1.0, 1000)        # observations (calibration period)
cm_hist = rng.normal(3.0, 1.0, 1000)    # model, calibration period
cm_future = rng.normal(4.0, 1.0, 1000)  # model, target period

gpu_model = GpuCDFt(n_quantiles=1001, tail="linear", delta_shift="additive")
ds_gpu = gpu_model.fit_transform(obs, cm_hist, cm_future)   # numpy array

# the numpy backend for comparison
ds_np = NumpyCDFt(n_quantiles=1001, tail="linear",
                  delta_shift="additive").fit_transform(obs, cm_hist,
                                                        cm_future)

diff = np.abs(ds_gpu - ds_np)
print("\n1-D series (n = 1000, additive shift, linear tails):")
print(f"  device: {gpu_model.device}")
print(f"  downscaled: mean={ds_gpu.mean():.4f}  std={ds_gpu.std():.4f}")
print(f"  max |gpu - numpy| = {diff.max():.3e}  (NaN masks equal: "
      f"{np.array_equal(np.isnan(ds_gpu), np.isnan(ds_np))})")

# ---------------------------------------------------------------------------
# (T, N) field: every cell (column) is corrected independently
# ---------------------------------------------------------------------------
obs2d = rng.normal(0.0, 1.0, (1000, 64)) + 0.00
ch2d = rng.normal(0.0, 1.0, (1000, 64)) + 3.00
cf2d = rng.normal(0.0, 1.0, (1000, 64)) + 4.00
ds2d = GpuCDFt(device=resolve_device()).fit_transform(obs2d, ch2d, cf2d)
print(f"\n(T, N) field (1000 x 64): output shape {ds2d.shape}, "
      f"all finite: {bool(np.isfinite(ds2d).all())}")

# ---------------------------------------------------------------------------
# Explicit device selection: the request must be satisfiable
# ---------------------------------------------------------------------------
devices = available_devices()
print(f"\navailable devices: {devices}")
if "cuda" in devices:
    print("CUDA available: run with `CDFt(device='cuda')`")
else:
    print("no accelerated device: the backend runs on CPU "
          "(install a CUDA or ROCm torch build to use a GPU)")
