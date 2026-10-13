"""pycdft.gpu -- torch (device-agnostic) backend for CDF-t.

A mirror of the :mod:`pycdft` numpy API implemented with torch operators, so
that the same CDF-t math can run on CUDA / AMD ROCm / Apple MPS / CPU
devices::

    from pycdft.gpu import CDFt          # mirrors pycdft.CDFt
    ds = CDFt(device="cuda").fit_transform(obs, cm_hist, cm_future)

Torch is an optional dependency: importing this module never fails without
torch (``TORCH_AVAILABLE`` is then ``False`` and the functions raise a clear
``ImportError`` when called).  See ``docs/gpu.md`` for the supported devices
and the verified numerical equivalence with the numpy backend.
"""

from ._torch import TORCH_AVAILABLE
from .devices import (available_devices, describe, is_rocm_build,
                      resolve_device, rocm_available)
from .api import (CDFt, Calibration, apply_cdft, apply_cdft_hist, calibrate,
                  calibrate_hist, cdft_transform, delta_from_means,
                  histograms, transfer)
from . import kernels

__all__ = [
    "TORCH_AVAILABLE",
    "available_devices",
    "resolve_device",
    "rocm_available",
    "is_rocm_build",
    "describe",
    "kernels",
    "Calibration",
    "calibrate",
    "calibrate_hist",
    "apply_cdft",
    "apply_cdft_hist",
    "histograms",
    "delta_from_means",
    "transfer",
    "CDFt",
    "cdft_transform",
]
