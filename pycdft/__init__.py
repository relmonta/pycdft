"""PyCDFt -- CDF-t statistical downscaling / bias correction (numpy only).

A vectorized, grid-ready implementation of the CDF-t (Cumulative
Distribution Function transform) method of Michelangeli, Vrac & Loukos
(2009), with the delta-shift and tail conventions of the R ``CDFt`` package
and ibicus.  See the README and ``docs/design.md`` for the math and the
comparison with the reference implementations.

An optional torch backend (:mod:`pycdft.gpu`) mirrors this API and runs the
same math on CUDA / ROCm / MPS / CPU devices; it is only available when
torch is installed (``pip install PyCDFt[gpu]``) and never affects the pure
numpy core.
"""

from .api import CDFt, cdft_transform
from .core import (Calibration, apply_cdft, apply_cdft_hist, calibrate,
                   calibrate_hist, delta_from_means, histograms, transfer)
from .math import (ecdf, ecdf_at, ecdf_hist, hist_quantile,
                   hist_quantile_pc, histogram_accumulate, interp_transfer,
                   nanquantile, quantile, quantile_from_sorted,
                   quantile_grid, searchsorted_columns, windowed_slope)

try:                                     # optional torch backend
    from . import gpu as gpu
    GPU_AVAILABLE = bool(gpu.TORCH_AVAILABLE)
except Exception:                        # pragma: no cover - torch-less install
    gpu = None                           # type: ignore[assignment]
    GPU_AVAILABLE = False

__version__ = "0.3.1"

__all__ = [
    "CDFt",
    "cdft_transform",
    "Calibration",
    "calibrate",
    "calibrate_hist",
    "apply_cdft",
    "apply_cdft_hist",
    "histograms",
    "delta_from_means",
    "quantile_grid",
    "searchsorted_columns",
    "ecdf",
    "ecdf_at",
    "quantile",
    "nanquantile",
    "quantile_from_sorted",
    "windowed_slope",
    "interp_transfer",
    "transfer",
    "histogram_accumulate",
    "hist_quantile",
    "hist_quantile_pc",
    "ecdf_hist",
    "gpu",
    "GPU_AVAILABLE",
    "__version__",
]
