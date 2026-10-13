"""High-level torch CDF-t API (``pycdft.gpu``).

A device-agnostic mirror of :mod:`pycdft.api` / :mod:`pycdft.core`: the same
CDF-t math (Michelangeli, Vrac & Loukos 2009, with the delta-shift and tail
conventions of the R ``CDFt`` package and ibicus) implemented with torch
operators, so that the per-cell kernels can run on a GPU (CUDA, AMD ROCm,
Apple MPS) or on CPU.  The high-level :class:`CDFt` class mirrors
:class:`pycdft.CDFt` (same parameters, same shapes, numpy in / numpy out by
default) and the low-level functions (:func:`calibrate`,
:func:`apply_cdft`, ...) mirror :mod:`pycdft.core`.

Devices
-------
- :func:`pycdft.gpu.available_devices` reports what torch exposes on the
  machine; :func:`pycdft.gpu.resolve_device` picks the best available
  device (``cuda`` -> ``mps`` -> ``cpu``) or validates an explicit request
  (an explicit unavailable device raises ``ValueError``, never a silent CPU
  fallback).
- On AMD hardware, a *ROCm build* of torch is required: ROCm builds expose
  the AMD GPUs through the ``cuda`` API (``torch.cuda.is_available()`` is
  true, ``torch.version.hip`` is set).  A CUDA-built torch simply reports no
  device and the backend runs on CPU.

Dtype / device handling
-----------------------
Inputs (numpy arrays or torch tensors) are converted to float64 tensors on
the target device -- the float64 CDF-t kernels then produce results that
match the numpy backend to the last significant digit for the same data
(see ``tests/test_gpu.py`` and ``docs/gpu.md`` for the verified tolerance).
Outputs are numpy float64 arrays by default (``as_numpy=True``), i.e. a
drop-in replacement for the numpy API; ``as_numpy=False`` returns torch
tensors on the compute device.

Memory notes
------------
The kernels are fully vectorized: peak GPU memory for a field is
``O((T + Q) * N * 8 bytes)`` (plus one sorted copy of the input), i.e. the
whole field is processed at once, unlike the chunked numpy core.  For very
large fields, pass ``chunk_cols`` to process the cells in column chunks
(identical results, bounded memory, e.g. ``chunk_cols=4096``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._torch import require_torch
from .devices import resolve_device
from . import kernels as _k

if TYPE_CHECKING:                        # annotations only, never runtime
    import torch


__all__ = [
    "Calibration",
    "SHIFT_MODES",
    "delta_from_means",
    "apply_shift",
    "calibrate",
    "calibrate_hist",
    "histograms",
    "transfer",
    "apply_cdft",
    "apply_cdft_hist",
    "CDFt",
    "cdft_transform",
]

SHIFT_MODES = ("additive", "multiplicative", "none")
_TAILS = ("linear", "clip")


def _as_matrix(x, device, name: str = "array"):
    """(T, ...) / (T,) input -> ((T, N) float64 tensor on ``device``, shape).

    Accepts numpy arrays (converted, never shared) and torch tensors (moved
    to ``device`` and converted to float64).
    """
    torch = require_torch()
    if isinstance(x, torch.Tensor):
        t = x.to(device=device, dtype=torch.float64)
    else:
        t = torch.as_tensor(x, device=device, dtype=torch.float64)
    if t.ndim < 1:
        raise ValueError(f"{name} must be at least 1-D, got shape {tuple(t.shape)}")
    shape = tuple(t.shape)
    return t.reshape(t.shape[0], -1), shape


def _as_1d(x, device, name: str = "array"):
    """Convert a 1-D array / tensor (numpy or torch) to a device tensor."""
    torch = require_torch()
    if isinstance(x, torch.Tensor):
        t = x.to(device=device, dtype=torch.float64)
    else:
        t = torch.as_tensor(x, device=device, dtype=torch.float64)
    if t.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {tuple(t.shape)}")
    return t


def _check_grid(p, n_quantiles: int = 1001, device=None):
    """Validate / default the probability grid (tensor on ``device``)."""
    torch = require_torch()
    if p is None:
        return _k.quantile_grid(n_quantiles, device=device)
    p = _as_1d(p, device, "p")
    if p.size(0) < 2:
        raise ValueError(f"p must be a 1-D grid with >= 2 points, got {p.size(0)}")
    if not bool(torch.all(torch.diff(p) > 0)):
        raise ValueError("p must be strictly increasing")
    return p


def _mean_axis0(a, nv):
    """Per-column mean of the finite values (NaN where a column is empty)."""
    torch = require_torch()
    s = torch.where(torch.isfinite(a), a, torch.zeros_like(a)).sum(dim=0)
    return torch.where(nv > 0, s / nv.clamp(min=1.0), torch.full_like(s, float("nan")))


def _cell_chunks(n: int, chunk_cols=None):
    """Column-chunk bounds [(s, e), ...] (single chunk when None)."""
    if chunk_cols is None or chunk_cols <= 0:
        return [(0, n)]
    chunk = max(1, int(chunk_cols))
    return [(s, min(s + chunk, n)) for s in range(0, n, chunk)]


# ---------------------------------------------------------------------------
# Delta shift
# ---------------------------------------------------------------------------

def delta_from_means(mean_lr, mean_hr, mode: str):
    """Per-cell shift ``d`` from the calibration means (float64 tensors).

    ``mean(HR_cal) - mean(LR_cal)`` (additive) or
    ``mean(HR_cal) / mean(LR_cal)`` (multiplicative); ``0`` for ``"none"``.
    Multiplicative falls back to ``1.0`` where the LR mean is ~0, the ratio
    is non-finite or non-positive (signed variables).  Identical to
    :func:`pycdft.core.delta_from_means`.
    """
    torch = require_torch()
    if mode == "additive":
        return mean_hr - mean_lr
    if mode == "multiplicative":
        ok = mean_lr.abs() > 1e-12
        d = torch.where(ok, mean_hr / torch.where(ok, mean_lr,
                                                  torch.ones_like(mean_lr)),
                        torch.ones_like(mean_lr))
        return torch.where(torch.isfinite(d) & (d > 0), d, torch.ones_like(d))
    return torch.zeros_like(mean_lr)


def apply_shift(x, d, mode: str):
    """Shift a (T, N) tensor by the per-cell delta (additive/multiplicative)."""
    if mode == "additive":
        return x + d[None, :]
    if mode == "multiplicative":
        return x * d[None, :]
    return x


# ---------------------------------------------------------------------------
# Calibration object
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    """The target-independent CDF-t calibration (torch tensors).

    Attributes
    ----------
    R : (Q, N) float64 tensor
        The calibration mapping ``R_k = F_m^cal(q_o^cal(p_k))``.
    delta : (N,) float64 tensor
        Per-cell shift (``mean(obs) - mean(cm_hist)`` for additive,
        ``mean(obs) / mean(cm_hist)`` for multiplicative, ``0`` for none).
    mode : str
        One of ``{"additive", "multiplicative", "none"}``.
    n_quantiles : int
        Size of the probability grid ``p``.
    """

    R: torch.Tensor
    delta: torch.Tensor
    mode: str
    n_quantiles: int

    def to(self, device):
        """Return a copy of the calibration on another device."""
        return Calibration(self.R.to(device), self.delta.to(device),
                           self.mode, self.n_quantiles)

    def cpu(self):
        """Return a copy of the calibration on CPU."""
        return self.to("cpu")

    def numpy(self):
        """Convert to a numpy-backed :class:`pycdft.core.Calibration`."""
        from ..core import Calibration as NumpyCalibration
        return NumpyCalibration(
            R=self.R.detach().cpu().numpy(),
            delta=self.delta.detach().cpu().numpy(),
            mode=self.mode, n_quantiles=self.n_quantiles)


# ---------------------------------------------------------------------------
# Exact calibration
# ---------------------------------------------------------------------------

def calibrate(lr_cal, hr_cal, p=None, delta_shift: str = "additive",
              device=None, chunk_cols=None, n_quantiles: int = 1001):
    """Build the CDF-t calibration ``R = F_m^cal(q_o^cal(p))`` + per-cell delta.

    Torch equivalent of :func:`pycdft.core.calibrate` (same maths, same
    conventions): per-column type-7 quantiles of the observation
    calibration series, per-column step ECDF of the shifted model series.
    Inputs may be numpy arrays or torch tensors.

    Parameters
    ----------
    lr_cal, hr_cal : (T, N) or (T,) array / tensor
        Model (coarse) / observation (fine) series over the calibration
        period (same T).
    p : (Q,) array / tensor, optional probability grid (default
        ``quantile_grid(1001)``).
    delta_shift : {"additive", "multiplicative", "none"}.
    device : str, torch.device or None
        Target device (default: best available, see
        :func:`pycdft.gpu.resolve_device`).
    chunk_cols : int, optional
        Process the cells in column chunks to bound the peak memory (the
        results are identical).

    Returns
    -------
    Calibration
    """
    torch = require_torch()
    if delta_shift not in SHIFT_MODES:
        raise ValueError(f"delta_shift must be one of {SHIFT_MODES}, "
                         f"got {delta_shift!r}")
    device = resolve_device(device)
    p = _check_grid(p, n_quantiles, device)
    lr, _ = _as_matrix(lr_cal, device, "lr_cal")
    hr, _ = _as_matrix(hr_cal, device, "hr_cal")
    if lr.shape[0] != hr.shape[0]:
        raise ValueError(
            f"calibration series lengths differ: {lr.shape[0]} vs "
            f"{hr.shape[0]}")
    N = lr.shape[1]
    nv_lr = torch.isfinite(lr).sum(dim=0).to(torch.float64)
    nv_hr = torch.isfinite(hr).sum(dim=0).to(torch.float64)
    mean_lr = _mean_axis0(lr, nv_lr)
    mean_hr = _mean_axis0(hr, nv_hr)
    delta = delta_from_means(mean_lr, mean_hr, delta_shift)
    lr_s, _ = torch.sort(lr, dim=0)       # NaNs to the end, private copy
    hr_s, _ = torch.sort(hr, dim=0)
    Q = p.size(0)
    R = torch.empty((Q, N), dtype=torch.float64, device=device)
    for s, e in _cell_chunks(N, chunk_cols):
        q_hr = _k.quantile_from_sorted(hr_s[:, s:e], p)         # (Q, chunk)
        lc = apply_shift(lr_s[:, s:e], delta[s:e], delta_shift)
        R[:, s:e] = _k._ecdf_at_sorted(lc, q_hr)
    return Calibration(R=R, delta=delta, mode=delta_shift, n_quantiles=Q)


# ---------------------------------------------------------------------------
# Histogram calibration
# ---------------------------------------------------------------------------

def histograms(values, n_bins: int = 2000, device=None):
    """Per-cell histograms with per-cell edges (in-memory, 2 passes).

    Torch equivalent of :func:`pycdft.core.histograms`: returns
    ``((B + 1, N) edges, (B, N) counts)``; NaNs excluded from the binning,
    degenerate (constant) cells get a +/-1 padding, empty (all-NaN) cells
    get zero counts.
    """
    torch = require_torch()
    values, _ = _as_matrix(values, resolve_device(device), "values")
    n_bins = int(n_bins)
    valid = torch.isfinite(values)
    nv = valid.sum(dim=0)
    lo = torch.where(valid, values, torch.full_like(values, float("inf"))).min(dim=0).values
    hi = torch.where(valid, values, torch.full_like(values, float("-inf"))).max(dim=0).values
    degenerate = hi <= lo
    lo = torch.where(degenerate, lo - 1.0, lo)
    hi = torch.where(degenerate, hi + 1.0, hi)
    empty = nv == 0
    lo = torch.where(empty, torch.zeros_like(lo), lo)
    hi = torch.where(empty, torch.ones_like(hi), hi)
    edges = (torch.arange(n_bins + 1, dtype=torch.float64,
                          device=values.device)[:, None]
             * ((hi - lo)[None, :] / n_bins)) + lo[None, :]
    width = (hi - lo)[None, :]
    idx = (((values - lo[None, :])
            / torch.where(width > 0, width, torch.ones_like(width))
            * n_bins).to(torch.int64)).clamp(0, n_bins - 1)
    idx = torch.where(valid, idx, torch.full_like(idx, n_bins))  # sentinel bin
    N = values.shape[1]
    counts = torch.zeros((n_bins + 1, N), dtype=torch.float64,
                         device=values.device)
    counts.scatter_add_(0, idx, torch.ones_like(values, dtype=torch.float64))
    return edges, counts[:-1]


def calibrate_hist(hist_lr, hist_hr, means, p=None, delta_shift: str = "additive",
                   device=None, n_quantiles: int = 1001):
    """Histogram calibration (same :class:`Calibration` object as ``calibrate``).

    Torch equivalent of :func:`pycdft.core.calibrate_hist`; ``means`` is the
    ``(mean_lr, mean_hr)`` pair of per-cell float64 means.
    """
    torch = require_torch()
    if delta_shift not in SHIFT_MODES:
        raise ValueError(f"delta_shift must be one of {SHIFT_MODES}, "
                         f"got {delta_shift!r}")
    device = resolve_device(device)
    p = _check_grid(p, n_quantiles, device)
    (edges_lr, counts_lr), (edges_hr, counts_hr) = hist_lr, hist_hr
    mean_lr, mean_hr = means
    mean_lr = _as_1d(mean_lr, device, "mean_lr")
    mean_hr = _as_1d(mean_hr, device, "mean_hr")
    if mean_lr.size(0) == 1 and counts_hr.shape[1] != 1:
        mean_lr = mean_lr.expand(counts_hr.shape[1])
        mean_hr = mean_hr.expand(counts_hr.shape[1])
    delta = delta_from_means(mean_lr, mean_hr, delta_shift)
    p_grid = p[:, None].expand(p.size(0), counts_hr.shape[1])
    q_hr = _k.hist_quantile_pc(edges_hr, counts_hr, p_grid)
    if delta_shift == "additive":
        q_unshift = q_hr - delta[None, :]
    elif delta_shift == "multiplicative":
        q_unshift = q_hr / torch.where(delta[None, :] != 0, delta[None, :],
                                       torch.ones_like(delta[None, :]))
    else:
        q_unshift = q_hr
    R = _k.ecdf_hist(edges_lr, counts_lr, q_unshift)
    return Calibration(R=R, delta=delta, mode=delta_shift, n_quantiles=p.size(0))


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------

def transfer(q_tar, R, p, chunk_cols=None):
    """Build the transfer curve on the target value grid.

    Torch equivalent of :func:`pycdft.core.transfer`: ``q_tar(q, s, e) ->
    (Q, e - s)`` is the quantile function of the (shifted) target series for
    the cell columns ``[s, e)``.  Returns ``(x_grid, T, k_lo, k_hi, bad)``.
    """
    torch = require_torch()
    R = R.to(torch.float64)
    p = p.to(torch.float64)
    Q, N = R.shape
    if p.size(0) != Q:
        raise ValueError(f"p size {p.size(0)} != R rows {Q}")
    bad = ~torch.isfinite(R).all(dim=0)   # NaN calibration cells
    R = torch.where(bad[None, :], torch.full_like(R, 0.5), R)
    x_grid = torch.empty((Q, N), dtype=torch.float64, device=R.device)
    T = torch.empty((Q, N), dtype=torch.float64, device=R.device)
    k_lo = torch.empty(N, dtype=torch.int64, device=R.device)
    k_hi = torch.empty(N, dtype=torch.int64, device=R.device)
    for s, e in _cell_chunks(N, chunk_cols):
        T[:, s:e] = q_tar(R[:, s:e], s, e)
        x_grid[:, s:e] = q_tar(p, s, e)
        k_lo[s:e] = torch.argmax((R[:, s:e] > 0).to(torch.int64), dim=0)
        k_hi[s:e] = Q - 1 - torch.argmax((R[:, s:e] < 1).flip(0).to(torch.int64),
                                         dim=0)
    return x_grid, T, k_lo, k_hi, bad


# ---------------------------------------------------------------------------
# Apply (target side)
# ---------------------------------------------------------------------------

def apply_cdft(lr_tar, calib: Calibration, p=None, tail: str = "linear",
               device=None, chunk_cols=None, as_numpy: bool = True,
               n_quantiles: int = 1001):
    """Apply a precomputed calibration to the target model series.

    Torch equivalent of :func:`pycdft.core.apply_cdft` (same maths, same
    conventions, same shape semantics: the output matches ``lr_tar`` and is
    in the shifted frame).  Inputs may be numpy arrays or torch tensors.

    Parameters
    ----------
    lr_tar : (T, N) or (T, ...) array / tensor
        Model series to downscale / bias-correct.
    calib : Calibration
        From :func:`calibrate` or :func:`calibrate_hist`.
    p : (Q,) array / tensor, optional probability grid.
    tail : {"linear", "clip"}.
    device : str, torch.device or None
        Target device for the computation (default: the calibration's
        device).
    chunk_cols : int, optional
        Process the cells in column chunks to bound the peak memory.
    as_numpy : bool, default True
        Return a numpy float64 array (as the numpy backend); ``False``
        returns a torch tensor on the compute device.

    Returns
    -------
    Array / tensor of the same shape as ``lr_tar`` (NaN inputs map to NaN;
    cells with an empty calibration map entirely to NaN).
    """
    torch = require_torch()
    if tail not in _TAILS:
        raise ValueError(f"tail must be 'linear' or 'clip', got {tail!r}")
    if device is None:
        device = calib.R.device
    else:
        device = resolve_device(device)
    if p is None:
        p = _k.quantile_grid(calib.R.shape[0], device=device)
    else:
        p = _check_grid(p, n_quantiles, device)
        if p.size(0) != calib.R.shape[0]:
            raise ValueError(
                f"probability grid size {p.size(0)} does not match the "
                f"calibration ({calib.R.shape[0]}); pass the same grid used "
                f"in calibrate()")
    x, shape = _as_matrix(lr_tar, device, "lr_tar")
    xs = apply_shift(x, calib.delta, calib.mode)
    v_tar, _ = torch.sort(xs, dim=0)      # NaNs to the end
    x_grid, T, k_lo, k_hi, bad = transfer(
        lambda q, s, e: _k.quantile_from_sorted(v_tar[:, s:e], q),
        calib.R, p, chunk_cols)
    pred = _k.interp_transfer(x_grid, T, k_lo, k_hi, xs, tail=tail)
    if bool(bad.any()):
        pred = pred.masked_fill(bad[None, :], float("nan"))
    pred = pred.reshape(shape)
    if as_numpy:
        return pred.detach().cpu().numpy()
    return pred


def apply_cdft_hist(lr_tar, calib: Calibration, p=None, n_bins: int = 2000,
                    tail: str = "linear", device=None, as_numpy: bool = True,
                    n_quantiles: int = 1001):
    """Same as :func:`apply_cdft`, with the target CDF accumulated as a
    per-cell histogram (bounded memory for very long records).

    Torch equivalent of :func:`pycdft.core.apply_cdft_hist` (approximate,
    exact in the limit of many bins).
    """
    if tail not in _TAILS:
        raise ValueError(f"tail must be 'linear' or 'clip', got {tail!r}")
    if device is None:
        device = calib.R.device
    else:
        device = resolve_device(device)
    if p is None:
        p = _k.quantile_grid(calib.R.shape[0], device=device)
    else:
        p = _check_grid(p, n_quantiles, device)
        if p.size(0) != calib.R.shape[0]:
            raise ValueError(
                f"probability grid size {p.size(0)} does not match the "
                f"calibration ({calib.R.shape[0]}); pass the same grid used "
                f"in calibrate()")
    x, shape = _as_matrix(lr_tar, device, "lr_tar")
    xs = apply_shift(x, calib.delta, calib.mode)
    edges, counts = _k.histogram_accumulate(xs, n_bins=n_bins)
    x_grid, T, k_lo, k_hi, bad = transfer(
        lambda q, s, e: _k.hist_quantile(edges, counts[:, s:e], q),
        calib.R, p, None)
    pred = _k.interp_transfer(x_grid, T, k_lo, k_hi, xs, tail=tail)
    if bool(bad.any()):
        pred = pred.masked_fill(bad[None, :], float("nan"))
    pred = pred.reshape(shape)
    if as_numpy:
        return pred.detach().cpu().numpy()
    return pred


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

class CDFt:
    """CDF-t downscaling / bias correction on a torch device (GPU backend).

    Mirrors :class:`pycdft.CDFt` (same parameters, same ``fit`` /
    ``transform`` / ``fit_transform`` interface, numpy in / numpy out by
    default) but runs the per-cell kernels with torch on the device selected
    by :func:`pycdft.gpu.resolve_device`.

    Parameters
    ----------
    n_quantiles : int, default 1001
        Number of points of the uniform probability grid ``p`` (endpoints 0
        and 1 included).
    tail : {"linear", "clip"}, default "linear"
        Behaviour outside the saturated region of the transfer.
    delta_shift : {"additive", "multiplicative", "none"}, default "additive"
        Per-cell mean shift applied to the model series.
    use_histograms : bool, default False
        Accumulate the target CDF as a per-cell histogram instead of an
        exact sort (approximate; bounded memory for very long records).
    n_bins : int, default 2000
        Number of bins per cell in histogram mode.
    device : str, torch.device or None, default None
        Compute device (default: best available -- ``cuda`` -> ``mps`` ->
        ``cpu``; an explicit unavailable device raises ``ValueError``).
    chunk_cols : int, optional
        Process the cells in column chunks to bound the peak memory.
    as_numpy : bool, default True
        Return numpy float64 arrays from ``transform`` /
        ``fit_transform`` (``False`` returns torch tensors on the compute
        device).

    Attributes
    ----------
    calibration_ : Calibration or None
        The calibration from the last :meth:`fit` / :meth:`fit_transform`.
    device : torch.device
        The resolved compute device.

    Notes
    -----
    NaN handling is identical to the numpy backend: NaNs are excluded from
    the ECDF counts and the quantile interpolation, NaN inputs pass through
    as NaN, and cells whose calibration is empty map entirely to NaN.  The
    math is float64 on the device: for the same input data the results match
    :class:`pycdft.CDFt` to the last significant digit (see
    ``tests/test_gpu.py``).
    """

    def __init__(self, n_quantiles: int = 1001, tail: str = "linear",
                 delta_shift: str = "additive", use_histograms: bool = False,
                 n_bins: int = 2000, device=None, chunk_cols=None,
                 as_numpy: bool = True):
        n_quantiles = int(n_quantiles)
        if n_quantiles < 2:
            raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
        if delta_shift not in SHIFT_MODES:
            raise ValueError(f"delta_shift must be one of {SHIFT_MODES}, "
                             f"got {delta_shift!r}")
        if tail not in _TAILS:
            raise ValueError(f"tail must be one of {_TAILS}, got {tail!r}")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        self.n_quantiles = n_quantiles
        self.tail = tail
        self.delta_shift = delta_shift
        self.use_histograms = bool(use_histograms)
        self.n_bins = int(n_bins)
        self.chunk_cols = chunk_cols
        self.as_numpy = bool(as_numpy)
        self.device = resolve_device(device)
        self.calibration_ = None

    # ------------------------------------------------------------------
    @property
    def probability_grid(self) -> np.ndarray:
        """The uniform probability grid ``linspace(0, 1, n_quantiles)``
        (numpy, identical to the numpy backend's grid)."""
        return np.linspace(0.0, 1.0, self.n_quantiles)

    # ------------------------------------------------------------------
    def to(self, device):
        """Move the fitted calibration to another device and return self."""
        if self.calibration_ is not None:
            self.calibration_ = self.calibration_.to(resolve_device(device))
        self.device = resolve_device(device)
        return self

    # ------------------------------------------------------------------
    def fit(self, obs, cm_hist) -> "CDFt":
        """Fit the calibration on ``(obs, cm_hist)`` (numpy arrays or
        torch tensors; returns self)."""
        self.calibration_ = calibrate(cm_hist, obs,
                                      p=_k.quantile_grid(self.n_quantiles,
                                                         device=self.device),
                                      delta_shift=self.delta_shift,
                                      device=self.device,
                                      chunk_cols=self.chunk_cols)
        return self

    def transform(self, cm_future):
        """Apply the fitted calibration to the target model series.

        Returns an array of the same shape as ``cm_future`` (numpy float64
        by default, or a tensor on the compute device with
        ``as_numpy=False``).
        """
        if self.calibration_ is None:
            raise RuntimeError("CDFt is not fitted: call fit() or "
                               "fit_transform() first")
        if self.use_histograms:
            return apply_cdft_hist(cm_future, self.calibration_,
                                   p=_k.quantile_grid(self.n_quantiles,
                                                      device=self.device),
                                   n_bins=self.n_bins, tail=self.tail,
                                   device=self.device, as_numpy=self.as_numpy)
        return apply_cdft(cm_future, self.calibration_,
                          p=_k.quantile_grid(self.n_quantiles,
                                             device=self.device),
                          tail=self.tail, device=self.device,
                          chunk_cols=self.chunk_cols, as_numpy=self.as_numpy)

    def fit_transform(self, obs, cm_hist, cm_future):
        """Fit on ``(obs, cm_hist)`` and transform ``cm_future`` in one call."""
        return self.fit(obs, cm_hist).transform(cm_future)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"CDFt(n_quantiles={self.n_quantiles}, tail={self.tail!r}, "
                f"delta_shift={self.delta_shift!r}, "
                f"use_histograms={self.use_histograms}, "
                f"n_bins={self.n_bins}, device={self.device!s})")


def cdft_transform(obs, cm_hist, cm_future, n_quantiles: int = 1001,
                   tail: str = "linear", delta_shift: str = "additive",
                   use_histograms: bool = False, n_bins: int = 2000,
                   device=None, chunk_cols=None, as_numpy: bool = True):
    """One-shot CDF-t transform (no persistent calibration object).

    Equivalent to ``CDFt(...).fit_transform(obs, cm_hist, cm_future)``;
    returns the same-shaped output array.
    """
    return CDFt(n_quantiles=n_quantiles, tail=tail, delta_shift=delta_shift,
                use_histograms=use_histograms, n_bins=n_bins, device=device,
                chunk_cols=chunk_cols, as_numpy=as_numpy) \
        .fit_transform(obs, cm_hist, cm_future)
