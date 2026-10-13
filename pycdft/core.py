"""CDF-t calibration objects and the transfer pipeline (numpy only).

The CDF-t composition (Michelangeli, Vrac & Loukos 2009)

    F_o^tar = F_o^cal o (F_m^cal)^{-1} o F_m^tar
    y(x)    = q_m^tar( F_m^cal( q_o^cal( F_m^tar(x) ) ) )

is computed per series (per grid cell) on a uniform probability grid
``p = linspace(0, 1, Q)`` (default Q = 1001, endpoints included), with:

- step ECDFs (``F(x) = #{finite <= x} / n``, NaN-excluded, right-continuous);
- type-7 linear quantiles (numpy/R convention);
- an optional per-series (per-cell) mean delta shift applied to the model
  series (calibration and target) before the transfer -- the standard CDF-t
  practice (R ``CDFt`` package, ibicus);
- the transfer curve ``(x_grid, T) = (q_m^tar(p), q_m^tar(R))`` evaluated on
  the target's own value grid, interpolated per cell in x-space, with the
  ``clip`` (constant) or ``linear`` (windowed least-squares extrapolation,
  slope capped at 2x the median interior slope) tail options;
- NaN robustness: NaNs never enter the counts or the interpolation; cells
  whose calibration is empty (all-NaN) give NaN output.

Pipeline::

    calib = calibrate(cm_hist, obs, p, delta_shift)      # (R, delta, mode)
    pred  = apply_cdft(cm_future, calib, p, tail)        # same shape

``calibrate`` keeps the calibration arrays in the caller's dtype (float32
storage with float64 per-chunk math is exact: float32 values convert exactly)
and processes cell chunks so that the ``(Q, chunk)`` float64 intermediates
stay below ``chunk_bytes`` (default 2 GiB) -- the memory-lean design of the
reference grid-scale implementation.  The result is identical to a pure
float64 computation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np

from . import math as _m

__all__ = [
    "Calibration",
    "DEFAULT_CHUNK_BYTES",
    "SHIFT_MODES",
    "delta_from_means",
    "apply_shift",
    "calibrate",
    "calibrate_hist",
    "histograms",
    "transfer",
    "apply_cdft",
    "apply_cdft_hist",
]

DEFAULT_CHUNK_BYTES = 2 ** 31          # 2 GiB per (Q, chunk) float64 block
SHIFT_MODES = ("additive", "multiplicative", "none")


@dataclass(frozen=True)
class Calibration:
    """The target-independent CDF-t calibration.

    Attributes
    ----------
    R : (Q, N) float64
        The calibration mapping ``R_k = F_m^cal(q_o^cal(p_k))`` (model
        ECDF of the obs calibration quantiles; the model series in the
        shifted frame).
    delta : (N,) float64
        Per-cell shift (``mean(obs) - mean(cm_hist)`` for additive,
        ``mean(obs) / mean(cm_hist)`` for multiplicative, ``0`` for none).
    mode : str
        One of ``{"additive", "multiplicative", "none"}``.
    n_quantiles : int
        Size of the probability grid ``p``.
    """

    R: np.ndarray
    delta: np.ndarray
    mode: str
    n_quantiles: int


# ---------------------------------------------------------------------------
# Delta shift
# ---------------------------------------------------------------------------

def delta_from_means(mean_lr: np.ndarray, mean_hr: np.ndarray,
                     mode: str) -> np.ndarray:
    """Per-cell shift ``d`` from the calibration means (float64 arrays).

    ``mean(HR_cal) - mean(LR_cal)`` (additive) or
    ``mean(HR_cal) / mean(LR_cal)`` (multiplicative); ``0`` for ``"none"``.
    Multiplicative falls back to ``1.0`` where the LR mean is ~0, the ratio
    is non-finite or non-positive (signed variables).
    """
    if mode == "additive":
        return mean_hr - mean_lr
    if mode == "multiplicative":
        ok = np.abs(mean_lr) > 1e-12
        d = np.where(ok, mean_hr / np.where(ok, mean_lr, 1.0), 1.0)
        return np.where(np.isfinite(d) & (d > 0), d, 1.0)
    return np.zeros_like(mean_lr)


def apply_shift(x, d, mode: str) -> np.ndarray:
    """Shift a (T, N) array by the per-cell delta (additive/multiplicative)."""
    x = np.asarray(x, np.float64)
    d = np.asarray(d, np.float64)
    if mode == "additive":
        return x + d[None, :]
    if mode == "multiplicative":
        return x * d[None, :]
    return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_matrix(x, name: str = "array") -> Tuple[np.ndarray, Tuple[int, ...]]:
    """(T, ...) or (T,) input -> ((T, N) float64 matrix, original shape)."""
    a = np.asarray(x, np.float64)
    if a.ndim < 1:
        raise ValueError(f"{name} must be at least 1-D, got shape {a.shape}")
    return a.reshape(a.shape[0], -1), a.shape


def _mean_axis0(a: np.ndarray, nv: np.ndarray) -> np.ndarray:
    """Per-column mean of the finite values (NaN where a column is empty)."""
    with np.errstate(invalid="ignore"):
        s = np.where(np.isfinite(a), a, 0.0).sum(axis=0, dtype=np.float64)
    return np.where(nv > 0, s / np.maximum(nv, 1), np.nan)


def _cell_chunks(n: int, t_or_q: int,
                 chunk_bytes: int = DEFAULT_CHUNK_BYTES):
    """Column-chunk bounds [(s, e), ...] keeping the (Q, chunk) float64
    intermediates below ``chunk_bytes``."""
    chunk = max(1, min(n, chunk_bytes // (8 * max(t_or_q, 1))))
    return [(s, min(s + chunk, n)) for s in range(0, n, chunk)]


def _check_grid(p, n_quantiles: int = 1001) -> np.ndarray:
    """Validate / default the probability grid."""
    if p is None:
        return _m.quantile_grid(n_quantiles)
    p = np.asarray(p, np.float64)
    if p.ndim != 1 or p.size < 2:
        raise ValueError(f"p must be a 1-D grid with >= 2 points, got {p.shape}")
    if not np.all(np.diff(p) > 0):
        raise ValueError("p must be strictly increasing")
    return p


# ---------------------------------------------------------------------------
# Exact calibration
# ---------------------------------------------------------------------------

def calibrate(lr_cal, hr_cal, p=None, delta_shift: str = "additive",
              workers: int = 1, chunk_bytes: int = DEFAULT_CHUNK_BYTES,
              n_quantiles: int = 1001) -> Calibration:
    """Build the CDF-t calibration ``R = F_m^cal(q_o^cal(p))`` + per-cell delta.

    Parameters
    ----------
    lr_cal : (T, N) or (T,) array
        Model (coarse) series over the calibration period.
    hr_cal : (T, N) or (T,) array
        Observation (fine) series over the calibration period (same T).
    p : (Q,) array, optional
        Probability grid (default ``quantile_grid(1001)``, endpoints 0 and 1).
    delta_shift : {"additive", "multiplicative", "none"}
        Per-cell mean shift applied to the model series before the transfer.
    workers : int
        Threads for the per-cell kernels.
    chunk_bytes : int
        Memory budget for the (Q, chunk) float64 intermediates.
    n_quantiles : int
        Default grid size when ``p`` is None.

    Returns
    -------
    Calibration
        ``R`` (Q, N), ``delta`` (N,), ``mode``, ``n_quantiles``.  Cells whose
        calibration is empty (all-NaN in either series) get ``R = NaN`` and
        produce NaN output downstream.

    Notes
    -----
    The arrays are sorted in place on a private copy (the caller's data is
    never modified); the sort keeps the input dtype (float32 stays float32,
    the per-chunk math is float64 and exact -- bit-identical to sorting
    float64 first), and all per-cell math runs on cell chunks so that the
    peak memory is ``O((T + Q) * chunk_bytes / 8)``, not ``O(T * N)``
    float64.
    """
    if delta_shift not in SHIFT_MODES:
        raise ValueError(f"delta_shift must be one of {SHIFT_MODES}, "
                         f"got {delta_shift!r}")
    p = _check_grid(p, n_quantiles)
    lr = np.asarray(lr_cal)
    hr = np.asarray(hr_cal)
    if lr.ndim == 1:
        lr = lr[:, None]
    if hr.ndim == 1:
        hr = hr[:, None]
    if lr.ndim < 2 or hr.ndim < 2:
        raise ValueError("lr_cal/hr_cal must be 1-D or (T, ...) arrays")
    if lr.shape[0] != hr.shape[0]:
        raise ValueError(
            f"calibration series lengths differ: {lr.shape[0]} vs "
            f"{hr.shape[0]}")
    # private copies: np.sort is done in place and must not touch the
    # caller's arrays; the input dtype is kept (float32 stays float32 ->
    # memory-lean sort; the per-chunk math is float64 and exact)
    lr = np.array(lr, copy=True).reshape(lr.shape[0], -1)
    hr = np.array(hr, copy=True).reshape(hr.shape[0], -1)
    N = lr.shape[1]
    nv_lr = np.isfinite(lr).sum(axis=0).astype(np.float64)
    nv_hr = np.isfinite(hr).sum(axis=0).astype(np.float64)
    mean_lr = _mean_axis0(lr, nv_lr)
    mean_hr = _mean_axis0(hr, nv_hr)
    delta = delta_from_means(mean_lr, mean_hr, delta_shift)
    lr.sort(axis=0)                      # NaNs to the end, excluded by counts
    hr.sort(axis=0)
    Q = p.size
    R = np.empty((Q, N), dtype=np.float64)
    for s, e in _cell_chunks(N, max(lr.shape[0], Q), chunk_bytes):
        # per-chunk float64 math (bit-identical to a full float64 run)
        q_hr = _m.quantile_from_sorted(
            np.asarray(hr[:, s:e], np.float64), p)               # (Q, chunk)
        lc = np.asarray(lr[:, s:e], np.float64)                  # (T, chunk)
        if delta_shift == "additive":
            lc = lc + delta[None, s:e]
        elif delta_shift == "multiplicative":
            lc = lc * delta[None, s:e]
        R[:, s:e] = _m._ecdf_at_sorted(lc, q_hr)
    return Calibration(R=R, delta=delta, mode=delta_shift, n_quantiles=Q)


# ---------------------------------------------------------------------------
# Histogram calibration
# ---------------------------------------------------------------------------

def histograms(values, n_bins: int = 2000):
    """Per-cell histograms with per-cell edges (in-memory, 2 passes).

    Parameters
    ----------
    values : (T, N) array.
    n_bins : int, number of bins per cell.

    Returns
    -------
    ((B + 1, N) edges, (B, N) counts): NaNs excluded from the binning;
    degenerate (constant) cells get a ±1 padding, empty (all-NaN) cells get
    zero counts (and NaN edges internally, quantiles -> NaN).
    """
    values = np.asarray(values, np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D (T, N) array, got {values.shape}")
    valid = np.isfinite(values)
    nv = valid.sum(axis=0)
    with np.errstate(invalid="ignore"):
        lo = np.where(valid, values, np.inf).min(axis=0)
        hi = np.where(valid, values, -np.inf).max(axis=0)
    degenerate = hi <= lo
    lo = np.where(degenerate, lo - 1.0, lo)
    hi = np.where(degenerate, hi + 1.0, hi)
    empty = nv == 0
    lo = np.where(empty, 0.0, lo)
    hi = np.where(empty, 1.0, hi)
    edges = np.linspace(lo, hi, n_bins + 1)                      # (B+1, N)
    width = (hi - lo)[None, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        idx = np.clip(((values - lo[None, :])
                       / np.where(width > 0, width, 1.0)
                       * n_bins).astype(np.int64), 0, n_bins - 1)
    idx = np.where(valid, idx, n_bins)                           # sentinel bin
    stride = n_bins + 1
    N = values.shape[1]
    counts = np.zeros((n_bins + 1, N), dtype=np.float64)
    flat = idx + (np.arange(N) * stride)[None]
    counts += np.bincount(flat.ravel(), minlength=stride * N) \
        .reshape(N, stride).T
    return edges, counts[:-1]


def calibrate_hist(hist_lr, hist_hr, means, p=None, delta_shift: str = "additive",
                   workers: int = 1, n_quantiles: int = 1001) -> Calibration:
    """Histogram calibration (same :class:`Calibration` object as ``calibrate``).

    Parameters
    ----------
    hist_lr, hist_hr : (edges, counts)
        Per-cell histograms of the model / observation calibration series
        (from :func:`histograms`; edges (B + 1, N), counts (B, N)).
    means : (mean_lr, mean_hr)
        Exact per-cell means (float64 (N,)) accumulated while streaming.
    p : (Q,) array, optional probability grid.
    delta_shift : {"additive", "multiplicative", "none"}.

    Returns
    -------
    Calibration
    """
    if delta_shift not in SHIFT_MODES:
        raise ValueError(f"delta_shift must be one of {SHIFT_MODES}, "
                         f"got {delta_shift!r}")
    p = _check_grid(p, n_quantiles)
    (edges_lr, counts_lr), (edges_hr, counts_hr) = hist_lr, hist_hr
    mean_lr, mean_hr = means
    mean_lr = np.atleast_1d(np.asarray(mean_lr, np.float64))
    mean_hr = np.atleast_1d(np.asarray(mean_hr, np.float64))
    if mean_lr.size == 1 and counts_hr.shape[1] != 1:
        mean_lr = np.full(counts_hr.shape[1], mean_lr[0])
        mean_hr = np.full(counts_hr.shape[1], mean_hr[0])
    delta = np.atleast_1d(delta_from_means(mean_lr, mean_hr, delta_shift))
    p_grid = np.broadcast_to(p[:, None], (p.size, counts_hr.shape[1]))
    q_hr = _m.hist_quantile_pc(edges_hr, counts_hr, p_grid, workers=workers)
    if delta_shift == "additive":
        q_unshift = q_hr - delta[None, :]
    elif delta_shift == "multiplicative":
        q_unshift = q_hr / np.where(delta[None, :] != 0, delta[None, :], 1.0)
    else:
        q_unshift = q_hr
    R = _m.ecdf_hist(edges_lr, counts_lr, q_unshift, workers=workers)
    return Calibration(R=R, delta=delta, mode=delta_shift, n_quantiles=p.size)


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------

def transfer(q_tar: Callable[[np.ndarray, int, int], np.ndarray], R, p,
             workers: int = 1, chunk_bytes: int = DEFAULT_CHUNK_BYTES):
    """Build the transfer curve on the target value grid.

    Parameters
    ----------
    q_tar : callable ``q_tar(q, s, e) -> (Q, e - s)``
        Quantile function of the (shifted) target series for cell columns
        ``[s, e)``, evaluated at the probabilities ``q`` (Q, chunk).
    R : (Q, N) calibration mapping (may contain NaN for empty cells).
    p : (Q,) probability grid.

    Returns
    -------
    (x_grid (Q, N), T (Q, N), k_lo (N,), k_hi (N,), bad (N,))
        ``x_grid = q_m^tar(p)``, ``T = q_m^tar(R)`` (target value grid and
        mapped values), the first / last unsaturated ranks of ``R`` and the
        flag of cells with a NaN calibration (R replaced by 0.5 in the
        lookup, output must be set to NaN downstream).
    """
    R = np.asarray(R, np.float64)
    p = np.asarray(p, np.float64)
    Q, N = R.shape
    if p.size != Q:
        raise ValueError(f"p size {p.size} != R rows {Q}")
    bad = ~np.isfinite(R).all(axis=0)                 # NaN calibration cells
    R = np.where(bad[None, :], 0.5, R)
    x_grid = np.empty((Q, N), dtype=np.float64)
    T = np.empty((Q, N), dtype=np.float64)
    k_lo = np.empty(N, dtype=np.int64)
    k_hi = np.empty(N, dtype=np.int64)
    for s, e in _cell_chunks(N, Q, chunk_bytes):
        T[:, s:e] = q_tar(R[:, s:e], s, e)
        x_grid[:, s:e] = q_tar(np.broadcast_to(p[:, None], (Q, e - s)), s, e)
        k_lo[s:e] = np.argmax(R[:, s:e] > 0, axis=0)
        k_hi[s:e] = Q - 1 - np.argmax((R[:, s:e] < 1)[::-1], axis=0)
    return x_grid, T, k_lo, k_hi, bad


# ---------------------------------------------------------------------------
# Apply (target side)
# ---------------------------------------------------------------------------

def apply_cdft(lr_tar, calib: Calibration, p=None, tail: str = "linear",
               workers: int = 1, chunk_bytes: int = DEFAULT_CHUNK_BYTES,
               n_quantiles: int = 1001):
    """Apply a precomputed calibration to the target model series.

    Parameters
    ----------
    lr_tar : (T, N) or (T, ...) array
        Model series to downscale / bias-correct (any shape; the first axis
        is time).
    calib : Calibration
        From :func:`calibrate` or :func:`calibrate_hist`.
    p : (Q,) array, optional probability grid (must match the calibration).
    tail : {"linear", "clip"}
    workers : int, threads for the per-cell kernels.
    chunk_bytes : int, memory budget for the (Q, chunk) intermediates.

    Returns
    -------
    Array of the same shape as ``lr_tar``, in the shifted frame (the mean
    bias model-vs-observation is removed, the model change between
    calibration and target is preserved).  NaN inputs map to NaN; cells with
    an empty (all-NaN) calibration map entirely to NaN.
    """
    if tail not in ("linear", "clip"):
        raise ValueError(f"tail must be 'linear' or 'clip', got {tail!r}")
    if p is None:
        p = _m.quantile_grid(calib.R.shape[0])
    else:
        p = _check_grid(p, n_quantiles)
        if p.size != calib.R.shape[0]:
            raise ValueError(
                f"probability grid size {p.size} does not match the "
                f"calibration ({calib.R.shape[0]}); pass the same grid used "
                f"in calibrate()")
    x, shape = _as_matrix(lr_tar, "lr_tar")
    delta, mode = calib.delta, calib.mode
    if mode == "additive":
        xs = x + delta[None, :]
    elif mode == "multiplicative":
        xs = x * delta[None, :]
    else:
        xs = x
    v_tar = np.sort(xs, axis=0)                      # NaNs to the end
    x_grid, T, k_lo, k_hi, bad = transfer(
        lambda q, s, e: _m.quantile_from_sorted(v_tar[:, s:e], q),
        calib.R, p, workers, chunk_bytes)
    pred = _m.interp_transfer(x_grid, T, k_lo, k_hi, xs, tail=tail,
                              workers=workers)
    if bad.any():
        pred[:, bad] = np.nan        # empty (NaN) calibration cells -> NaN
    return pred.reshape(shape)


def apply_cdft_hist(lr_tar, calib: Calibration, p=None, n_bins: int = 2000,
                    tail: str = "linear", workers: int = 1,
                    n_quantiles: int = 1001):
    """Same as :func:`apply_cdft`, with the target CDF accumulated as a
    per-cell histogram (bounded memory for very long records).

    Parameters
    ----------
    lr_tar : (T, N) or (T, ...) array
    calib : Calibration
    p : (Q,) array, optional probability grid.
    n_bins : int, bins for the target histogram (global edges).
    tail : {"linear", "clip"}
    workers : int

    Returns
    -------
    Array of the same shape as ``lr_tar`` (histogram approximation of
    :func:`apply_cdft`, exact in the limit of many bins).
    """
    if tail not in ("linear", "clip"):
        raise ValueError(f"tail must be 'linear' or 'clip', got {tail!r}")
    if p is None:
        p = _m.quantile_grid(calib.R.shape[0])
    else:
        p = _check_grid(p, n_quantiles)
        if p.size != calib.R.shape[0]:
            raise ValueError(
                f"probability grid size {p.size} does not match the "
                f"calibration ({calib.R.shape[0]}); pass the same grid used "
                f"in calibrate()")
    x, shape = _as_matrix(lr_tar, "lr_tar")
    delta, mode = calib.delta, calib.mode
    if mode == "additive":
        xs = x + delta[None, :]
    elif mode == "multiplicative":
        xs = x * delta[None, :]
    else:
        xs = x
    edges, counts = _m.histogram_accumulate(xs, n_bins=n_bins)
    x_grid, T, k_lo, k_hi, bad = transfer(
        lambda q, s, e: _m.hist_quantile(edges, counts[:, s:e], q,
                                         workers=workers),
        calib.R, p, workers)
    pred = _m.interp_transfer(x_grid, T, k_lo, k_hi, xs, tail=tail,
                              workers=workers)
    if bad.any():
        pred[:, bad] = np.nan
    return pred.reshape(shape)
