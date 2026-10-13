"""Low-level pure-numpy kernels for the CDF-t transform.

This module contains the numerical building blocks of the CDF-t method,
vectorized over an arbitrary number of independent series (columns):

- ECDF: right-continuous step function ``F(x) = #{finite values <= x} / n``
  (NaNs excluded from the count and sorted to the end);
- quantiles: numpy / R ``type-7`` linear interpolation
  (``np.quantile(..., method="linear")`` == R ``quantile(..., type=7)``),
  re-implemented bit-for-bit on pre-sorted arrays (including numpy's
  ``_lerp`` ``w >= 0.5`` branch), with the index grid scaled by the per-cell
  number of *valid* values so that NaNs can never enter the interpolation;
- the transfer interpolation (``np.interp`` per cell) with the two tail
  options (``clip`` / ``linear``);
- histogram-based ECDF / quantile variants.

All kernels accept 2-D arrays ``(T, N)`` (or ``(Q, N)``) of float64 and
operate independently per column (per series / per grid cell).  The 1-D
convenience wrappers (:func:`ecdf`, :func:`quantile`) accept ``(T,)`` input.

The conventions are identical to the reference grid-scale implementation the
package is derived from (a production CERRA-downscaling CDF-t baseline) and
to the R ``CDFt`` package with the same options.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

__all__ = [
    "quantile_grid",
    "searchsorted_columns",
    "ecdf",
    "ecdf_at",
    "quantile",
    "nanquantile",
    "quantile_from_sorted",
    "windowed_slope",
    "interp_transfer",
    "histogram_accumulate",
    "hist_quantile",
    "hist_quantile_pc",
    "ecdf_hist",
]


# ---------------------------------------------------------------------------
# Probability grid
# ---------------------------------------------------------------------------

def quantile_grid(n_quantiles: int = 1001) -> np.ndarray:
    """Uniform probability grid ``linspace(0, 1, n_quantiles)``.

    Parameters
    ----------
    n_quantiles : int
        Number of grid points (``>= 2``); the endpoints 0 and 1 are always
        included (the grid is the support of the transfer curve).

    Returns
    -------
    np.ndarray of shape ``(n_quantiles,)``.
    """
    n_quantiles = int(n_quantiles)
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
    return np.linspace(0.0, 1.0, n_quantiles)


# ---------------------------------------------------------------------------
# Batched per-column searchsorted
# ---------------------------------------------------------------------------

def searchsorted_columns(a, v, side="right", workers: int = 1) -> np.ndarray:
    """Per-column ``np.searchsorted`` on the (T, N) / (M, N) layouts.

    Equivalent to looping ``np.searchsorted(a[:, j], v[:, j], side=side)``
    over columns (identical results), optionally with a thread pool (numpy
    releases the GIL, so ``workers > 1`` scales the per-cell kernels).

    Parameters
    ----------
    a : (T, N) array, sorted along axis 0 (NaNs allowed, sorted last).
    v : (M, N) array of query points.
    side : {"left", "right"}, default "right" (``F(x) = #(<= x)``).
    workers : int, number of threads (``<= 1`` = single-threaded).

    Returns
    -------
    (M, N) int64 array of insertion indices.
    """
    a = np.asarray(a)
    v = np.asarray(v)
    if a.ndim != 2 or v.ndim != 2 or a.shape[1] != v.shape[1]:
        raise ValueError(
            f"expected (T, N) and (M, N) arrays with the same N, "
            f"got {a.shape} and {v.shape}")
    M, N = v.shape
    out = np.empty((M, N), dtype=np.int64)
    if workers > 1 and N >= 256:
        n_chunks = min(workers * 4, N)
        edges = np.linspace(0, N, n_chunks + 1).astype(np.int64)

        def _run(lo: int, hi: int) -> None:
            for j in range(lo, hi):
                out[:, j] = np.searchsorted(a[:, j], v[:, j], side=side)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_run, edges[:-1], edges[1:]))
    else:
        for j in range(N):
            out[:, j] = np.searchsorted(a[:, j], v[:, j], side=side)
    return out


# ---------------------------------------------------------------------------
# ECDF
# ---------------------------------------------------------------------------

def _ecdf_at_sorted(v_sorted, points) -> np.ndarray:
    """Step ECDF of a sorted (T, N) array (NaNs at the end) at (M, N) points.

    ``F(x) = #{finite values <= x} / n_valid`` (right-continuous, side="right");
    NaNs excluded from the count; NaN query points and empty (all-NaN)
    columns give NaN.
    """
    v = np.asarray(v_sorted, np.float64)
    pts = np.asarray(points, np.float64)
    nv = np.isfinite(v).sum(axis=0).astype(np.float64)
    idx = searchsorted_columns(v, pts, side="right")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(nv[None, :] > 0, idx / np.maximum(nv[None, :], 1.0),
                       np.nan)
    # NaN query points must give NaN, not the end-index (numpy treats NaN
    # queries as +inf in searchsorted)
    out = np.where(np.isfinite(pts), out, np.nan)
    return np.clip(out, 0.0, 1.0)


def ecdf_at(values, points) -> np.ndarray:
    """Step ECDF of ``values`` (T, N) evaluated at ``points`` (M, N).

    Returns the (M, N) matrix ``F(points)`` with
    ``F(x) = #{finite values <= x} / n`` (NaN-excluded); NaN points and
    all-NaN columns give NaN.
    """
    values = np.asarray(values, np.float64)
    points = np.asarray(points, np.float64)
    if values.ndim != 2 or points.ndim != 2:
        raise ValueError("expected 2-D (T, N) values and (M, N) points")
    v = np.sort(values, axis=0)          # NaNs sort to the end
    return _ecdf_at_sorted(v, points)


def ecdf(values, points=None) -> np.ndarray:
    """Empirical (step) CDF of a sample, at the sample or at given points.

    ``F(x) = #{finite values <= x} / n`` (right-continuous step; NaNs
    excluded from the count).  With ``points=None`` returns the ECDF
    evaluated at the (finite) sample values themselves.

    Accepts 1-D ``(T,)`` (returns ``(T,)``) or 2-D ``(T, N)`` (returns
    ``(T, N)``) input.
    """
    values = np.asarray(values, np.float64)
    if values.ndim == 1:
        v = np.sort(values, axis=0)[:, None]
        pts = (values if points is None else np.asarray(points, np.float64))
        pts = np.asarray(pts, np.float64)[:, None]
        return _ecdf_at_sorted(v, pts)[:, 0]
    v = np.sort(values, axis=0)
    if points is None:
        return _ecdf_at_sorted(v, values)
    return ecdf_at(values, np.asarray(points, np.float64))


# ---------------------------------------------------------------------------
# Quantiles (type-7 linear, bit-identical to np.quantile(..., method="linear"))
# ---------------------------------------------------------------------------

def quantile_from_sorted(v_sorted, q) -> np.ndarray:
    """Type-7 linear quantiles of a sorted (T, N) array at probabilities ``q``.

    The array is sorted along axis 0 with NaNs at the end (e.g. the output of
    ``np.sort(..., axis=0)``); NaNs are excluded via the per-column valid
    counts, so they can never enter the interpolation (empty columns -> NaN).

    ``q`` may be ``(Q,)`` (same probabilities for all columns) or ``(Q, N)``
    (per-column probabilities); the result is ``(Q, N)``.

    Bit-identical to ``np.quantile(v, q, method="linear", axis=0)`` on the
    NaN-free prefix (numpy's ``_lerp`` ``w >= 0.5`` branch is replicated).
    """
    v = np.asarray(v_sorted, np.float64)
    q = np.asarray(q, np.float64)
    if v.ndim != 2:
        raise ValueError(f"expected a 2-D (T, N) sorted array, got {v.shape}")
    if q.ndim == 1:
        q = np.broadcast_to(q[:, None], (q.size, v.shape[1]))
    elif q.ndim != 2 or q.shape[1] != v.shape[1]:
        raise ValueError("q must be (Q,) or (Q, N) with the same N as v")
    nv = np.isfinite(v).sum(axis=0).astype(np.float64)
    scale = np.maximum(nv - 1.0, 0.0)[None, :]
    idx = np.clip(q * scale, 0.0, scale)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.clip(np.ceil(idx).astype(np.int64), 0,
                 np.maximum(nv.astype(np.int64) - 1, 0)[None, :])
    w = idx - i0
    a = np.take_along_axis(v, i0, axis=0)
    b = np.take_along_axis(v, i1, axis=0)
    diff = b - a
    out = a + diff * w
    out = np.where(w >= 0.5, b - diff * (1.0 - w), out)
    return np.where(nv[None, :] > 0, out, np.nan)


def quantile(values, p) -> np.ndarray:
    """Type-7 quantiles of ``values`` (T, N) at probabilities ``p`` (Q,).

    Returns ``(Q, N)``; identical to ``np.quantile(values, p, method="linear",
    axis=0)`` with NaNs excluded per column (empty columns -> NaN).
    """
    values = np.asarray(values, np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D (T, N) array, got {values.shape}")
    v = np.sort(values, axis=0)          # NaNs to the end
    return quantile_from_sorted(v, np.asarray(p, np.float64))


def nanquantile(values, p) -> np.ndarray:
    """Alias of :func:`quantile` (NaN-excluded type-7 quantiles)."""
    return quantile(values, p)


def _nanmedian_axis0(a: np.ndarray) -> np.ndarray:
    """Column-wise nanmedian without RuntimeWarnings on all-NaN columns."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=0)


# ---------------------------------------------------------------------------
# Windowed least-squares slope (tail extrapolation)
# ---------------------------------------------------------------------------

def windowed_slope(x_grid, T, lo, hi, W: int = 30) -> np.ndarray:
    """Least-squares slope of ``T`` vs ``x_grid`` over the rank window per cell.

    For each column j, the slope of the linear least-squares fit of
    ``T[lo[j]:hi[j]+1, j]`` on ``x_grid[lo[j]:hi[j]+1, j]`` (closed form,
    equivalent to ``np.polyfit(..., 1)[0]`` to ~1e-12 relative, without
    RankWarnings on degenerate windows: 0 when the window is degenerate).

    Parameters
    ----------
    x_grid, T : (Q, N) transfer arrays.
    lo, hi : (N,) rank bounds (inclusive; the window is the full ``[lo, hi]``
        rank range per cell -- the ``W`` argument is kept for interface
        compatibility with the reference implementation, where the callers
        pass ``hi = min(lo + W, Q - 1)``).
    W : typical window length (informational; the effective window is the
        full ``[lo, hi]`` range of each cell).

    Returns
    -------
    (N,) slopes (``0`` where the window is degenerate or empty).
    """
    x_grid = np.asarray(x_grid, np.float64)
    T = np.asarray(T, np.float64)
    lo = np.asarray(lo, np.int64)
    hi = np.asarray(hi, np.int64)
    Q, N = x_grid.shape
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, Q - 1)
    r = np.arange(Q)[:, None]
    mask = (r >= lo[None, :]) & (r <= hi[None, :])        # (Q, N)
    gw = np.where(mask, x_grid, np.nan)
    tw = np.where(mask, T, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore"):
            gm = np.nansum(gw, axis=0) / np.maximum(mask.sum(axis=0), 1)
            g0 = gw - gm[None, :]
            tm = np.nansum(tw, axis=0) / np.maximum(mask.sum(axis=0), 1)
            t0 = tw - tm[None, :]
            num = np.nansum(g0 * t0, axis=0)
            den = np.nansum(g0 * g0, axis=0)
    slope = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return np.where(np.isfinite(slope), slope, 0.0)


# ---------------------------------------------------------------------------
# Transfer interpolation (per cell) + tail handling
# ---------------------------------------------------------------------------

def interp_transfer(x_grid, T, k_lo, k_hi, x, tail: str = "linear",
                    workers: int = 1) -> np.ndarray:
    """Evaluate the CDF-t transfer at the target values ``x`` (T, N).

    ``y(x)`` is the per-cell piecewise-linear interpolation of the transfer
    curve ``(x_grid, T)`` (``np.interp``), with ``T[0]`` / ``T[-1]`` as the
    constant (clip) extrapolation below / above the grid.

    Parameters
    ----------
    x_grid, T : (Q, N) transfer arrays (``x_grid`` = target quantiles at the
        probability grid, ``T`` = target quantiles at the mapping ``R``).
    k_lo, k_hi : (N,) first / last *unsaturated* ranks of ``R`` per cell
        (``k_lo = argmax(R > 0)``, ``k_hi = last rank with ``R < 1``).
    x : (T, N) target values (already delta-shifted).
    tail : {"linear", "clip"}
        - "linear": outside ``[x_grid[k_lo], x_grid[k_hi]]`` the transfer is
          extrapolated linearly with the 30-rank windowed least-squares slope
          at the boundary, clipped to ``[0, 2 x median interior slope]``;
        - "clip": constant at ``T[0]`` / ``T[-1]`` (np.interp defaults).
    workers : threads for the per-cell interpolation.

    Returns
    -------
    (T, N) mapped values (NaN inputs propagate to NaN outputs).
    """
    x_grid = np.asarray(x_grid, np.float64)
    T = np.asarray(T, np.float64)
    x = np.asarray(x, np.float64)
    if not (x_grid.ndim == T.ndim == 2 and x.ndim == 2
            and x_grid.shape == T.shape and x_grid.shape[1] == x.shape[1]):
        raise ValueError(
            f"expected (Q, N) x_grid/T and (T, N) x, got "
            f"{x_grid.shape}, {T.shape}, {x.shape}")
    Q, N = x_grid.shape
    y = np.empty(x.shape, dtype=np.float64)

    def _run(cols) -> None:
        for j in cols:
            y[:, j] = np.interp(x[:, j], x_grid[:, j], T[:, j],
                                left=T[0, j], right=T[-1, j])

    if workers > 1 and N >= 256:
        n_chunks = min(workers * 4, N)
        edges = np.linspace(0, N, n_chunks + 1).astype(np.int64)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_run, [range(a, b) for a, b in
                               zip(edges[:-1], edges[1:])]))
    else:
        _run(range(N))

    if tail == "linear":
        # ---- vectorized interior median slope + 2x cap (as in the
        # reference implementation) ---------------------------------------
        a, b = int(0.2 * (Q - 1)), int(0.8 * (Q - 1)) + 1
        dg = np.diff(x_grid[a:b], axis=0)
        dT = np.diff(T[a:b], axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(dg > 0, dT / np.where(dg > 0, dg, np.nan),
                             np.nan)
        s_med = _nanmedian_axis0(ratio)
        s_med = np.where(np.isnan(s_med), 0.0, s_med)
        cap = np.where(s_med > 0, 2.0 * s_med, 0.0)
        s_lo = np.clip(windowed_slope(x_grid, T, k_lo,
                                      np.minimum(k_lo + 30, Q - 1)),
                       0.0, cap)
        s_hi = np.clip(windowed_slope(x_grid, T, np.maximum(k_hi - 30, 0),
                                      k_hi),
                       0.0, cap)
        # ---- linear tail override (same where() as the reference) -------
        g_lo = np.take_along_axis(x_grid, k_lo[None, :], axis=0)
        T_lo = np.take_along_axis(T, k_lo[None, :], axis=0)
        g_hi = np.take_along_axis(x_grid, k_hi[None, :], axis=0)
        T_hi = np.take_along_axis(T, k_hi[None, :], axis=0)
        ok_lo = (k_lo > 0) & (k_lo < Q - 1)
        ok_hi = (k_hi > 0) & (k_hi < Q - 1)
        y = np.where((x < g_lo) & ok_lo[None, :],
                     T_lo + s_lo[None, :] * (x - g_lo), y)
        y = np.where((x > g_hi) & ok_hi[None, :],
                     T_hi + s_hi[None, :] * (x - g_hi), y)
    elif tail != "clip":
        raise ValueError(f"tail must be 'linear' or 'clip', got {tail!r}")
    return y


# ---------------------------------------------------------------------------
# Histogram-based ECDF / quantile variants (bounded memory)
# ---------------------------------------------------------------------------

def histogram_accumulate(values, n_bins: int = 2000):
    """Per-cell histogram with global (field-wide) edges.

    Parameters
    ----------
    values : (T, N) array.
    n_bins : int, number of bins (the edges are ``n_bins + 1``).

    Returns
    -------
    (edges (n_bins + 1,), counts (n_bins, N)): NaNs are excluded from the
    binning (sentinel bin); all-NaN fields give ``edges = linspace(0, 1, ...)``
    and zero counts.
    """
    values = np.asarray(values, np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D (T, N) array, got {values.shape}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        lo, hi = 0.0, 1.0          # all-NaN field: neutral edges (counts 0)
    elif hi <= lo:
        lo, hi = lo - 1.0, hi + 1.0
    edges = np.linspace(lo, hi, n_bins + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        idx = np.clip(((values - lo) / (hi - lo) * n_bins).astype(np.int64),
                      0, n_bins - 1)
    idx = np.where(np.isfinite(values), idx, n_bins)     # sentinel bin
    N = values.shape[1]
    counts = np.zeros((n_bins + 1, N), dtype=np.float64)
    stride = n_bins + 1
    for s in range(0, values.shape[0], 500):
        block = idx[s:s + 500]
        flat = block + (np.arange(N) * stride)[None]
        counts += np.bincount(flat.ravel(),
                              minlength=stride * N).reshape(N, stride).T
    return edges, counts[:-1]


def hist_quantile(edges, counts, q, workers: int = 1) -> np.ndarray:
    """Quantile function of per-cell histograms with GLOBAL 1-D edges.

    Same convention as :func:`quantile_from_sorted` (type-7-style linear
    within-bin interpolation scaled by ``total - 1``); ``q=0`` interpolates
    inside the first non-empty bin instead of extrapolating below the global
    minimum; all-empty cells give NaN.

    Parameters
    ----------
    edges : (B + 1,) bin edges.
    counts : (B, N) bin counts.
    q : (Q,) or (Q, N) probabilities.
    workers : int, threads for the per-cell searchsorted.

    Returns
    -------
    (Q, N) quantile values.
    """
    edges = np.asarray(edges, np.float64)
    counts = np.asarray(counts, np.float64)
    q = np.asarray(q, np.float64)
    B, N = counts.shape
    if edges.ndim != 1 or edges.size != B + 1:
        raise ValueError(f"expected edges of shape ({B + 1},), got {edges.shape}")
    if q.ndim == 1:
        q = np.broadcast_to(q[:, None], (q.size, N))
    C = np.cumsum(counts, axis=0)
    total = C[-1]
    pos = q * (total - 1.0)[None]
    b = searchsorted_columns(C, pos, side="right", workers=workers)
    b_safe = np.clip(b, 1, B - 1)
    lo = np.take_along_axis(C, b_safe - 1, axis=0)
    hi = np.take_along_axis(C, b_safe, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(hi > lo, (pos - lo) / np.where(hi > lo, hi - lo, 1.0),
                        0.0)
    out = edges[b_safe] + frac * (edges[b_safe + 1] - edges[b_safe])
    # q=0 / below the first non-empty bin: interpolate within bin 0
    c0 = np.maximum(C[0], 1.0)[None, :]
    frac0 = np.where(C[0][None, :] > 0, pos / c0, 0.0)
    out0 = edges[0] + frac0 * (edges[1] - edges[0])
    out = np.where(b == 0, out0, out)
    return np.where(total[None, :] > 0, out, np.nan)


def hist_quantile_pc(edges, counts, q, workers: int = 1) -> np.ndarray:
    """Quantile function of per-cell histograms with PER-CELL edges.

    Same convention as :func:`hist_quantile`, but ``edges`` is ``(B + 1, N)``
    (a different edge vector per column).

    Parameters
    ----------
    edges : (B + 1, N) bin edges.
    counts : (B, N) bin counts.
    q : (Q,) or (Q, N) probabilities.
    workers : int, threads for the per-cell searchsorted.

    Returns
    -------
    (Q, N) quantile values.
    """
    edges = np.asarray(edges, np.float64)
    counts = np.asarray(counts, np.float64)
    q = np.asarray(q, np.float64)
    B, N = counts.shape
    if edges.shape != (B + 1, N):
        raise ValueError(f"expected edges of shape ({B + 1}, {N}), "
                         f"got {edges.shape}")
    if q.ndim == 1:
        q = np.broadcast_to(q[:, None], (q.size, N))
    C = np.cumsum(counts, axis=0)
    total = C[-1]
    pos = q * (total - 1.0)[None]
    b = searchsorted_columns(C, pos, side="right", workers=workers)
    b_safe = np.clip(b, 1, B - 1)
    lo = np.take_along_axis(C, b_safe - 1, axis=0)
    hi = np.take_along_axis(C, b_safe, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(hi > lo, (pos - lo) / np.where(hi > lo, hi - lo, 1.0),
                        0.0)
    e_lo = np.take_along_axis(edges, b_safe, axis=0)
    e_hi = np.take_along_axis(edges, b_safe + 1, axis=0)
    out = e_lo + frac * (e_hi - e_lo)
    # q=0 / below the first non-empty bin: interpolate within bin 0
    c0 = np.maximum(C[0], 1.0)[None, :]
    frac0 = np.where(C[0][None, :] > 0, pos / c0, 0.0)
    out0 = edges[0][None, :] + frac0 * (edges[1][None, :] - edges[0][None, :])
    out = np.where(b == 0, out0, out)
    return np.where(total[None, :] > 0, out, np.nan)


def ecdf_hist(edges, counts, points, workers: int = 1) -> np.ndarray:
    """Empirical CDF of per-cell histograms at ``points`` (M, N) -> (M, N).

    Linear interpolation inside a bin (consistent with
    :func:`hist_quantile_pc`): ``F(v) = (C[b-1] + frac * counts[b]) / total``;
    ``F = 0`` below the first edge, ``1`` at/above the last edge; all-empty
    cells and NaN points give NaN.  ``edges`` may be 1-D (global) or
    ``(B + 1, N)`` (per cell).
    """
    edges = np.asarray(edges, np.float64)
    counts = np.asarray(counts, np.float64)
    points = np.asarray(points, np.float64)
    B, N = counts.shape
    if edges.ndim == 1:
        if edges.size != B + 1:
            raise ValueError(f"expected edges of shape ({B + 1},), "
                             f"got {edges.shape}")
        edges = np.broadcast_to(edges[:, None], (B + 1, N))
    if edges.shape != (B + 1, N):
        raise ValueError(f"expected edges of shape ({B + 1}, {N}), "
                         f"got {edges.shape}")
    C = np.cumsum(counts, axis=0)
    total = C[-1]
    idx = searchsorted_columns(edges, points, side="right",
                               workers=workers) - 1
    idx_safe = np.clip(idx, 0, B - 1)
    lo = np.take_along_axis(edges, idx_safe, axis=0)
    hi = np.take_along_axis(edges, idx_safe + 1, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(hi > lo, (points - lo) / np.where(hi > lo, hi - lo, 1.0),
                        0.0)
    c_below = np.take_along_axis(C, np.clip(idx_safe - 1, 0, B - 1), axis=0)
    c_below = np.where(idx_safe > 0, c_below, 0.0)
    cnt = np.take_along_axis(counts, idx_safe, axis=0)
    out = (c_below + frac * cnt) / np.where(total > 0, total, 1.0)
    out = np.where(points <= edges[0], 0.0, out)
    out = np.where(points >= edges[-1], 1.0, out)
    out = np.where(np.isfinite(points), out, np.nan)
    return np.where(total[None, :] > 0, np.clip(out, 0.0, 1.0), np.nan)
