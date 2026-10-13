"""Low-level torch kernels for the CDF-t transform (GPU backend).

This module mirrors :mod:`pycdft.math` with torch operators, so that the
same math can run on CUDA / ROCm / MPS / CPU devices:

- ECDF: right-continuous step function ``F(x) = #{finite values <= x} / n``
  (NaNs excluded from the count; ``torch.sort`` puts NaNs last, exactly
  like ``numpy.sort``);
- quantiles: numpy / R ``type-7`` linear interpolation, re-implemented on
  pre-sorted arrays with the same index / weight arithmetic as
  :func:`pycdft.math.quantile_from_sorted` (including the ``w >= 0.5``
  branch), so that the results agree with the numpy backend to the last
  significant digit on the same data;
- the transfer interpolation: a vectorized ``np.interp`` equivalent
  (``torch.searchsorted(..., right=True)`` with the same duplicate /
  endpoint convention), with the two tail options (``clip`` / ``linear``);
- histogram-based ECDF / quantile variants.

All kernels accept 2-D tensors ``(T, N)`` (or ``(Q, N)``) of float64 and
operate independently per column (per series / per grid cell), i.e. the
same conventions as the numpy backend.  All math is done in float64 (the
CDF-t kernels are memory-bound; float64 keeps the results identical to the
numpy backend -- the input dtype is widened on conversion, see
:mod:`pycdft.gpu.api`).

Notes
-----
The batched :func:`searchsorted_columns` uses a transposed ``(N, ...)``
layout internally (torch searches along the last dimension); the transpose
is a view and the explicit ``contiguous()`` copy is the only extra
allocation of that kernel.
"""

from __future__ import annotations

import numpy as np

from ._torch import require_torch


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

def quantile_grid(n_quantiles: int = 1001, device=None):
    """Uniform probability grid ``linspace(0, 1, n_quantiles)`` on a device.

    Bit-identical to ``numpy.linspace(0.0, 1.0, n_quantiles)`` (computed as
    ``arange(n) * (1 / (n - 1))``, the same arithmetic numpy uses).
    """
    torch = require_torch()
    n_quantiles = int(n_quantiles)
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
    return torch.arange(n_quantiles, dtype=torch.float64,
                        device=device) * (1.0 / (n_quantiles - 1))


# ---------------------------------------------------------------------------
# Batched per-column searchsorted
# ---------------------------------------------------------------------------

def _as_2d(x, name: str = "array"):
    torch = require_torch()
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    if t.ndim != 2:
        raise ValueError(f"expected a 2-D tensor, got shape {tuple(t.shape)}")
    return t


def searchsorted_columns(a, v, side: str = "right"):
    """Per-column ``torch.searchsorted`` on the (T, N) / (M, N) layouts.

    Equivalent to looping ``torch.searchsorted(a[:, j], v[:, j],
    side=side)`` over columns; identical to
    :func:`pycdft.math.searchsorted_columns` (same ``side`` convention,
    NaNs treated as ``+inf`` in both the sequence and the queries, as numpy
    does).

    Parameters
    ----------
    a : (T, N) tensor, sorted along dim 0 (NaNs allowed, sorted last).
    v : (M, N) tensor of query points.
    side : {"left", "right"}, default "right" (``F(x) = #(<= x)``).

    Returns
    -------
    (M, N) int64 tensor of insertion indices.
    """
    torch = require_torch()
    a = _as_2d(a, "a")
    v = _as_2d(v, "v")
    if a.shape[1] != v.shape[1]:
        raise ValueError(
            f"expected (T, N) and (M, N) arrays with the same N, "
            f"got {tuple(a.shape)} and {tuple(v.shape)}")
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    # torch.searchsorted is not well-defined for NaN entries; numpy treats
    # NaN as +inf in both the sequence and the query (hence a NaN query
    # lands at the end and is masked to NaN by the callers).
    a_ss = torch.nan_to_num(a, nan=float("inf"), posinf=float("inf"),
                            neginf=float("-inf"))
    v_ss = torch.nan_to_num(v, nan=float("inf"), posinf=float("inf"),
                            neginf=float("-inf"))
    # torch searches along the last dim: use the transposed (N, ...) layout
    # so that the batch dimension is the number of columns.
    idx = torch.searchsorted(a_ss.transpose(0, 1).contiguous(),
                             v_ss.transpose(0, 1).contiguous(),
                             right=(side == "right"))
    return idx.transpose(0, 1)


# ---------------------------------------------------------------------------
# ECDF
# ---------------------------------------------------------------------------

def _ecdf_at_sorted(v_sorted, points):
    """Step ECDF of a sorted (T, N) tensor (NaNs at the end) at (M, N) points.

    ``F(x) = #{finite values <= x} / n_valid`` (right-continuous,
    side="right"); NaNs excluded from the count; NaN query points and empty
    (all-NaN) columns give NaN.  Identical to
    :func:`pycdft.math._ecdf_at_sorted`.
    """
    torch = require_torch()
    v = v_sorted.to(torch.float64)
    pts = points.to(torch.float64)
    nv = torch.isfinite(v).sum(dim=0).to(torch.float64)
    idx = searchsorted_columns(v, pts, side="right")
    out = idx.to(torch.float64) / nv.clamp(min=1.0)[None, :]
    out = out.masked_fill(nv[None, :] <= 0, float("nan"))
    out = out.masked_fill(~torch.isfinite(pts), float("nan"))
    return out.clamp(0.0, 1.0)


def ecdf_at(values, points):
    """Step ECDF of ``values`` (T, N) evaluated at ``points`` (M, N).

    Returns the (M, N) tensor ``F(points)`` with
    ``F(x) = #{finite values <= x} / n`` (NaN-excluded); NaN points and
    all-NaN columns give NaN.
    """
    torch = require_torch()
    values = _as_2d(values, "values")
    points = _as_2d(points, "points")
    v, _ = torch.sort(values, dim=0)      # NaNs sort to the end
    return _ecdf_at_sorted(v, points)


def ecdf(values, points=None):
    """Empirical (step) CDF of a sample, at the sample or at given points.

    ``F(x) = #{finite values <= x} / n`` (right-continuous step; NaNs
    excluded from the count).  With ``points=None`` returns the ECDF
    evaluated at the (finite) sample values themselves.  Accepts 1-D
    ``(T,)`` (returns ``(T,)``) or 2-D ``(T, N)`` (returns ``(T, N)``).
    """
    torch = require_torch()
    values = torch.as_tensor(values) if not isinstance(values, torch.Tensor) \
        else values
    squeeze = values.ndim == 1
    if squeeze:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D tensor, got {tuple(values.shape)}")
    v, _ = torch.sort(values, dim=0)
    if points is None:
        out = _ecdf_at_sorted(v, values)
    else:
        pts = torch.as_tensor(points) if not isinstance(points, torch.Tensor) \
            else points
        if pts.ndim == 1:
            pts = pts[:, None]
        out = _ecdf_at_sorted(v, pts)
    return out[:, 0] if squeeze else out


# ---------------------------------------------------------------------------
# Quantiles (type-7 linear, identical to np.quantile(..., method="linear"))
# ---------------------------------------------------------------------------

def quantile_from_sorted(v_sorted, q):
    """Type-7 linear quantiles of a sorted (T, N) tensor at probabilities ``q``.

    The tensor is sorted along dim 0 with NaNs at the end; NaNs are excluded
    via the per-column valid counts, so they can never enter the
    interpolation (empty columns -> NaN).  ``q`` may be ``(Q,)`` (same
    probabilities for all columns) or ``(Q, N)`` (per-column probabilities);
    the result is ``(Q, N)``.

    Bit-identical to :func:`pycdft.math.quantile_from_sorted` (same index
    and weight arithmetic, including the ``w >= 0.5`` branch).
    """
    torch = require_torch()
    v = v_sorted.to(torch.float64)
    q = q.to(torch.float64)
    if v.ndim != 2:
        raise ValueError(f"expected a 2-D (T, N) sorted tensor, got {v.shape}")
    if q.ndim == 1:
        q = q[:, None].expand(q.size(0), v.shape[1])
    elif q.ndim != 2 or q.shape[1] != v.shape[1]:
        raise ValueError("q must be (Q,) or (Q, N) with the same N as v")
    nv = torch.isfinite(v).sum(dim=0).to(torch.float64)
    scale = (nv - 1.0).clamp(min=0.0)[None, :]
    idx = (q * scale).clamp(min=0.0).clamp(max=scale)
    i0 = torch.floor(idx).to(torch.int64)
    i1 = torch.ceil(idx).to(torch.int64).clamp(min=0).clamp(
        max=(nv.to(torch.int64) - 1).clamp(min=0)[None, :])
    w = idx - i0.to(torch.float64)
    a = torch.take_along_dim(v, i0, dim=0)
    b = torch.take_along_dim(v, i1, dim=0)
    diff = b - a
    out = a + diff * w
    out = torch.where(w >= 0.5, b - diff * (1.0 - w), out)
    return out.masked_fill(nv[None, :] <= 0, float("nan"))


def quantile(values, p):
    """Type-7 quantiles of ``values`` (T, N) at probabilities ``p`` (Q,).

    Returns ``(Q, N)``; identical to ``np.quantile(values, p, method=
    "linear", axis=0)`` with NaNs excluded per column (empty columns ->
    NaN).
    """
    torch = require_torch()
    values = _as_2d(values, "values")
    v, _ = torch.sort(values, dim=0)      # NaNs to the end
    return quantile_from_sorted(v, p)


def nanquantile(values, p):
    """Alias of :func:`quantile` (NaN-excluded type-7 quantiles)."""
    return quantile(values, p)


# ---------------------------------------------------------------------------
# Windowed least-squares slope (tail extrapolation)
# ---------------------------------------------------------------------------

def _nanmedian_axis0(x):
    """Column-wise nanmedian (average of the two middle values, as numpy).

    ``torch.nanmedian`` returns the *lower* median for an even number of
    values (documented torch behaviour), so the median is computed from the
    sorted finite values to match ``numpy.nanmedian`` exactly.
    """
    torch = require_torch()
    s, _ = torch.sort(x, dim=0)           # NaNs at the end
    if s.shape[0] == 0:                   # empty window -> NaN (as numpy)
        return torch.full((s.shape[1],), float("nan"), dtype=s.dtype,
                          device=s.device)
    nv = torch.isfinite(s).sum(dim=0).to(torch.int64)
    lo = ((nv - 1) // 2).clamp(min=0)
    hi = (nv // 2).clamp(min=0)
    v_lo = torch.take_along_dim(s, lo[None, :], dim=0)
    v_hi = torch.take_along_dim(s, hi[None, :], dim=0)
    med = (v_lo + v_hi) / 2.0
    return med.masked_fill(nv[None, :] <= 0, float("nan"))[0]


def windowed_slope(x_grid, T, lo, hi):
    """Least-squares slope of ``T`` vs ``x_grid`` over the rank window per cell.

    Same closed-form computation as :func:`pycdft.math.windowed_slope`:
    per column, the slope of the linear least-squares fit of the window
    ``[lo[j], hi[j]]`` (``0`` when the window is degenerate or empty).
    """
    torch = require_torch()
    x_grid = x_grid.to(torch.float64)
    T = T.to(torch.float64)
    lo = lo.to(torch.int64).clamp(min=0)
    hi = hi.to(torch.int64).clamp(max=x_grid.shape[0] - 1)
    Q, N = x_grid.shape
    r = torch.arange(Q, dtype=torch.int64, device=x_grid.device)[:, None]
    mask = (r >= lo[None, :]) & (r <= hi[None, :])        # (Q, N)
    gw = x_grid.masked_fill(~mask, float("nan"))
    tw = T.masked_fill(~mask, float("nan"))
    counts = mask.sum(dim=0).to(torch.float64).clamp(min=1.0)
    gm = torch.nansum(gw, dim=0) / counts
    g0 = gw - gm[None, :]
    tm = torch.nansum(tw, dim=0) / counts
    t0 = tw - tm[None, :]
    num = torch.nansum(g0 * t0, dim=0)
    den = torch.nansum(g0 * g0, dim=0)
    slope = torch.where(den > 0, num / torch.where(den > 0, den,
                                                   torch.ones_like(den)),
                        torch.zeros_like(den))
    return torch.where(torch.isfinite(slope), slope, torch.zeros_like(slope))


# ---------------------------------------------------------------------------
# Transfer interpolation (per cell) + tail handling
# ---------------------------------------------------------------------------

def interp_transfer(x_grid, T, k_lo, k_hi, x, tail: str = "linear"):
    """Evaluate the CDF-t transfer at the target values ``x`` (T, N).

    Vectorized equivalent of :func:`pycdft.math.interp_transfer`:
    ``y(x)`` is the per-cell piecewise-linear interpolation of the transfer
    curve ``(x_grid, T)`` with the same endpoint / duplicate-value
    convention as ``numpy.interp`` (``T[0]`` / ``T[-1]`` as the constant
    clip extrapolation below / above the grid; for duplicated grid values
    the value of the last occurrence is used).

    Parameters
    ----------
    x_grid, T : (Q, N) transfer tensors (``x_grid`` = target quantiles at
        the probability grid, ``T`` = target quantiles at the mapping ``R``).
    k_lo, k_hi : (N,) first / last *unsaturated* ranks of ``R`` per cell.
    x : (T, N) target values (already delta-shifted).
    tail : {"linear", "clip"}: as in the numpy backend ("linear" adds the
        windowed least-squares extrapolation, slope capped at 2x the median
        interior slope).

    Returns
    -------
    (T, N) mapped values (NaN inputs propagate to NaN outputs).
    """
    torch = require_torch()
    x_grid = x_grid.to(torch.float64)
    T = T.to(torch.float64)
    x = x.to(torch.float64)
    if not (x_grid.ndim == T.ndim == 2 and x.ndim == 2
            and x_grid.shape == T.shape and x_grid.shape[1] == x.shape[1]):
        raise ValueError(
            f"expected (Q, N) x_grid/T and (T, N) x, got "
            f"{tuple(x_grid.shape)}, {tuple(T.shape)}, {tuple(x.shape)}")
    Q, N = x_grid.shape
    # ---- np.interp equivalent (per cell, vectorized) --------------------
    # torch.searchsorted is not well-defined for NaN; replace with +inf in
    # the sorted grid and the queries (NaN queries are masked to NaN below,
    # matching np.interp).  Ties / duplicated grid values follow the numpy
    # convention: right=True lands after the duplicate run, so the value of
    # the last occurrence is used.
    xg_ss = torch.nan_to_num(x_grid, nan=float("inf"), posinf=float("inf"),
                             neginf=float("-inf"))
    x_ss = torch.nan_to_num(x, nan=float("inf"), posinf=float("inf"),
                            neginf=float("-inf"))
    idx = searchsorted_columns(xg_ss, x_ss, side="right")     # (M, N)
    idx_cl = idx.clamp(1, Q - 1)
    x0 = torch.take_along_dim(x_grid, idx_cl - 1, dim=0)
    x1 = torch.take_along_dim(x_grid, idx_cl, dim=0)
    y0 = torch.take_along_dim(T, idx_cl - 1, dim=0)
    y1 = torch.take_along_dim(T, idx_cl, dim=0)
    frac = (y1 - y0) / (x1 - x0)
    y = y0 + frac * (x - x0)
    y = torch.where(idx == 0, T[0][None, :], y)               # x < x_grid[0]
    y = torch.where(idx >= Q, T[-1][None, :], y)              # x >= x_grid[-1]
    y = y.masked_fill(torch.isnan(x), float("nan"))           # NaN queries

    if tail == "linear":
        # ---- interior median slope + 2x cap (as the numpy backend) ------
        a, b = int(0.2 * (Q - 1)), int(0.8 * (Q - 1)) + 1
        dg = torch.diff(x_grid[a:b], dim=0)
        dT = torch.diff(T[a:b], dim=0)
        ratio = torch.where(dg > 0, dT / torch.where(dg > 0, dg,
                                                     torch.full_like(dg, np.nan)),
                            torch.full_like(dg, np.nan))
        s_med = _nanmedian_axis0(ratio)
        s_med = torch.where(torch.isnan(s_med), torch.zeros_like(s_med), s_med)
        cap = torch.where(s_med > 0, 2.0 * s_med, torch.zeros_like(s_med))
        s_lo = windowed_slope(x_grid, T, k_lo, (k_lo + 30).clamp(max=Q - 1))
        s_lo = s_lo.clamp(min=0.0).clamp(max=cap)
        s_hi = windowed_slope(x_grid, T, (k_hi - 30).clamp(min=0), k_hi)
        s_hi = s_hi.clamp(min=0.0).clamp(max=cap)
        # ---- linear tail override ---------------------------------------
        g_lo = torch.take_along_dim(x_grid, k_lo[None, :], dim=0)
        T_lo = torch.take_along_dim(T, k_lo[None, :], dim=0)
        g_hi = torch.take_along_dim(x_grid, k_hi[None, :], dim=0)
        T_hi = torch.take_along_dim(T, k_hi[None, :], dim=0)
        ok_lo = (k_lo > 0) & (k_lo < Q - 1)
        ok_hi = (k_hi > 0) & (k_hi < Q - 1)
        y = torch.where((x < g_lo) & ok_lo[None, :],
                        T_lo + s_lo[None, :] * (x - g_lo), y)
        y = torch.where((x > g_hi) & ok_hi[None, :],
                        T_hi + s_hi[None, :] * (x - g_hi), y)
    elif tail != "clip":
        raise ValueError(f"tail must be 'linear' or 'clip', got {tail!r}")
    return y


# ---------------------------------------------------------------------------
# Histogram-based ECDF / quantile variants (bounded memory)
# ---------------------------------------------------------------------------

def histogram_accumulate(values, n_bins: int = 2000):
    """Per-cell histogram with global (field-wide) edges.

    Same convention as :func:`pycdft.math.histogram_accumulate`: NaNs are
    excluded from the binning (sentinel bin), all-NaN fields give
    ``edges = linspace(0, 1, n_bins + 1)`` and zero counts, degenerate
    (constant) fields get a ``+/-1`` padding.  Returns
    ``(edges (n_bins + 1,), counts (n_bins, N))``.
    """
    torch = require_torch()
    values = _as_2d(values, "values")
    n_bins = int(n_bins)
    valid = torch.isfinite(values)
    lo = torch.where(valid, values, torch.full_like(values, float("inf"))).min()
    hi = torch.where(valid, values, torch.full_like(values, float("-inf"))).max()
    if not (torch.isfinite(lo) and torch.isfinite(hi)):
        lo, hi = torch.tensor(0.0, dtype=values.dtype,
                              device=values.device), \
            torch.tensor(1.0, dtype=values.dtype, device=values.device)
    elif bool(hi <= lo):
        lo, hi = lo - 1.0, hi + 1.0
    edges = (torch.arange(n_bins + 1, dtype=torch.float64,
                          device=values.device)
             * ((hi - lo) / n_bins)) + lo
    width = (hi - lo)
    idx = (((values - lo) / width * n_bins)
           .to(torch.int64).clamp(0, n_bins - 1))
    idx = torch.where(valid, idx,
                      torch.full_like(idx, n_bins))          # sentinel bin
    N = values.shape[1]
    counts = torch.zeros((n_bins + 1, N), dtype=torch.float64,
                         device=values.device)
    counts.scatter_add_(0, idx, torch.ones_like(values, dtype=torch.float64))
    return edges, counts[:-1]


def _hist_quantile(edges, counts, q):
    """Shared type-7-style within-bin interpolation (see hist_quantile*)."""
    torch = require_torch()
    B, N = counts.shape
    C = torch.cumsum(counts, dim=0)
    total = C[-1]
    pos = q * (total - 1.0)[None, :]
    b = searchsorted_columns(C, pos, side="right")           # (Q, N)
    b_safe = b.clamp(1, B - 1)
    lo = torch.take_along_dim(C, b_safe - 1, dim=0)
    hi = torch.take_along_dim(C, b_safe, dim=0)
    frac = torch.where(hi > lo, (pos - lo) / torch.where(hi > lo, hi - lo,
                                                         torch.ones_like(hi)),
                       torch.zeros_like(hi))
    return C, total, b, b_safe, frac


def hist_quantile(edges, counts, q):
    """Quantile function of per-cell histograms with GLOBAL 1-D edges.

    Same convention as :func:`pycdft.math.hist_quantile` (type-7-style
    linear within-bin interpolation scaled by ``total - 1``; ``q=0``
    interpolates inside the first non-empty bin instead of extrapolating
    below the global minimum; all-empty cells give NaN).
    """
    torch = require_torch()
    edges = edges.to(torch.float64)
    counts = counts.to(torch.float64)
    q = q.to(torch.float64)
    B, N = counts.shape
    if edges.ndim != 1 or edges.size(0) != B + 1:
        raise ValueError(f"expected edges of shape ({B + 1},), got "
                         f"{tuple(edges.shape)}")
    if q.ndim == 1:
        q = q[:, None].expand(q.size(0), N)
    C, total, b, b_safe, frac = _hist_quantile(edges, counts, q)
    out = edges[b_safe] + frac * (edges[b_safe + 1] - edges[b_safe])
    # q=0 / below the first non-empty bin: interpolate within bin 0
    c0 = C[0].clamp(min=1.0)[None, :]
    frac0 = torch.where(C[0][None, :] > 0, q * (total - 1.0)[None, :] / c0,
                        torch.zeros_like(c0))
    out0 = edges[0] + frac0 * (edges[1] - edges[0])
    out = torch.where(b == 0, out0, out)
    return out.masked_fill(total[None, :] <= 0, float("nan"))


def hist_quantile_pc(edges, counts, q):
    """Quantile function of per-cell histograms with PER-CELL edges.

    Same convention as :func:`hist_quantile`, but ``edges`` is ``(B + 1, N)``
    (a different edge vector per column).
    """
    torch = require_torch()
    edges = edges.to(torch.float64)
    counts = counts.to(torch.float64)
    q = q.to(torch.float64)
    B, N = counts.shape
    if edges.shape != (B + 1, N):
        raise ValueError(f"expected edges of shape ({B + 1}, {N}), "
                         f"got {tuple(edges.shape)}")
    if q.ndim == 1:
        q = q[:, None].expand(q.size(0), N)
    C, total, b, b_safe, frac = _hist_quantile(edges, counts, q)
    e_lo = torch.take_along_dim(edges, b_safe, dim=0)
    e_hi = torch.take_along_dim(edges, b_safe + 1, dim=0)
    out = e_lo + frac * (e_hi - e_lo)
    c0 = C[0].clamp(min=1.0)[None, :]
    frac0 = torch.where(C[0][None, :] > 0, q * (total - 1.0)[None, :] / c0,
                        torch.zeros_like(c0))
    out0 = edges[0][None, :] + frac0 * (edges[1][None, :] - edges[0][None, :])
    out = torch.where(b == 0, out0, out)
    return out.masked_fill(total[None, :] <= 0, float("nan"))


def ecdf_hist(edges, counts, points):
    """Empirical CDF of per-cell histograms at ``points`` (M, N) -> (M, N).

    Linear interpolation inside a bin (consistent with
    :func:`hist_quantile_pc`): ``F(v) = (C[b-1] + frac * counts[b]) / total``;
    ``F = 0`` below the first edge, ``1`` at/above the last edge; all-empty
    cells and NaN points give NaN.  ``edges`` may be 1-D (global) or
    ``(B + 1, N)`` (per cell).
    """
    torch = require_torch()
    edges = edges.to(torch.float64)
    counts = counts.to(torch.float64)
    points = points.to(torch.float64)
    B, N = counts.shape
    if edges.ndim == 1:
        if edges.size(0) != B + 1:
            raise ValueError(f"expected edges of shape ({B + 1},), "
                             f"got {tuple(edges.shape)}")
        edges = edges[:, None].expand(B + 1, N)
    if edges.shape != (B + 1, N):
        raise ValueError(f"expected edges of shape ({B + 1}, {N}), "
                         f"got {tuple(edges.shape)}")
    C = torch.cumsum(counts, dim=0)
    total = C[-1]
    idx = searchsorted_columns(edges, points, side="right") - 1
    idx_safe = idx.clamp(0, B - 1)
    lo = torch.take_along_dim(edges, idx_safe, dim=0)
    hi = torch.take_along_dim(edges, idx_safe + 1, dim=0)
    frac = torch.where(hi > lo, (points - lo) / torch.where(hi > lo, hi - lo,
                                                            torch.ones_like(hi)),
                       torch.zeros_like(hi))
    c_below = torch.take_along_dim(C, (idx_safe - 1).clamp(min=0), dim=0)
    c_below = torch.where(idx_safe > 0, c_below, torch.zeros_like(c_below))
    cnt = torch.take_along_dim(counts, idx_safe, dim=0)
    out = (c_below + frac * cnt) / torch.where(total > 0, total,
                                               torch.ones_like(total))
    out = torch.where(points <= edges[0], torch.zeros_like(out), out)
    out = torch.where(points >= edges[-1], torch.ones_like(out), out)
    out = out.masked_fill(~torch.isfinite(points), float("nan"))
    out = out.clamp(0.0, 1.0)
    return out.masked_fill(total[None, :] <= 0, float("nan"))
