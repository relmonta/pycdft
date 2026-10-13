"""Math-kernel correctness tests (pure numpy, no external packages)."""

import numpy as np
import pytest

from pycdft import math as m
from pycdft.math import (ecdf, ecdf_at, ecdf_hist, hist_quantile,
                         hist_quantile_pc, histogram_accumulate,
                         interp_transfer, quantile, quantile_from_sorted,
                         quantile_grid, searchsorted_columns,
                         windowed_slope)


# ---------------------------------------------------------------------------
# Probability grid
# ---------------------------------------------------------------------------

def test_quantile_grid_endpoints_and_size():
    p = quantile_grid(1001)
    assert p.shape == (1001,)
    assert p[0] == 0.0 and p[-1] == 1.0
    assert np.all(np.diff(p) > 0)
    assert quantile_grid(5).tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert np.allclose(quantile_grid(), quantile_grid(1001))


def test_quantile_grid_rejects_small():
    with pytest.raises(ValueError):
        quantile_grid(1)


# ---------------------------------------------------------------------------
# Quantiles (type-7)
# ---------------------------------------------------------------------------

def test_quantile_from_sorted_matches_numpy():
    rng = np.random.default_rng(0)
    v = rng.normal(0.5, 2.0, (500, 6))
    v_sorted = np.sort(v, axis=0)
    q = np.linspace(0.0, 1.0, 201)
    got = quantile_from_sorted(v_sorted, q)
    for j in range(v.shape[1]):
        np.testing.assert_allclose(got[:, j], np.quantile(v[:, j], q), atol=0)
    # per-column probabilities
    qq = rng.random((201, 6))
    got2 = quantile_from_sorted(v_sorted, qq)
    for j in range(v.shape[1]):
        np.testing.assert_allclose(got2[:, j], np.quantile(v[:, j], qq[:, j]),
                                   atol=0)


def test_quantile_matches_nanquantile():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(400, 4))
    mask = rng.random((400, 4)) < 0.1
    v[mask] = np.nan
    p = np.linspace(0.0, 1.0, 101)
    got = quantile(v, p)
    for j in range(v.shape[1]):
        np.testing.assert_allclose(got[:, j],
                                   np.nanquantile(v[:, j], p, method="linear"),
                                   atol=1e-12)


def test_quantile_endpoints_are_min_max():
    rng = np.random.default_rng(2)
    v = rng.normal(size=(300, 5))
    q = quantile(v, np.array([0.0, 1.0]))
    for j in range(v.shape[1]):
        assert q[0, j] == v[:, j].min()
        assert q[1, j] == v[:, j].max()


def test_quantile_nan_not_in_counts():
    # NaNs sorted to the end must never enter the interpolation: adding NaNs
    # to a NaN-free column does not change the valid quantiles
    rng = np.random.default_rng(3)
    v = rng.normal(size=(300,))
    w = np.concatenate([v, [np.nan] * 50])
    p = np.linspace(0.05, 0.95, 19)
    np.testing.assert_allclose(quantile(w[:, None], p)[:, 0],
                               quantile(v[:, None], p)[:, 0], atol=0)


def test_quantile_empty_column_is_nan():
    v = np.full((10, 2), np.nan)
    q = quantile(v, np.array([0.5]))
    assert np.all(np.isnan(q))


# ---------------------------------------------------------------------------
# ECDF
# ---------------------------------------------------------------------------

def test_ecdf_step_convention():
    v = np.array([1.0, 2.0, 2.0, 4.0])
    f = ecdf(v)
    np.testing.assert_allclose(f, [0.25, 0.75, 0.75, 1.0])
    pts = np.array([0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
    f2 = ecdf(v, pts)
    np.testing.assert_allclose(f2, [0.0, 0.25, 0.25, 0.75, 0.75, 1.0, 1.0])


def test_ecdf_excludes_nan():
    v = np.array([1.0, np.nan, 3.0])
    f = ecdf(v)
    np.testing.assert_allclose(f[:1], [0.5])
    assert np.isnan(f[1])            # NaN evaluation point -> NaN
    assert f[2] == 1.0
    assert ecdf(v, np.array([2.0]))[0] == 0.5


def test_ecdf_at_nan_points_and_empty_columns():
    v = np.array([[1.0, np.nan], [2.0, np.nan], [3.0, np.nan]])
    pts = np.array([[1.5, 0.0], [np.nan, 0.0]])
    out = ecdf_at(v, pts)
    assert np.isnan(out[0, 1])           # empty column
    assert np.isnan(out[1, 0])           # NaN point
    assert out[0, 0] == pytest.approx(1.0 / 3)


def test_ecdf_matches_manual_2d():
    rng = np.random.default_rng(4)
    v = rng.normal(size=(200, 5))
    pts = rng.normal(size=(50, 5))
    got = ecdf_at(v, pts)
    for j in range(5):
        want = np.searchsorted(np.sort(v[:, j]), pts[:, j],
                               side="right") / v.shape[0]
        np.testing.assert_allclose(got[:, j], want)


def test_searchsorted_columns_matches_loop():
    rng = np.random.default_rng(5)
    a = np.sort(rng.normal(size=(100, 7)), axis=0)
    v = rng.normal(size=(13, 7))
    got = searchsorted_columns(a, v, side="right")
    for j in range(7):
        np.testing.assert_array_equal(got[:, j],
                                      np.searchsorted(a[:, j], v[:, j],
                                                      side="right"))
    got_l = searchsorted_columns(a, v, side="left")
    for j in range(7):
        np.testing.assert_array_equal(got_l[:, j],
                                      np.searchsorted(a[:, j], v[:, j],
                                                      side="left"))


# ---------------------------------------------------------------------------
# Windowed slope
# ---------------------------------------------------------------------------

def test_windowed_slope_matches_polyfit():
    rng = np.random.default_rng(6)
    Q, N = 200, 5
    x_grid = np.sort(rng.normal(size=(Q, N)), axis=0) + \
        np.arange(Q)[:, None] * 0.1
    T = 2.0 * x_grid - 1.0 + rng.normal(0, 0.01, (Q, N))
    lo = np.full(N, 20)
    hi = np.full(N, 50)
    got = windowed_slope(x_grid, T, lo, hi)
    for j in range(N):
        want = np.polyfit(x_grid[20:51, j], T[20:51, j], 1)[0]
        assert got[j] == pytest.approx(want, abs=1e-10)


def test_windowed_slope_degenerate_is_zero():
    x_grid = np.ones((50, 2))
    T = np.arange(50.0)[:, None] * np.ones((1, 2))
    got = windowed_slope(x_grid, T, np.zeros(2, dtype=int),
                         np.full(2, 49))
    np.testing.assert_allclose(got, 0.0)


# ---------------------------------------------------------------------------
# Transfer interpolation + tails
# ---------------------------------------------------------------------------

def _make_transfer(rng, Q=301, N=3, slope=1.5):
    """Strictly increasing x_grid and T with a kink at the saturated ranks."""
    p = np.linspace(0.0, 1.0, Q)
    x_grid = np.sort(rng.normal(size=(Q, N)), axis=0)
    x_grid += np.arange(Q)[:, None] * 0.02
    T = slope * (x_grid - x_grid.mean(axis=0)) + rng.normal(0, 0.005, (Q, N))
    k_lo = np.full(N, 60)
    k_hi = np.full(N, Q - 61)
    return x_grid, T, k_lo, k_hi


def test_interp_transfer_clip_equals_np_interp():
    rng = np.random.default_rng(7)
    x_grid, T, k_lo, k_hi = _make_transfer(rng)
    x = rng.normal(size=(200, 3))
    got = interp_transfer(x_grid, T, k_lo, k_hi, x, tail="clip")
    for j in range(3):
        want = np.interp(x[:, j], x_grid[:, j], T[:, j], left=T[0, j],
                         right=T[-1, j])
        np.testing.assert_allclose(got[:, j], want, atol=1e-12)
    # clip: output within the transfer's value range
    assert np.nanmin(got) >= np.nanmin(T) - 1e-12
    assert np.nanmax(got) <= np.nanmax(T) + 1e-12


def test_interp_transfer_linear_uses_windowed_slope_formula():
    rng = np.random.default_rng(8)
    x_grid, T, k_lo, k_hi = _make_transfer(rng)
    x = rng.normal(size=(300, 3)) * 4.0        # wide range: both tails hit
    got = interp_transfer(x_grid, T, k_lo, k_hi, x, tail="linear")
    # outside the saturated region the output is exactly the anchor + slope
    # formula; replicate the slope logic of interp_transfer to verify
    Q = x_grid.shape[0]
    a, b = int(0.2 * (Q - 1)), int(0.8 * (Q - 1)) + 1
    dg = np.diff(x_grid[a:b], axis=0)
    dT = np.diff(T[a:b], axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(dg > 0, dT / np.where(dg > 0, dg, np.nan), np.nan)
    s_med = np.nanmedian(ratio, axis=0)
    cap = np.where(s_med > 0, 2.0 * s_med, 0.0)
    s_lo = np.clip(windowed_slope(x_grid, T, k_lo,
                                  np.minimum(k_lo + 30, Q - 1)), 0.0, cap)
    s_hi = np.clip(windowed_slope(x_grid, T, np.maximum(k_hi - 30, 0),
                                  k_hi), 0.0, cap)
    g_lo = np.take_along_axis(x_grid, k_lo[None, :], axis=0)
    T_lo = np.take_along_axis(T, k_lo[None, :], axis=0)
    g_hi = np.take_along_axis(x_grid, k_hi[None, :], axis=0)
    T_hi = np.take_along_axis(T, k_hi[None, :], axis=0)
    for j in range(3):
        below = x[:, j] < g_lo[0, j]
        above = x[:, j] > g_hi[0, j]
        assert below.any() and above.any()
        np.testing.assert_allclose(
            got[below, j], T_lo[0, j] + s_lo[j] * (x[below, j] - g_lo[0, j]),
            atol=1e-12)
        np.testing.assert_allclose(
            got[above, j], T_hi[0, j] + s_hi[j] * (x[above, j] - g_hi[0, j]),
            atol=1e-12)
        # positive capped slopes -> the extrapolation extends the clip range
        assert s_lo[j] >= 0 and s_hi[j] >= 0
        assert s_lo[j] <= 2 * s_med[j] + 1e-12
        assert s_hi[j] <= 2 * s_med[j] + 1e-12


def test_interp_transfer_linear_extrapolates_monotonically():
    rng = np.random.default_rng(9)
    x_grid, T, k_lo, k_hi = _make_transfer(rng)
    x = np.linspace(x_grid.min() - 1, x_grid.max() + 1, 500)
    x = np.tile(x[:, None], (1, 3))
    got = interp_transfer(x_grid, T, k_lo, k_hi, x, tail="linear")
    for j in range(3):
        assert np.all(np.diff(got[:, j]) >= -1e-12)


def test_interp_transfer_rejects_bad_tail():
    rng = np.random.default_rng(10)
    x_grid, T, k_lo, k_hi = _make_transfer(rng)
    with pytest.raises(ValueError):
        interp_transfer(x_grid, T, k_lo, k_hi, x_grid, tail="cubic")


# ---------------------------------------------------------------------------
# Histogram variants
# ---------------------------------------------------------------------------

def test_histogram_accumulate_excludes_nan():
    rng = np.random.default_rng(11)
    v = rng.normal(size=(1000, 3))
    v[0, 0] = np.nan
    edges, counts = histogram_accumulate(v, n_bins=100)
    total0 = counts[:, 0].sum()
    assert total0 == 999
    assert counts[:, 1].sum() == 1000
    # all-NaN field: no crash, zero counts
    w = np.full((10, 2), np.nan)
    e2, c2 = histogram_accumulate(w, n_bins=50)
    assert c2.sum() == 0
    assert np.all(np.isfinite(e2))


def test_hist_quantile_matches_exact_quantile():
    rng = np.random.default_rng(12)
    v = rng.normal(size=(20000, 3))
    edges, counts = histogram_accumulate(v, n_bins=2000)
    p = np.linspace(0.001, 0.999, 101)
    got = hist_quantile(edges, counts, p)
    for j in range(3):
        np.testing.assert_allclose(got[:, j], np.quantile(v[:, j], p),
                                   atol=0.05)


def test_hist_quantile_pc_matches_hist_quantile():
    rng = np.random.default_rng(13)
    v = rng.normal(size=(5000, 4))
    edges, counts = histogram_accumulate(v, n_bins=500)
    edges_pc = np.broadcast_to(edges[:, None], (edges.size, 4)).copy()
    p = np.linspace(0.0, 1.0, 77)
    np.testing.assert_allclose(hist_quantile(edges, counts, p),
                               hist_quantile_pc(edges_pc, counts, p),
                               atol=0)


def test_ecdf_hist_matches_step_ecdf():
    rng = np.random.default_rng(14)
    v = rng.normal(size=(20000, 3))
    edges, counts = histogram_accumulate(v, n_bins=2000)
    pts = rng.normal(size=(200, 3))
    got = ecdf_hist(edges, counts, pts)
    for j in range(3):
        want = np.searchsorted(np.sort(v[:, j]), pts[:, j],
                               side="right") / v.shape[0]
        np.testing.assert_allclose(got[:, j], want, atol=0.02)


def test_hist_quantile_ecdf_inverse_roundtrip():
    rng = np.random.default_rng(15)
    v = rng.normal(size=(20000, 2))
    edges, counts = histogram_accumulate(v, n_bins=2000)
    p = np.linspace(0.01, 0.99, 51)
    q = hist_quantile(edges, counts, p)
    f = ecdf_hist(edges, counts, q)
    np.testing.assert_allclose(f, np.broadcast_to(p[:, None], f.shape),
                               atol=0.02)


def test_hist_quantile_empty_cells_nan():
    edges = np.linspace(0.0, 1.0, 11)
    counts = np.zeros((10, 2))
    counts[:, 1] = 1.0
    q = hist_quantile(edges, counts, np.array([0.5]))
    assert np.isnan(q[0, 0])
    assert np.isfinite(q[0, 1])


def test_ecdf_hist_per_cell_edges_and_clamps():
    edges = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0]])
    counts = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    pts = np.array([[0.5, 10.5], [2.5, 12.5], [99.0, 99.0]])
    got = ecdf_hist(edges, counts, pts)
    # linear inside a bin: F(0.5) = 0.5 x (1/2 of the mass) = 0.25;
    # at/above the last edge -> 1
    np.testing.assert_allclose(got[:, 0], [0.25, 1.0, 1.0])
    np.testing.assert_allclose(got[:, 1], [0.25, 1.0, 1.0])
