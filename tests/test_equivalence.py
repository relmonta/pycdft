"""Equivalence / regression tests: batched vs per-cell, dtypes, chunking,
threads, and the naive reference implementation."""

import numpy as np
import pytest

from pycdft import CDFt, calibrate, apply_cdft
from pycdft.core import DEFAULT_CHUNK_BYTES
from pycdft.math import quantile_grid
from naive_cdft import naive_cdft


@pytest.fixture
def gaussian():
    rng = np.random.default_rng(7)
    return (rng.normal(0.0, 1.0, 1000),
            rng.normal(3.0, 1.0, 1000),
            rng.normal(4.0, 1.0, 1000))


@pytest.fixture
def grid2d():
    rng = np.random.default_rng(42)
    n_cells, t = 7, 600
    cell_scale = np.linspace(-2.0, 2.0, n_cells)
    obs = rng.normal(0.0, 1.0, (t, n_cells)) + cell_scale
    cm_hist = rng.normal(3.0, 1.0, (t, n_cells)) + cell_scale
    cm_future = rng.normal(4.0, 1.0, (t, n_cells)) + cell_scale
    return obs, cm_hist, cm_future


def test_batched_matches_per_cell(grid2d):
    """(T, N) batched call == loop over the 1-D API per cell."""
    obs, cm_hist, cm_future = grid2d
    ds = CDFt().fit_transform(obs, cm_hist, cm_future)
    for j in range(obs.shape[1]):
        one = CDFt().fit_transform(obs[:, j], cm_hist[:, j], cm_future[:, j])
        np.testing.assert_allclose(ds[:, j], one, atol=1e-12)


def test_float32_matches_float64(grid2d):
    """float32 inputs (memory-lean storage) agree with float64 to rounding."""
    obs, cm_hist, cm_future = grid2d
    ds64 = CDFt().fit_transform(obs, cm_hist, cm_future)
    ds32 = CDFt().fit_transform(obs.astype(np.float32),
                                cm_hist.astype(np.float32),
                                cm_future.astype(np.float32))
    scale = np.abs(ds64).max() + 1e-8
    assert np.nanmax(np.abs(ds32 - ds64)) / scale < 1e-5


def test_chunked_matches_unchunked(grid2d):
    """Small chunk_bytes (forcing cell chunks) must not change the result."""
    obs, cm_hist, cm_future = grid2d
    a = apply_cdft(cm_future,
                   calibrate(cm_hist, obs, delta_shift="additive"),
                   tail="linear")
    cal = calibrate(cm_hist, obs, delta_shift="additive",
                    chunk_bytes=8_000)
    b = apply_cdft(cm_future, cal, tail="linear", chunk_bytes=8_000)
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_workers_do_not_change_results(grid2d):
    obs, cm_hist, cm_future = grid2d
    a = CDFt(workers=1).fit_transform(obs, cm_hist, cm_future)
    b = CDFt(workers=4).fit_transform(obs, cm_hist, cm_future)
    np.testing.assert_array_equal(a, b)


def test_default_chunk_bytes_is_2gib():
    assert DEFAULT_CHUNK_BYTES == 2 ** 31


# ---------------------------------------------------------------------------
# vs the naive direct composition (ibicus-style exact chain)
# ---------------------------------------------------------------------------

def test_naive_agreement_gaussian(gaussian):
    obs, cm_hist, cm_future = gaussian
    ds = CDFt(tail="clip").fit_transform(obs, cm_hist, cm_future)
    naive = naive_cdft(obs, cm_hist, cm_future, delta_shift="additive")
    srt = np.sort(cm_future)
    interior = (cm_future > srt[20]) & (cm_future < srt[980])
    assert np.sqrt(np.mean((ds[interior] - naive[interior]) ** 2)) < 1e-2
    assert np.abs(ds[interior] - naive[interior]).max() < 0.1
    assert np.sqrt(np.mean((ds - naive) ** 2)) < 0.05


def test_naive_agreement_skewed():
    rng = np.random.default_rng(11)
    obs = rng.gamma(2.0, 1.5, 1000)
    cm_hist = rng.gamma(2.0, 1.5, 1000) + 3.0
    cm_future = rng.gamma(2.0, 1.8, 1000) + 4.5
    ds = CDFt(tail="clip").fit_transform(obs, cm_hist, cm_future)
    naive = naive_cdft(obs, cm_hist, cm_future, delta_shift="additive")
    srt = np.sort(cm_future)
    interior = (cm_future > srt[20]) & (cm_future < srt[980])
    assert np.sqrt(np.mean((ds[interior] - naive[interior]) ** 2)) < 2e-2
    assert np.abs(ds[interior] - naive[interior]).max() < 0.1
    assert np.sqrt(np.mean((ds - naive) ** 2)) < 0.05


def test_naive_agreement_multiplicative():
    rng = np.random.default_rng(12)
    obs = rng.gamma(2.0, 2.0, 1000)
    cm_hist = rng.gamma(2.0, 4.0, 1000)
    cm_future = rng.gamma(2.0, 5.0, 1000)
    ds = CDFt(tail="clip", delta_shift="multiplicative") \
        .fit_transform(obs, cm_hist, cm_future)
    naive = naive_cdft(obs, cm_hist, cm_future, delta_shift="multiplicative")
    srt = np.sort(cm_future)
    interior = (cm_future > srt[20]) & (cm_future < srt[980])
    assert np.sqrt(np.mean((ds[interior] - naive[interior]) ** 2)) < 2e-2
    assert np.abs(ds[interior] - naive[interior]).max() < 0.2
    assert np.sqrt(np.mean((ds - naive) ** 2)) < 0.05


def test_naive_agreement_no_shift(gaussian):
    obs, cm_hist, cm_future = gaussian
    ds = CDFt(tail="clip", delta_shift="none").fit_transform(obs, cm_hist,
                                                             cm_future)
    naive = naive_cdft(obs, cm_hist, cm_future, delta_shift="none")
    srt = np.sort(cm_future)
    interior = (cm_future > srt[20]) & (cm_future < srt[980])
    # disjoint supports -> steep transfer -> documented larger tolerance
    assert np.sqrt(np.mean((ds[interior] - naive[interior]) ** 2)) < 0.05
    assert np.sqrt(np.mean((ds - naive) ** 2)) < 0.05
