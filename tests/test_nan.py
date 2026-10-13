"""NaN-handling tests: scattered NaNs, empty calibration cells, empty
target columns, histogram path."""

import numpy as np
import pytest

from pycdft import CDFt, apply_cdft, apply_cdft_hist, calibrate
from pycdft.math import ecdf, quantile_grid, histogram_accumulate
from naive_cdft import naive_cdft


@pytest.fixture
def data():
    rng = np.random.default_rng(17)
    return (rng.normal(0.0, 1.0, 1000),
            rng.normal(3.0, 1.0, 1000),
            rng.normal(4.0, 1.0, 1000))


def test_scattered_nan_output_positions(data):
    """NaN inputs pass through as NaN; all other positions are finite."""
    obs, cm_hist, cm_future = data
    rng = np.random.default_rng(18)
    cm_future_nan = cm_future.copy()
    cm_future_nan[rng.random(1000) < 0.05] = np.nan
    ds = CDFt(tail="clip").fit_transform(obs, cm_hist, cm_future_nan)
    assert np.isnan(ds).sum() == np.isnan(cm_future_nan).sum()
    assert np.isfinite(ds[~np.isnan(cm_future_nan)]).all()


def test_scattered_nan_calibration(data):
    """NaNs in the calibration series are excluded from counts and means."""
    obs, cm_hist, cm_future = data
    rng = np.random.default_rng(19)
    obs_nan = obs.copy()
    obs_nan[rng.random(1000) < 0.03] = np.nan
    cm_hist_nan = cm_hist.copy()
    cm_hist_nan[rng.random(1000) < 0.03] = np.nan
    ds = CDFt(tail="clip").fit_transform(obs_nan, cm_hist_nan, cm_future)
    assert np.isfinite(ds).all()
    # matches the NaN-aware naive reference (same exclusion convention)
    naive = naive_cdft(obs_nan, cm_hist_nan, cm_future, delta_shift="additive")
    assert np.sqrt(np.mean((ds - naive) ** 2)) < 1e-2


def test_nan_excluded_from_ecdf_counts(data):
    obs, cm_hist, _ = data
    v = cm_hist.copy()
    v[::3] = np.nan
    valid = np.sort(v[~np.isnan(v)])
    f = ecdf(valid)
    assert np.all(np.isfinite(f))
    assert np.allclose(f, (np.arange(valid.size) + 1) / valid.size)


def test_all_nan_calibration_1d_returns_nan(data):
    """Empty calibration series -> all-NaN output, no crash."""
    obs, cm_hist, cm_future = data
    obs_empty = np.full_like(obs, np.nan)
    ds = CDFt().fit_transform(obs_empty, cm_hist, cm_future)
    assert np.isnan(ds).all()
    cm_empty = np.full_like(cm_hist, np.nan)
    ds2 = CDFt().fit_transform(obs, cm_empty, cm_future)
    assert np.isnan(ds2).all()


def test_all_nan_calibration_cell_2d():
    """One empty cell among good ones: that column is NaN, the rest fine."""
    rng = np.random.default_rng(20)
    t, n = 300, 5
    obs = rng.normal(0.0, 1.0, (t, n))
    cm_hist = rng.normal(3.0, 1.0, (t, n))
    cm_future = rng.normal(4.0, 1.0, (t, n))
    obs[:, 2] = np.nan
    cm_hist[:, 2] = np.nan
    ds = CDFt().fit_transform(obs, cm_hist, cm_future)
    assert np.isnan(ds[:, 2]).all()
    assert np.isfinite(ds[:, [0, 1, 3, 4]]).all()


def test_all_nan_target_cell_2d():
    rng = np.random.default_rng(22)
    t, n = 200, 4
    obs = rng.normal(0.0, 1.0, (t, n))
    cm_hist = rng.normal(3.0, 1.0, (t, n))
    cm_future = rng.normal(4.0, 1.0, (t, n))
    cm_future[:, 1] = np.nan
    ds = CDFt().fit_transform(obs, cm_hist, cm_future)
    assert np.isnan(ds[:, 1]).all()
    assert np.isfinite(ds[:, [0, 2, 3]]).all()


def test_nan_only_hr_calibration_cell():
    """Only the observation (HR) calibration cell is empty -> NaN output."""
    rng = np.random.default_rng(23)
    t = 200
    obs = rng.normal(0.0, 1.0, (t, 2))
    cm_hist = rng.normal(3.0, 1.0, (t, 2))
    obs[:, 0] = np.nan                      # empty obs cell
    cm_hist[:, 1] = np.nan                  # empty model cell
    ds = CDFt().fit_transform(obs, cm_hist, cm_hist)
    assert np.isnan(ds[:, 0]).all()
    assert np.isnan(ds[:, 1]).all()


def test_histogram_path_with_nan():
    """The histogram target path must also be NaN-robust."""
    rng = np.random.default_rng(24)
    t, n = 500, 3
    obs = rng.normal(0.0, 1.0, (t, n))
    cm_hist = rng.normal(3.0, 1.0, (t, n))
    cm_future = rng.normal(4.0, 1.0, (t, n))
    cm_future[rng.random((t, n)) < 0.05] = np.nan
    cm_hist[:, 1] = np.nan
    cal = calibrate(cm_hist, obs, delta_shift="additive")
    ds = apply_cdft_hist(cm_future, cal, n_bins=2000, tail="clip")
    assert np.isnan(ds[:, 1]).all()                       # empty calib cell
    # elsewhere: NaN only where the target was NaN
    assert np.isnan(ds[:, [0, 2]]).sum() == \
        np.isnan(cm_future[:, [0, 2]]).sum()
    assert np.isfinite(ds[~np.isnan(cm_future)]).sum() == t * 2 - \
        np.isnan(cm_future[:, [0, 2]]).sum()


def test_histogram_all_nan_target():
    rng = np.random.default_rng(25)
    obs = rng.normal(0.0, 1.0, (100, 2))
    cm_hist = rng.normal(3.0, 1.0, (100, 2))
    cm_future = np.full((100, 2), np.nan)
    cal = calibrate(cm_hist, obs, delta_shift="additive")
    ds = apply_cdft_hist(cm_future, cal, n_bins=200, tail="clip")
    assert np.isnan(ds).all()


def test_histogram_accumulate_all_nan_does_not_warn():
    v = np.full((50, 2), np.nan)
    edges, counts = histogram_accumulate(v, n_bins=100)
    assert counts.sum() == 0
    assert np.isfinite(edges).all()
