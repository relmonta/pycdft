"""High-level API tests (CDFt class)."""

import numpy as np
import pytest

from pycdft import CDFt, calibrate, apply_cdft, cdft_transform
from pycdft.math import quantile_grid


@pytest.fixture
def gaussian():
    rng = np.random.default_rng(7)
    return (rng.normal(0.0, 1.0, 1000),
            rng.normal(3.0, 1.0, 1000),
            rng.normal(4.0, 1.0, 1000))


def test_fit_transform_1d(gaussian):
    obs, cm_hist, cm_future = gaussian
    model = CDFt(n_quantiles=1001, tail="linear", delta_shift="additive")
    ds = model.fit_transform(obs, cm_hist, cm_future)
    assert ds.shape == cm_future.shape
    assert np.isfinite(ds).all()
    assert model.calibration_ is not None
    # matches the low-level pipeline exactly
    cal = calibrate(cm_hist, obs, p=quantile_grid(1001), delta_shift="additive")
    want = apply_cdft(cm_future, cal, p=quantile_grid(1001), tail="linear")
    np.testing.assert_array_equal(ds, want)


def test_fit_transform_2d_and_3d():
    rng = np.random.default_rng(1)
    obs = rng.normal(0.0, 1.0, (400, 6))
    cm_hist = rng.normal(3.0, 1.0, (400, 6))
    cm_future = rng.normal(4.0, 1.0, (400, 6))
    ds = CDFt().fit_transform(obs, cm_hist, cm_future)
    assert ds.shape == (400, 6)
    assert np.isfinite(ds).all()
    obs3 = obs.reshape(400, 2, 3)
    ch3 = cm_hist.reshape(400, 2, 3)
    cf3 = cm_future.reshape(400, 2, 3)
    ds3 = CDFt().fit_transform(obs3, ch3, cf3)
    assert ds3.shape == (400, 2, 3)
    np.testing.assert_allclose(ds3.reshape(400, 6), ds, atol=1e-12)


def test_fit_transform_accepts_float32():
    rng = np.random.default_rng(2)
    obs = rng.normal(0.0, 1.0, 500).astype(np.float32)
    cm_hist = rng.normal(3.0, 1.0, 500).astype(np.float32)
    cm_future = rng.normal(4.0, 1.0, 500).astype(np.float32)
    ds = CDFt().fit_transform(obs, cm_hist, cm_future)
    assert np.isfinite(ds).all()
    assert ds.dtype == np.float64


def test_fit_and_transform_are_cached(gaussian):
    obs, cm_hist, cm_future = gaussian
    model = CDFt().fit(obs, cm_hist)
    a = model.transform(cm_future)
    b = model.transform(cm_future)
    np.testing.assert_array_equal(a, b)


def test_transform_requires_fit(gaussian):
    obs, cm_hist, cm_future = gaussian
    with pytest.raises(RuntimeError):
        CDFt().transform(cm_future)


def test_histogram_mode(gaussian):
    obs, cm_hist, cm_future = gaussian
    ds_exact = CDFt().fit_transform(obs, cm_hist, cm_future)
    ds_hist = CDFt(use_histograms=True, n_bins=4000) \
        .fit_transform(obs, cm_hist, cm_future)
    assert np.sqrt(np.mean((ds_exact - ds_hist) ** 2)) < 0.05


def test_defaults():
    m = CDFt()
    assert m.n_quantiles == 1001
    assert m.tail == "linear"
    assert m.delta_shift == "additive"
    assert m.use_histograms is False
    assert m.n_bins == 2000
    assert m.workers == 1
    assert m.calibration_ is None
    p = m.probability_grid
    assert p.shape == (1001,) and p[0] == 0.0 and p[-1] == 1.0


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        CDFt(n_quantiles=1)
    with pytest.raises(ValueError):
        CDFt(tail="cubic")
    with pytest.raises(ValueError):
        CDFt(delta_shift="no_shift")
    with pytest.raises(ValueError):
        CDFt(n_bins=1)


def test_repr_and_str(gaussian):
    m = CDFt()
    assert "CDFt(" in repr(m)
    assert "n_quantiles=1001" in repr(m)


def test_cdft_transform_one_shot(gaussian):
    obs, cm_hist, cm_future = gaussian
    a = cdft_transform(obs, cm_hist, cm_future)
    b = CDFt().fit_transform(obs, cm_hist, cm_future)
    np.testing.assert_array_equal(a, b)


def test_delta_shift_none_mode(gaussian):
    obs, cm_hist, cm_future = gaussian
    ds = CDFt(delta_shift="none").fit_transform(obs, cm_hist, cm_future)
    assert np.isfinite(ds).all()
    # same as low-level with explicit none
    cal = calibrate(cm_hist, obs, delta_shift="none")
    want = apply_cdft(cm_future, cal, tail="linear")
    np.testing.assert_array_equal(ds, want)


def test_tail_clip_mode(gaussian):
    obs, cm_hist, cm_future = gaussian
    ds = CDFt(tail="clip").fit_transform(obs, cm_hist, cm_future)
    # clip: the output stays within the (shifted) target range
    d = np.mean(obs) - np.mean(cm_hist)
    assert np.nanmin(ds) >= np.nanmin(cm_future) + d - 1e-9
    assert np.nanmax(ds) <= np.nanmax(cm_future) + d + 1e-9
