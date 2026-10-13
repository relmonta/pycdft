"""Core CDF-t correctness tests (calibration, shift, composition, histograms)."""

import numpy as np
import pytest

from pycdft import (apply_cdft, apply_cdft_hist, calibrate, calibrate_hist,
                    delta_from_means, histograms, transfer)
from pycdft.math import quantile_grid
from naive_cdft import naive_cdft, quantile_mapping, step_ecdf


@pytest.fixture
def gaussian():
    """The R-package example data: obs N(0,1), cm_hist N(3,1), cm_future N(4,1)."""
    rng = np.random.default_rng(7)
    return (rng.normal(0.0, 1.0, 1000),
            rng.normal(3.0, 1.0, 1000),
            rng.normal(4.0, 1.0, 1000))


@pytest.fixture
def skewed():
    """Gamma-distributed case with drift and variance change."""
    rng = np.random.default_rng(11)
    return (rng.gamma(2.0, 1.5, 1000),
            rng.gamma(2.0, 1.5, 1000) + 3.0,
            rng.gamma(2.0, 1.8, 1000) + 4.5)


# ---------------------------------------------------------------------------
# Composition / reduction properties
# ---------------------------------------------------------------------------

def test_reduces_to_classical_quantile_mapping():
    """When F_m^tar = F_m^cal (cm_future == cm_hist) CDF-t must reduce to
    classical quantile mapping y = q_o^cal(F_m^cal(x)) -- exactly when the
    classical QM output lies inside the model support (here obs and cm_hist
    share the same support, so the q_m o F_m step is the identity up to the
    ECDF discretization)."""
    rng = np.random.default_rng(29)
    obs = rng.normal(0.0, 1.0, 2000)
    cm_hist = rng.normal(0.0, 1.0, 2000)
    cm_future = cm_hist.copy()
    out = apply_cdft(cm_future, calibrate(cm_hist, obs, delta_shift="none"),
                     p=quantile_grid(1001), tail="clip")
    qm = quantile_mapping(obs, cm_hist, cm_future)
    # interior ranks: exact to the ECDF-step resolution; extremes slightly
    # looser (sparse order statistics)
    srt = np.sort(cm_future)
    interior = (cm_future > srt[50]) & (cm_future < srt[1950])
    assert np.sqrt(np.mean((out[interior] - qm[interior]) ** 2)) < 0.02
    assert np.abs(out[interior] - qm[interior]).max() < 0.05
    assert np.sqrt(np.mean((out - qm) ** 2)) < 0.05


def test_composition_identity_naive(gaussian):
    """Grid-based CDF-t equals the direct per-value composition (the exact
    chain q_m^tar(F_m^cal(q_o^cal(F_m^tar(x))))) to grid resolution."""
    obs, cm_hist, cm_future = gaussian
    out = apply_cdft(cm_future, calibrate(cm_hist, obs, delta_shift="none"),
                     p=quantile_grid(1001), tail="clip")
    naive = naive_cdft(obs, cm_hist, cm_future, delta_shift="none")
    # disjoint supports -> steep transfer -> larger grid-interpolation error
    srt = np.sort(cm_future)
    interior = (cm_future > srt[20]) & (cm_future < srt[980])
    assert np.sqrt(np.mean((out[interior] - naive[interior]) ** 2)) < 0.05
    assert np.abs(out[interior] - naive[interior]).max() < 0.5
    assert np.sqrt(np.mean((out - naive) ** 2)) < 0.05


def test_composition_identity_with_additive_shift(gaussian):
    obs, cm_hist, cm_future = gaussian
    out = apply_cdft(cm_future, calibrate(cm_hist, obs, delta_shift="additive"),
                     p=quantile_grid(1001), tail="clip")
    naive = naive_cdft(obs, cm_hist, cm_future, delta_shift="additive")
    srt = np.sort(cm_future)
    interior = (cm_future > srt[20]) & (cm_future < srt[980])
    assert np.sqrt(np.mean((out[interior] - naive[interior]) ** 2)) < 1e-2
    assert np.abs(out[interior] - naive[interior]).max() < 0.1
    assert np.sqrt(np.mean((out - naive) ** 2)) < 0.05


def test_identity_self_consistency():
    """obs == cm_hist == cm_future: CDF-t must approximately reproduce the
    input (no bias to correct)."""
    rng = np.random.default_rng(21)
    x = rng.gamma(2.0, 1.5, 800)
    out = apply_cdft(x, calibrate(x, x, delta_shift="none"),
                     p=quantile_grid(1001), tail="clip")
    assert np.sqrt(np.mean((out - x) ** 2)) < 0.05


# ---------------------------------------------------------------------------
# Delta shift
# ---------------------------------------------------------------------------

def test_delta_from_means_modes():
    ml = np.array([3.0, 5.0, np.nan, 0.0])
    mh = np.array([1.0, 5.0, np.nan, 2.0])
    np.testing.assert_allclose(delta_from_means(ml, mh, "additive"),
                               [-2.0, 0.0, np.nan, 2.0])
    np.testing.assert_allclose(delta_from_means(ml, mh, "multiplicative"),
                               [1.0 / 3, 1.0, 1.0, 1.0])  # nan -> 1.0, 0 -> 1.0
    np.testing.assert_allclose(delta_from_means(ml, mh, "none"), [0.0] * 4)


def test_delta_shift_additive_semantics(gaussian):
    """Additive shift commutes with the CDF-t composition: shifting the model
    series by d then running with delta_shift='none' is identical to running
    with delta_shift='additive'."""
    obs, cm_hist, cm_future = gaussian
    d = np.mean(obs) - np.mean(cm_hist)
    assert np.array_equal(np.sort(cm_hist) + d, np.sort(cm_hist + d))
    cal = calibrate(cm_hist, obs, delta_shift="additive")
    out = apply_cdft(cm_future, cal, p=quantile_grid(1001), tail="clip")
    cal2 = calibrate(cm_hist + d, obs, delta_shift="none")
    out2 = apply_cdft(cm_future + d, cal2, p=quantile_grid(1001), tail="clip")
    np.testing.assert_allclose(out, out2, atol=1e-12)
    # delta is the mean difference
    np.testing.assert_allclose(cal.delta, np.full(1, d), atol=1e-12)


def test_delta_shift_multiplicative_semantics(skewed):
    """Multiplicative shift commutes with the composition (positive data)."""
    obs, cm_hist, cm_future = skewed
    d = np.mean(obs) / np.mean(cm_hist)
    cal = calibrate(cm_hist, obs, delta_shift="multiplicative")
    out = apply_cdft(cm_future, cal, p=quantile_grid(1001), tail="clip")
    cal2 = calibrate(cm_hist * d, obs, delta_shift="none")
    out2 = apply_cdft(cm_future * d, cal2, p=quantile_grid(1001), tail="clip")
    np.testing.assert_allclose(out, out2, atol=1e-12)
    np.testing.assert_allclose(cal.delta, np.full(1, d), atol=1e-12)


def test_delta_shift_none_means_no_shift(gaussian):
    obs, cm_hist, cm_future = gaussian
    cal = calibrate(cm_hist, obs, delta_shift="none")
    np.testing.assert_allclose(cal.delta, [0.0])
    out = apply_cdft(cm_future, cal, p=quantile_grid(1001), tail="clip")
    naive = naive_cdft(obs, cm_hist, cm_future, delta_shift="none")
    # with disjoint supports (obs ~ N(0,1) vs model ~ N(3,1)) the transfer is
    # steep in the left tail of the model, so the grid-interpolation error is
    # larger than with the mean shift (documented: no-shift is the least
    # comparable configuration)
    srt = np.sort(cm_future)
    interior = (cm_future > srt[20]) & (cm_future < srt[980])
    assert np.sqrt(np.mean((out[interior] - naive[interior]) ** 2)) < 0.05
    assert np.sqrt(np.mean((out - naive) ** 2)) < 0.05


def test_delta_shift_multiplicative_guard_zero_mean():
    """Near-zero model mean: the multiplicative shift falls back to 1.0."""
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    cm_hist = np.array([1e-13, -1e-13, 2e-13, -2e-13])
    d = delta_from_means(np.array([np.mean(cm_hist)]), np.array([np.mean(obs)]),
                         "multiplicative")
    assert d[0] == 1.0
    p = quantile_grid(101)
    cal = calibrate(cm_hist, obs, delta_shift="multiplicative", p=p)
    out = apply_cdft(cm_hist, cal, p=p)
    assert np.isfinite(out).all()


def test_shift_applied_to_both_calibration_and_target(gaussian):
    """The output is in the shifted frame: the mean bias is removed, the
    model change (calibration -> target) is preserved."""
    obs, cm_hist, cm_future = gaussian
    d = np.mean(obs) - np.mean(cm_hist)
    out = apply_cdft(cm_future, calibrate(cm_hist, obs, delta_shift="additive"),
                     p=quantile_grid(1001), tail="clip")
    # mean(output) ~= mean(obs) + [mean(cm_future) - mean(cm_hist)]
    target_mean = np.mean(obs) + (np.mean(cm_future) - np.mean(cm_hist))
    assert np.mean(out) == pytest.approx(target_mean, abs=0.1)
    # the shift equals the calibration mean difference
    assert d == pytest.approx(np.mean(obs) - np.mean(cm_hist), abs=1e-12)


def test_monotonicity(gaussian):
    obs, cm_hist, cm_future = gaussian
    out = apply_cdft(np.sort(cm_future),
                     calibrate(cm_hist, obs, delta_shift="additive"),
                     p=quantile_grid(1001), tail="linear")
    assert np.all(np.diff(out) >= -1e-12)


def test_inputs_not_mutated(gaussian):
    obs, cm_hist, cm_future = gaussian
    obs_c, ch_c, cf_c = obs.copy(), cm_hist.copy(), cm_future.copy()
    calibrate(cm_hist, obs, delta_shift="additive")
    apply_cdft(cm_future, calibrate(cm_hist, obs, delta_shift="additive"),
               p=quantile_grid(1001))
    np.testing.assert_array_equal(obs, obs_c)
    np.testing.assert_array_equal(cm_hist, ch_c)
    np.testing.assert_array_equal(cm_future, cf_c)


def test_inputs_not_mutated_float32():
    rng = np.random.default_rng(31)
    obs = rng.normal(size=(100, 3)).astype(np.float32)
    cm_hist = rng.normal(size=(100, 3)).astype(np.float32)
    obs_c, ch_c = obs.copy(), cm_hist.copy()
    calibrate(cm_hist, obs, delta_shift="additive")
    np.testing.assert_array_equal(obs, obs_c)
    np.testing.assert_array_equal(cm_hist, ch_c)


# ---------------------------------------------------------------------------
# Calibration object
# ---------------------------------------------------------------------------

def test_calibration_shapes_and_grid(gaussian):
    obs, cm_hist, _ = gaussian
    p = quantile_grid(501)
    cal = calibrate(cm_hist, obs, p=p, delta_shift="additive")
    assert cal.R.shape == (501, 1)
    assert cal.delta.shape == (1,)
    assert cal.mode == "additive"
    assert cal.n_quantiles == 501
    assert np.all(np.diff(cal.R[:, 0]) >= -1e-15)      # R is a CDF: monotone


def test_calibration_matches_manual_R(gaussian):
    """R = F_m^cal(q_o^cal(p)) computed by hand with the package kernels."""
    obs, cm_hist, _ = gaussian
    p = quantile_grid(101)
    cal = calibrate(cm_hist, obs, delta_shift="none", p=p)
    q_obs = np.quantile(obs, p)
    want = step_ecdf(cm_hist)(q_obs)
    np.testing.assert_allclose(cal.R[:, 0], want, atol=1e-12)


# ---------------------------------------------------------------------------
# Histogram path
# ---------------------------------------------------------------------------

def test_calibrate_hist_matches_calibrate(skewed):
    obs, cm_hist, cm_future = skewed
    p = quantile_grid(501)
    cal = calibrate(cm_hist, obs, p=p, delta_shift="additive")
    hist = histograms(cm_hist, n_bins=4000)
    hist_o = histograms(obs, n_bins=4000)
    cal_h = calibrate_hist(hist, hist_o,
                           (np.mean(cm_hist), np.mean(obs)), p=p,
                           delta_shift="additive")
    np.testing.assert_allclose(cal_h.delta, cal.delta, atol=1e-12)
    np.testing.assert_allclose(cal_h.R, cal.R, atol=1e-2)


def test_apply_cdft_hist_matches_apply_cdft(skewed):
    obs, cm_hist, cm_future = skewed
    p = quantile_grid(501)
    cal = calibrate(cm_hist, obs, p=p, delta_shift="additive")
    exact = apply_cdft(cm_future, cal, p=p, tail="linear")
    hist = apply_cdft_hist(cm_future, cal, p=p, n_bins=4000, tail="linear")
    assert np.sqrt(np.mean((exact - hist) ** 2)) < 0.02


def test_transfer_outputs(gaussian):
    obs, cm_hist, cm_future = gaussian
    p = quantile_grid(101)
    cal = calibrate(cm_hist, obs, p=p, delta_shift="none")

    def q_tar(q, s, e):
        # per-column type-7 quantiles (q is (Q, e - s)); the test data is
        # 1-D (single cell)
        x = cm_future[:, None] if cm_future.ndim == 1 else cm_future
        return np.stack([np.quantile(x[:, j], q[:, j - s])
                         for j in range(s, e)], axis=1)

    x_grid, T, k_lo, k_hi, bad = transfer(q_tar, cal.R, p)
    assert x_grid.shape == T.shape == (101, 1)
    assert not bad.any()
    assert np.all(np.diff(x_grid[:, 0]) > 0)
    assert 0 <= k_lo[0] <= k_hi[0] < 101
    # the transfer maps the target's own quantiles to the mapped quantiles
    np.testing.assert_allclose(x_grid[:, 0], np.quantile(cm_future, p))
    np.testing.assert_allclose(T[:, 0], np.quantile(cm_future, cal.R[:, 0]))


def test_grid_size_mismatch_raises(gaussian):
    obs, cm_hist, _ = gaussian
    cal = calibrate(cm_hist, obs, p=quantile_grid(501))
    with pytest.raises(ValueError):
        apply_cdft(cm_hist, cal, p=quantile_grid(1001))


def test_apply_defaults_grid_from_calibration(gaussian):
    """apply_cdft without an explicit p uses the calibration's own grid."""
    obs, cm_hist, cm_future = gaussian
    p = quantile_grid(501)
    cal = calibrate(cm_hist, obs, p=p, delta_shift="additive")
    a = apply_cdft(cm_future, cal, tail="clip")
    b = apply_cdft(cm_future, cal, p=p, tail="clip")
    np.testing.assert_array_equal(a, b)


def test_invalid_delta_shift_raises(gaussian):
    obs, cm_hist, _ = gaussian
    with pytest.raises(ValueError):
        calibrate(cm_hist, obs, delta_shift="no_shift")
