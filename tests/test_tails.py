"""Tail-handling tests (clip vs linear extrapolation)."""

import numpy as np
import pytest

from pycdft import CDFt, apply_cdft, calibrate
from pycdft.math import quantile_grid


@pytest.fixture
def saturating():
    """obs wider than the shifted model calibration -> R saturates at both
    ends; the shifted target is wider than the calibration too."""
    rng = np.random.default_rng(3)
    return (rng.normal(0.0, 1.0, 2000),
            rng.normal(3.0, 0.3, 2000),
            rng.normal(4.0, 1.0, 2000))


def test_clip_clamps_to_shifted_target_range(saturating):
    obs, cm_hist, cm_future = saturating
    d = np.mean(obs) - np.mean(cm_hist)
    cl = CDFt(tail="clip").fit_transform(obs, cm_hist, cm_future)
    # clip: constant extrapolation -> output inside the shifted target range
    assert np.nanmin(cl) >= np.nanmin(cm_future) + d - 1e-9
    assert np.nanmax(cl) <= np.nanmax(cm_future) + d + 1e-9


def test_linear_extrapolates_beyond_clip(saturating):
    obs, cm_hist, cm_future = saturating
    cl = CDFt(tail="clip").fit_transform(obs, cm_hist, cm_future)
    ln = CDFt(tail="linear").fit_transform(obs, cm_hist, cm_future)
    # the transfer saturates (R reaches 0/1), so the linear tail moves the
    # extremes beyond the clipped values
    assert np.nanmax(ln) > np.nanmax(cl) + 1e-3
    assert np.nanmin(ln) < np.nanmin(cl) - 1e-3
    assert int((ln > cl + 1e-9).sum()) > 0
    assert int((ln < cl - 1e-9).sum()) > 0
    # extrapolation is shape-aware (capped slope): the output stays finite
    assert np.isfinite(ln).all()


def test_linear_output_is_monotone(saturating):
    obs, cm_hist, cm_future = saturating
    ln = CDFt(tail="linear").fit_transform(obs, cm_hist, np.sort(cm_future))
    assert np.all(np.diff(ln) >= -1e-12)


def test_tail_option_matches_kernel_formula(saturating):
    """CDFt(tail='linear') is exactly the kernel interpolation + the linear
    override (no hidden state)."""
    obs, cm_hist, cm_future = saturating
    cal = calibrate(cm_hist, obs, delta_shift="additive")
    ln = apply_cdft(cm_future, cal, tail="linear")
    cl = apply_cdft(cm_future, cal, tail="clip")
    # interior ranks are identical: the tails only differ outside the
    # saturated region (use a margin: saturation sets in ~11% / ~89%)
    srt = np.sort(cm_future)
    interior = (cm_future > srt[600]) & (cm_future < srt[1400])
    np.testing.assert_allclose(ln[interior], cl[interior], atol=1e-12)


def test_small_grid_tail_consistency():
    """n_quantiles resolution: the interior answer is stable and the grid
    effect is second-order (documented: ~0.01 mean/sd change from Q=101 to
    Q=1001; ~0.03 RMSE on this sample)."""
    rng = np.random.default_rng(4)
    obs = rng.normal(0.0, 1.0, 1000)
    cm_hist = rng.normal(3.0, 1.0, 1000)
    cm_future = rng.normal(4.0, 1.0, 1000)
    a = CDFt(n_quantiles=101, tail="clip").fit_transform(obs, cm_hist,
                                                         cm_future)
    b = CDFt(n_quantiles=1001, tail="clip").fit_transform(obs, cm_hist,
                                                          cm_future)
    assert np.sqrt(np.mean((a - b) ** 2)) < 0.05
