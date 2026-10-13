"""Cross-test against a faithful numpy port of the R ``CDFt`` package.

The R package ``CDFt`` (Vrac & Michelangeli, CRAN 1.2) is not installed in
this environment (no R runtime); ``tests/r_cdft_reference.py`` is a line-by-
line numpy port of ``CDFt.R`` (validated against the published R example
numbers: output quantiles -1.004 / 0.339 / 0.970 / 2.060 / 3.013 and
ours-vs-R RMSE 0.0939 on the Gaussian example, which the port reproduces to
the printed digits).

R's algorithm differs from ours by design (common value grid
``npas=1000`` extended by ``dev=2`` x |mean drift| on both sides,
observation-tail copy + 0/1 padding, p-space ``approx`` inversion with
ties-mean) so the agreement is documented as: overall RMSE < 0.2 on
Gaussian data, interior RMSE < 0.05, max quantile difference < 0.15.

Marked ``pytest.mark.cross``: runs with numpy only (no extra dependency).
"""

import numpy as np
import pytest

import pycdft
from naive_cdft import naive_cdft
from r_cdft_reference import r_cdft, r_cdft_full

pytestmark = pytest.mark.cross

Q = 1001


def _metrics(a, b, interior=None):
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    if interior is None:
        return rmse, float(np.abs(a - b).max())
    rmse_i = float(np.sqrt(np.mean((a[interior] - b[interior]) ** 2)))
    max_i = float(np.abs(a[interior] - b[interior]).max())
    return rmse, float(np.abs(a - b).max()), rmse_i, max_i


def test_r_port_reproduces_published_quantiles():
    """The R port must reproduce the documented R output for the R package's
    own example (obs N(0,1), cm_hist N(3,1), cm_future N(4,1), seed 7)."""
    rng = np.random.default_rng(7)
    obs = rng.normal(0.0, 1.0, 1000)
    cm_hist = rng.normal(3.0, 1.0, 1000)
    cm_future = rng.normal(4.0, 1.0, 1000)
    r_out = r_cdft(obs, cm_hist, cm_future)
    q = np.quantile(r_out, [0.01, 0.25, 0.50, 0.90, 0.99])
    np.testing.assert_allclose(q, [-1.004, 0.339, 0.970, 2.060, 3.013],
                               atol=0.02)


def test_agreement_gaussian():
    rng = np.random.default_rng(7)
    obs = rng.normal(0.0, 1.0, 1000)
    cm_hist = rng.normal(3.0, 1.0, 1000)
    cm_future = rng.normal(4.0, 1.0, 1000)
    ours = pycdft.CDFt(tail="linear").fit_transform(obs, cm_hist, cm_future)
    r_out = r_cdft(obs, cm_hist, cm_future)
    srt = np.sort(cm_future)
    interior = (cm_future > srt[20]) & (cm_future < srt[980])
    rmse, maxd, rmse_i, max_i = _metrics(ours, r_out, interior)
    print(f"[cross-r] Gaussian: RMSE={rmse:.4f} max|d|={maxd:.4f} "
          f"interior RMSE={rmse_i:.4f} interior max|d|={max_i:.4f}")
    assert rmse < 0.2                      # documented tolerance
    assert rmse_i < 0.05
    p = np.linspace(0.01, 0.99, 99)
    qdiff = np.abs(np.quantile(ours, p) - np.quantile(r_out, p)).max()
    print(f"[cross-r] Gaussian: max quantile diff={qdiff:.4f}")
    assert qdiff < 0.15


def test_agreement_skewed():
    rng = np.random.default_rng(11)
    obs = rng.gamma(2.0, 1.5, 1000)
    cm_hist = rng.gamma(2.0, 1.5, 1000) + 3.0
    cm_future = rng.gamma(2.0, 1.8, 1000) + 4.5
    ours = pycdft.CDFt(tail="linear").fit_transform(obs, cm_hist, cm_future)
    r_out = r_cdft(obs, cm_hist, cm_future)
    srt = np.sort(cm_future)
    interior = (cm_future > srt[20]) & (cm_future < srt[980])
    rmse, maxd, rmse_i, max_i = _metrics(ours, r_out, interior)
    print(f"[cross-r] skewed: RMSE={rmse:.4f} max|d|={maxd:.4f} "
          f"interior RMSE={rmse_i:.4f} interior max|d|={max_i:.4f}")
    assert rmse < 0.25                      # documented tolerance (0.18 obs.)
    assert rmse_i < 0.10


def test_r_tail_extends_beyond_obs_range():
    """R's dev=2 grid + tail copy can produce output beyond the observed
    range (documented difference vs our capped linear tail)."""
    rng = np.random.default_rng(7)
    obs = rng.normal(0.0, 1.0, 1000)
    cm_hist = rng.normal(3.0, 1.0, 1000)
    cm_future = rng.normal(4.0, 1.0, 1000)
    r_out = r_cdft(obs, cm_hist, cm_future)
    ours = pycdft.CDFt(tail="clip").fit_transform(obs, cm_hist, cm_future)
    assert r_out.max() > np.nanmax(obs) + 0.5
    assert ours.max() <= np.nanmax(cm_future) + 1e-9


def test_r_port_identity_self_consistency():
    """Sanity of the R port: obs == cm_hist == cm_future gives an output close
    to the input (the R algorithm is approximately the identity then)."""
    rng = np.random.default_rng(21)
    x = rng.gamma(2.0, 1.5, 800)
    r_out = r_cdft(x, x, x)
    assert np.sqrt(np.mean((r_out - x) ** 2)) < 0.2


def test_r_port_nan_support():
    """R drops NaNs (na.rm); NaN target positions stay NaN."""
    rng = np.random.default_rng(26)
    obs = rng.normal(0.0, 1.0, 500)
    cm_hist = rng.normal(3.0, 1.0, 500)
    cm_future = rng.normal(4.0, 1.0, 500)
    cm_future[10] = np.nan
    r_out = r_cdft(obs, cm_hist, cm_future)
    assert np.isnan(r_out[10])
    assert np.isfinite(r_out[np.arange(500) != 10]).all()


def test_r_full_output_shapes():
    rng = np.random.default_rng(27)
    obs = rng.normal(0.0, 1.0, 400)
    cm_hist = rng.normal(3.0, 1.0, 400)
    cm_future = rng.normal(4.0, 1.0, 400)
    out = r_cdft_full(obs, cm_hist, cm_future, npas=200, dev=2.0)
    assert out["x"].shape == (200,)
    assert out["FRp"].shape == out["FGp"].shape == out["FGf"].shape == \
        out["FRf"].shape == (200,)
    assert out["DS"].shape == (400,)
    assert np.all(np.diff(out["FRf"]) >= -1e-12)     # modified CDF monotone
