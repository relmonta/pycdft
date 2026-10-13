"""Cross-test against ibicus (ECMWF; Apache-2.0).

ibicus implements the same CDF-t composition but *without* a probability
grid: each value goes through the exact chain
``q_m^tar(F_m^cal(q_o^cal(F_m^tar(x))))`` (with ``ecdf_method="step_function"``
and ``iecdf_method="linear"`` matching our step ECDF + type-7 quantile
conventions).  The residual difference with our grid-based version is the
discretization of the transfer curve on the Q-point grid (documented:
interior RMSE ~0.006, tail-rank RMSE ~0.06 on Gaussian data, versus ~0.5
RMSE for the delta-shift choice and ~0.3 for running windows).

Skipped automatically when ibicus is not installed; marked
``pytest.mark.cross``.
"""

import numpy as np
import pytest

ibicus = pytest.importorskip("ibicus")
from ibicus.debias._cdft import CDFt as IbicusCDFt  # noqa: E402

import pycdft  # noqa: E402

pytestmark = pytest.mark.cross

IBICUS_KW = dict(
    running_window_mode=False,
    running_window_mode_over_years_of_cm_future=False,
    ecdf_method="step_function",
    iecdf_method="linear",
    SSR=False,
)


def _ibicus(obs, cm_hist, cm_future, delta_shift="additive"):
    """Run ibicus CDFt on 1-D series (ibicus needs (T, x, y) input)."""
    d = IbicusCDFt(delta_shift=delta_shift, **IBICUS_KW)
    return d.apply(obs[:, None, None], cm_hist[:, None, None],
                   cm_future[:, None, None], progressbar=False)[:, 0, 0]


def _interior_mask(x, lo=0.02, hi=0.98):
    srt = np.sort(x)
    return (x > srt[int(lo * x.size)]) & (x < srt[int(hi * x.size) - 1])


def test_agreement_gaussian_additive():
    rng = np.random.default_rng(7)
    obs = rng.normal(0.0, 1.0, 1000)
    cm_hist = rng.normal(3.0, 1.0, 1000)
    cm_future = rng.normal(4.0, 1.0, 1000)
    ours = pycdft.CDFt(tail="linear").fit_transform(obs, cm_hist, cm_future)
    theirs = _ibicus(obs, cm_hist, cm_future, delta_shift="additive")
    interior = _interior_mask(cm_future)
    rmse = float(np.sqrt(np.mean((ours - theirs) ** 2)))
    rmse_i = float(np.sqrt(np.mean((ours[interior] - theirs[interior]) ** 2)))
    max_i = float(np.abs(ours[interior] - theirs[interior]).max())
    print(f"[cross-ibicus] Gaussian/additive: RMSE={rmse:.4f} "
          f"interior RMSE={rmse_i:.4f} interior max|d|={max_i:.4f}")
    assert rmse_i < 1e-2                    # documented interior agreement
    assert rmse < 0.05
    # mean bias removed by both (same shifted frame)
    assert abs(np.mean(ours) - np.mean(theirs)) < 0.05


def test_agreement_gaussian_no_shift():
    rng = np.random.default_rng(7)
    obs = rng.normal(0.0, 1.0, 1000)
    cm_hist = rng.normal(3.0, 1.0, 1000)
    cm_future = rng.normal(4.0, 1.0, 1000)
    # with disjoint supports the no-shift transfer is steep; clip (constant)
    # tails keep the two implementations comparable (the documented no-shift
    # divergence 0.17-0.56 RMSE is dominated by the tail models)
    ours = pycdft.CDFt(tail="clip", delta_shift="none") \
        .fit_transform(obs, cm_hist, cm_future)
    theirs = _ibicus(obs, cm_hist, cm_future, delta_shift="no_shift")
    interior = _interior_mask(cm_future)
    rmse_i = float(np.sqrt(np.mean((ours[interior] - theirs[interior]) ** 2)))
    print(f"[cross-ibicus] Gaussian/no_shift: interior RMSE={rmse_i:.4f}")
    assert rmse_i < 0.05


def test_agreement_multiplicative():
    rng = np.random.default_rng(12)
    obs = rng.gamma(2.0, 2.0, 1000)
    cm_hist = rng.gamma(2.0, 4.0, 1000)
    cm_future = rng.gamma(2.0, 5.0, 1000)
    ours = pycdft.CDFt(tail="linear", delta_shift="multiplicative") \
        .fit_transform(obs, cm_hist, cm_future)
    theirs = _ibicus(obs, cm_hist, cm_future, delta_shift="multiplicative")
    interior = _interior_mask(cm_future)
    rmse_i = float(np.sqrt(np.mean((ours[interior] - theirs[interior]) ** 2)))
    print(f"[cross-ibicus] gamma/multiplicative: interior RMSE={rmse_i:.4f}")
    assert rmse_i < 2e-2


def test_agreement_skewed():
    rng = np.random.default_rng(11)
    obs = rng.gamma(2.0, 1.5, 1000)
    cm_hist = rng.gamma(2.0, 1.5, 1000) + 3.0
    cm_future = rng.gamma(2.0, 1.8, 1000) + 4.5
    ours = pycdft.CDFt(tail="linear").fit_transform(obs, cm_hist, cm_future)
    theirs = _ibicus(obs, cm_hist, cm_future, delta_shift="additive")
    interior = _interior_mask(cm_future)
    rmse_i = float(np.sqrt(np.mean((ours[interior] - theirs[interior]) ** 2)))
    print(f"[cross-ibicus] skewed/additive: interior RMSE={rmse_i:.4f}")
    assert rmse_i < 2e-2


def test_agreement_multicell_grid():
    rng = np.random.default_rng(42)
    t, n = 500, 4
    scale = np.linspace(-2.0, 2.0, n)
    obs = rng.normal(0.0, 1.0, (t, n)) + scale
    cm_hist = rng.normal(3.0, 1.0, (t, n)) + scale
    cm_future = rng.normal(4.0, 1.0, (t, n)) + scale
    ours = pycdft.CDFt(tail="linear").fit_transform(obs, cm_hist, cm_future)
    d = IbicusCDFt(delta_shift="additive", **IBICUS_KW)
    theirs = d.apply(obs.reshape(t, 2, 2), cm_hist.reshape(t, 2, 2),
                     cm_future.reshape(t, 2, 2),
                     progressbar=False).reshape(t, n)
    for j in range(n):
        interior = _interior_mask(cm_future[:, j])
        rmse_i = float(np.sqrt(
            np.mean((ours[interior, j] - theirs[interior, j]) ** 2)))
        print(f"[cross-ibicus] grid cell {j}: interior RMSE={rmse_i:.4f}")
        assert rmse_i < 2e-2
