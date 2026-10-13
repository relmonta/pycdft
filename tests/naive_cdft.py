"""Reference (naive) CDF-t implementation used as a regression baseline.

This is the *direct per-value composition* of the CDF-t chain -- the exact
evaluation of

    y(x) = q_m^tar( F_m^cal( q_o^cal( F_m^tar(x) ) ) )

with the same conventions as :mod:`pycdft` (right-continuous step ECDF,
type-7 linear quantiles, optional per-series mean delta shift), but with no
probability grid and no transfer interpolation: each output value is
obtained by composing the four maps directly (this is exactly how ibicus
with ``ecdf_method="step_function"`` / ``iecdf_method="linear"`` computes
the same expression).

The grid-based implementation therefore differs from this reference only by
the discretization of the transfer curve on the Q-point probability grid
(and by the tail model, which the reference does not have -- the exact chain
clamps at the observed/model extremes, i.e. ``tail="clip"`` behaviour).

Reference implementations live under ``tests/`` on purpose: they are
deliberately simple, 1-D and slow, and are only used to cross-check the
vectorized package code.
"""

from __future__ import annotations

import numpy as np


def step_ecdf(sample: np.ndarray):
    """Right-continuous step ECDF function of a 1-D sample (NaNs dropped).

    ``F(x) = #{finite values <= x} / n`` (the ``PyCDFt`` / R ``CDFt`` /
    ibicus ``step_function`` convention).
    """
    s = np.sort(np.asarray(sample, np.float64)[np.isfinite(sample)])
    n = s.size

    def F(v):
        return np.searchsorted(s, np.atleast_1d(v), side="right") / n

    return F


def naive_cdft(obs, cm_hist, cm_future, delta_shift: str = "none",
               obs_shift_frame: bool = False):
    """Direct per-value CDF-t composition on 1-D series.

    Parameters
    ----------
    obs, cm_hist, cm_future : 1-D float64 arrays (NaNs allowed).
    delta_shift : {"additive", "multiplicative", "none"}
        Per-series mean shift of the model series (same convention as
        ``PyCDFt``: ``d = mean(obs) - mean(cm_hist)`` (additive) or
        ``mean(obs) / mean(cm_hist)`` (multiplicative); the shift is applied
        to ``cm_hist`` and ``cm_future``; the output is in the shifted frame).
    obs_shift_frame : bool
        If True, also shift ``obs`` (used only for testing shift semantics).

    Returns
    -------
    (n,) mapped values (NaN where ``cm_future`` is NaN).
    """
    obs = np.asarray(obs, np.float64)
    cm_hist = np.asarray(cm_hist, np.float64)
    cm_future = np.asarray(cm_future, np.float64)
    if delta_shift == "additive":
        d = np.nanmean(obs) - np.nanmean(cm_hist)
        cm_hist = cm_hist + d
        cm_future = cm_future + d
        if obs_shift_frame:
            obs = obs + d
    elif delta_shift == "multiplicative":
        d = np.nanmean(obs) / np.nanmean(cm_hist)
        cm_hist = cm_hist * d
        cm_future = cm_future * d
        if obs_shift_frame:
            obs = obs * d
    elif delta_shift != "none":
        raise ValueError(f"unknown delta_shift {delta_shift!r}")

    F_mf = step_ecdf(cm_future)      # F_m^tar
    F_mc = step_ecdf(cm_hist)        # F_m^cal
    p1 = F_mf(cm_future)             # F_m^tar(x)
    u = np.nanquantile(obs, p1)      # q_o^cal
    p2 = F_mc(u)                     # F_m^cal(q_o^cal(...))
    y = np.nanquantile(cm_future, p2)  # q_m^tar
    return y


def quantile_mapping(obs, cm_hist, x):
    """Classical (empirical) quantile mapping ``y = q_o^cal(F_m^cal(x))``.

    Used to test that CDF-t reduces to classical QM when ``F_m^tar =
    F_m^cal`` (i.e. when ``cm_future`` has the same ECDF as ``cm_hist``).
    """
    obs = np.asarray(obs, np.float64)
    cm_hist = np.asarray(cm_hist, np.float64)
    x = np.asarray(x, np.float64)
    F = step_ecdf(cm_hist)
    return np.nanquantile(obs, F(x))
