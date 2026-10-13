"""High-level CDF-t API.

The :class:`CDFt` class wraps the low-level pipeline
(:func:`pycdft.core.calibrate` + :func:`pycdft.core.apply_cdft`) in a
scikit-learn-style ``fit`` / ``transform`` / ``fit_transform`` interface and
works on plain numpy arrays: 1-D series ``(T,)`` or fields ``(T, N)`` /
``(T, Y, X)`` (the trailing dimensions are flattened into independent
series / grid cells, and the output shape matches the input).
"""

from __future__ import annotations

import numpy as np

from .core import (Calibration, apply_cdft, apply_cdft_hist, calibrate,
                   calibrate_hist)
from .math import quantile_grid

__all__ = ["CDFt", "cdft_transform"]

_SHIFT_MODES = ("additive", "multiplicative", "none")
_TAILS = ("linear", "clip")


class CDFt:
    """CDF-t statistical downscaling / bias correction (Michelangeli et al.
    2009; Vrac et al. 2016).

    Parameters
    ----------
    n_quantiles : int, default 1001
        Number of points of the uniform probability grid ``p`` (endpoints 0
        and 1 included), on which the transfer curve is evaluated.
    tail : {"linear", "clip"}, default "linear"
        Behaviour outside the saturated region of the transfer:
        "linear" extrapolates with a 30-rank windowed least-squares slope
        (capped at 2x the median interior slope); "clip" holds the constant
        end values of the transfer.
    delta_shift : {"additive", "multiplicative", "none"}, default "additive"
        Per-cell mean shift applied to the model series (calibration and
        target) before the transfer -- the standard CDF-t practice:

        - "additive": ``x -> x + (mean(obs) - mean(cm_hist))``
          (default, as in the R ``CDFt`` package and ibicus);
        - "multiplicative": ``x -> x * (mean(obs) / mean(cm_hist))``
          (for strictly positive fluxes; falls back to 1.0 where the model
          mean is ~0);
        - "none": no shift (ibicus ``"no_shift"``).

        The output is left in the shifted frame: the mean bias
        model-vs-observation is removed, the model change between the
        calibration and target periods is preserved.
    use_histograms : bool, default False
        Accumulate the target CDF as a per-cell histogram instead of an
        exact sort (bounded memory for very long records; approximate).
    n_bins : int, default 2000
        Number of bins per cell in histogram mode.
    workers : int, default 1
        Threads for the per-cell kernels (identical results; numpy releases
        the GIL, so ``workers > 1`` speeds up large grids).

    Attributes
    ----------
    calibration_ : Calibration or None
        The calibration object from the last :meth:`fit` /
        :meth:`fit_transform` call (``None`` before fitting).
    probability_grid : (Q,) np.ndarray
        The probability grid ``linspace(0, 1, n_quantiles)``.

    Notes
    -----
    NaN handling: NaNs are excluded from the ECDF counts and from the
    quantile interpolation; NaN inputs pass through as NaN; series (cells)
    whose calibration is empty (all-NaN) map entirely to NaN.  The core is
    pure numpy (no scipy, no xarray); the memory-lean design keeps the
    calibration arrays in the input dtype (float32 stays float32) with
    float64 per-chunk math.

    Examples
    --------
    >>> import numpy as np
    >>> from pycdft import CDFt
    >>> rng = np.random.default_rng(7)
    >>> obs = rng.normal(0.0, 1.0, 1000)        # observations (calibration)
    >>> cm_hist = rng.normal(3.0, 1.0, 1000)    # model, calibration period
    >>> cm_future = rng.normal(4.0, 1.0, 1000)  # model, target period
    >>> ds = CDFt().fit_transform(obs, cm_hist, cm_future)
    >>> ds.shape
    (1000,)
    """

    def __init__(self, n_quantiles: int = 1001, tail: str = "linear",
                 delta_shift: str = "additive", use_histograms: bool = False,
                 n_bins: int = 2000, workers: int = 1):
        n_quantiles = int(n_quantiles)
        if n_quantiles < 2:
            raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
        if delta_shift not in _SHIFT_MODES:
            raise ValueError(f"delta_shift must be one of {_SHIFT_MODES}, "
                             f"got {delta_shift!r}")
        if tail not in _TAILS:
            raise ValueError(f"tail must be one of {_TAILS}, got {tail!r}")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2, got {n_bins}")
        self.n_quantiles = n_quantiles
        self.tail = tail
        self.delta_shift = delta_shift
        self.use_histograms = bool(use_histograms)
        self.n_bins = int(n_bins)
        self.workers = int(workers)
        self.calibration_: Calibration | None = None

    # ------------------------------------------------------------------
    @property
    def probability_grid(self) -> np.ndarray:
        """The uniform probability grid ``linspace(0, 1, n_quantiles)``."""
        return quantile_grid(self.n_quantiles)

    # ------------------------------------------------------------------
    def fit(self, obs, cm_hist) -> "CDFt":
        """Fit the calibration on ``(obs, cm_hist)``.

        Parameters
        ----------
        obs : (T,) or (T, ...) array
            Observations (fine / high-resolution series) over the
            calibration period.
        cm_hist : (T,) or (T, ...) array
            Model (coarse / low-resolution) series over the same period.

        Returns
        -------
        self
        """
        self.calibration_ = calibrate(cm_hist, obs, p=self.probability_grid,
                                      delta_shift=self.delta_shift,
                                      workers=self.workers)
        return self

    def transform(self, cm_future):
        """Apply the fitted calibration to the target model series.

        Parameters
        ----------
        cm_future : (T,) or (T, ...) array
            Model series to downscale / bias-correct.

        Returns
        -------
        Array of the same shape as ``cm_future`` (in the shifted frame).
        """
        if self.calibration_ is None:
            raise RuntimeError("CDFt is not fitted: call fit() or "
                               "fit_transform() first")
        if self.use_histograms:
            return apply_cdft_hist(cm_future, self.calibration_,
                                   p=self.probability_grid, n_bins=self.n_bins,
                                   tail=self.tail, workers=self.workers)
        return apply_cdft(cm_future, self.calibration_, p=self.probability_grid,
                          tail=self.tail, workers=self.workers)

    def fit_transform(self, obs, cm_hist, cm_future):
        """Fit on ``(obs, cm_hist)`` and transform ``cm_future`` in one call.

        Parameters
        ----------
        obs : (T,) or (T, ...) array
            Observations over the calibration period.
        cm_hist : (T,) or (T, ...) array
            Model series over the calibration period.
        cm_future : (T,) or (T, ...) array
            Model series to downscale / bias-correct.

        Returns
        -------
        Array of the same shape as ``cm_future``.
        """
        return self.fit(obs, cm_hist).transform(cm_future)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"CDFt(n_quantiles={self.n_quantiles}, tail={self.tail!r}, "
                f"delta_shift={self.delta_shift!r}, "
                f"use_histograms={self.use_histograms}, "
                f"n_bins={self.n_bins}, workers={self.workers})")


def cdft_transform(obs, cm_hist, cm_future, n_quantiles: int = 1001,
                   tail: str = "linear", delta_shift: str = "additive",
                   use_histograms: bool = False, n_bins: int = 2000,
                   workers: int = 1):
    """One-shot CDF-t transform (no persistent calibration object).

    Equivalent to ``CDFt(...).fit_transform(obs, cm_hist, cm_future)``;
    returns the same-shaped output array.
    """
    return CDFt(n_quantiles=n_quantiles, tail=tail, delta_shift=delta_shift,
                use_histograms=use_histograms, n_bins=n_bins,
                workers=workers).fit_transform(obs, cm_hist, cm_future)
