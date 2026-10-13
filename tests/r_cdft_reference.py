"""Faithful numpy port of the R ``CDFt`` package algorithm (CRAN, version 1.2).

The R package ``CDFt`` (Vrac & Michelangeli, CRAN 2021-02-18, GPL >= 2)
implements CDF-t with:

- an **additive mean shift** of the model series
  (``DataGp2 = DataGp + (mean(Obs) - mean(DataGp))``, same for the future);
- ECDFs as right-continuous step functions (R ``ecdf``), evaluated on a
  **common value grid** ``x = seq(min(all) - dev*a, max(all) + dev*a,
  length.out = npas)`` where ``a = |mean(DataGf) - mean(DataGp)|`` (raw
  means) and ``dev = 2`` by default;
- the composition ``FRF = F_obs(q_m^cal(F_m^tar(x)))`` on the grid, followed
  by two **tail-copy blocks** that graft the observation-CDF tail onto the
  target CDF below ``min(DataGf2)`` and above the saturated region (with
  0/1 padding);
- the final inversion with R ``approx(FRF, x, qntl, yleft = x[1],
  yright = x[n], ties = "mean")``: piecewise-linear interpolation in
  **p-space** with duplicate ``FRF`` values collapsed by averaging ``x``.

This module is a line-by-line port of ``CDFt.R`` (the exact source is
reproduced in the docstring comments), so that the R algorithm can be
cross-checked without an R installation.  Only 1-D series are supported
(the R package itself is 1-D only).

The port deviates from R only where R's semantics are undefined or
error-prone:

- termination guards in the ``while`` loops (R would raise subscript errors
  or loop forever in degenerate cases); the ``"In CDFt, dev must be higher"``
  error is reproduced verbatim;
- NaN handling: NaNs are dropped by the ECDFs and the means (R ``na.rm``);
  NaN positions in ``DataGf`` map to NaN in the output (as in R).

Reference source (CRAN ``CDFt`` 1.2, ``R/CDFt.R``)::

    CDFt <- function(ObsRp, DataGp, DataGf, npas = 1000, dev = 2){
      mO = mean(ObsRp, na.rm=TRUE)
      mGp= mean(DataGp, na.rm=TRUE)
      DataGp2 = DataGp + (mO-mGp)
      DataGf2 = DataGf + (mO-mGp)

      FRp=ecdf(ObsRp)
      FGp=ecdf(DataGp2)
      FGf=ecdf(DataGf2)

      a=abs(mean(DataGf, na.rm=TRUE)-mean(DataGp, na.rm=TRUE))
      m=min(ObsRp, DataGp, DataGf, na.rm=TRUE)-dev*a
      M=max(ObsRp, DataGp, DataGf, na.rm=TRUE)+dev*a
      x=seq(m,M,,npas)

      FGF=FGf(x); FGP=FGp(x); FRP=FRp(x)
      FGPm1.FGF=quantile(DataGp2,probs=FGF, na.rm=TRUE)
      FRF=FRp(FGPm1.FGF)

      ## tail copy below min(DataGf2)
      if(min(ObsRp, na.rm=TRUE)<min(DataGf2, na.rm=TRUE)){
        i=1; while(x[i]<=quantile(ObsRp,probs=FRF[1],na.rm=TRUE)){i=i+1}
        j=1; while(x[j]<min(DataGf2, na.rm=TRUE)){j=j+1}
        k=i
        while(j>0 && k>0){FRF[j]=FRP[k]; j=j-1; k=k-1}
        if(j>0){for(k in j:1){FRF[k]=0}}
      }

      ## tail copy above the saturated region
      if(FRF[length(x)]<1){
        i=length(x); QQ=quantile(ObsRp,probs=FRF[length(x)], na.rm=TRUE)
        while(x[i]>=QQ){i=i-1}
        i=i+1
        j=length(x)-1
        while(j>0 && FRF[j]==FRF[length(x)]){j=j-1}
        if(j==0){stop("In CDFt, dev must be higher\\n")}
        dif=min((length(x)-j),(length(x)-i))
        FRF[j:(j+dif)]=FRP[i:(i+dif)]
        k=j+dif
        if(k<length(x)){FRF[k:(length(x))]=1}
      }

      ## quantile-matching
      NaNs.indices = which(is.na(DataGf2))
      No.NaNs.indices = which(!is.na(DataGf2))
      qntl = array(NaN, dim=length(DataGf2))
      qntl[No.NaNs.indices] = FGf(DataGf2[No.NaNs.indices])
      xx = approx(FRF,x,qntl,yleft=x[1],yright=x[length(x)],ties='mean')
      return(list(x=x,FRp=FRP,FGp=FGP,FGf=FGF,FRf=FRF,DS=xx$y))
    }
"""

from __future__ import annotations

import numpy as np

__all__ = ["r_cdft", "r_cdft_full", "approx_ties_mean"]


def _ecdf_step(sample: np.ndarray):
    """R ``ecdf``: right-continuous step ECDF, NaNs dropped."""
    s = np.sort(np.asarray(sample, np.float64)[np.isfinite(sample)])
    n = s.size

    def F(v):
        return np.searchsorted(s, np.atleast_1d(v), side="right") / n

    return F


def approx_ties_mean(y, x, xout, yleft, yright):
    """R ``approx(..., ties = 'mean')``: linear interpolation of ``x`` vs
    ``y`` at ``xout``, duplicate ``y`` values collapsed by averaging ``x``.

    ``y`` must be non-decreasing (a CDF evaluated on an increasing grid);
    ``yleft``/``yright`` clamp the output outside ``[y[0], y[-1]]``; NaN in
    ``xout`` gives NaN (as in R).
    """
    y = np.asarray(y, np.float64)
    x = np.asarray(x, np.float64)
    xout = np.asarray(xout, np.float64)
    uniq, inv = np.unique(y, return_inverse=True)
    sums = np.zeros(uniq.size, dtype=np.float64)
    cnts = np.zeros(uniq.size, dtype=np.float64)
    np.add.at(sums, inv, x)
    np.add.at(cnts, inv, 1.0)
    x_uniq = sums / cnts
    return np.interp(xout, uniq, x_uniq, left=yleft, right=yright)


def r_cdft_full(obs, data_gp, data_gf, npas: int = 1000, dev: float = 2.0):
    """Port of ``CDFt()`` -- returns the full list of the R function.

    Returns
    -------
    dict with the R output fields: ``x`` (value grid), ``FRp``, ``FGp``,
    ``FGf`` (ECDFs on the grid), ``FRf`` (modified target CDF) and ``DS``
    (the downscaled series).
    """
    obs = np.asarray(obs, np.float64)
    data_gp = np.asarray(data_gp, np.float64)
    data_gf = np.asarray(data_gf, np.float64)
    mO = np.nanmean(obs)
    mGp = np.nanmean(data_gp)
    d = mO - mGp
    DataGp2 = data_gp + d
    DataGf2 = data_gf + d

    FRp = _ecdf_step(obs)
    FGp = _ecdf_step(DataGp2)
    FGf = _ecdf_step(DataGf2)

    a = abs(np.nanmean(data_gf) - np.nanmean(data_gp))
    m = min(np.nanmin(obs), np.nanmin(data_gp), np.nanmin(data_gf)) - dev * a
    M = max(np.nanmax(obs), np.nanmax(data_gp), np.nanmax(data_gf)) + dev * a
    x = np.linspace(m, M, npas)

    FGF = FGf(x)
    FGP = FGp(x)
    FRP = FRp(x)

    FGPm1FGF = np.quantile(DataGp2, FGF)          # R quantile, type 7, na.rm
    FRF = FRp(FGPm1FGF)
    n = x.size

    # ---- tail copy for x < min(DataGf2) ---------------------------------
    if np.nanmin(obs) < np.nanmin(DataGf2):
        # i = 1; while(x[i] <= quantile(ObsRp, probs = FRF[1])) i = i + 1
        QQ = np.quantile(obs, FRF[0])
        i = 0
        while i < n and x[i] <= QQ:
            i += 1
        # j = 1; while(x[j] < min(DataGf2)) j = j + 1
        j = 0
        while j < n and x[j] < np.nanmin(DataGf2):
            j += 1
        # k = i; while(j > 0 && k > 0) { FRF[j] = FRP[k]; j = j - 1; k = k - 1 }
        k = i
        while j > 0 and k > 0:
            FRF[j - 1] = FRP[k - 1]
            j -= 1
            k -= 1
        # if(j > 0) { for(k in j:1) FRF[k] = 0 }
        if j > 0:
            FRF[:j] = 0.0

    # ---- tail copy for x > max(DataGf2) ---------------------------------
    if FRF[n - 1] < 1.0:
        # i = length(x); QQ = quantile(ObsRp, probs = FRF[n]); while(x[i] >= QQ)
        QQ = np.quantile(obs, FRF[n - 1])
        i = n - 1
        while i >= 0 and x[i] >= QQ:
            i -= 1
        if i < 0:
            raise ValueError("In CDFt, dev must be higher (value grid does "
                             "not reach the observation quantile)")
        i += 1                                  # first index with x >= QQ
        # j = length(x) - 1; while(j > 0 && FRF[j] == FRF[n]) j = j - 1
        j = n - 2
        while j >= 0 and FRF[j] == FRF[n - 1]:
            j -= 1
        if j == -1:
            raise ValueError("In CDFt, dev must be higher")
        dif = min(n - (j + 1), n - (i + 1))
        FRF[j:j + dif + 1] = FRP[i:i + dif + 1]
        k = j + dif
        if k < n - 1:
            FRF[k + 1:] = 1.0

    # ---- final quantile matching ----------------------------------------
    qntl = FGf(DataGf2)
    DS = approx_ties_mean(FRF, x, qntl, yleft=x[0], yright=x[n - 1])
    DS = np.where(np.isfinite(DataGf2), DS, np.nan)
    return {"x": x, "FRp": FRP, "FGp": FGP, "FGf": FGF, "FRf": FRF, "DS": DS}


def r_cdft(obs, data_gp, data_gf, npas: int = 1000, dev: float = 2.0):
    """Port of the R ``CDFt`` function -- returns only the downscaled series.

    See :func:`r_cdft_full` for the arguments and the full output.
    """
    return r_cdft_full(obs, data_gp, data_gf, npas=npas, dev=dev)["DS"]
