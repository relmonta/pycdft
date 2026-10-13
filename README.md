# PyCDFt

[![CI](https://github.com/relmonta/pycdft/actions/workflows/ci.yml/badge.svg)](https://github.com/relmonta/pycdft/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/PyCDFt.svg)](https://pypi.org/project/PyCDFt/)
[![Python versions](https://img.shields.io/pypi/pyversions/PyCDFt.svg)](https://pypi.org/project/PyCDFt/)
[![License](https://img.shields.io/pypi/l/PyCDFt.svg)](https://github.com/relmonta/pycdft/blob/main/LICENSE)

**CDF-t statistical downscaling / bias correction, vectorized over grid cells
(pure numpy).**

**PyCDFt** is a generic, dependency-light implementation of the **CDF-t**
(Cumulative Distribution Function transform) method of
[Michelangeli, Vrac & Loukos (2009)](#references), with the delta-shift and
tail conventions of the R `CDFt` package and of
[ibicus](https://github.com/ecmwf-projects/ibicus). It operates on plain
numpy arrays — 1-D time series or `(T, N)` fields (any trailing dimensions
are treated as independent cells) — in one vectorized call, and is derived
from a production grid-scale implementation that has been run at
1056 × 1056 cells × 29-year calibration × 86-year target.

```python
import numpy as np
from pycdft import CDFt

rng = np.random.default_rng(7)
obs       = rng.normal(0.0, 1.0, 1000)   # observations (calibration period)
cm_hist   = rng.normal(3.0, 1.0, 1000)   # model, calibration period
cm_future = rng.normal(4.0, 1.0, 1000)   # model, target period

model = CDFt(n_quantiles=1001, tail="linear", delta_shift="additive")
ds = model.fit_transform(obs, cm_hist, cm_future)   # (1000,) downscaled series
```

## Installation

```bash
pip install pycdft              # from PyPI (numpy only)
pip install "pycdft[gpu]"       # + optional torch GPU backend
# or from this repository:
pip install .                   # core only
pip install -e ".[test]"        # editable, with the test tools
```

Requires Python ≥ 3.9 and `numpy>=1.20`. The core has **no other
dependencies** (scipy, xarray, pandas and matplotlib are never required;
matplotlib is only used by the optional `examples/`, ibicus only by the
optional cross-validation tests).

**Optional GPU backend.** `pip install .[gpu]` adds an optional torch-based
backend (`pycdft.gpu`) that mirrors this API and runs the same math on CUDA
/ AMD ROCm / Apple MPS / CPU devices (numpy in, numpy out by default; the
results match the numpy backend to ~1e-15). See [docs/gpu.md](docs/gpu.md)
for the supported devices, the installation per hardware (CUDA / ROCm / CPU
torch builds), and the verified equivalence.

## Documentation

- [docs/design.md](docs/design.md) — the math, conventions, design choices
  and the measured comparison with the reference implementations;
- [docs/gpu.md](docs/gpu.md) — the optional torch GPU backend (`pycdft.gpu`);
- [docs/guide.html](docs/guide.html) — a plain-language companion guide
  (no statistics background needed).

## What is CDF-t?

CDF-t (Michelangeli et al. 2009) corrects both the distribution **and** its
change between the calibration and target periods, by defining the target
observation CDF as

```
F_o^tar = F_o^cal ∘ (F_m^cal)^{-1} ∘ F_m^tar
```

(here `m` = model, `o` = observations, `cal` = calibration period,
`tar` = target period, `F` = CDF and `q` = its inverse, the quantile
function), so that each model value `x` of the target period is mapped to

```
y(x) = q_m^tar( F_m^cal( q_o^cal( F_m^tar(x) ) ) )
```

where `F_m`, `F_o` are the model and observation CDFs, `q` the quantile
(inverse CDF) functions, `cal`/`tar` the calibration and target periods.
In practice the composition is evaluated on a uniform probability grid
`p = linspace(0, 1, Q)` (default `Q = 1001`, endpoints included): the
calibration mapping `R = F_m^cal(q_o^cal(p))` is computed once, and the
transfer curve `(x_grid, T) = (q_m^tar(p), q_m^tar(R))` is interpolated per
cell at the target values. See [docs/design.md](docs/design.md) for the
full math, conventions and a detailed comparison with the reference
implementations.

### Conventions (matching the reference implementations)

| aspect | convention |
|---|---|
| ECDF | right-continuous step `F(x) = #{finite values ≤ x} / n` (NaNs excluded) — identical to the R `CDFt` package and to ibicus `ecdf_method="step_function"` |
| quantiles | type-7 linear (numpy default `method="linear"` = R `quantile(type=7)`), re-implemented bit-for-bit on pre-sorted arrays |
| delta shift | per-cell additive `x + (mean(obs) − mean(cm_hist))` (default) or multiplicative `x * (mean(obs) / mean(cm_hist))`, applied to the model series (calibration and target) *before* the transfer; `"none"` disables it (ibicus `"no_shift"`) |
| output frame | the shifted frame (the mean bias is removed, the model change calibration → target is preserved) |
| grid | `linspace(0, 1, Q)` with `Q = 1001`; the transfer is interpolated in x-space on the target's own value grid |
| tails | `"clip"` = constant at the transfer endpoints; `"linear"` = 30-rank windowed least-squares slope at the first/last unsaturated rank, capped at `2 ×` the median interior slope (default) |
| NaN | excluded from ECDF counts and quantile interpolation; NaN inputs pass through; cells with an empty (all-NaN) calibration map entirely to NaN |

## Quick start

```python
import numpy as np
from pycdft import CDFt

rng = np.random.default_rng(0)
obs, cm_hist, cm_future = (rng.normal(m, 1.0, 2000) for m in (0.0, 3.0, 4.0))

# 1-D series ---------------------------------------------------------------
ds = CDFt().fit_transform(obs, cm_hist, cm_future)

# (T, N) field: every cell (column) is corrected independently -------------
per_cell = np.linspace(-2.0, 2.0, 64)             # per-cell spatial offset
obs2d, ch2d, cf2d = (rng.normal(0.0, 1.0, (1000, 64)) + per_cell * j
                     for j in (0, 3, 4))          # dataset bias grows with j
ds2d = CDFt().fit_transform(obs2d, ch2d, cf2d)    # (1000, 64)

# options ------------------------------------------------------------------
ds_clip = CDFt(tail="clip").fit_transform(obs, cm_hist, cm_future)
ds_none = CDFt(delta_shift="none").fit_transform(obs, cm_hist, cm_future)
ds_hist = CDFt(use_histograms=True, n_bins=2000).fit_transform(obs, cm_hist,
                                                                cm_future)

# fit / transform separately (reuse the calibration) -----------------------
model = CDFt().fit(obs, cm_hist)
ds_a = model.transform(cm_future)          # identical to fit_transform
cm_future2 = rng.normal(4.5, 1.0, 1000)    # another scenario / member
ds_b = model.transform(cm_future2)         # same calibration, new target

# one-shot functional API --------------------------------------------------
from pycdft import cdft_transform
ds = cdft_transform(obs, cm_hist, cm_future, n_quantiles=1001)
```

### Low-level API

The building blocks are exposed in `pycdft.math` (kernels) and `pycdft.core`
(pipeline). Every kernel accepts `(T, N)` arrays and is vectorized per column
(cell); `workers > 1` threads the per-cell kernels (results are identical,
numpy releases the GIL):

```python
from pycdft import calibrate, apply_cdft
from pycdft.math import quantile_grid

p = quantile_grid(1001)
calib = calibrate(cm_hist, obs, p=p, delta_shift="additive")
#   -> Calibration(R=(Q, N), delta=(N,), mode="additive", n_quantiles=1001)
pred = apply_cdft(cm_future, calib, p=p, tail="linear")   # (T, N)
```

`calibrate` takes the model first, the observations second (the reverse of
`CDFt.fit`). The histogram variants (`calibrate_hist`, `apply_cdft_hist`,
`use_histograms=True`) and the individual kernels are documented in
`pycdft.math` / `pycdft.core`.

## Comparison with the reference implementations

PyCDFt follows the conventions of the R `CDFt` package and ibicus, so it
agrees closely with both in the middle of the distribution; the differences
are at the extremes (tail models) and in the exact grid:

| aspect | PyCDFt | R `CDFt` | ibicus | SBCK |
|---|---|---|---|---|
| composition | same | same | same | same (histogram margins) |
| ECDF / quantiles | step + type-7 | step + type-7 | configurable (step + linear in the tests) | rank-based + interp1d |
| delta shift | per-cell, 3 modes | additive only | per-series, 3 modes | per-column normalization |
| grid | p-grid Q=1001, x-space interp | value grid npas=1000, dev=2 | none (per value) | adaptive support, dsupp=1000 |
| tails | capped LSQ linear / clip | obs-tail copy + 0/1 padding | clamp | support extension |
| seasonality | none (a monthly mode exists in the production codebase this package is derived from) | none | 31-day + 17y/9y windows (default ON) | none |
| precipitation SSR | no | no | yes (Vrac et al. 2016) | no |
| NaN | robust | robust | not supported | n/a |
| grid scale | vectorized, chunked, cacheable | single series | per-location loop, full arrays | full arrays |

The measured agreement (interior RMSE ~0.005–0.012 vs ibicus; the residual
is the probability-grid discretization of the transfer curve) and the full
analysis of *where* the implementations differ are in
[docs/design.md](docs/design.md) §4, which is reproduced by
`pytest -m cross -s`.

## Examples

Run from the repository root (with `pip install -e .`, or set
`PYTHONPATH=.`):

```bash
python examples/example_r_gaussian.py        # the R package's Gaussian example
python examples/example_skewed.py            # non-Gaussian / skewed case
python examples/example_grid.py              # small (T, N) grid with QQ plots
python examples/example_ibicus_compare.py    # vs ibicus (needs `pip install ibicus`)
python examples/gpu_example.py               # torch GPU backend (needs torch)
```

Each example prints the summary statistics and saves figures to
`examples/figures/` when matplotlib is available.

## Tests

```bash
pip install -e .[test]
pytest                          # core suite (numpy only, no extras)
pytest -m cross                 # cross-validation vs ibicus + the R port
pytest -m "not cross"           # core suite only (CI)
pytest -m gpu                   # torch GPU backend (skipped without torch)
```

The core suite is pure numpy and does not require ibicus: the *ibicus*
cross tests skip automatically when ibicus is missing, while the R-port
cross tests need only numpy (R is not required — the R algorithm is ported
in `tests/r_cdft_reference.py`). The GPU tests are device-agnostic: they
run on CUDA / ROCm / MPS if available and always pass on CPU.

## Limitations

- **Univariate**: each cell (series) is corrected independently; no
  multivariate, spatial or temporal-consistency constraints are imposed.
- **No seasonality**: the calibration does not distinguish months or
  seasons (a monthly mode exists in the production codebase this package
  is derived from, but it is not part of PyCDFt).
- **No precipitation-specific handling**: the SSR (singularity stochastic
  removal) zero-inflation of Vrac et al. (2016) is not implemented.
- **Histogram mode is approximate**: `use_histograms=True` and the `_hist`
  functions trade a small accuracy for bounded memory (exact in the limit
  of many bins; see `docs/design.md` §3.1).
- **No autocorrelation modelling**: each series is treated as a single
  sample; temporal structure (daily/monthly persistence) is not modelled.

## References

- Michelangeli, P.-A., Vrac, M. & Loukos, H. (2009). *Probabilistic
  downscaling approaches: Application to wind cumulative distribution
  functions.* Geophys. Res. Lett., 36, L11708.
  doi:[10.1029/2009GL038401](https://doi.org/10.1029/2009GL038401)
- Vrac, M., Noël, T. & Vautard, R. (2016). *Bias correction of
  precipitation through singularity stochastic removal: Method and
  application to the CORDEX-Africa domain.* J. Geophys. Res. Atmos.
- Vrac, M. & Michelangeli, P.-A. (2021). *CDFt: Downscaling and bias
  correction via CDF transform.* R package version 1.2, CRAN.
  https://cran.r-project.org/package=CDFt
- Cucchi, M. et al. (2024). *ibicus: A Python package for statistical bias
  correction and downscaling.* Geosci. Model Dev. (Apache-2.0)
  https://github.com/ecmwf-projects/ibicus
- Robin, Y. (2018+). *SBCK: Statistical Bias Correction Kit.*
  https://github.com/yrobink/SBCK (GPL-3, needs Eigen)

The R `CDFt` source is GPL (≥ 2); it is reproduced in the tests
(`tests/r_cdft_reference.py`) solely as a validation reference for the
Apache-2.0 PyCDFt package.

## License

Apache-2.0. See [LICENSE](LICENSE).
