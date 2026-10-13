# PyCDFt — math and design notes

This document describes the CDF-t formulation implemented by **PyCDFt**, the
numerical conventions, the design choices (grid, tails, delta shift, memory
layout, NaN handling), and a measured comparison with the reference
implementations (R `CDFt`, ibicus, SBCK). It is the math/design companion
of the README.

## 1. The method

CDF-t (Michelangeli, Vrac & Loukos 2009) downscales/bias-corrects a model
(low-resolution, "m") series using observations (high-resolution, "o")
over a calibration period, such that the *change* of the model distribution
between the calibration and target periods is transferred to the
observations. The target observation CDF is defined as

```
F_o^tar = F_o^cal ∘ (F_m^cal)^{-1} ∘ F_m^tar
```

so each target model value `x` maps to

```
y(x) = q_m^tar( F_m^cal( q_o^cal( F_m^tar(x) ) ) )          (1)
```

where `F_m^tar`, `F_m^cal` are the model CDFs in the target and calibration
periods, `F_o^cal` the observation calibration CDF, and `q` denotes the
quantile (inverse-CDF) functions. When `F_m^tar = F_m^cal` (no model
change) the composition collapses to classical quantile mapping
`y = q_o^cal(F_m^cal(x))` (up to the ECDF discretization; verified in
`tests/test_core.py::test_reduces_to_classical_quantile_mapping`).

### 1.1 Delta shift

Following the standard CDF-t practice (R `CDFt`, ibicus), the model series
are re-centred on the observation calibration mean *before* the transfer,
per series (per cell):

```
additive:       x -> x + (mean(o_cal) - mean(m_cal))
multiplicative: x -> x * (mean(o_cal) / mean(m_cal))
```

applied to the model calibration **and** target series. The output is left
in the shifted frame: the mean bias is removed, the model change is
preserved. The multiplicative mode falls back to `1.0` where the model mean
is ~0, non-finite or non-positive (signed variables); ibicus has no such
guard. `delta_shift="none"` disables the shift (ibicus `"no_shift"`).

### 1.2 ECDF and quantile conventions

- **ECDF**: right-continuous step `F(x) = #{finite values ≤ x} / n_valid`
  (NaNs excluded, sorted to the end). Identical to R's `ecdf` and to ibicus
  `ecdf_method="step_function"`. (ibicus's *default* `linear_interpolation`
  ECDF is the exception: with that setting ibicus's results change by
  ~0.01–0.03 RMSE (see §4.3).)
- **Quantiles**: type-7 linear interpolation — numpy
  `quantile(method="linear")` = R `quantile(type=7)` — re-implemented
  bit-for-bit on pre-sorted arrays, including numpy's `_lerp` `w >= 0.5`
  branch (`pycdft.math.quantile_from_sorted`). The probability index is
  scaled by the per-cell number of *valid* values, so NaNs can never enter
  the interpolation. All-empty columns return NaN.

## 2. Grid-based evaluation (the discretization choice)

Equation (1) is evaluated on a uniform probability grid
`p = linspace(0, 1, Q)`, `Q = n_quantiles` (default 1001, endpoints 0 and 1
included):

```
calibration (target-independent):
    R_k   = F_m^cal( q_o^cal(p_k) )            (Q, N)

transfer on the target's own value grid:
    x_grid_k = q_m^tar(p_k)                     (Q, N)
    T_k      = q_m^tar(R_k)                     (Q, N)

output:
    y(x)  = interp(x, x_grid, T)   per cell     (piecewise linear, np.interp)
```

Saturation ranks: `k_lo = argmax(R > 0)` (first unsaturated rank) and
`k_hi = last rank with R < 1`; outside `[x_grid[k_lo], x_grid[k_hi]]` the
transfer is either held constant (`tail="clip"`) or extrapolated linearly
(`tail="linear"`).

The key property: the calibration `R` (plus the per-cell `delta`) is
**target-independent**, so it can be computed once and reused for any
number of target periods, scenarios and ensemble members
(`CDFt.fit` / `CDFt.transform`).

### 2.1 Tail model

`tail="linear"` (default): at the first/last unsaturated rank the transfer
is extended with the least-squares slope of the 30-rank boundary window
(`pycdft.math.windowed_slope`, closed form, ~1e-12 relative agreement with
`np.polyfit`, no RankWarnings on degenerate windows), clipped to
`[0, 2 × median interior slope]` (interior = ranks 20%–80% of the grid).
The cap keeps the extrapolation shape-aware but bounded. `tail="clip"`
holds the constant end values (the ibicus behaviour).

### 2.2 Why a grid?

The grid formulation is the standard one in the R `CDFt` package (which
uses a *value* grid of `npas = 1000` points extended by `dev = 2 × |Δmean|`
on both sides and interpolates in p-space). PyCDFt uses a *probability*
grid and interpolates in x-space on the target's own value grid: the
transfer curve is evaluated exactly at the grid probabilities and the
output is a piecewise-linear function of the input value (monotone, no
re-inversion needed). The grid-sensitivity is second-order: on the
Gaussian test sample (seed 7), Q = 101 → 1001 changes the output mean by
~0.002 and the sd by ~0.011 (RMSE between the two outputs ~0.04; ~0.08 on
the skewed sample) — far smaller than the effect of the delta-shift
choice (0.17–0.56 RMSE on the same data) or of ibicus's running-window
option (~0.28 RMSE on seasonal data, an ibicus-side effect).

## 3. Design choices

### 3.1 Vectorized per cell, memory-lean

- The package works on `(T, N)` arrays (any trailing dims flattened) and
  every kernel is vectorized over the `N` independent cells (columns); the
  high-level `CDFt` accepts 1-D series, `(T, N)` fields or `(T, Y, X)`
  grids.
- The calibration arrays are sorted in place on a **private copy** (the
  caller's data is never mutated) and keep their input dtype: float32
  stays float32 (memory lean), the per-chunk math is float64 and exact
  (float32 values convert exactly → bit-identical to sorting float64
  first). All per-cell math runs on cell chunks so the `(Q, chunk)`
  float64 intermediates stay below `chunk_bytes` (default 2 GiB).
- `workers > 1` threads the per-cell kernels (searchsorted, interp) with
  identical results.
- For very long records, `use_histograms=True` (or the low-level
  `apply_cdft_hist` / `calibrate_hist`) accumulates the target CDF as
  per-cell histograms (`histogram_accumulate`, global edges; or
  `histograms` + `hist_quantile_pc` / `ecdf_hist` with per-cell edges for
  the calibration). The histogram path is exact in the limit of many bins
  (verified in
  `tests/test_core.py::test_apply_cdft_hist_matches_apply_cdft`: 4000 bins
  on the skewed n = 1000 sample → RMSE < 0.02 vs the exact path) and is
  NaN-robust (sentinel bin excluded from the counts).

### 3.2 NaN robustness

Four NaN classes are handled and tested (`tests/test_nan.py`):

1. scattered NaNs in the calibration series: excluded from the counts, the
   means and the interpolation;
2. NaN inputs in the target series: pass through (only the NaN positions);
3. all-NaN calibration series/cells: `R = NaN` → the whole cell maps to
   NaN (a common failure mode of naive implementations here is
   `floor(NaN)` → `INT64_MIN`; PyCDFt returns NaN instead);
4. all-NaN target fields: NaN output (exact and histogram paths).

ibicus cannot handle missing data (all-NaN output or a crash with the
default `linear_interpolation` ECDF); R drops NaNs (`na.rm=TRUE`).

### 3.3 Determinism / precision

The package is deterministic (threads only parallelize identical per-cell
kernels). The float32-input path differs from the float64 path by at most
float32 input rounding (measured < 1e-5 relative); the chunked and
unchunked paths are identical to 1e-12. ibicus's SSR (precipitation
zero-inflation) is the only non-deterministic step in the reference
implementations and is not implemented here.

## 4. Comparison with the reference implementations

### 4.1 What was run

1-D synthetic series (float64, n = 1000 per series unless noted), seed 7
for the Gaussian case (the R package's own example data), plus a
multi-cell grid. PyCDFt invoked as `CDFt(...).fit_transform(obs, cm_hist,
cm_future)` with `Q = 1001`; ibicus as
`CDFt(delta_shift=..., running_window_mode=False,
running_window_mode_over_years_of_cm_future=False,
ecdf_method="step_function", iecdf_method="linear", SSR=False)`; the R
algorithm as the faithful numpy port in `tests/r_cdft_reference.py`
(`npas = 1000, dev = 2`). Metrics: RMSE / max|Δ| between output series,
interior = ranks 2%–98% of `cm_future`, plus max|Δ| over the 99 output
quantiles p = 0.01…0.99.

### 4.2 Measured agreement (reproduced by `pytest -m cross`)

| comparison | case | RMSE | max\|Δ\| | interior RMSE | interior max\|Δ\| | max qdiff |
|---|---|---|---|---|---|---|
| PyCDFt vs ibicus | Gaussian (R example), additive | 0.0249 | 0.632 | 0.0050 | 0.0595 | — |
| PyCDFt vs ibicus | skewed (gamma), additive | — | — | 0.0082 | — | — |
| PyCDFt vs ibicus | gamma, multiplicative | — | — | 0.0112 | — | — |
| PyCDFt vs ibicus | Gaussian, no shift (clip) | — | — | 0.0119 | — | — |
| PyCDFt vs ibicus | 4-cell grid, additive (n = 500) | — | — | 0.010–0.014/cell | — | — |
| PyCDFt vs R `CDFt` | Gaussian (R example) | 0.0939 | 2.928 | 0.0076 | 0.0594 | 0.0609 |
| PyCDFt vs R `CDFt` | skewed (gamma) | 0.2193 | 6.275 | 0.0374 | 0.4554 | — |

Independent reference points (validation of the ports): ibicus vs R
Gaussian RMSE 0.0964 (reproduced by `pytest -m cross -s`); R's published
example output quantiles −1.004 / 0.339 / 0.970 / 2.060 / 3.013 and
per-quantile agreement with PyCDFt to ~1e-3 in the interior (p = 0.01 …
0.99) — the R port in `tests/r_cdft_reference.py` reproduces all of these
to the printed digits. The README shows the headline numbers; this table
is the complete record — keep the two in sync (both are reproduced by
`pytest -m cross -s`).

### 4.3 Where the implementations differ (and why)

1. **Tail model** (dominant at the extremes): PyCDFt capped LSQ linear
   extrapolation vs ibicus clamp vs R observation-tail copy + 0/1 padding
   + `dev=2` grid extension. This is where `max|Δ|` concentrates
   (0.03–0.3 RMSE, up to 5.6 units in the skewed case). R's tail-copy can
   push the output beyond both the observed and model ranges (max|Δ| ≈
   2.9 units in the Gaussian example) — the largest single source of
   disagreement with the other two.
2. **Grid / parameterization** (interior): PyCDFt p-grid + x-space interp
   vs R value-grid + p-space `approx(ties='mean')` vs ibicus no grid
   (per-value composition). Interior agreement with ibicus ~0.005–0.012
   RMSE; with R ~0.008–0.04 interior RMSE.
3. **ECDF convention**: PyCDFt / R / ibicus-`step_function` identical;
   ibicus's default `linear_interpolation` ECDF changes its results by
   ~0.01–0.03 RMSE (and crashes on NaN).
4. **Delta shift**: identical formulas everywhere; `none`/`no_shift` is
   only a *name* difference, but its *choice* shifts the comparison by
   0.17–0.56 RMSE when the supports are disjoint (the no-shift transfer is
   steep, so the grid discretization error also grows: interior RMSE
   0.0119 with clip tails).
5. **Running windows**: ibicus defaults ON (31-day within-year + 17y/9y
   over the future years; ~0.28 RMSE effect on seasonal data) — the
   comparisons above run ibicus with windows OFF (the PyCDFt default; the
   a coarser monthly mode exists in the production codebase this package is
    derived from, but it is not part of PyCDFt).

### 4.4 SBCK

SBCK (Robin; GPL-3) implements CDF-t with histogram margins
(`rv_histogram` default), its own adaptive support grid (`dsupp = 1000`)
and a normalization option; the build requires Eigen (C++ header library).
SBCK is **not installed and not tested here** (no conda Eigen, no
`EIGEN_INCLUDE_PATH` in this environment): the cross-test matrix covers
ibicus and the R `CDFt` port instead. The `pytest.mark.cross` tests skip
cleanly when their dependencies are missing.

## 5. API map

| level | module | contents |
|---|---|---|
| high-level | `pycdft` (`api.py`) | `CDFt` (fit / transform / fit_transform), `cdft_transform` |
| pipeline | `pycdft.core` | `Calibration`, `calibrate`, `calibrate_hist`, `histograms`, `transfer`, `apply_cdft`, `apply_cdft_hist`, `delta_from_means`, `apply_shift` |
| kernels | `pycdft.math` | `quantile_grid`, `searchsorted_columns`, `ecdf`, `ecdf_at`, `quantile`, `nanquantile`, `quantile_from_sorted`, `windowed_slope`, `interp_transfer`, `histogram_accumulate`, `hist_quantile`, `hist_quantile_pc`, `ecdf_hist` |

All kernels take `(T, N)` float64 arrays and are vectorized per column;
1-D series are supported at the `CDFt`, `calibrate`, `apply_cdft` and
`ecdf` levels. The package is self-contained (numpy only) and imports no
project-specific modules.

## 6. Reproducing the measurements

```bash
pip install -e .[test] ibicus   # ibicus only for the cross tests
pytest -m cross -s              # prints the measured numbers
```
