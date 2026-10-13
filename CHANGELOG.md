# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and versioning follows
[Semantic Versioning](https://semver.org/).

## [0.3.1] - 2026-10-13

### Added

- README: CI, PyPI version, Python versions and license badges.

## [0.3.0] - 2026-10-13

### Added

- PyPI publishing via GitHub Actions trusted publishing (OIDC):
  `.github/workflows/release.yml` builds the sdist and wheel and uploads on
  tag push (`v*`), with no API tokens or stored secrets.

### Documentation

- README: PyPI install instructions (`pip install pycdft`, with the
  `[gpu]` extra for the optional torch backend).

## [0.2.0] - 2026-10-12

### Documentation

- README: added a Documentation section (links to `docs/design.md`,
  `docs/gpu.md` and `docs/guide.html`), a glossary note for the CDF-t
  notation, a Limitations section, and honest cross-test skip wording;
  fixed the grid-field example ("per-cell drift" was a field-wide offset)
  and the calibration-reuse example (was a no-op); removed the
  maintainer-only PyPI-name paragraph and the multivariate references
  unrelated to the package; added a note that the R `CDFt` source is GPL
  and reproduced in the tests for validation only.
- `docs/design.md`: replaced unverifiable "measured" claims with the
  actual numbers and test references; removed informal wording and
  un-attributed "reference implementation" / "parent project" phrasing;
  cross-referenced the README comparison table.
- `docs/gpu.md`: added the measured MI210 speedup table (23× vs numpy);
  corrected the hardware/test-count claims to what the logs show.
- `docs/guide.html`: linked from the README, removed the hardcoded version
  badge, removed the invented "verification period" inputs, fixed the
  no-op quick-start example, trimmed the reference list.
- `CONTRIBUTING.md`: fixed the release instructions (version is
  single-sourced), the setup comment, added the GPU test command and a
  documentation-maintenance note.
- `docs/guide.html`: removed the duplication with the README — quick start
  shrunk to a teaser, the comparison table and measured numbers dropped
  (README/`docs/design.md` §4 only), the reference list replaced by a
  pointer, the FAQ removed (its answers restated earlier sections), and the
  GPU install table removed (see `docs/gpu.md`). The notation table now ties
  the symbols to the `obs`/`cm_hist`/`cm_future` names.
- `README.md`: removed the "Memory-lean design", "Tail semantics" and
  "NaN semantics" sections (conventions live in `docs/design.md` and the
  plain-language guide §6) and the measured comparison table (now
  `docs/design.md` §4 only); kept the aspect comparison table and trimmed
  the low-level API example.

### Changed

- The package is now called **PyCDFt**. The import name stays `pycdft`
  (lowercase, Python module convention), so existing imports are unchanged.

### Packaging

- Added `MANIFEST.in` so the sdist ships the docs, examples, benchmarks,
  the CI workflow and **all** test helpers (`tests/naive_cdft.py`,
  `tests/r_cdft_reference.py`), which the previous sdist silently omitted
  (tests from the sdist failed at collection).
- Added the `py.typed` marker: the package is fully type-annotated, so
  type checkers now treat it as typed.
- Switched to the SPDX license expression (`license = "Apache-2.0"` +
  `license-files`) and bumped the setuptools build requirement to `>=77`
  (the deprecated `license = { text = ... }` form is gone).
- Single-sourced the version: `pycdft/__init__.py` is now the only place
  that defines it (`[tool.setuptools.dynamic] version`).
- Extended the CI matrix to Python 3.9–3.13.
- Added the `dev` extra (pytest + pyflakes).

## [0.1.0] - 2026-10-12

### Added

- Vectorized CDF-t (Michelangeli, Vrac & Loukos 2009) on a numpy core:
  `pycdft.CDFt`, `calibrate` / `apply_cdft` / `calibrate_hist` /
  `apply_cdft_hist`, `cdft_transform`, `transfer`.
- Grid-ready kernels (sorted ECDF, type-7 quantiles, windowed-slope tails,
  histogram path), NaN conventions, float32 support, chunked processing and
  a `workers` thread pool.
- Optional torch GPU backend `pycdft.gpu` (CUDA / ROCm / MPS / CPU)
  mirroring the numpy API, verified numerically identical to the numpy core.
- Cross-validation against ibicus and a reference R CDFt port; tests,
  benchmarks, examples and this documentation set.
