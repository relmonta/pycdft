# Contributing to PyCDFt

Thanks for contributing! This package is intentionally small, numpy-only and
dependency-light: keep it that way.

## Setup

```bash
pip install -e .[test]        # numpy + pytest (matplotlib via .[examples])
pip install ibicus            # only needed for the ibicus cross tests
```

## Tests

```bash
pytest                        # everything
pytest -m "not cross"         # core suite (no external deps, CI runs this)
pytest -m cross -s            # cross-validation (ibicus + R port, prints metrics)
pytest -m gpu                 # torch GPU backend (skipped without torch)
```

- New kernels go in `pycdft/math.py` (numpy only, vectorized over the cell
  axis, docstrings with the convention); the pipeline in `pycdft/core.py`;
  the public API in `pycdft/api.py`.
- Any change to the numerics must keep the conventions documented in
  `docs/design.md` (step ECDF, type-7 quantiles, delta shift, grid, tails,
  NaN semantics) — and the cross-test tolerances must stay honest: if a
  tolerance is loosened, update the numbers in `README.md` and
  `docs/design.md` (they are reproduced by `pytest -m cross -s`).
- Documentation: the public docs are `README.md`, `docs/design.md`,
  `docs/gpu.md` and `docs/guide.html` (the plain-language companion).
  Any API or convention change must update all of them consistently —
  `docs/guide.html` is hand-maintained, so check it for drift too.

## Conventions

- numpy only in the core; scipy/xarray/pandas/matplotlib must stay optional.
- 1-D series and `(T, N)` fields both supported (any trailing dims are
  flattened into cells).
- Never mutate the caller's arrays (`np.sort` only on private copies).
- Deterministic: `workers > 1` must not change results.
- No project-specific paths, no network access in the tests.

## Releasing (maintainers)

1. Bump `__version__` in `pycdft/__init__.py` (single-sourced:
   `pyproject.toml` reads it via `[tool.setuptools.dynamic]`).
2. `python -m build` (or `pip install build && python -m build`).
3. `twine check dist/* && twine upload dist/*` (the name `pycdft` was free
   at the time of writing; re-check `https://pypi.org/pypi/pycdft/json`
   before publishing).
