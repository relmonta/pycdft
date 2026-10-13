# GPU backend (`pycdft.gpu`)

The optional torch backend (`pycdft.gpu`) is a device-agnostic mirror of the
numpy API: the **same CDF-t math** (step ECDFs, type-7 quantiles, per-cell
delta shift, transfer interpolation with `clip` / capped-LSQ `linear` tails,
NaN-safe handling) implemented with torch operators, so the per-cell kernels
can run on a GPU or on CPU with the same interface:

```python
from pycdft.gpu import CDFt

ds = CDFt(device="cuda").fit_transform(obs, cm_hist, cm_future)
```

## Installation

```bash
# torch is an optional dependency: the numpy core never needs it
pip install .                 # numpy-only core
pip install .[gpu]            # + torch (pip's default torch build is CUDA)
```

Install a torch build matching your hardware:

| hardware | torch build |
|---|---|
| NVIDIA GPU | `pip install torch` (CUDA build, default) |
| AMD GPU (ROCm) | `pip install torch --index-url https://download.pytorch.org/whl/rocm6.3` (adjust the ROCm version) |
| Apple Silicon | `pip install torch` (MPS backend, `mps` device) |
| CPU only | `pip install torch --index-url https://download.pytorch.org/whl/cpu` |

The GPU backend is never imported by the numpy core; without torch,
`pycdft.gpu` is still importable (`pycdft.gpu.TORCH_AVAILABLE == False`) and
its functions raise a clear `ImportError` when called.

## Devices

`pycdft.gpu.available_devices()` reports what torch exposes on the machine
and `pycdft.gpu.resolve_device()` picks the device:

```python
from pycdft.gpu import available_devices, resolve_device

available_devices()      # e.g. ['cuda', 'cpu'] or ['mps', 'cpu'] or ['cpu']
dev = resolve_device()   # best available: cuda -> mps -> cpu
dev = resolve_device("cpu")   # explicit; raises ValueError if unavailable
```

- **CUDA** (`"cuda"`, `"cuda:0"`, ...): NVIDIA GPUs and **AMD ROCm builds**
  of torch, which expose the AMD GPUs through the same `cuda` API
  (`torch.version.hip` is set, `torch.backends.rocm.is_available()` is
  true).
- **ROCm** (`"rocm"`): accepted as an alias for the `cuda` device of a ROCm
  build of torch.
- **MPS** (`"mps"`): Apple Metal.
- **CPU** (`"cpu"`): always available; also the fallback of the automatic
  selection on machines without an accelerated device.

An explicit request for an unavailable device raises `ValueError` (never a
silent CPU fallback); `resolve_device(None)` selects the best available
device. `pycdft.gpu.describe()` returns a small diagnostic dict (torch /
CUDA / HIP build versions, available devices, device names).

## Equivalence with the numpy backend

The GPU backend converts its inputs to float64 tensors on the compute
device and reproduces the numpy arithmetic (same type-7 index/weight
formulas, same `np.interp` endpoint/duplicate convention, same
`np.nanmedian` two-middle-values convention — note that `torch.nanmedian`
returns the *lower* median, so the median is computed explicitly).

Verified against the numpy backend in `tests/test_gpu.py`:

- exact pipeline: **max |Δ| ≤ ~1e-15** (typically 0, i.e. bit-identical;
  a few ULPs in the additive delta mean);
- histogram mode: **max |Δ| ≤ ~1e-13**;
- the tests assert a documented tolerance of `atol=1e-8, rtol=1e-6`
  (NaN masks must match exactly).

```bash
pip install -e .[test,gpu]
pytest tests/test_gpu.py -v     # device-agnostic; runs on CPU if no GPU
pytest                          # full suite (core + GPU backend)
```

## API

`pycdft.gpu` mirrors the numpy API:

```python
from pycdft.gpu import (CDFt, cdft_transform, calibrate, apply_cdft,
                        Calibration, histograms, calibrate_hist,
                        apply_cdft_hist)

# high level (same parameters as pycdft.CDFt + device/chunk_cols/as_numpy)
ds = CDFt(device="cuda", n_quantiles=1001, tail="linear",
          delta_shift="additive").fit_transform(obs, cm_hist, cm_future)

# low level
calib = calibrate(cm_hist, obs, device="cuda")            # GPU Calibration
pred = apply_cdft(cm_future, calib, tail="linear")        # numpy array
pred_t = apply_cdft(cm_future, calib, as_numpy=False)     # tensor on device
calib_np = calib.numpy()          # -> numpy-backed Calibration (pycdft.core)
```

- Inputs: numpy arrays **or** torch tensors (moved to the compute device,
  converted to float64; inputs are never modified).
- Outputs: numpy float64 arrays by default (`as_numpy=True`), torch tensors
  with `as_numpy=False` — the numpy API is a drop-in replacement.
- `use_histograms=True` (histogram target CDF) is supported and matches the
  numpy histogram path.

## Memory notes

The kernels are fully vectorized: peak memory for a field is
`O((T + Q) * N * 8 bytes)` (plus one sorted copy of the input), i.e. the
whole field is processed at once — unlike the chunked numpy core. For very
large fields, pass `chunk_cols` (e.g. `CDFt(chunk_cols=4096)` or
`calibrate(..., chunk_cols=4096)`) to process the cells in column chunks:
the results are identical, the peak memory is bounded by
`O((T + Q) * chunk_cols * 8 bytes)`.

## Verified on a hybrid GPU node (AMD MI210, ROCm)

Tested on a SLURM `hybrid` partition node (AMD Instinct MI210, allocated
with `--gres=gpu:mi210:1`):

- `rocm-smi` on the node reports the AMD Instinct MI210 device (ROCm
  driver stack present); `nvidia-smi` does not exist.
- **CUDA-built torch** (`torch 2.11.0+cu130`, the environment's default):
  `torch.cuda.is_available()` → `False` on the AMD node,
  `torch.version.hip` → `None`, and `torch.backends.rocm` does not exist,
  so `pycdft.gpu` auto-selected `cpu`.  The GPU test suite (45 tests) and
  the full suite (134 passed, 1 skipped at the time) **passed on CPU on
  the GPU node** — a CUDA-built torch does not accelerate an AMD GPU.
- **ROCm-built torch** (`torch 2.9.1+rocm6.3` in a scratch venv, matching
  the node's ROCm 6.3.1 stack): `torch.cuda.is_available()` → `True`,
  `torch.version.hip` → `6.3.42134`, the device is `cuda:0`
  (`AMD Instinct MI210`, capability (9, 0)),
  `available_devices()` → `["cuda", "rocm", "cpu"]` and the whole
  `tests/test_gpu.py` suite (45 tests) **passed on the MI210 GPU**.  The
  backend required no changes; a float64 (1000×1000) tensor round-trip on
  `cuda` works.

Note: `torch.backends.rocm` does not exist in torch 2.9 ROCm builds; the
reliable markers are `torch.version.hip` (set) and
`torch.cuda.is_available()` (true), which is what `pycdft.gpu` uses.

## Measured speedup (MI210, ROCm)

From `benchmarks/benchmark_gpu.py` on the MI210 node (synthetic field:
T_cal = 10000, T_tar = 5000, N = 50000 cells, Q = 1001, float32 inputs;
reproduce with `bash benchmarks/run_hybrid_benchmark.sh`):

| backend | wall time | vs numpy (1 thread) |
|---|---|---|
| numpy, 1 thread | 45.0 s | 1.0× |
| numpy, 16 threads | 27.1 s | 1.7× |
| torch, CPU | 87.2 s | 0.52× |
| torch, CUDA (MI210) | 1.96 s | **23×** |

The speedup is on the *transform* of a large field; the first GPU call
also pays the (one-time) kernel/stream initialization. For small fields
the numpy core is faster (torch CPU is ~0.5× here) and, on consumer GPUs,
float64 kernels may not beat numpy — see the caveats below.

## Caveats

- AMD MI210 / ROCm: use a ROCm build of torch whose ROCm version matches
  the node stack (e.g. `pip install torch==2.9.1+rocm6.3 --index-url
  https://download.pytorch.org/whl/rocm6.3`); the device is then named
  `cuda` (torch's ROCm port uses the CUDA API) and `pycdft.gpu` reports it
  as `cuda` (plus `rocm` as an alias).  The GPU backend itself needs no
  changes.
- float64 kernels on consumer GPUs can be slower than float32 kernels; the
  backend computes in float64 deliberately, for bit-compatibility with the
  numpy backend. For pure float32 use cases, the numpy core (with its
  float32-keeping sort) is the more economical option.
- `torch.searchsorted` is not well-defined for NaN entries: the kernels
  treat NaN as `+inf` (as numpy does) and mask NaN queries back to NaN.
