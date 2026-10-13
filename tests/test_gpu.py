"""Tests of the optional torch GPU backend (``pycdft.gpu``).

The GPU backend is a device-agnostic mirror of the numpy backend; these
tests verify that it produces the same results as :class:`pycdft.CDFt` on
the same data (random 1-D / (T, N) / 3-D cases, Gaussian and skewed
distributions, all delta-shift modes, both tail options, NaN cases, grid
endpoints, ties / constant series, and the histogram mode).

Device handling is device-agnostic: the tests run wherever torch exposes a
device (CUDA, ROCm, MPS or CPU) and always pass on a CPU-only machine.
They are skipped entirely when torch is not installed.

Verified tolerance (float64 math on the device): the GPU backend matches
the numpy backend to ~1e-15 (a few ULPs) for the exact pipeline and to
~1e-13 in the histogram mode; the tests assert a documented tolerance of
``ATOL=1e-8, RTOL=1e-6``, which leaves ample headroom for cross-device
float64 accumulation-order differences while still being far tighter than
any scientific use.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import pycdft
from pycdft import CDFt as NumpyCDFt
from pycdft import apply_cdft as numpy_apply_cdft
from pycdft import calibrate as numpy_calibrate
from pycdft.gpu import (CDFt as GpuCDFt, TORCH_AVAILABLE, apply_cdft as gpu_apply_cdft,
                        available_devices, calibrate as gpu_calibrate,
                        describe, resolve_device)
from pycdft.gpu import kernels as gpu_kernels

pytestmark = pytest.mark.gpu

#: Documented equivalence tolerance (see the module docstring).
ATOL = 1e-8
RTOL = 1e-6

SHIFT_MODES = ("additive", "multiplicative", "none")
TAILS = ("linear", "clip")


def assert_equivalent(a, b, atol=ATOL, rtol=RTOL):
    """Assert numpy/gpu results agree (NaN masks included)."""
    a = np.asarray(a)
    b = np.asarray(b)
    assert a.shape == b.shape, f"shape mismatch: {a.shape} vs {b.shape}"
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b),
                                  err_msg="NaN masks differ")
    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, equal_nan=True)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def test_torch_available_flag():
    assert TORCH_AVAILABLE is True
    assert pycdft.GPU_AVAILABLE is True
    assert pycdft.gpu is not None


def test_available_devices_always_include_cpu():
    devices = available_devices()
    assert "cpu" in devices
    if torch.cuda.is_available():
        assert "cuda" in devices
    if torch.backends.mps.is_available():
        assert "mps" in devices


def test_resolve_device_auto_and_explicit():
    dev = resolve_device()                 # best available
    assert str(dev) in available_devices()
    assert str(resolve_device("cpu")) == "cpu"
    if "cuda" in available_devices():
        assert str(resolve_device("cuda")) == "cuda"
        if _has_rocm():
            assert str(resolve_device("rocm")) == "cuda"
        else:
            with pytest.raises(ValueError):
                resolve_device("rocm")


def _has_rocm():
    """True when a ROCm device is exposed (see pycdft.gpu.rocm_available)."""
    from pycdft.gpu import rocm_available
    return rocm_available()


def test_resolve_device_unavailable_raises():
    if "cuda" not in available_devices():
        with pytest.raises(ValueError):
            resolve_device("cuda")
        with pytest.raises(ValueError):
            resolve_device("rocm")
    if "mps" not in available_devices():
        with pytest.raises(ValueError):
            resolve_device("mps")


def test_describe_reports_build():
    info = describe()
    assert info["torch_available"] is True
    assert "cpu" in info["devices"]
    if torch.cuda.is_available():
        assert info["cuda_device_count"] >= 1
        assert len(info["cuda_device_names"]) == info["cuda_device_count"]


# ---------------------------------------------------------------------------
# Equivalence with the numpy backend
# ---------------------------------------------------------------------------

def _gaussian_1d(seed=7, t=1000):
    rng = np.random.default_rng(seed)
    return (rng.normal(0.0, 1.0, t), rng.normal(3.0, 1.0, t),
            rng.normal(4.0, 1.0, t))


@pytest.mark.parametrize("delta_shift", SHIFT_MODES)
@pytest.mark.parametrize("tail", TAILS)
def test_equivalence_1d_gaussian(delta_shift, tail):
    obs, cm_hist, cm_future = _gaussian_1d()
    want = NumpyCDFt(delta_shift=delta_shift, tail=tail) \
        .fit_transform(obs, cm_hist, cm_future)
    got = GpuCDFt(delta_shift=delta_shift, tail=tail) \
        .fit_transform(obs, cm_hist, cm_future)
    assert got.shape == cm_future.shape
    assert got.dtype == np.float64
    assert_equivalent(want, got)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_equivalence_2d_gaussian(dtype):
    rng = np.random.default_rng(1)
    obs = rng.normal(0.0, 1.0, (400, 6)).astype(dtype)
    cm_hist = rng.normal(3.0, 1.0, (400, 6)).astype(dtype)
    cm_future = rng.normal(4.0, 1.0, (400, 6)).astype(dtype)
    want = NumpyCDFt().fit_transform(obs, cm_hist, cm_future)
    got = GpuCDFt().fit_transform(obs, cm_hist, cm_future)
    assert_equivalent(want, got)


def test_equivalence_3d_shape_preserved():
    rng = np.random.default_rng(2)
    obs = rng.normal(0.0, 1.0, (200, 2, 3))
    cm_hist = rng.normal(3.0, 1.0, (200, 2, 3))
    cm_future = rng.normal(4.0, 1.0, (200, 2, 3))
    want = NumpyCDFt().fit_transform(obs, cm_hist, cm_future)
    got = GpuCDFt().fit_transform(obs, cm_hist, cm_future)
    assert got.shape == (200, 2, 3)
    assert_equivalent(want, got)


@pytest.mark.parametrize("delta_shift", SHIFT_MODES)
def test_equivalence_skewed(delta_shift):
    rng = np.random.default_rng(11)
    obs = rng.lognormal(0.0, 0.5, 2000)
    cm_hist = rng.lognormal(1.0, 0.6, 2000)
    cm_future = rng.lognormal(1.5, 0.7, 2000)
    want = NumpyCDFt(delta_shift=delta_shift).fit_transform(obs, cm_hist, cm_future)
    got = GpuCDFt(delta_shift=delta_shift).fit_transform(obs, cm_hist, cm_future)
    assert_equivalent(want, got)


def test_equivalence_skewed_2d():
    rng = np.random.default_rng(12)
    obs = rng.gamma(2.0, 1.0, (500, 4))
    cm_hist = rng.gamma(3.0, 1.5, (500, 4))
    cm_future = rng.gamma(4.0, 2.0, (500, 4))
    want = NumpyCDFt(delta_shift="multiplicative").fit_transform(obs, cm_hist, cm_future)
    got = GpuCDFt(delta_shift="multiplicative").fit_transform(obs, cm_hist, cm_future)
    assert_equivalent(want, got)


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def test_equivalence_nan_sprinkled():
    rng = np.random.default_rng(21)
    obs = rng.normal(0.0, 1.0, 500)
    obs[10:30] = np.nan
    cm_hist = rng.normal(2.0, 1.0, 500)
    cm_hist[100:110] = np.nan
    cm_future = rng.normal(3.0, 1.0, 500)
    cm_future[50] = np.nan
    want = NumpyCDFt().fit_transform(obs, cm_hist, cm_future)
    got = GpuCDFt().fit_transform(obs, cm_hist, cm_future)
    assert np.isnan(want).any() and np.isnan(got).any()
    assert_equivalent(want, got)


def test_equivalence_nan_calibration_column():
    rng = np.random.default_rng(22)
    obs = rng.normal(0.0, 1.0, (300, 4))
    cm_hist = rng.normal(2.0, 1.0, (300, 4))
    cm_future = rng.normal(3.0, 1.0, (300, 4))
    obs[:, 1] = np.nan                    # empty calibration cell -> NaN
    cm_future[::3, 2] = np.nan            # sprinkled target NaNs
    want = NumpyCDFt().fit_transform(obs, cm_hist, cm_future)
    got = GpuCDFt().fit_transform(obs, cm_hist, cm_future)
    assert np.isnan(want[:, 1]).all() and np.isnan(got[:, 1]).all()
    assert_equivalent(want, got)


def test_equivalence_all_nan_inputs():
    obs = np.full(300, np.nan)
    cm_hist = np.full(300, np.nan)
    cm_future = np.full(300, np.nan)
    want = NumpyCDFt().fit_transform(obs, cm_hist, cm_future)
    got = GpuCDFt().fit_transform(obs, cm_hist, cm_future)
    assert np.isnan(want).all() and np.isnan(got).all()


# ---------------------------------------------------------------------------
# Grid endpoints, ties, degenerate series
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_quantiles", [2, 11, 101])
@pytest.mark.parametrize("tail", TAILS)
def test_equivalence_grid_endpoints(n_quantiles, tail):
    rng = np.random.default_rng(31)
    obs = rng.normal(0.0, 1.0, 300)
    cm_hist = rng.normal(1.0, 1.0, 300)
    cm_future = rng.normal(0.5, 1.0, 300)
    cm_future[0] = 100.0                  # far above the data range
    cm_future[1] = -100.0                 # far below the data range
    want = NumpyCDFt(n_quantiles=n_quantiles, tail=tail) \
        .fit_transform(obs, cm_hist, cm_future)
    got = GpuCDFt(n_quantiles=n_quantiles, tail=tail) \
        .fit_transform(obs, cm_hist, cm_future)
    assert_equivalent(want, got)


def test_equivalence_ties_and_duplicates():
    rng = np.random.default_rng(41)
    obs = rng.integers(-5, 5, 800).astype(np.float64)
    cm_hist = rng.integers(0, 10, 800).astype(np.float64)
    cm_future = rng.integers(-2, 8, 800).astype(np.float64)
    for tail in TAILS:
        want = NumpyCDFt(tail=tail).fit_transform(obs, cm_hist, cm_future)
        got = GpuCDFt(tail=tail).fit_transform(obs, cm_hist, cm_future)
        assert_equivalent(want, got)


def test_equivalence_constant_column():
    rng = np.random.default_rng(42)
    obs = rng.normal(0.0, 1.0, (300, 3))
    cm_hist = rng.normal(1.0, 1.0, (300, 3))
    cm_future = rng.normal(2.0, 1.0, (300, 3))
    obs[:, 0] = 7.0
    cm_hist[:, 0] = 3.0
    cm_future[:, 0] = 5.0
    want = NumpyCDFt().fit_transform(obs, cm_hist, cm_future)
    got = GpuCDFt().fit_transform(obs, cm_hist, cm_future)
    assert_equivalent(want, got)


def test_equivalence_tiny_series():
    obs = np.array([1.0, 2.0, 3.0])
    cm_hist = np.array([2.0, 2.5, 3.5])
    cm_future = np.array([2.2, 2.8, 3.8])
    want = NumpyCDFt().fit_transform(obs, cm_hist, cm_future)
    got = GpuCDFt().fit_transform(obs, cm_hist, cm_future)
    assert_equivalent(want, got)


def test_equivalence_single_point():
    obs = np.array([5.0])
    cm_hist = np.array([5.0])
    cm_future = np.array([5.0])
    want = NumpyCDFt().fit_transform(obs, cm_hist, cm_future)
    got = GpuCDFt().fit_transform(obs, cm_hist, cm_future)
    assert_equivalent(want, got)


# ---------------------------------------------------------------------------
# Low-level API
# ---------------------------------------------------------------------------

def test_low_level_calibration_equivalence():
    rng = np.random.default_rng(51)
    obs = rng.normal(0.0, 1.0, 600)
    cm_hist = rng.normal(2.0, 1.0, 600)
    obs[20:25] = np.nan
    want = numpy_calibrate(cm_hist, obs, delta_shift="additive")
    got = gpu_calibrate(cm_hist, obs, delta_shift="additive")
    np.testing.assert_allclose(want.R, got.R.cpu().numpy(), atol=ATOL, rtol=RTOL,
                               equal_nan=True)
    np.testing.assert_allclose(want.delta, got.delta.cpu().numpy(),
                               atol=ATOL, rtol=RTOL)
    assert got.mode == want.mode and got.n_quantiles == want.n_quantiles


def test_low_level_apply_equivalence():
    rng = np.random.default_rng(52)
    obs = rng.normal(0.0, 1.0, 400)
    cm_hist = rng.normal(2.0, 1.0, 400)
    cm_future = rng.normal(3.0, 1.0, 400)
    cal_np = numpy_calibrate(cm_hist, obs, delta_shift="additive")
    cal_gpu = gpu_calibrate(cm_hist, obs, delta_shift="additive")
    want = numpy_apply_cdft(cm_future, cal_np, tail="linear")
    got = gpu_apply_cdft(cm_future, cal_gpu, tail="linear")
    assert_equivalent(want, got)
    # numpy() round-trip of the GPU calibration
    cal_rt = cal_gpu.numpy()
    np.testing.assert_allclose(cal_rt.R, cal_np.R, atol=ATOL, rtol=RTOL,
                               equal_nan=True)


def test_chunked_matches_unchunked():
    rng = np.random.default_rng(61)
    obs = rng.normal(0.0, 1.0, (600, 100))
    cm_hist = rng.normal(2.0, 1.0, (600, 100))
    cm_future = rng.normal(3.0, 1.0, (600, 100))
    want = GpuCDFt().fit_transform(obs, cm_hist, cm_future)
    got = GpuCDFt(chunk_cols=16).fit_transform(obs, cm_hist, cm_future)
    np.testing.assert_array_equal(want, got)


# ---------------------------------------------------------------------------
# Kernel-level equivalence
# ---------------------------------------------------------------------------

def test_kernels_match_numpy_math():
    from pycdft.math import (ecdf_at as np_ecdf_at, hist_quantile as np_hq,
                             quantile as np_quantile, windowed_slope as np_ws)
    rng = np.random.default_rng(71)
    v = rng.normal(0.0, 1.0, (200, 7))
    v[3, 1] = np.nan
    pts = rng.normal(0.0, 1.0, (50, 7))
    pts[7, 0] = np.nan
    q = np.linspace(0.0, 1.0, 33)
    np.testing.assert_allclose(
        np_ecdf_at(v, pts), gpu_kernels.ecdf_at(torch.tensor(v), torch.tensor(pts)).numpy(),
        atol=ATOL, rtol=RTOL, equal_nan=True)
    np.testing.assert_allclose(
        np_quantile(v, q), gpu_kernels.quantile(torch.tensor(v), torch.tensor(q)).numpy(),
        atol=ATOL, rtol=RTOL, equal_nan=True)
    xg = np.sort(rng.normal(0.0, 1.0, (101, 5)), axis=0)
    T = rng.normal(0.0, 1.0, (101, 5))
    lo = np.array([0, 5, 10, 20, 40])
    hi = np.array([30, 35, 40, 50, 70])
    np.testing.assert_allclose(
        np_ws(xg, T, lo, hi),
        gpu_kernels.windowed_slope(torch.tensor(xg), torch.tensor(T),
                                   torch.tensor(lo), torch.tensor(hi)).numpy(),
        atol=ATOL, rtol=RTOL)
    edges, counts = gpu_kernels.histogram_accumulate(torch.tensor(v), n_bins=50)
    np.testing.assert_array_equal(edges.numpy(), np.linspace(np.nanmin(v), np.nanmax(v), 51))
    np.testing.assert_allclose(
        np_hq(np.linspace(np.nanmin(v), np.nanmax(v), 51),
              counts.numpy(), q),
        gpu_kernels.hist_quantile(edges, counts, torch.tensor(q)).numpy(),
        atol=ATOL, rtol=RTOL, equal_nan=True)


# ---------------------------------------------------------------------------
# Histogram mode
# ---------------------------------------------------------------------------

def test_histogram_mode_equivalence():
    rng = np.random.default_rng(81)
    obs = rng.normal(0.0, 1.0, 1500)
    cm_hist = rng.normal(2.0, 1.0, 1500)
    cm_future = rng.normal(3.0, 1.0, 1500)
    for tail in TAILS:
        want = NumpyCDFt(use_histograms=True, n_bins=1000, tail=tail) \
            .fit_transform(obs, cm_hist, cm_future)
        got = GpuCDFt(use_histograms=True, n_bins=1000, tail=tail) \
            .fit_transform(obs, cm_hist, cm_future)
        assert_equivalent(want, got)


def test_histogram_mode_equivalence_2d_nan():
    rng = np.random.default_rng(82)
    obs = rng.normal(0.0, 1.0, (400, 5))
    obs[5:9, 2] = np.nan
    cm_hist = rng.normal(2.0, 1.0, (400, 5))
    cm_future = rng.normal(3.0, 1.0, (400, 5))
    cm_future[:, 4] = np.nan
    want = NumpyCDFt(use_histograms=True, n_bins=500) \
        .fit_transform(obs, cm_hist, cm_future)
    got = GpuCDFt(use_histograms=True, n_bins=500) \
        .fit_transform(obs, cm_hist, cm_future)
    assert_equivalent(want, got)


# ---------------------------------------------------------------------------
# Input / output conventions
# ---------------------------------------------------------------------------

def test_numpy_and_tensor_inputs_agree():
    rng = np.random.default_rng(91)
    obs = rng.normal(0.0, 1.0, 500)
    cm_hist = rng.normal(2.0, 1.0, 500)
    cm_future = rng.normal(3.0, 1.0, 500)
    a = GpuCDFt().fit_transform(obs, cm_hist, cm_future)
    b = GpuCDFt().fit_transform(torch.tensor(obs), torch.tensor(cm_hist),
                                torch.tensor(cm_future))
    np.testing.assert_array_equal(a, b)


def test_as_numpy_false_returns_tensor_on_device():
    rng = np.random.default_rng(92)
    obs = rng.normal(0.0, 1.0, 300)
    cm_hist = rng.normal(2.0, 1.0, 300)
    cm_future = rng.normal(3.0, 1.0, 300)
    model = GpuCDFt(as_numpy=False)
    out = model.fit_transform(obs, cm_hist, cm_future)
    assert isinstance(out, torch.Tensor)
    assert out.device.type == model.device.type          # cuda / cpu / mps
    assert out.dtype == torch.float64
    assert model.calibration_.R.device.type == model.device.type
    # numpy output equals the tensor output
    model2 = GpuCDFt(as_numpy=True).fit_transform(obs, cm_hist, cm_future)
    np.testing.assert_array_equal(out.cpu().numpy(), model2)


def test_inputs_not_modified():
    rng = np.random.default_rng(93)
    obs = rng.normal(0.0, 1.0, 300)
    cm_hist = rng.normal(2.0, 1.0, 300)
    cm_future = rng.normal(3.0, 1.0, 300)
    obs0, ch0, cf0 = obs.copy(), cm_hist.copy(), cm_future.copy()
    GpuCDFt().fit_transform(obs, cm_hist, cm_future)
    np.testing.assert_array_equal(obs, obs0)
    np.testing.assert_array_equal(cm_hist, ch0)
    np.testing.assert_array_equal(cm_future, cf0)


def test_fit_transform_matches_fit_then_transform():
    rng = np.random.default_rng(94)
    obs = rng.normal(0.0, 1.0, 300)
    cm_hist = rng.normal(2.0, 1.0, 300)
    cm_future = rng.normal(3.0, 1.0, 300)
    a = GpuCDFt().fit_transform(obs, cm_hist, cm_future)
    b = GpuCDFt().fit(obs, cm_hist).transform(cm_future)
    np.testing.assert_array_equal(a, b)


def test_transform_requires_fit():
    with pytest.raises(RuntimeError):
        GpuCDFt().transform(np.zeros(10))


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        GpuCDFt(n_quantiles=1)
    with pytest.raises(ValueError):
        GpuCDFt(tail="cubic")
    with pytest.raises(ValueError):
        GpuCDFt(delta_shift="no_shift")
    with pytest.raises(ValueError):
        GpuCDFt(n_bins=1)


def test_cdft_transform_one_shot_matches_class():
    rng = np.random.default_rng(95)
    obs = rng.normal(0.0, 1.0, 300)
    cm_hist = rng.normal(2.0, 1.0, 300)
    cm_future = rng.normal(3.0, 1.0, 300)
    from pycdft.gpu import cdft_transform
    a = cdft_transform(obs, cm_hist, cm_future)
    b = GpuCDFt().fit_transform(obs, cm_hist, cm_future)
    np.testing.assert_array_equal(a, b)


def test_cpu_device_explicit():
    rng = np.random.default_rng(96)
    obs = rng.normal(0.0, 1.0, 200)
    cm_hist = rng.normal(2.0, 1.0, 200)
    cm_future = rng.normal(3.0, 1.0, 200)
    model = GpuCDFt(device="cpu")
    out = model.fit_transform(obs, cm_hist, cm_future)
    assert str(model.device) == "cpu"
    assert model.calibration_.R.device.type == "cpu"
    assert_equivalent(NumpyCDFt().fit_transform(obs, cm_hist, cm_future), out)
