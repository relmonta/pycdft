"""Device detection and resolution for :mod:`pycdft.gpu`.

The torch backend is device-agnostic: it runs wherever torch exposes a
device.  Detection order (first available wins for the automatic default):

- ``cuda`` -- NVIDIA CUDA, and also AMD ROCm builds of torch, which expose
  the ROCm devices through the same ``cuda`` API (a ROCm build is
  identified by ``torch.version.hip`` being set and
  ``torch.cuda.is_available()`` being true);
- ``rocm`` -- reported (as an alias of the same ``cuda`` device) when a
  ROCm build of torch is detected;
- ``mps`` -- Apple Metal (MacOS);
- ``cpu`` -- always available.

A specific device can be requested (``"cuda"``, ``"cuda:0"``, ``"mps"``,
``"cpu"``, or ``"rocm"`` as an alias for the CUDA device of a ROCm build);
``resolve_device`` raises ``ValueError`` when the requested device is not
available, so explicit requests never silently fall back to CPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._torch import TORCH_AVAILABLE, require_torch

if TYPE_CHECKING:                        # annotations only, never runtime
    import torch

__all__ = [
    "TORCH_AVAILABLE",
    "available_devices",
    "rocm_available",
    "is_rocm_build",
    "resolve_device",
    "describe",
]

#: Preferred order for the automatic default device.
_DEFAULT_ORDER = ("cuda", "mps", "cpu")


def is_rocm_build() -> bool:
    """True when the installed torch was built for AMD ROCm."""
    if not TORCH_AVAILABLE:
        return False
    torch = require_torch()
    return getattr(torch.version, "hip", None) is not None


def rocm_available() -> bool:
    """True when torch exposes an AMD ROCm device.

    A ROCm build of torch is identified by ``torch.version.hip`` (set) and
    exposes the AMD devices through the ``cuda`` API.  ``torch.backends.rocm``
    exists only in some builds; the HIP build string plus
    ``torch.cuda.is_available()`` is the reliable marker.
    """
    if not TORCH_AVAILABLE:
        return False
    torch = require_torch()
    backend = getattr(torch.backends, "rocm", None)
    if backend is not None and getattr(backend, "is_available", lambda: False)():
        return True
    return bool(is_rocm_build() and torch.cuda.is_available())


def available_devices() -> list:
    """Names of the devices torch can use, best first (``["cuda", "cpu"]``,
    ``["cuda", "rocm", "cpu"]`` for a ROCm build, ``["mps", "cpu"]``, ...).

    An empty list means torch is not installed.
    """
    if not TORCH_AVAILABLE:
        return []
    torch = require_torch()
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
        if rocm_available():
            devices.append("rocm")      # same physical device, ROCm flavour
    if torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def resolve_device(preferred=None) -> "torch.device":
    """Resolve a device name (or ``None`` for the best available device).

    Parameters
    ----------
    preferred : str, torch.device or None
        ``"cuda"`` / ``"cuda:0"``, ``"mps"``, ``"cpu"``, or ``"rocm"``
        (alias for the CUDA device of a ROCm build).  ``None`` picks the
        best available device automatically.

    Returns
    -------
    torch.device

    Raises
    ------
    ImportError
        If torch is not installed.
    ValueError
        If the requested device is not available on this machine (an
        explicit request never silently falls back).
    """
    torch = require_torch()
    available = available_devices()
    if preferred is None:
        for name in _DEFAULT_ORDER:
            if name in available:
                return torch.device(name)
        raise RuntimeError("torch is installed but exposes no usable device")
    if isinstance(preferred, torch.device):
        name = preferred.type
        index = preferred.index
    else:
        name, _, index_part = str(preferred).partition(":")
        index = int(index_part) if index_part else None
    if name == "rocm":
        if "rocm" not in available:
            raise ValueError(
                f"torch device 'rocm' is not available on this machine "
                f"(available: {available}); a ROCm build of torch is "
                f"required on AMD GPUs")
        return torch.device("cuda" if index is None else f"cuda:{index}")
    if name not in available:
        raise ValueError(
            f"torch device {preferred!r} is not available on this machine "
            f"(available: {available})")
    return torch.device(name if index is None else f"{name}:{index}")


def describe() -> dict:
    """Human-readable description of the torch installation and devices.

    Returns a dict with the torch version, the CUDA / ROCm build versions,
    the available devices and, for CUDA devices, ``torch.cuda.get_device_name``
    of each device.  Useful for diagnostics (e.g. on a compute node).
    """
    if not TORCH_AVAILABLE:
        return {"torch_available": False, "devices": []}
    torch = require_torch()
    info = {
        "torch_available": True,
        "torch_version": torch.__version__,
        "cuda_build": getattr(torch.version, "cuda", None),
        "hip_build": getattr(torch.version, "hip", None),
        "rocm_backend": rocm_available(),
        "devices": available_devices(),
    }
    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_names"] = [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
    return info
