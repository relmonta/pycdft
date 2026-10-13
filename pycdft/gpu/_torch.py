"""Internal torch accessor for :mod:`pycdft.gpu` (optional dependency).

Torch is an *optional* dependency of pycdft: the package itself is pure
numpy and imports fine without torch.  This module centralizes the guarded
import so that every other module of the GPU backend can rely on
:func:`require_torch` to raise a clear error only when torch is actually
needed.
"""

from __future__ import annotations

try:                                    # pragma: no cover - exercised only
    import torch                        # when torch is missing
    TORCH_AVAILABLE = True
except ImportError:                     # pragma: no cover
    torch = None                        # type: ignore[assignment]
    TORCH_AVAILABLE = False

__all__ = ["TORCH_AVAILABLE", "require_torch"]


def require_torch():
    """Return the ``torch`` module or raise an informative ``ImportError``."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "pycdft.gpu requires torch: install it with "
            "`pip install pycdft[gpu]` (or a torch build matching your "
            "hardware, see docs/gpu.md)")
    return torch
