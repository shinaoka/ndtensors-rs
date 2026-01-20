"""Python bindings for ndtensors-rs tensor library.

This package provides Python bindings to the Rust ndtensors library,
allowing high-performance tensor operations with automatic differentiation
support.

Examples:
    >>> from ndtensors_rs import TensorF64, contract
    >>> a = TensorF64.ones((2, 3))
    >>> b = TensorF64.ones((3, 4))
    >>> c = contract(a, (1, -1), b, (-1, 2))
    >>> c.shape
    (2, 4)

For JAX integration:
    >>> from ndtensors_rs.jax_ops import jax_contract  # requires jax

For PyTorch integration:
    >>> from ndtensors_rs.torch_ops import torch_contract  # requires torch
"""

from .tensor import TensorF64
from .ops import contract, contract_vjp
from ._status import NDTensorsError, StatusCode

__all__ = [
    "TensorF64",
    "contract",
    "contract_vjp",
    "NDTensorsError",
    "StatusCode",
]

__version__ = "0.1.0"


def __getattr__(name: str):
    """Lazy import for optional dependencies."""
    if name == "jax_contract":
        from .jax_ops import jax_contract
        return jax_contract
    if name == "torch_contract":
        from .torch_ops import torch_contract
        return torch_contract
    if name == "JAX_AVAILABLE":
        from .jax_ops import JAX_AVAILABLE
        return JAX_AVAILABLE
    if name == "TORCH_AVAILABLE":
        from .torch_ops import TORCH_AVAILABLE
        return TORCH_AVAILABLE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
