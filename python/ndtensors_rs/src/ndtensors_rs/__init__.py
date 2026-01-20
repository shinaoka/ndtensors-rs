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
