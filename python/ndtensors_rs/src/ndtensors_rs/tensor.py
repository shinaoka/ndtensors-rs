"""TensorF64 class for dense tensors backed by Rust."""

from __future__ import annotations

import ctypes
from typing import Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from ._lib import get_lib
from ._status import NDTensorsError, StatusCode, check_status


class TensorF64:
    """Dense tensor of float64 values backed by Rust.

    This class wraps a Rust tensor and provides a Pythonic interface.
    Memory is automatically managed - when the Python object is garbage
    collected, the underlying Rust memory is released.

    Examples:
        >>> t = TensorF64.zeros((2, 3))
        >>> t.shape
        (2, 3)
        >>> t.fill(1.0)
        >>> t[0]
        1.0
    """

    __slots__ = ("_ptr", "_lib")

    def __init__(self, ptr: int) -> None:
        """Internal constructor. Use class methods to create tensors.

        Args:
            ptr: Raw pointer to the Rust tensor (as integer)
        """
        self._lib = get_lib()
        self._ptr = ptr

    def __del__(self) -> None:
        """Release Rust memory when Python object is garbage collected."""
        if hasattr(self, "_ptr") and self._ptr:
            self._lib.ndt_tensor_f64_release(self._ptr)
            self._ptr = None

    @classmethod
    def zeros(cls, shape: Tuple[int, ...]) -> TensorF64:
        """Create a tensor filled with zeros.

        Args:
            shape: Shape of the tensor

        Returns:
            New tensor filled with zeros
        """
        lib = get_lib()
        ndim = len(shape)
        if ndim == 0:
            shape_arr = None
        else:
            shape_arr = (ctypes.c_size_t * ndim)(*shape)
        status = ctypes.c_int(-999)
        ptr = lib.ndt_tensor_f64_zeros(shape_arr, ndim, ctypes.byref(status))
        check_status(status.value, "zeros")
        return cls(ptr)

    @classmethod
    def ones(cls, shape: Tuple[int, ...]) -> TensorF64:
        """Create a tensor filled with ones.

        Args:
            shape: Shape of the tensor

        Returns:
            New tensor filled with ones
        """
        lib = get_lib()
        ndim = len(shape)
        if ndim == 0:
            shape_arr = None
        else:
            shape_arr = (ctypes.c_size_t * ndim)(*shape)
        status = ctypes.c_int(-999)
        ptr = lib.ndt_tensor_f64_ones(shape_arr, ndim, ctypes.byref(status))
        check_status(status.value, "ones")
        return cls(ptr)

    @classmethod
    def rand(cls, shape: Tuple[int, ...]) -> TensorF64:
        """Create a tensor with uniform random values in [0, 1).

        Args:
            shape: Shape of the tensor

        Returns:
            New tensor with random values
        """
        lib = get_lib()
        ndim = len(shape)
        if ndim == 0:
            shape_arr = None
        else:
            shape_arr = (ctypes.c_size_t * ndim)(*shape)
        status = ctypes.c_int(-999)
        ptr = lib.ndt_tensor_f64_rand(shape_arr, ndim, ctypes.byref(status))
        check_status(status.value, "rand")
        return cls(ptr)

    @classmethod
    def randn(cls, shape: Tuple[int, ...]) -> TensorF64:
        """Create a tensor with standard normal random values (mean=0, std=1).

        Args:
            shape: Shape of the tensor

        Returns:
            New tensor with standard normal random values
        """
        lib = get_lib()
        ndim = len(shape)
        if ndim == 0:
            shape_arr = None
        else:
            shape_arr = (ctypes.c_size_t * ndim)(*shape)
        status = ctypes.c_int(-999)
        ptr = lib.ndt_tensor_f64_randn(shape_arr, ndim, ctypes.byref(status))
        check_status(status.value, "randn")
        return cls(ptr)

    @classmethod
    def from_numpy(cls, arr: NDArray[np.float64]) -> TensorF64:
        """Create a tensor from a NumPy array (copies data).

        The array will be converted to column-major (Fortran) order
        for compatibility with the Rust backend.

        Args:
            arr: NumPy array to convert

        Returns:
            New tensor with copied data
        """
        lib = get_lib()
        # Ensure column-major (Fortran) order for compatibility
        arr = np.asfortranarray(arr, dtype=np.float64)
        shape = arr.shape
        ndim = len(shape)
        if ndim == 0:
            shape_arr = None
        else:
            shape_arr = (ctypes.c_size_t * ndim)(*shape)
        status = ctypes.c_int(-999)
        ptr = lib.ndt_tensor_f64_from_data(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            arr.size,
            shape_arr,
            ndim,
            ctypes.byref(status),
        )
        check_status(status.value, "from_numpy")
        return cls(ptr)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._lib.ndt_tensor_f64_ndim(self._ptr)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the tensor."""
        nd = self.ndim
        if nd == 0:
            return ()
        shape_arr = (ctypes.c_size_t * nd)()
        status = self._lib.ndt_tensor_f64_shape(self._ptr, shape_arr)
        check_status(status, "shape")
        return tuple(int(x) for x in shape_arr)

    def __len__(self) -> int:
        """Total number of elements."""
        return self._lib.ndt_tensor_f64_len(self._ptr)

    def to_numpy(self) -> NDArray[np.float64]:
        """Convert to NumPy array (copies data).

        Returns:
            NumPy array with copied data in column-major order
        """
        data_ptr = self._lib.ndt_tensor_f64_data(self._ptr)
        # Create numpy array from pointer (zero-copy view)
        arr = np.ctypeslib.as_array(data_ptr, shape=(len(self),))
        # Copy and reshape to proper shape (column-major)
        return arr.copy().reshape(self.shape, order="F")

    def __getitem__(self, idx: int) -> float:
        """Get element by linear index (0-based).

        Args:
            idx: Linear index into the flattened tensor

        Returns:
            Value at the given index
        """
        out = ctypes.c_double()
        status = self._lib.ndt_tensor_f64_get_linear(self._ptr, idx, ctypes.byref(out))
        if status == StatusCode.INDEX_OUT_OF_BOUNDS:
            raise IndexError(f"index {idx} out of bounds for tensor of length {len(self)}")
        check_status(status, "getitem")
        return out.value

    def __setitem__(self, idx: int, value: float) -> None:
        """Set element by linear index (0-based).

        Args:
            idx: Linear index into the flattened tensor
            value: Value to set
        """
        status = self._lib.ndt_tensor_f64_set_linear(self._ptr, idx, float(value))
        if status == StatusCode.INDEX_OUT_OF_BOUNDS:
            raise IndexError(f"index {idx} out of bounds for tensor of length {len(self)}")
        check_status(status, "setitem")

    def fill(self, value: float) -> TensorF64:
        """Fill tensor with a value (in-place).

        Args:
            value: Value to fill the tensor with

        Returns:
            self (for method chaining)
        """
        status = self._lib.ndt_tensor_f64_fill(self._ptr, float(value))
        check_status(status, "fill")
        return self

    def copy(self) -> TensorF64:
        """Create a copy of the tensor.

        Returns:
            New tensor with copied data
        """
        ptr = self._lib.ndt_tensor_f64_clone(self._ptr)
        if not ptr:
            raise NDTensorsError("Failed to clone tensor")
        return TensorF64(ptr)

    def permutedims(self, perm: Sequence[int]) -> TensorF64:
        """Permute dimensions (returns new tensor).

        Args:
            perm: Permutation of dimensions. perm[i] specifies which
                dimension of the input becomes dimension i of the output.

        Returns:
            New tensor with permuted dimensions
        """
        ndim = len(perm)
        perm_arr = (ctypes.c_size_t * ndim)(*perm)
        status = ctypes.c_int(-999)
        ptr = self._lib.ndt_tensor_f64_permutedims(
            self._ptr, perm_arr, ndim, ctypes.byref(status)
        )
        if status.value == StatusCode.INVALID_PERMUTATION:
            raise ValueError(f"Invalid permutation: {perm}")
        check_status(status.value, "permutedims")
        return TensorF64(ptr)

    def __repr__(self) -> str:
        """String representation."""
        return f"TensorF64(shape={self.shape})"

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"TensorF64(shape={self.shape}):\n{self.to_numpy()}"
