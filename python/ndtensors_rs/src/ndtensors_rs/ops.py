"""Tensor operations for ndtensors-rs."""

from __future__ import annotations

import ctypes
from typing import Sequence, Tuple

from ._lib import get_lib
from ._status import StatusCode, check_status
from .tensor import TensorF64


def contract(
    a: TensorF64,
    labels_a: Sequence[int],
    b: TensorF64,
    labels_b: Sequence[int],
) -> TensorF64:
    """Contract two tensors using label-based contraction.

    Labels determine how dimensions are contracted:
    - Negative values: contracted indices (matched between tensors)
    - Positive values: uncontracted indices (appear in output)

    The output dimensions are ordered by their positive labels in ascending order.

    Args:
        a: First tensor
        labels_a: Labels for each dimension of `a`
        b: Second tensor
        labels_b: Labels for each dimension of `b`

    Returns:
        Contracted result tensor

    Raises:
        ValueError: If contracted dimensions have mismatched sizes

    Examples:
        >>> # Matrix multiplication: C[i,k] = A[i,j] * B[j,k]
        >>> a = TensorF64.ones((2, 3))
        >>> b = TensorF64.ones((3, 4))
        >>> c = contract(a, (1, -1), b, (-1, 2))
        >>> c.shape
        (2, 4)

        >>> # Inner product: sum over all indices
        >>> v1 = TensorF64.from_numpy(np.array([1.0, 2.0, 3.0]))
        >>> v2 = TensorF64.from_numpy(np.array([4.0, 5.0, 6.0]))
        >>> result = contract(v1, (-1,), v2, (-1,))
        >>> result[0]  # 1*4 + 2*5 + 3*6 = 32
        32.0
    """
    lib = get_lib()
    ndim_a = len(labels_a)
    ndim_b = len(labels_b)
    labels_a_arr = (ctypes.c_long * ndim_a)(*labels_a)
    labels_b_arr = (ctypes.c_long * ndim_b)(*labels_b)
    status = ctypes.c_int(-999)

    ptr = lib.ndt_tensor_f64_contract(
        a._ptr,
        labels_a_arr,
        ndim_a,
        b._ptr,
        labels_b_arr,
        ndim_b,
        ctypes.byref(status),
    )

    if status.value == StatusCode.SHAPE_MISMATCH:
        raise ValueError("Contracted dimensions must have matching sizes")
    check_status(status.value, "contract")
    return TensorF64(ptr)


def contract_vjp(
    a: TensorF64,
    labels_a: Sequence[int],
    b: TensorF64,
    labels_b: Sequence[int],
    grad_output: TensorF64,
) -> Tuple[TensorF64, TensorF64]:
    """Compute VJP (Vector-Jacobian Product) for tensor contraction.

    Given the forward pass `c = contract(a, labels_a, b, labels_b)` and the
    gradient of the loss with respect to `c` (`grad_output`), this computes
    the gradients with respect to `a` and `b`.

    Args:
        a: First tensor from forward pass
        labels_a: Labels for each dimension of `a`
        b: Second tensor from forward pass
        labels_b: Labels for each dimension of `b`
        grad_output: Gradient of loss with respect to output

    Returns:
        Tuple of (grad_a, grad_b) where:
        - grad_a has the same shape as `a`
        - grad_b has the same shape as `b`

    Raises:
        ValueError: If shapes don't match

    Examples:
        >>> a = TensorF64.ones((2, 3))
        >>> b = TensorF64.ones((3, 4))
        >>> c = contract(a, (1, -1), b, (-1, 2))
        >>> grad_c = TensorF64.ones((2, 4))
        >>> grad_a, grad_b = contract_vjp(a, (1, -1), b, (-1, 2), grad_c)
        >>> grad_a.shape
        (2, 3)
        >>> grad_b.shape
        (3, 4)
    """
    lib = get_lib()
    ndim_a = len(labels_a)
    ndim_b = len(labels_b)
    labels_a_arr = (ctypes.c_long * ndim_a)(*labels_a)
    labels_b_arr = (ctypes.c_long * ndim_b)(*labels_b)
    status = ctypes.c_int(-999)
    grad_a_ptr = ctypes.c_void_p()
    grad_b_ptr = ctypes.c_void_p()

    lib.ndt_tensor_f64_contract_vjp(
        a._ptr,
        labels_a_arr,
        ndim_a,
        b._ptr,
        labels_b_arr,
        ndim_b,
        grad_output._ptr,
        ctypes.byref(grad_a_ptr),
        ctypes.byref(grad_b_ptr),
        ctypes.byref(status),
    )

    if status.value == StatusCode.SHAPE_MISMATCH:
        raise ValueError("Contracted dimensions must have matching sizes")
    check_status(status.value, "contract_vjp")
    return TensorF64(grad_a_ptr.value), TensorF64(grad_b_ptr.value)
