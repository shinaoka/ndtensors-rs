"""JAX integration for ndtensors-rs tensor operations."""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import custom_vjp

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

from .ops import contract, contract_vjp
from .tensor import TensorF64


def _check_jax_available() -> None:
    """Check if JAX is available."""
    if not JAX_AVAILABLE:
        raise ImportError(
            "JAX integration requires jax. "
            "Install with: pip install ndtensors-rs[jax]"
        )


if JAX_AVAILABLE:

    def _contract_impl(
        a_np: np.ndarray,
        b_np: np.ndarray,
        labels_a: Tuple[int, ...],
        labels_b: Tuple[int, ...],
    ) -> np.ndarray:
        """Pure Python implementation of contract for use with pure_callback."""
        a_tensor = TensorF64.from_numpy(a_np.astype(np.float64))
        b_tensor = TensorF64.from_numpy(b_np.astype(np.float64))
        result_tensor = contract(a_tensor, labels_a, b_tensor, labels_b)
        return result_tensor.to_numpy()

    def _contract_vjp_impl(
        a_np: np.ndarray,
        b_np: np.ndarray,
        grad_output_np: np.ndarray,
        labels_a: Tuple[int, ...],
        labels_b: Tuple[int, ...],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Pure Python implementation of contract_vjp for use with pure_callback."""
        a_tensor = TensorF64.from_numpy(a_np.astype(np.float64))
        b_tensor = TensorF64.from_numpy(b_np.astype(np.float64))
        grad_output_tensor = TensorF64.from_numpy(grad_output_np.astype(np.float64))
        grad_a_tensor, grad_b_tensor = contract_vjp(
            a_tensor, labels_a, b_tensor, labels_b, grad_output_tensor
        )
        return grad_a_tensor.to_numpy(), grad_b_tensor.to_numpy()

    def _compute_output_shape(
        a_shape: Tuple[int, ...],
        b_shape: Tuple[int, ...],
        labels_a: Tuple[int, ...],
        labels_b: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        """Compute the output shape of a contraction."""
        # Collect positive labels and their dimensions
        positive_dims = {}
        for i, label in enumerate(labels_a):
            if label > 0:
                positive_dims[label] = a_shape[i]
        for i, label in enumerate(labels_b):
            if label > 0:
                positive_dims[label] = b_shape[i]

        # Sort by label and return dimensions
        if not positive_dims:
            return (1,)  # Scalar result (as 1D tensor)
        sorted_labels = sorted(positive_dims.keys())
        return tuple(positive_dims[label] for label in sorted_labels)

    def _jax_contract_impl(
        a: jnp.ndarray,
        labels_a: Tuple[int, ...],
        b: jnp.ndarray,
        labels_b: Tuple[int, ...],
    ) -> jnp.ndarray:
        """Internal implementation of jax_contract."""
        # Compute output shape for pure_callback
        output_shape = _compute_output_shape(a.shape, b.shape, labels_a, labels_b)

        # Use the dtype of input a (use float32 if JAX x64 is disabled)
        output_dtype = a.dtype

        def _callback_fn(args):
            result = _contract_impl(args[0], args[1], labels_a, labels_b)
            # Convert to the expected output dtype
            return result.astype(np.dtype(output_dtype))

        # Use pure_callback for JIT compatibility
        result = jax.pure_callback(
            _callback_fn,
            jax.ShapeDtypeStruct(output_shape, output_dtype),
            (a, b),
        )
        return result

    def _jax_contract_fwd(
        a: jnp.ndarray,
        labels_a: Tuple[int, ...],
        b: jnp.ndarray,
        labels_b: Tuple[int, ...],
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Forward pass for jax_contract, returns output and residuals.

        Note: fwd function has SAME signature as the original function.
        """
        result = _jax_contract_impl(a, labels_a, b, labels_b)
        # Save inputs for backward pass (don't need to save labels, they're static)
        return result, (a, b)

    def _jax_contract_bwd(
        labels_a: Tuple[int, ...],
        labels_b: Tuple[int, ...],
        residuals: Tuple[jnp.ndarray, jnp.ndarray],
        grad_output: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Backward pass using contract_vjp.

        Note: nondiff args come first, then residuals, then grad_output.
        """
        a, b = residuals

        # Use the dtype of input a
        output_dtype_a = a.dtype
        output_dtype_b = b.dtype

        def _bwd_callback(args):
            a_np, b_np, grad_np = args
            grad_a, grad_b = _contract_vjp_impl(a_np, b_np, grad_np, labels_a, labels_b)
            # Convert to the expected output dtypes
            return grad_a.astype(np.dtype(output_dtype_a)), grad_b.astype(np.dtype(output_dtype_b))

        # Use pure_callback for JIT compatibility
        grad_a, grad_b = jax.pure_callback(
            _bwd_callback,
            (
                jax.ShapeDtypeStruct(a.shape, output_dtype_a),
                jax.ShapeDtypeStruct(b.shape, output_dtype_b),
            ),
            (a, b, grad_output),
        )

        # Return gradients for (a, b) only - labels are nondiff
        return (grad_a, grad_b)

    # Create the custom_vjp wrapper with nondiff_argnums
    jax_contract = custom_vjp(_jax_contract_impl, nondiff_argnums=(1, 3))
    jax_contract.defvjp(_jax_contract_fwd, _jax_contract_bwd)

    # Add docstring to jax_contract
    jax_contract.__doc__ = """JAX-compatible tensor contraction.

    This function can be used with JAX's automatic differentiation
    (jax.grad, jax.vjp) and JIT compilation (jax.jit).

    Labels determine how dimensions are contracted:
    - Negative values: contracted indices (matched between tensors)
    - Positive values: uncontracted indices (appear in output)

    Args:
        a: First tensor as JAX array
        labels_a: Labels for each dimension of `a` (must be a tuple for JIT)
        b: Second tensor as JAX array
        labels_b: Labels for each dimension of `b` (must be a tuple for JIT)

    Returns:
        Contracted result as JAX array

    Note:
        When using with jax.jit, use static_argnums for labels:
        `jax.jit(jax_contract, static_argnums=(1, 3))`

    Examples:
        >>> import jax.numpy as jnp
        >>> from ndtensors_rs.jax_ops import jax_contract
        >>> a = jnp.ones((2, 3))
        >>> b = jnp.ones((3, 4))
        >>> c = jax_contract(a, (1, -1), b, (-1, 2))
        >>> c.shape
        (2, 4)
    """

else:
    # Placeholder when JAX is not available
    def jax_contract(*args, **kwargs):
        """JAX-compatible tensor contraction (JAX not installed)."""
        _check_jax_available()


__all__ = ["jax_contract", "JAX_AVAILABLE"]
