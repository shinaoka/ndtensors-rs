"""JAX tensor contraction example with automatic differentiation.

This example demonstrates using ndtensors-rs with JAX for tensor contractions
that support automatic differentiation (jax.grad, jax.vjp) and JIT compilation.

Requirements:
    pip install ndtensors-rs[jax]
"""

import jax

# Enable 64-bit floating point (required for ndtensors-rs which uses f64 internally)
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from ndtensors_rs.jax_ops import jax_contract


def example_matrix_multiply():
    """Matrix multiplication using tensor contraction.

    C[i,k] = A[i,j] * B[j,k]  (sum over j)

    Labels:
    - A has labels (1, -1): dim 0 -> output label 1, dim 1 -> contract label -1
    - B has labels (-1, 2): dim 0 -> contract label -1, dim 1 -> output label 2
    """
    print("=== Matrix Multiplication ===")
    a = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # shape (2, 3)
    b = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # shape (3, 2)

    c = jax_contract(a, (1, -1), b, (-1, 2))

    print(f"A shape: {a.shape}")
    print(f"B shape: {b.shape}")
    print(f"C = A @ B shape: {c.shape}")
    print(f"C:\n{c}")

    # Verify against numpy
    expected = jnp.dot(a, b)
    np.testing.assert_allclose(c, expected)
    print("Verified against jnp.dot!\n")


def example_inner_product():
    """Inner product (dot product) of two vectors.

    result = v1[i] * v2[i]  (sum over i)

    Both vectors have contract label -1.
    """
    print("=== Inner Product ===")
    v1 = jnp.array([1.0, 2.0, 3.0])
    v2 = jnp.array([4.0, 5.0, 6.0])

    result = jax_contract(v1, (-1,), v2, (-1,))

    print(f"v1: {v1}")
    print(f"v2: {v2}")
    print(f"v1 · v2 = {result[0]}")
    print(f"Expected: {jnp.dot(v1, v2)}\n")


def example_outer_product():
    """Outer product of two vectors.

    C[i,j] = v1[i] * v2[j]

    Both vectors have positive (output) labels, no contraction.
    """
    print("=== Outer Product ===")
    v1 = jnp.array([1.0, 2.0])
    v2 = jnp.array([3.0, 4.0, 5.0])

    result = jax_contract(v1, (1,), v2, (2,))

    print(f"v1: {v1}")
    print(f"v2: {v2}")
    print(f"v1 ⊗ v2 shape: {result.shape}")
    print(f"v1 ⊗ v2:\n{result}\n")


def example_3d_contraction():
    """Contract a 3D tensor with a 2D tensor.

    C[i,j,l] = A[i,j,k] * B[k,l]  (sum over k)
    """
    print("=== 3D Tensor Contraction ===")
    a = jnp.ones((2, 3, 4))
    b = jnp.ones((4, 5))

    c = jax_contract(a, (1, 2, -1), b, (-1, 3))

    print(f"A shape: {a.shape}")
    print(f"B shape: {b.shape}")
    print(f"C shape: {c.shape}")
    print(f"Each element of C sums over 4 ones: {c[0, 0, 0]}\n")


def example_gradient():
    """Compute gradients through tensor contraction.

    Demonstrates that jax_contract supports automatic differentiation.
    """
    print("=== Gradient Computation ===")

    def loss_fn(a, b):
        c = jax_contract(a, (1, -1), b, (-1, 2))
        return jnp.sum(c)

    a = jnp.ones((2, 3))
    b = jnp.ones((3, 4))

    # Compute gradients
    grad_a, grad_b = jax.grad(loss_fn, argnums=(0, 1))(a, b)

    print(f"A shape: {a.shape}")
    print(f"B shape: {b.shape}")
    print(f"grad_A shape: {grad_a.shape}")
    print(f"grad_B shape: {grad_b.shape}")
    print(f"grad_A[0,0] = sum over B's columns = {grad_a[0, 0]}")
    print(f"grad_B[0,0] = sum over A's rows = {grad_b[0, 0]}\n")


def example_jit():
    """JIT compilation with jax_contract.

    Note: Use static_argnums for label arguments when JIT compiling directly.
    """
    print("=== JIT Compilation ===")

    # Method 1: JIT the contract function directly
    jit_contract = jax.jit(jax_contract, static_argnums=(1, 3))

    a = jnp.ones((2, 3))
    b = jnp.ones((3, 4))
    c = jit_contract(a, (1, -1), b, (-1, 2))
    print(f"JIT result shape: {c.shape}")

    # Method 2: JIT a function that uses jax_contract
    @jax.jit
    def matmul(a, b):
        return jax_contract(a, (1, -1), b, (-1, 2))

    c2 = matmul(a, b)
    print(f"JIT wrapped result shape: {c2.shape}")

    # Method 3: JIT gradient computation
    @jax.jit
    def compute_grads(a, b):
        def loss_fn(a, b):
            c = jax_contract(a, (1, -1), b, (-1, 2))
            return jnp.sum(c)
        return jax.grad(loss_fn, argnums=(0, 1))(a, b)

    grad_a, grad_b = compute_grads(a, b)
    print(f"JIT gradient shapes: {grad_a.shape}, {grad_b.shape}\n")


def example_multiple_contractions():
    """Contract over multiple indices simultaneously.

    C[i,l] = A[i,j,k] * B[j,k,l]  (sum over j and k)
    """
    print("=== Multiple Index Contraction ===")
    a = jnp.ones((2, 3, 4))
    b = jnp.ones((3, 4, 5))

    c = jax_contract(a, (1, -1, -2), b, (-1, -2, 2))

    print(f"A shape: {a.shape}")
    print(f"B shape: {b.shape}")
    print(f"C shape (after contracting 2 indices): {c.shape}")
    print(f"Each element sums over 3*4=12 ones: {c[0, 0]}\n")


if __name__ == "__main__":
    example_matrix_multiply()
    example_inner_product()
    example_outer_product()
    example_3d_contraction()
    example_gradient()
    example_jit()
    example_multiple_contractions()
