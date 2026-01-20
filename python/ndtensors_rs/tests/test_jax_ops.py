"""Tests for JAX integration."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from ndtensors_rs.jax_ops import jax_contract


class TestJaxContractForward:
    """Tests for jax_contract forward pass."""

    def test_matrix_multiply(self):
        """Test matrix multiplication via contraction."""
        a = jnp.ones((2, 3))
        b = jnp.ones((3, 4))
        c = jax_contract(a, (1, -1), b, (-1, 2))

        assert c.shape == (2, 4)
        assert c[0, 0] == 3.0  # sum of 3 ones

    def test_matrix_multiply_values(self):
        """Test matrix multiplication with specific values."""
        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]])
        c = jax_contract(a, (1, -1), b, (-1, 2))

        expected = jnp.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_almost_equal(c, expected)

    def test_inner_product(self):
        """Test inner product (dot product) of vectors."""
        v1 = jnp.array([1.0, 2.0, 3.0])
        v2 = jnp.array([4.0, 5.0, 6.0])
        result = jax_contract(v1, (-1,), v2, (-1,))

        # 1*4 + 2*5 + 3*6 = 32
        assert result.shape == (1,)
        assert result[0] == 32.0

    def test_outer_product(self):
        """Test outer product of vectors."""
        v1 = jnp.array([1.0, 2.0])
        v2 = jnp.array([3.0, 4.0, 5.0])
        result = jax_contract(v1, (1,), v2, (2,))

        assert result.shape == (2, 3)
        expected = jnp.outer(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0, 5.0]))
        np.testing.assert_array_almost_equal(result, expected)


class TestJaxContractGradient:
    """Tests for jax_contract gradient computation."""

    def test_gradient_shapes(self):
        """Test that gradients have correct shapes."""
        def loss_fn(a, b):
            c = jax_contract(a, (1, -1), b, (-1, 2))
            return jnp.sum(c)

        a = jnp.ones((2, 3))
        b = jnp.ones((3, 4))
        grad_fn = jax.grad(loss_fn, argnums=(0, 1))
        grad_a, grad_b = grad_fn(a, b)

        assert grad_a.shape == (2, 3)
        assert grad_b.shape == (3, 4)

    def test_gradient_values(self):
        """Test gradient values for matrix multiplication."""
        def loss_fn(a, b):
            c = jax_contract(a, (1, -1), b, (-1, 2))
            return jnp.sum(c)

        a = jnp.ones((2, 3))
        b = jnp.ones((3, 4))
        grad_fn = jax.grad(loss_fn, argnums=(0, 1))
        grad_a, grad_b = grad_fn(a, b)

        # grad_a[i,j] = sum_k B[j,k] = 4
        assert grad_a[0, 0] == 4.0
        # grad_b[j,k] = sum_i A[i,j] = 2
        assert grad_b[0, 0] == 2.0

    def test_gradient_chain_rule(self):
        """Test that gradients satisfy chain rule."""
        np.random.seed(42)
        a_np = np.random.rand(2, 3).astype(np.float32)
        b_np = np.random.rand(3, 4).astype(np.float32)
        grad_c_np = np.random.rand(2, 4).astype(np.float32)

        a = jnp.array(a_np)
        b = jnp.array(b_np)

        def loss_fn(a, b):
            c = jax_contract(a, (1, -1), b, (-1, 2))
            return jnp.sum(c * jnp.array(grad_c_np))

        grad_fn = jax.grad(loss_fn, argnums=(0, 1))
        grad_a, grad_b = grad_fn(a, b)

        # Expected: grad_a = grad_c @ B^T
        expected_grad_a = np.dot(grad_c_np, b_np.T)
        # Use decimal=5 for float32 precision
        np.testing.assert_array_almost_equal(grad_a, expected_grad_a, decimal=5)

        # Expected: grad_b = A^T @ grad_c
        expected_grad_b = np.dot(a_np.T, grad_c_np)
        np.testing.assert_array_almost_equal(grad_b, expected_grad_b, decimal=5)

    def test_numerical_gradient(self):
        """Verify gradients against numerical approximation."""
        from jax.test_util import check_grads

        def f(a, b):
            return jax_contract(a, (1, -1), b, (-1, 2))

        a = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        b = jnp.array([[5.0, 6.0], [7.0, 8.0]])

        # Check first-order gradients in reverse mode
        check_grads(f, (a, b), order=1, modes=["rev"])


class TestJaxContractJit:
    """Tests for JIT compilation of jax_contract."""

    def test_jit_basic(self):
        """Test basic JIT compilation."""
        jit_contract = jax.jit(jax_contract, static_argnums=(1, 3))

        a = jnp.ones((2, 3))
        b = jnp.ones((3, 4))
        c = jit_contract(a, (1, -1), b, (-1, 2))

        assert c.shape == (2, 4)
        assert c[0, 0] == 3.0

    def test_jit_with_gradient(self):
        """Test JIT compilation with gradient computation."""
        @jax.jit
        def loss_fn(a, b):
            c = jax_contract(a, (1, -1), b, (-1, 2))
            return jnp.sum(c)

        grad_fn = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))

        a = jnp.ones((2, 3))
        b = jnp.ones((3, 4))
        grad_a, grad_b = grad_fn(a, b)

        assert grad_a.shape == (2, 3)
        assert grad_b.shape == (3, 4)


class TestJaxContractEdgeCases:
    """Tests for edge cases."""

    def test_1d_contraction(self):
        """Test contracting two 1D tensors."""
        v1 = jnp.ones((5,))
        v2 = jnp.ones((5,))
        result = jax_contract(v1, (-1,), v2, (-1,))
        assert result[0] == 5.0

    def test_3d_contraction(self):
        """Test 3D tensor contraction."""
        a = jnp.ones((2, 3, 4))
        b = jnp.ones((4, 5))
        c = jax_contract(a, (1, 2, -1), b, (-1, 3))

        assert c.shape == (2, 3, 5)
        assert c[0, 0, 0] == 4.0  # sum of 4 ones

    def test_multiple_contractions(self):
        """Test contraction with multiple contracted indices."""
        a = jnp.ones((2, 3, 4))
        b = jnp.ones((3, 4, 5))
        c = jax_contract(a, (1, -1, -2), b, (-1, -2, 2))

        assert c.shape == (2, 5)
        assert c[0, 0] == 12.0  # sum over 3*4 = 12 ones
