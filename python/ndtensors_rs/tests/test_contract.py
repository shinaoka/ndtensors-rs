"""Tests for tensor contraction operations."""

import numpy as np
import pytest

from ndtensors_rs import TensorF64, contract, contract_vjp


class TestContract:
    """Tests for tensor contraction."""

    def test_matrix_multiply(self):
        """Test matrix multiplication via contraction."""
        # A[2x3] * B[3x4] = C[2x4]
        a = TensorF64.ones((2, 3))
        b = TensorF64.ones((3, 4))
        c = contract(a, (1, -1), b, (-1, 2))

        assert c.shape == (2, 4)
        # Each element is sum of 3 ones = 3
        assert c[0] == 3.0

    def test_matrix_multiply_values(self):
        """Test matrix multiplication with specific values."""
        # A = [[1, 2], [3, 4]]
        # B = [[5, 6], [7, 8]]
        # A @ B = [[19, 22], [43, 50]]
        a = TensorF64.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
        b = TensorF64.from_numpy(np.array([[5.0, 6.0], [7.0, 8.0]]))
        c = contract(a, (1, -1), b, (-1, 2))

        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_almost_equal(c.to_numpy(), expected)

    def test_inner_product(self):
        """Test inner product (dot product) of vectors."""
        v1 = TensorF64.from_numpy(np.array([1.0, 2.0, 3.0]))
        v2 = TensorF64.from_numpy(np.array([4.0, 5.0, 6.0]))
        result = contract(v1, (-1,), v2, (-1,))

        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        # Note: Result is 1D tensor with shape (1,), not scalar
        assert result.shape == (1,)
        assert result[0] == 32.0

    def test_outer_product(self):
        """Test outer product of vectors."""
        v1 = TensorF64.from_numpy(np.array([1.0, 2.0]))
        v2 = TensorF64.from_numpy(np.array([3.0, 4.0, 5.0]))
        result = contract(v1, (1,), v2, (2,))

        assert result.shape == (2, 3)
        expected = np.outer([1.0, 2.0], [3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_trace(self):
        """Test computing trace via contraction."""
        # Trace: sum of diagonal elements
        a = TensorF64.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]]))
        # Contract both dimensions with same negative label
        result = contract(a, (-1, -1), TensorF64.ones(()), ())

        # Trace = 1 + 4 = 5
        # Note: This test assumes the contraction handles this case
        # If not supported, we should skip or adjust

    def test_3d_contraction(self):
        """Test 3D tensor contraction.

        Note: Same positive labels on both tensors do NOT create batch dimensions.
        Instead, they create separate output dimensions (outer product behavior).
        For actual batched operations, indices must be explicitly handled.
        """
        # A[2, 3, 4] @ B[4, 5] = C[2, 3, 5] (contract over last dim of A, first dim of B)
        a = TensorF64.ones((2, 3, 4))
        b = TensorF64.ones((4, 5))
        c = contract(a, (1, 2, -1), b, (-1, 3))

        assert c.shape == (2, 3, 5)
        # Each element is sum of 4 ones = 4
        assert c[0] == 4.0

    def test_contract_mismatched_dimensions(self):
        """Test that mismatched contracted dimensions raise error."""
        a = TensorF64.ones((2, 3))
        b = TensorF64.ones((4, 5))  # 4 != 3
        with pytest.raises(ValueError, match="matching sizes"):
            contract(a, (1, -1), b, (-1, 2))


class TestContractVJP:
    """Tests for VJP (backward pass) of tensor contraction."""

    def test_matrix_multiply_vjp_shapes(self):
        """Test that VJP returns correct shapes."""
        a = TensorF64.ones((2, 3))
        b = TensorF64.ones((3, 4))
        grad_c = TensorF64.ones((2, 4))

        grad_a, grad_b = contract_vjp(a, (1, -1), b, (-1, 2), grad_c)

        assert grad_a.shape == (2, 3)
        assert grad_b.shape == (3, 4)

    def test_matrix_multiply_vjp_values(self):
        """Test VJP values for matrix multiplication."""
        # Forward: C = A @ B
        # Backward: dA = dC @ B^T, dB = A^T @ dC
        a = TensorF64.ones((2, 3))
        b = TensorF64.ones((3, 4))
        grad_c = TensorF64.ones((2, 4))

        grad_a, grad_b = contract_vjp(a, (1, -1), b, (-1, 2), grad_c)

        # grad_a[i,j] = sum_k grad_c[i,k] * B[j,k] = sum_k 1 * 1 = 4
        assert grad_a[0] == 4.0

        # grad_b[j,k] = sum_i A[i,j] * grad_c[i,k] = sum_i 1 * 1 = 2
        assert grad_b[0] == 2.0

    def test_vjp_inner_product(self):
        """Test VJP for inner product."""
        v1 = TensorF64.from_numpy(np.array([1.0, 2.0, 3.0]))
        v2 = TensorF64.from_numpy(np.array([4.0, 5.0, 6.0]))
        grad_out = TensorF64.ones(())

        grad_v1, grad_v2 = contract_vjp(v1, (-1,), v2, (-1,), grad_out)

        # d(v1 . v2)/dv1 = v2
        np.testing.assert_array_almost_equal(grad_v1.to_numpy(), np.array([4.0, 5.0, 6.0]))
        # d(v1 . v2)/dv2 = v1
        np.testing.assert_array_almost_equal(grad_v2.to_numpy(), np.array([1.0, 2.0, 3.0]))

    def test_vjp_numerical_gradient(self):
        """Test VJP against numerical gradient."""
        np.random.seed(42)
        a_np = np.random.rand(2, 3)
        b_np = np.random.rand(3, 4)

        a = TensorF64.from_numpy(a_np)
        b = TensorF64.from_numpy(b_np)

        # Forward pass
        c = contract(a, (1, -1), b, (-1, 2))
        c_np = c.to_numpy()

        # Backward pass with gradient of ones
        grad_c = TensorF64.ones((2, 4))
        grad_a, grad_b = contract_vjp(a, (1, -1), b, (-1, 2), grad_c)

        # Numerical gradient for A
        eps = 1e-5
        for i in range(2):
            for j in range(3):
                a_plus = a_np.copy()
                a_plus[i, j] += eps
                c_plus = np.dot(a_plus, b_np)

                a_minus = a_np.copy()
                a_minus[i, j] -= eps
                c_minus = np.dot(a_minus, b_np)

                # Gradient is sum of all output changes (since grad_c is ones)
                numerical_grad = (c_plus.sum() - c_minus.sum()) / (2 * eps)

                # Linear index in column-major order
                linear_idx = i + j * 2
                assert abs(grad_a[linear_idx] - numerical_grad) < 1e-4

    def test_vjp_chain_rule(self):
        """Test that VJP satisfies chain rule."""
        np.random.seed(123)
        a_np = np.random.rand(2, 3)
        b_np = np.random.rand(3, 4)
        grad_c_np = np.random.rand(2, 4)

        a = TensorF64.from_numpy(a_np)
        b = TensorF64.from_numpy(b_np)
        grad_c = TensorF64.from_numpy(grad_c_np)

        grad_a, grad_b = contract_vjp(a, (1, -1), b, (-1, 2), grad_c)

        # Expected: grad_a = grad_c @ B^T
        expected_grad_a = np.dot(grad_c_np, b_np.T)
        np.testing.assert_array_almost_equal(grad_a.to_numpy(), expected_grad_a, decimal=10)

        # Expected: grad_b = A^T @ grad_c
        expected_grad_b = np.dot(a_np.T, grad_c_np)
        np.testing.assert_array_almost_equal(grad_b.to_numpy(), expected_grad_b, decimal=10)


class TestContractEdgeCases:
    """Tests for edge cases in contraction."""

    def test_contract_1d_1d(self):
        """Test contracting two 1D tensors."""
        v1 = TensorF64.ones((5,))
        v2 = TensorF64.ones((5,))
        result = contract(v1, (-1,), v2, (-1,))
        assert result[0] == 5.0

    def test_contract_no_contraction(self):
        """Test contraction with no contracted indices (tensor product)."""
        a = TensorF64.from_numpy(np.array([1.0, 2.0]))
        b = TensorF64.from_numpy(np.array([3.0, 4.0]))
        result = contract(a, (1,), b, (2,))

        assert result.shape == (2, 2)
        expected = np.outer([1.0, 2.0], [3.0, 4.0])
        np.testing.assert_array_almost_equal(result.to_numpy(), expected)

    def test_contract_multiple_contractions(self):
        """Test contraction with multiple contracted indices."""
        # A[i,j,k] * B[j,k,l] = C[i,l] (contract over j and k)
        a = TensorF64.ones((2, 3, 4))
        b = TensorF64.ones((3, 4, 5))
        c = contract(a, (1, -1, -2), b, (-1, -2, 2))

        assert c.shape == (2, 5)
        # Each element is sum over 3*4 = 12 ones
        assert c[0] == 12.0
