"""Tests for README.md examples.

This file verifies that all code examples in README.md actually work.
"""

import numpy as np
import pytest

from ndtensors_rs import TensorF64, contract, contract_vjp


class TestReadmeUsageExamples:
    """Tests for the Usage section examples."""

    def test_create_tensors(self):
        """Test tensor creation examples."""
        # Create tensors
        a = TensorF64.zeros((2, 3))
        b = TensorF64.ones((3, 4))
        c = TensorF64.rand((2, 2))
        d = TensorF64.randn((3, 3))

        assert a.shape == (2, 3)
        assert b.shape == (3, 4)
        assert c.shape == (2, 2)
        assert d.shape == (3, 3)

    def test_numpy_conversion(self):
        """Test NumPy conversion examples."""
        # Create from NumPy array
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = TensorF64.from_numpy(arr)

        # Convert back to NumPy
        arr_out = t.to_numpy()

        np.testing.assert_array_equal(arr_out, arr)

    def test_basic_operations(self):
        """Test basic operations examples."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = TensorF64.from_numpy(arr)

        # Basic operations
        t.fill(5.0)
        assert t[0] == 5.0

        t[0] = 1.0
        assert t[0] == 1.0

        val = t[0]
        assert val == 1.0

    def test_copy_and_permute(self):
        """Test copy and permute examples."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = TensorF64.from_numpy(arr)

        # Copy and permute
        t2 = t.copy()
        assert t2.shape == t.shape

        t3 = t.permutedims((1, 0))  # Transpose
        assert t3.shape == (2, 2)
        # Check transpose is correct
        np.testing.assert_array_equal(t3.to_numpy(), arr.T)

    def test_tensor_contraction(self):
        """Test tensor contraction (matrix multiplication) example."""
        # Tensor contraction (matrix multiplication)
        # C[i,k] = A[i,j] * B[j,k]
        a = TensorF64.ones((2, 3))
        b = TensorF64.ones((3, 4))
        c = contract(a, (1, -1), b, (-1, 2))

        assert c.shape == (2, 4)
        # Each element should be 3 (sum of 3 ones)
        assert c[0] == 3.0

    def test_vjp(self):
        """Test VJP for automatic differentiation example."""
        a = TensorF64.ones((2, 3))
        b = TensorF64.ones((3, 4))
        c = contract(a, (1, -1), b, (-1, 2))

        # VJP for automatic differentiation
        grad_c = TensorF64.ones((2, 4))
        grad_a, grad_b = contract_vjp(a, (1, -1), b, (-1, 2), grad_c)

        assert grad_a.shape == (2, 3)
        assert grad_b.shape == (3, 4)


class TestReadmeContractionExamples:
    """Tests for the Label-Based Contraction examples."""

    def test_matrix_multiplication(self):
        """Test matrix multiplication example."""
        # Matrix multiplication: C[i,k] = A[i,j] * B[j,k]
        a = TensorF64.ones((2, 3))
        b = TensorF64.ones((3, 4))
        c = contract(a, (1, -1), b, (-1, 2))

        assert c.shape == (2, 4)

    def test_inner_product(self):
        """Test inner product example.

        Note: The result is a 1D tensor with shape (1,), not a scalar.
        """
        v1 = TensorF64.from_numpy(np.array([1.0, 2.0, 3.0]))
        v2 = TensorF64.from_numpy(np.array([4.0, 5.0, 6.0]))

        # Inner product: result = sum_i(v1[i] * v2[i])
        result = contract(v1, (-1,), v2, (-1,))

        # Result is 1D tensor, not scalar
        assert result.shape == (1,)
        assert result[0] == 32.0  # 1*4 + 2*5 + 3*6 = 32

    def test_outer_product(self):
        """Test outer product example."""
        v1 = TensorF64.from_numpy(np.array([1.0, 2.0]))
        v2 = TensorF64.from_numpy(np.array([3.0, 4.0, 5.0]))

        # Outer product: C[i,j] = v1[i] * v2[j]
        c = contract(v1, (1,), v2, (2,))

        assert c.shape == (2, 3)
        expected = np.outer([1.0, 2.0], [3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(c.to_numpy(), expected)
