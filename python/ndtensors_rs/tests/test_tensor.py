"""Tests for TensorF64 class."""

import numpy as np
import pytest

from ndtensors_rs import TensorF64


class TestTensorCreation:
    """Tests for tensor creation methods."""

    def test_zeros(self):
        """Test creating a tensor filled with zeros."""
        t = TensorF64.zeros((2, 3))
        assert t.shape == (2, 3)
        assert len(t) == 6
        assert t.ndim == 2
        assert t[0] == 0.0
        assert t[5] == 0.0

    def test_zeros_scalar(self):
        """Test creating a scalar (0-dimensional) tensor."""
        t = TensorF64.zeros(())
        assert t.shape == ()
        assert len(t) == 1
        assert t.ndim == 0

    def test_ones(self):
        """Test creating a tensor filled with ones."""
        t = TensorF64.ones((2, 3))
        assert t.shape == (2, 3)
        assert all(t[i] == 1.0 for i in range(6))

    def test_rand_shape(self):
        """Test that rand creates tensor with correct shape."""
        t = TensorF64.rand((3, 4, 5))
        assert t.shape == (3, 4, 5)
        assert len(t) == 60

    def test_rand_range(self):
        """Test that rand values are in [0, 1)."""
        t = TensorF64.rand((100,))
        arr = t.to_numpy()
        assert np.all((arr >= 0) & (arr < 1))

    def test_randn_shape(self):
        """Test that randn creates tensor with correct shape."""
        t = TensorF64.randn((3, 4))
        assert t.shape == (3, 4)
        assert len(t) == 12

    def test_randn_distribution(self):
        """Test that randn values are roughly normal."""
        t = TensorF64.randn((1000,))
        arr = t.to_numpy()
        # Mean should be close to 0
        assert abs(arr.mean()) < 0.2
        # Std should be close to 1
        assert abs(arr.std() - 1.0) < 0.2

    def test_from_numpy_1d(self):
        """Test creating tensor from 1D numpy array."""
        arr = np.array([1.0, 2.0, 3.0])
        t = TensorF64.from_numpy(arr)
        assert t.shape == (3,)
        np.testing.assert_array_equal(t.to_numpy(), arr)

    def test_from_numpy_2d(self):
        """Test creating tensor from 2D numpy array."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = TensorF64.from_numpy(arr)
        assert t.shape == (2, 2)
        np.testing.assert_array_equal(t.to_numpy(), arr)

    def test_from_numpy_fortran_order(self):
        """Test creating tensor from Fortran-ordered array."""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], order="F")
        t = TensorF64.from_numpy(arr)
        assert t.shape == (2, 3)
        np.testing.assert_array_equal(t.to_numpy(), arr)

    def test_from_numpy_c_order(self):
        """Test creating tensor from C-ordered array (should be converted)."""
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], order="C")
        t = TensorF64.from_numpy(arr)
        assert t.shape == (2, 3)
        np.testing.assert_array_equal(t.to_numpy(), arr)


class TestTensorProperties:
    """Tests for tensor properties."""

    def test_shape(self):
        """Test shape property."""
        t = TensorF64.zeros((2, 3, 4))
        assert t.shape == (2, 3, 4)

    def test_ndim(self):
        """Test ndim property."""
        assert TensorF64.zeros((2,)).ndim == 1
        assert TensorF64.zeros((2, 3)).ndim == 2
        assert TensorF64.zeros((2, 3, 4)).ndim == 3

    def test_len(self):
        """Test len function."""
        assert len(TensorF64.zeros((2, 3))) == 6
        assert len(TensorF64.zeros((2, 3, 4))) == 24


class TestTensorIndexing:
    """Tests for tensor indexing operations."""

    def test_getitem(self):
        """Test getting element by linear index."""
        arr = np.array([[1.0, 3.0], [2.0, 4.0]], order="F")
        t = TensorF64.from_numpy(arr)
        # Column-major order: [1, 2, 3, 4]
        assert t[0] == 1.0
        assert t[1] == 2.0
        assert t[2] == 3.0
        assert t[3] == 4.0

    def test_setitem(self):
        """Test setting element by linear index."""
        t = TensorF64.zeros((2, 2))
        t[0] = 1.0
        t[1] = 2.0
        t[2] = 3.0
        t[3] = 4.0
        assert t[0] == 1.0
        assert t[1] == 2.0
        assert t[2] == 3.0
        assert t[3] == 4.0

    def test_getitem_out_of_bounds(self):
        """Test that out of bounds access raises IndexError."""
        t = TensorF64.zeros((2, 2))
        with pytest.raises(IndexError):
            _ = t[10]

    def test_setitem_out_of_bounds(self):
        """Test that out of bounds assignment raises IndexError."""
        t = TensorF64.zeros((2, 2))
        with pytest.raises(IndexError):
            t[10] = 1.0


class TestTensorOperations:
    """Tests for tensor operations."""

    def test_fill(self):
        """Test filling tensor with a value."""
        t = TensorF64.zeros((2, 3))
        t.fill(5.0)
        assert all(t[i] == 5.0 for i in range(6))

    def test_fill_returns_self(self):
        """Test that fill returns self for chaining."""
        t = TensorF64.zeros((2, 2))
        result = t.fill(3.0)
        assert result is t

    def test_copy(self):
        """Test copying a tensor."""
        t1 = TensorF64.ones((2, 2))
        t2 = t1.copy()
        # Modify copy
        t2.fill(0.0)
        # Original should be unchanged
        assert t1[0] == 1.0
        assert t2[0] == 0.0

    def test_permutedims_transpose(self):
        """Test permuting dimensions (transpose)."""
        # Create 2x3 tensor in column-major order
        arr = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
        t = TensorF64.from_numpy(arr)
        assert t.shape == (2, 3)

        # Transpose
        t2 = t.permutedims((1, 0))
        assert t2.shape == (3, 2)

        # Check values
        expected = arr.T
        np.testing.assert_array_equal(t2.to_numpy(), expected)

    def test_permutedims_3d(self):
        """Test permuting dimensions of 3D tensor."""
        t = TensorF64.zeros((2, 3, 4))
        t2 = t.permutedims((2, 0, 1))
        assert t2.shape == (4, 2, 3)

    def test_permutedims_invalid(self):
        """Test that invalid permutation raises error."""
        t = TensorF64.zeros((2, 3))
        with pytest.raises(ValueError):
            t.permutedims((0, 0))  # Duplicate indices


class TestTensorConversion:
    """Tests for tensor conversion."""

    def test_to_numpy(self):
        """Test converting tensor to numpy array."""
        t = TensorF64.ones((2, 3))
        arr = t.to_numpy()
        assert arr.shape == (2, 3)
        assert arr.dtype == np.float64
        np.testing.assert_array_equal(arr, np.ones((2, 3)))

    def test_to_numpy_copies(self):
        """Test that to_numpy creates a copy."""
        t = TensorF64.ones((2, 2))
        arr = t.to_numpy()
        arr[0, 0] = 999.0
        # Original tensor should be unchanged
        assert t[0] == 1.0

    def test_roundtrip(self):
        """Test numpy -> tensor -> numpy roundtrip."""
        original = np.random.rand(3, 4, 5)
        t = TensorF64.from_numpy(original)
        result = t.to_numpy()
        np.testing.assert_array_almost_equal(result, original)


class TestTensorRepr:
    """Tests for tensor string representation."""

    def test_repr(self):
        """Test repr output."""
        t = TensorF64.zeros((2, 3))
        assert repr(t) == "TensorF64(shape=(2, 3))"

    def test_str(self):
        """Test str output includes array."""
        t = TensorF64.ones((2, 2))
        s = str(t)
        assert "TensorF64" in s
        assert "(2, 2)" in s
