"""Tests for PyTorch integration."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ndtensors_rs.torch_ops import torch_contract


class TestTorchContractForward:
    """Tests for torch_contract forward pass."""

    def test_matrix_multiply(self):
        """Test matrix multiplication via contraction."""
        a = torch.ones((2, 3))
        b = torch.ones((3, 4))
        c = torch_contract(a, (1, -1), b, (-1, 2))

        assert c.shape == (2, 4)
        assert c[0, 0].item() == 3.0  # sum of 3 ones

    def test_matrix_multiply_values(self):
        """Test matrix multiplication with specific values."""
        a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        c = torch_contract(a, (1, -1), b, (-1, 2))

        # Result is float64 (from Rust computation)
        expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]], dtype=torch.float64)
        torch.testing.assert_close(c, expected)

    def test_inner_product(self):
        """Test inner product (dot product) of vectors."""
        v1 = torch.tensor([1.0, 2.0, 3.0])
        v2 = torch.tensor([4.0, 5.0, 6.0])
        result = torch_contract(v1, (-1,), v2, (-1,))

        # 1*4 + 2*5 + 3*6 = 32
        assert result.shape == (1,)
        assert result[0].item() == 32.0

    def test_outer_product(self):
        """Test outer product of vectors."""
        v1 = torch.tensor([1.0, 2.0])
        v2 = torch.tensor([3.0, 4.0, 5.0])
        result = torch_contract(v1, (1,), v2, (2,))

        assert result.shape == (2, 3)
        # Result is float64 (from Rust computation)
        expected = torch.outer(torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0])).to(torch.float64)
        torch.testing.assert_close(result, expected)


class TestTorchContractBackward:
    """Tests for torch_contract backward pass (autograd)."""

    def test_backward_shapes(self):
        """Test that gradients have correct shapes."""
        a = torch.ones((2, 3), requires_grad=True)
        b = torch.ones((3, 4), requires_grad=True)
        c = torch_contract(a, (1, -1), b, (-1, 2))

        loss = c.sum()
        loss.backward()

        assert a.grad is not None
        assert b.grad is not None
        assert a.grad.shape == (2, 3)
        assert b.grad.shape == (3, 4)

    def test_backward_values(self):
        """Test gradient values for matrix multiplication."""
        a = torch.ones((2, 3), requires_grad=True)
        b = torch.ones((3, 4), requires_grad=True)
        c = torch_contract(a, (1, -1), b, (-1, 2))

        loss = c.sum()
        loss.backward()

        # grad_a[i,j] = sum_k B[j,k] = 4
        assert a.grad[0, 0].item() == 4.0
        # grad_b[j,k] = sum_i A[i,j] = 2
        assert b.grad[0, 0].item() == 2.0

    def test_backward_chain_rule(self):
        """Test that gradients satisfy chain rule."""
        np.random.seed(42)
        a_np = np.random.rand(2, 3).astype(np.float64)
        b_np = np.random.rand(3, 4).astype(np.float64)
        grad_c_np = np.random.rand(2, 4).astype(np.float64)

        a = torch.tensor(a_np, requires_grad=True)
        b = torch.tensor(b_np, requires_grad=True)
        grad_c = torch.tensor(grad_c_np)

        c = torch_contract(a, (1, -1), b, (-1, 2))
        loss = (c * grad_c).sum()
        loss.backward()

        # Expected: grad_a = grad_c @ B^T
        expected_grad_a = np.dot(grad_c_np, b_np.T)
        np.testing.assert_array_almost_equal(a.grad.numpy(), expected_grad_a, decimal=10)

        # Expected: grad_b = A^T @ grad_c
        expected_grad_b = np.dot(a_np.T, grad_c_np)
        np.testing.assert_array_almost_equal(b.grad.numpy(), expected_grad_b, decimal=10)

    def test_numerical_gradient(self):
        """Verify gradients against numerical approximation."""
        from torch.autograd import gradcheck

        a = torch.randn((2, 3), dtype=torch.float64, requires_grad=True)
        b = torch.randn((3, 4), dtype=torch.float64, requires_grad=True)

        def f(a, b):
            return torch_contract(a, (1, -1), b, (-1, 2))

        assert gradcheck(f, (a, b), eps=1e-6, atol=1e-4)


class TestTorchContractGradFunction:
    """Tests using torch.autograd.grad."""

    def test_autograd_grad(self):
        """Test using torch.autograd.grad directly."""
        a = torch.ones((2, 3), requires_grad=True)
        b = torch.ones((3, 4), requires_grad=True)
        c = torch_contract(a, (1, -1), b, (-1, 2))

        loss = c.sum()
        grads = torch.autograd.grad(loss, [a, b])

        assert len(grads) == 2
        assert grads[0].shape == (2, 3)
        assert grads[1].shape == (3, 4)


class TestTorchContractEdgeCases:
    """Tests for edge cases."""

    def test_1d_contraction(self):
        """Test contracting two 1D tensors."""
        v1 = torch.ones((5,), requires_grad=True)
        v2 = torch.ones((5,), requires_grad=True)
        result = torch_contract(v1, (-1,), v2, (-1,))

        assert result[0].item() == 5.0

        # Test backward
        result.sum().backward()
        assert v1.grad is not None
        assert v2.grad is not None

    def test_3d_contraction(self):
        """Test 3D tensor contraction."""
        a = torch.ones((2, 3, 4), requires_grad=True)
        b = torch.ones((4, 5), requires_grad=True)
        c = torch_contract(a, (1, 2, -1), b, (-1, 3))

        assert c.shape == (2, 3, 5)
        assert c[0, 0, 0].item() == 4.0  # sum of 4 ones

        # Test backward
        c.sum().backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_multiple_contractions(self):
        """Test contraction with multiple contracted indices."""
        a = torch.ones((2, 3, 4), requires_grad=True)
        b = torch.ones((3, 4, 5), requires_grad=True)
        c = torch_contract(a, (1, -1, -2), b, (-1, -2, 2))

        assert c.shape == (2, 5)
        assert c[0, 0].item() == 12.0  # sum over 3*4 = 12 ones

        # Test backward
        c.sum().backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_no_grad_context(self):
        """Test that forward pass works in no_grad context."""
        a = torch.ones((2, 3))
        b = torch.ones((3, 4))

        with torch.no_grad():
            c = torch_contract(a, (1, -1), b, (-1, 2))

        assert c.shape == (2, 4)
        assert not c.requires_grad

    def test_detached_tensors(self):
        """Test with detached tensors."""
        a = torch.ones((2, 3), requires_grad=True).detach()
        b = torch.ones((3, 4), requires_grad=True).detach()
        c = torch_contract(a, (1, -1), b, (-1, 2))

        assert c.shape == (2, 4)
