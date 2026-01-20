"""PyTorch tensor contraction example with automatic differentiation.

This example demonstrates using ndtensors-rs with PyTorch for tensor contractions
that support automatic differentiation (loss.backward(), torch.autograd.grad).

Requirements:
    pip install ndtensors-rs[torch]
"""

import numpy as np
import torch

from ndtensors_rs.torch_ops import torch_contract


def example_matrix_multiply():
    """Matrix multiplication using tensor contraction.

    C[i,k] = A[i,j] * B[j,k]  (sum over j)

    Labels:
    - A has labels (1, -1): dim 0 -> output label 1, dim 1 -> contract label -1
    - B has labels (-1, 2): dim 0 -> contract label -1, dim 1 -> output label 2
    """
    print("=== Matrix Multiplication ===")
    a = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # shape (2, 3)
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # shape (3, 2)

    c = torch_contract(a, (1, -1), b, (-1, 2))

    print(f"A shape: {a.shape}")
    print(f"B shape: {b.shape}")
    print(f"C = A @ B shape: {c.shape}")
    print(f"C:\n{c}")

    # Verify against torch
    expected = torch.mm(a, b)
    torch.testing.assert_close(c, expected.to(torch.float64))
    print("Verified against torch.mm!\n")


def example_inner_product():
    """Inner product (dot product) of two vectors.

    result = v1[i] * v2[i]  (sum over i)

    Both vectors have contract label -1.
    """
    print("=== Inner Product ===")
    v1 = torch.tensor([1.0, 2.0, 3.0])
    v2 = torch.tensor([4.0, 5.0, 6.0])

    result = torch_contract(v1, (-1,), v2, (-1,))

    print(f"v1: {v1}")
    print(f"v2: {v2}")
    print(f"v1 · v2 = {result[0].item()}")
    print(f"Expected: {torch.dot(v1, v2).item()}\n")


def example_outer_product():
    """Outer product of two vectors.

    C[i,j] = v1[i] * v2[j]

    Both vectors have positive (output) labels, no contraction.
    """
    print("=== Outer Product ===")
    v1 = torch.tensor([1.0, 2.0])
    v2 = torch.tensor([3.0, 4.0, 5.0])

    result = torch_contract(v1, (1,), v2, (2,))

    print(f"v1: {v1}")
    print(f"v2: {v2}")
    print(f"v1 ⊗ v2 shape: {result.shape}")
    print(f"v1 ⊗ v2:\n{result}\n")


def example_3d_contraction():
    """Contract a 3D tensor with a 2D tensor.

    C[i,j,l] = A[i,j,k] * B[k,l]  (sum over k)
    """
    print("=== 3D Tensor Contraction ===")
    a = torch.ones((2, 3, 4))
    b = torch.ones((4, 5))

    c = torch_contract(a, (1, 2, -1), b, (-1, 3))

    print(f"A shape: {a.shape}")
    print(f"B shape: {b.shape}")
    print(f"C shape: {c.shape}")
    print(f"Each element of C sums over 4 ones: {c[0, 0, 0].item()}\n")


def example_gradient_backward():
    """Compute gradients through tensor contraction using .backward().

    Demonstrates that torch_contract supports automatic differentiation.
    """
    print("=== Gradient via backward() ===")

    a = torch.ones((2, 3), requires_grad=True)
    b = torch.ones((3, 4), requires_grad=True)

    c = torch_contract(a, (1, -1), b, (-1, 2))
    loss = c.sum()
    loss.backward()

    print(f"A shape: {a.shape}")
    print(f"B shape: {b.shape}")
    print(f"grad_A shape: {a.grad.shape}")
    print(f"grad_B shape: {b.grad.shape}")
    print(f"grad_A[0,0] = sum over B's columns = {a.grad[0, 0].item()}")
    print(f"grad_B[0,0] = sum over A's rows = {b.grad[0, 0].item()}\n")


def example_gradient_autograd():
    """Compute gradients using torch.autograd.grad."""
    print("=== Gradient via autograd.grad ===")

    a = torch.ones((2, 3), requires_grad=True)
    b = torch.ones((3, 4), requires_grad=True)

    c = torch_contract(a, (1, -1), b, (-1, 2))
    loss = c.sum()

    # Use autograd.grad instead of backward
    grad_a, grad_b = torch.autograd.grad(loss, [a, b])

    print(f"grad_A shape: {grad_a.shape}")
    print(f"grad_B shape: {grad_b.shape}")
    print(f"grad_A:\n{grad_a}")
    print(f"grad_B:\n{grad_b}\n")


def example_no_grad():
    """Forward pass without gradient tracking."""
    print("=== No Gradient Mode ===")

    a = torch.ones((2, 3))
    b = torch.ones((3, 4))

    with torch.no_grad():
        c = torch_contract(a, (1, -1), b, (-1, 2))

    print(f"Result shape: {c.shape}")
    print(f"requires_grad: {c.requires_grad}\n")


def example_multiple_contractions():
    """Contract over multiple indices simultaneously.

    C[i,l] = A[i,j,k] * B[j,k,l]  (sum over j and k)
    """
    print("=== Multiple Index Contraction ===")
    a = torch.ones((2, 3, 4), requires_grad=True)
    b = torch.ones((3, 4, 5), requires_grad=True)

    c = torch_contract(a, (1, -1, -2), b, (-1, -2, 2))

    print(f"A shape: {a.shape}")
    print(f"B shape: {b.shape}")
    print(f"C shape (after contracting 2 indices): {c.shape}")
    print(f"Each element sums over 3*4=12 ones: {c[0, 0].item()}")

    # Verify backward works
    c.sum().backward()
    print(f"Backward pass completed, grad_A shape: {a.grad.shape}\n")


def example_chain_rule():
    """Verify gradients satisfy chain rule for matrix multiplication."""
    print("=== Chain Rule Verification ===")

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

    # Expected gradients from chain rule
    # grad_a = grad_c @ B^T
    expected_grad_a = np.dot(grad_c_np, b_np.T)
    # grad_b = A^T @ grad_c
    expected_grad_b = np.dot(a_np.T, grad_c_np)

    np.testing.assert_allclose(a.grad.numpy(), expected_grad_a, rtol=1e-10)
    np.testing.assert_allclose(b.grad.numpy(), expected_grad_b, rtol=1e-10)
    print("Chain rule verified!")
    print(f"grad_A matches grad_C @ B^T: True")
    print(f"grad_B matches A^T @ grad_C: True\n")


if __name__ == "__main__":
    example_matrix_multiply()
    example_inner_product()
    example_outer_product()
    example_3d_contraction()
    example_gradient_backward()
    example_gradient_autograd()
    example_no_grad()
    example_multiple_contractions()
    example_chain_rule()
