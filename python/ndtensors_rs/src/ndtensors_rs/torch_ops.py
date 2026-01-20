"""PyTorch integration for ndtensors-rs tensor operations."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch.autograd import Function

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .ops import contract, contract_vjp
from .tensor import TensorF64


def _check_torch_available() -> None:
    """Check if PyTorch is available."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch integration requires torch. "
            "Install with: pip install ndtensors-rs[torch]"
        )


if TORCH_AVAILABLE:

    class _TorchContractFunction(Function):
        """PyTorch autograd Function for tensor contraction."""

        @staticmethod
        def forward(
            ctx: Any,
            a: torch.Tensor,
            b: torch.Tensor,
            labels_a: Tuple[int, ...],
            labels_b: Tuple[int, ...],
        ) -> torch.Tensor:
            """Forward pass for tensor contraction.

            Args:
                ctx: Context object for saving tensors
                a: First tensor
                b: Second tensor
                labels_a: Labels for dimensions of a
                labels_b: Labels for dimensions of b

            Returns:
                Contracted result tensor
            """
            # Save for backward
            ctx.save_for_backward(a, b)
            ctx.labels_a = labels_a
            ctx.labels_b = labels_b

            # Convert to numpy (move to CPU if on GPU)
            a_np = a.detach().cpu().numpy().astype(np.float64)
            b_np = b.detach().cpu().numpy().astype(np.float64)

            # Convert to TensorF64 and perform contraction
            a_tensor = TensorF64.from_numpy(a_np)
            b_tensor = TensorF64.from_numpy(b_np)
            result_tensor = contract(a_tensor, labels_a, b_tensor, labels_b)

            # Convert back to PyTorch tensor
            result_np = result_tensor.to_numpy()
            result = torch.from_numpy(result_np.copy())

            # Move to same device as input
            if a.is_cuda:
                result = result.to(a.device)

            return result

        @staticmethod
        def backward(
            ctx: Any, grad_output: torch.Tensor
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None, None]:
            """Backward pass using contract_vjp.

            Args:
                ctx: Context with saved tensors
                grad_output: Gradient of loss w.r.t. output

            Returns:
                Tuple of gradients for (a, b, labels_a, labels_b)
            """
            a, b = ctx.saved_tensors
            labels_a = ctx.labels_a
            labels_b = ctx.labels_b

            # Convert to numpy
            a_np = a.detach().cpu().numpy().astype(np.float64)
            b_np = b.detach().cpu().numpy().astype(np.float64)
            grad_output_np = grad_output.detach().cpu().numpy().astype(np.float64)

            # Convert to TensorF64
            a_tensor = TensorF64.from_numpy(a_np)
            b_tensor = TensorF64.from_numpy(b_np)
            grad_output_tensor = TensorF64.from_numpy(grad_output_np)

            # Call Rust VJP
            grad_a_tensor, grad_b_tensor = contract_vjp(
                a_tensor, labels_a, b_tensor, labels_b, grad_output_tensor
            )

            # Convert back to PyTorch tensors
            grad_a = torch.from_numpy(grad_a_tensor.to_numpy().copy())
            grad_b = torch.from_numpy(grad_b_tensor.to_numpy().copy())

            # Move to same device as input
            if a.is_cuda:
                grad_a = grad_a.to(a.device)
                grad_b = grad_b.to(b.device)

            # Return gradients: (grad_a, grad_b, None for labels_a, None for labels_b)
            return grad_a, grad_b, None, None

    def torch_contract(
        a: torch.Tensor,
        labels_a: Tuple[int, ...],
        b: torch.Tensor,
        labels_b: Tuple[int, ...],
    ) -> torch.Tensor:
        """PyTorch-compatible tensor contraction with autograd support.

        This function can be used with PyTorch's automatic differentiation
        (loss.backward(), torch.autograd.grad).

        Labels determine how dimensions are contracted:
        - Negative values: contracted indices (matched between tensors)
        - Positive values: uncontracted indices (appear in output)

        Args:
            a: First tensor as PyTorch tensor
            labels_a: Labels for each dimension of `a`
            b: Second tensor as PyTorch tensor
            labels_b: Labels for each dimension of `b`

        Returns:
            Contracted result as PyTorch tensor

        Note:
            GPU tensors will be moved to CPU for computation and back to GPU.
            For best performance with GPU tensors, consider using native
            PyTorch operations.

        Examples:
            >>> import torch
            >>> from ndtensors_rs.torch_ops import torch_contract
            >>> a = torch.ones((2, 3), requires_grad=True)
            >>> b = torch.ones((3, 4), requires_grad=True)
            >>> c = torch_contract(a, (1, -1), b, (-1, 2))
            >>> c.shape
            torch.Size([2, 4])
            >>> loss = c.sum()
            >>> loss.backward()
            >>> a.grad.shape
            torch.Size([2, 3])
        """
        return _TorchContractFunction.apply(a, b, labels_a, labels_b)

else:
    # Placeholder when PyTorch is not available
    def torch_contract(*args, **kwargs):
        """PyTorch-compatible tensor contraction (PyTorch not installed)."""
        _check_torch_available()


__all__ = ["torch_contract", "TORCH_AVAILABLE"]
