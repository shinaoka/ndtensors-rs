# ndtensors-rs Python Bindings

Python bindings for the [ndtensors-rs](https://github.com/shinaoka/ndtensors-rs) tensor library.

## Installation

### Development Installation

```bash
cd python/ndtensors_rs
uv sync --extra dev
```

The Rust library is automatically built when running tests (via `conftest.py`).

### Prerequisites

- Python 3.12+
- Rust toolchain (for building the native library)
- uv (Python package manager)

## Usage

```python
from ndtensors_rs import TensorF64, contract, contract_vjp
import numpy as np

# Create tensors
a = TensorF64.zeros((2, 3))
b = TensorF64.ones((3, 4))
c = TensorF64.rand((2, 2))
d = TensorF64.randn((3, 3))

# Create from NumPy array
arr = np.array([[1.0, 2.0], [3.0, 4.0]])
t = TensorF64.from_numpy(arr)

# Convert back to NumPy
arr_out = t.to_numpy()

# Basic operations
t.fill(5.0)
t[0] = 1.0
val = t[0]

# Copy and permute
t2 = t.copy()
t3 = t.permutedims((1, 0))  # Transpose

# Tensor contraction (matrix multiplication)
# C[i,k] = A[i,j] * B[j,k]
a = TensorF64.ones((2, 3))
b = TensorF64.ones((3, 4))
c = contract(a, (1, -1), b, (-1, 2))
# c.shape == (2, 4)

# VJP for automatic differentiation
grad_c = TensorF64.ones((2, 4))
grad_a, grad_b = contract_vjp(a, (1, -1), b, (-1, 2), grad_c)
```

## Label-Based Contraction

The `contract` function uses labels to specify which dimensions to contract:

- **Negative labels**: Contracted (summed over) dimensions. Matching negative labels between tensors are contracted together.
- **Positive labels**: Output dimensions. The result tensor's dimensions are ordered by their positive labels.

### Examples

```python
# Matrix multiplication: C[i,k] = A[i,j] * B[j,k]
# A has labels (1, -1): dimension 0 -> output dim 1, dimension 1 -> contract
# B has labels (-1, 2): dimension 0 -> contract, dimension 1 -> output dim 2
c = contract(a, (1, -1), b, (-1, 2))

# Inner product: result = sum_i(v1[i] * v2[i])
# Note: Returns a 1D tensor with shape (1,), not a scalar
result = contract(v1, (-1,), v2, (-1,))

# Outer product: C[i,j] = v1[i] * v2[j]
c = contract(v1, (1,), v2, (2,))

# 3D tensor contraction: C[i,j,l] = A[i,j,k] * B[k,l]
c = contract(a, (1, 2, -1), b, (-1, 3))
```

## Automatic Differentiation Integration

### JAX Integration

```python
import jax
import jax.numpy as jnp
from ndtensors_rs.jax_ops import jax_contract

a = jnp.ones((2, 3))
b = jnp.ones((3, 4))

# Forward pass
c = jax_contract(a, (1, -1), b, (-1, 2))

# Backward pass with jax.grad
def loss_fn(a, b):
    return jax_contract(a, (1, -1), b, (-1, 2)).sum()

grad_a, grad_b = jax.grad(loss_fn, argnums=(0, 1))(a, b)

# JIT compilation (labels must be static)
jit_contract = jax.jit(jax_contract, static_argnums=(1, 3))
c = jit_contract(a, (1, -1), b, (-1, 2))
```

### PyTorch Integration

```python
import torch
from ndtensors_rs.torch_ops import torch_contract

a = torch.ones((2, 3), requires_grad=True)
b = torch.ones((3, 4), requires_grad=True)

# Forward pass
c = torch_contract(a, (1, -1), b, (-1, 2))

# Backward pass
loss = c.sum()
loss.backward()
print(a.grad, b.grad)
```

### Installation with AD Support

```bash
# JAX support
uv sync --extra jax

# PyTorch support
uv sync --extra torch

# Both JAX and PyTorch
uv sync --extra all
```

## Testing

```bash
cd python/ndtensors_rs
uv run pytest
```
