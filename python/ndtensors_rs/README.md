# ndtensors-rs Python Bindings

Python bindings for the [ndtensors-rs](https://github.com/...) tensor library.

## Installation

### Development Installation

```bash
# From the repository root
cd python/ndtensors_rs
pip install -e ".[dev]"
```

This will:
1. Build the Rust library (`cargo build --release -p ndtensors-capi`)
2. Copy the shared library to the package
3. Install the Python package in development mode

### Prerequisites

- Python 3.9+
- Rust toolchain (for building the native library)
- NumPy

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

# Inner product: scalar = sum_i(v1[i] * v2[i])
result = contract(v1, (-1,), v2, (-1,))

# Outer product: C[i,j] = v1[i] * v2[j]
c = contract(v1, (1,), v2, (2,))

# Batched matrix multiplication: C[b,i,k] = A[b,i,j] * B[b,j,k]
c = contract(a, (1, 2, -1), b, (1, -1, 3))
```

## Testing

```bash
cd python/ndtensors_rs
pytest
```

## License

MIT
