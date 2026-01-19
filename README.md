# ndtensors-rs

**Unofficial experimental Rust port of [NDTensors.jl](https://github.com/ITensor/ITensors.jl/tree/main/NDTensors)**

This is a technical feasibility study exploring whether NDTensors.jl can be ported to Rust while maintaining API compatibility, enabling ITensors.jl to use a Rust backend.

## Goal

```
Current:
  ITensors.jl → NDTensors.jl (Pure Julia)

Target:
  ITensors.jl → NDTensors.jl (Julia wrapper) → ndtensors-rs (Rust via C API)
```

## Why Rust?

- **Faster precompilation**: Eliminate JIT overhead, enable rapid trial-and-error (vibe coding) for ITensors.jl-dependent libraries
- **Maintainability**: Rust's type system catches bugs at compile time, making refactoring safer

## Key Challenges

- Zero-copy data sharing between Julia and Rust
- Automatic differentiation integration (ChainRules.jl)
- Performance parity with pure Julia implementation

## NDTensors.jl Correspondence

| NDTensors.jl | ndtensors-rs | Notes |
|--------------|--------------|-------|
| `TensorStorage{ElT}` | `TensorStorage<T>` trait | Storage abstraction |
| `Dense{ElT, DataT}` | `Dense<T>` | Dense storage type |
| `Tensor{ElT,N,StoreT,IndsT}` | `Tensor<T, S>` | Generic tensor |
| `DenseTensor` type alias | `DenseTensor<T>` type alias | `Tensor<T, Dense<T>>` |
| Storage-specific dispatch | `impl<T> Tensor<T, Dense<T>>` | Methods on specific storage |
| `permutedims` | `permutedims` | Dimension permutation |
| `contract` | `contract` | Tensor contraction |

## Usage

### Rust

```rust
use ndtensors::{Tensor, contract};

// Create tensors
let a = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
let b = Tensor::<f64>::ones(&[3, 4]);

// Permute dimensions (transpose)
let a_t = a.permutedims(&[1, 0]).unwrap();
assert_eq!(a_t.shape(), &[3, 2]);

// Tensor contraction: C[i,k] = A[i,j] * B[j,k]
// Negative labels indicate contracted indices
let c = contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();
assert_eq!(c.shape(), &[2, 4]);
```

### Julia (via C API)

```julia
using NDTensorsRS

# Create tensors
a = TensorF64(2, 3)
fill!(a, 1.0)
b = TensorF64(3, 4)
fill!(b, 1.0)

# Tensor contraction with AD support
c = contract(a, (1, -1), b, (-1, 2))

# Reverse-mode AD via ChainRules.jl
using Zygote
loss(a, b) = sum(Array(contract(a, (1, -1), b, (-1, 2))))
grad_a, grad_b = Zygote.gradient(loss, a, b)
```

## Design Document

See [docs/design.md](docs/design.md) for technical details.

## Citation

This is a port of **NDTensors.jl** from **ITensors.jl**. Please cite:

> M. Fishman, S. R. White, E. M. Stoudenmire, "The ITensor Software Library for Tensor Network Calculations", SciPost Phys. Codebases 4 (2022), arXiv:2007.14822

## License

Apache License 2.0 (see [LICENSE-APACHE](LICENSE-APACHE))
