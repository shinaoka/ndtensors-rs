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

### 1. Zero Runtime Compilation → AI-Accelerated Development

- **Near-zero startup time**: Native code eliminates JIT compilation overhead
- **Enables practical vibe coding**: AI tools (Claude Code, etc.) can iterate rapidly
  - Edit code → Run tests immediately → Get feedback
  - Current Julia: precompile wait is a major bottleneck for AI-assisted development

### 2. 100% Compatibility via C-Pointer Injection

- Inject BLAS/LAPACK function pointers from Julia at runtime
- Detect CUDA/Metal at runtime and select appropriate backend dynamically
- **Existing Julia code requires no changes**

### 3. Single C API for All Languages

- Julia, Python, C++, and any other language via one C API
- Each language only needs a thin wrapper
- One core implementation, minimal maintenance cost
- **Full ITensors.jl functionality in Python** becomes achievable quickly

### 4. C API Maintenance is Not Hard

- **Rust's memory safety**: No segfaults, no memory leaks—Rust guarantees safety at compile time
- **AI agents can manage C API**: Repetitive wrapper code is easy for AI to generate and maintain
- Far simpler than traditional C/C++ FFI development

### 5. Monorepo with Rust Workspace

Current ITensors.jl ecosystem challenges:
- TensorAlgebra.jl refactoring in progress
- Multi-package version synchronization is tedious (especially with official registry)

**Rust workspace solves this**:
- Manage related crates + Julia bindings in one repository
- `cargo publish` registers all crates at once
- CI runs fast across the entire workspace

### 6. No "Two-Language Problem" with Rust

Julia aimed to solve the "Python + C++" two-language problem. Does introducing Rust bring it back?

**No—Rust avoids the two-language problem**:
- **Excellent package system**: `cargo` rivals Julia's Pkg.jl in usability
- **JLL package integration**: Distribute Rust binaries via BinaryBuilder.jl
  - Users just `Pkg.add()`—no compilation required
  - Cross-platform binaries distributed automatically
- **User experience unchanged**: Julia users don't notice the Rust backend

### 7. Foundation for Rust Tensor Network Ecosystem

ndtensors-rs is not a standalone project—it becomes the **shared foundation for Rust-based tensor network libraries**:

- **tensor4all-rs**: TCI (Tensor Cross Interpolation), Quantics TT, etc. in Rust
- Future Rust tensor network libraries share a common core (ndtensors-rs)
- Avoid duplicate implementations, improve ecosystem quality

### 8. Maintainability

- Rust's type system catches bugs at compile time
- Explicit ownership/borrowing prevents memory bugs
- Refactoring is safer with compiler guarantees

## Implementation Status

### Storage Types

| NDTensors.jl | ndtensors-rs | Status |
|--------------|--------------|--------|
| `Dense{ElT, DataT}` | `Dense<ElT, D>` | ✅ Implemented |
| `BlockSparse{ElT, ...}` | `BlockSparse<ElT, D>` | ✅ Implemented |
| `Diag{ElT, DataT}` | `Diag<ElT, D>` | ✅ Implemented |
| `DiagBlockSparse{ElT, ...}` | `DiagBlockSparse<ElT, D>` | ✅ Implemented |
| `Combiner` | `CombinerStorage` | ✅ Implemented |
| `EmptyStorage{ElT}` | `EmptyStorage<ElT>` | ✅ Implemented |

### Tensor Types

| NDTensors.jl | ndtensors-rs | Status |
|--------------|--------------|--------|
| `Tensor{ElT,N,StoreT,IndsT}` | `Tensor<ElT, StoreT>` | ✅ Implemented |
| `DenseTensor` (alias) | `DenseTensor<ElT>` (alias) | ✅ Implemented |
| `BlockSparseTensor` | `BlockSparseTensor<ElT, D>` | ✅ Implemented |
| `DiagBlockSparseTensor` | `DiagBlockSparseTensor<ElT, D>` | ✅ Implemented |
| `CombinerTensor` | `CombinerTensor` | ✅ Implemented |
| `DiagTensor` (alias) | - | ❌ Not yet (issue #44) |
| `EmptyTensor` | - | ❌ Not yet (issue #45) |

### Operations (DenseTensor)

| Operation | Status | Notes |
|-----------|--------|-------|
| `zeros`, `ones`, `fill` | ✅ | Basic construction |
| `from_vec` | ✅ | Create from data |
| `get`, `set` | ✅ | Element access |
| `reshape` | ✅ | View with new shape |
| `permutedims` | ✅ | Dimension permutation |
| `contract` | ✅ | Tensor contraction (GEMM-based) |
| `norm`, `norm_sqr` | ✅ | Frobenius norm |
| `conj` | ✅ | Complex conjugate |
| `scale` | ✅ | Scalar multiplication |
| `outer` | ✅ | Outer product |
| `diag` | ✅ | Extract/create diagonal |
| `slice` | ✅ | Tensor slicing |
| `copy_into` | ✅ | Copy data between tensors |
| `apply`, `apply_binary` | ✅ | Element-wise operations |
| `real`, `imag` | ✅ | Extract real/imaginary parts |

### Linear Algebra (DenseTensor)

| Operation | Status | Notes |
|-----------|--------|-------|
| `svd` | ✅ | Full and truncated SVD |
| `qr` | ✅ | QR decomposition |
| `ql` | ✅ | QL decomposition |
| `polar` | ✅ | Polar decomposition |
| `matrix_exp` | ✅ | Matrix exponential |
| `qr_positive`, `ql_positive` | ✅ | Positive R/L diagonal |
| `random_orthog` | ✅ | Random orthogonal matrix |
| `random_unitary` | ✅ | Random unitary matrix |
| `eigen` | ❌ | Not yet (issue #50) |

### Operations (BlockSparseTensor)

| Operation | Status | Notes |
|-----------|--------|-------|
| `zeros`, construction | ✅ | With block structure |
| `get`, `set` | ✅ | Element access |
| `blockview` | ✅ | Access individual blocks |
| `isblocknz`, `nnzblocks` | ✅ | Block queries |
| `permutedims` | ✅ | Block-aware permutation |
| `contract` | ✅ | Block-wise contraction |
| `scale`, `add` | ✅ | Elementwise operations |
| `norm` | ✅ | Frobenius norm |
| `to_dense` | ✅ | Convert to dense |

### C API

| Feature | Status | Notes |
|---------|--------|-------|
| f64 DenseTensor | ✅ | Basic operations |
| Complex64 | ❌ | Not yet (issue #48) |
| BlockSparseTensor | ❌ | Not yet (issue #48) |
| Decompositions (SVD, QR) | ❌ | Not yet (issue #48) |
| AD functions (VJP/JVP) | ❌ | See issue #52 |

### Automatic Differentiation (issue #52)

| Phase | Feature | Status | Notes |
|-------|---------|--------|-------|
| 1 | VJP primitives (`contract_vjp`, `svd_vjp`, etc.) | 🔶 | `contract_vjp` done, others pending |
| 2 | JVP primitives (forward-mode) | 🔶 | `contract_jvp` done, others pending |
| 3a | Native Rust backward-mode AD | ✅ | `TrackedTensor` + tape (issue #54) |
| 3b | Native Rust forward-mode AD | ✅ | `DualTensor` + JVP (issue #55) |
| 4 | Hessian-vector products | ❌ | Requires combining forward + backward |

Host language integration:
- **Julia**: ChainRules.jl rrule/frule
- **Python**: JAX custom_vjp/jvp, PyTorch autograd.Function

**Native Rust AD** enables automatic differentiation without relying on host language AD systems.
This is useful for pure Rust applications and provides the foundation for Hessian-vector products (Phase 4).

## Usage Examples

### Basic Tensor Operations

```rust
use ndtensors::{DenseTensor, Tensor};

// Create tensors
let a: DenseTensor<f64> = Tensor::zeros(&[2, 3]);
let b: DenseTensor<f64> = Tensor::ones(&[3, 4]);
let c: DenseTensor<f64> = Tensor::from_vec(
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    &[2, 3]
).unwrap();

// Element access
assert_eq!(c.get(&[0, 0]), Some(&1.0));
assert_eq!(c.get(&[1, 0]), Some(&2.0));  // Column-major order
```

### Tensor Contraction

```rust
use ndtensors::{DenseTensor, Tensor, contract};

// Create matrices
let a: DenseTensor<f64> = Tensor::from_vec(
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    &[2, 3]
).unwrap();
let b: DenseTensor<f64> = Tensor::ones(&[3, 4]);

// Contract: C[i,k] = A[i,j] * B[j,k]
// Negative labels indicate contracted indices
let c = contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();
assert_eq!(c.shape(), &[2, 4]);

// Matrix-vector multiplication: y[i] = A[i,j] * x[j]
let x: DenseTensor<f64> = Tensor::ones(&[3]);
let y = contract(&a, &[1, -1], &x, &[-1]).unwrap();
assert_eq!(y.shape(), &[2]);
```

### Dimension Permutation

```rust
use ndtensors::{DenseTensor, Tensor};

let a: DenseTensor<f64> = Tensor::from_vec(
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    &[2, 3]
).unwrap();

// Transpose: swap dimensions 0 and 1
let a_t = a.permutedims(&[1, 0]).unwrap();
assert_eq!(a_t.shape(), &[3, 2]);

// 3D tensor permutation
let t: DenseTensor<f64> = Tensor::zeros(&[2, 3, 4]);
let t_perm = t.permutedims(&[2, 0, 1]).unwrap();
assert_eq!(t_perm.shape(), &[4, 2, 3]);
```

### Linear Algebra Decompositions

```rust
use ndtensors::{DenseTensor, Tensor};
use ndtensors::decomposition::{svd, qr, polar};

// Create a matrix
let a: DenseTensor<f64> = Tensor::from_vec(
    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    &[2, 3]
).unwrap();

// SVD: A = U * S * Vt
let (u, s, vt) = svd(&a, &[0]).unwrap();
// u: [2, 2], s: [2] (diagonal), vt: [2, 3]

// QR decomposition: A = Q * R
let (q, r) = qr(&a, &[0]).unwrap();
// q: [2, 2] orthogonal, r: [2, 3] upper triangular

// Polar decomposition: A = U * P (unitary * positive)
let (u_polar, p) = polar(&a, &[0]).unwrap();
```

### Complex Tensors

```rust
use ndtensors::{DenseTensor, Tensor};
use ndtensors::operations::{conj, real, imag};
use faer::c64;

// Create complex tensor
let z: DenseTensor<c64> = Tensor::from_vec(
    vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)],
    &[2]
).unwrap();

// Complex conjugate
let z_conj = conj(&z);

// Extract real and imaginary parts
let re = real(&z);  // DenseTensor<f64>
let im = imag(&z);  // DenseTensor<f64>
```

### BlockSparse Tensors

```rust
use ndtensors::blocksparse_tensor::BlockSparseTensor;
use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims};

// Create block structure: 5x9 tensor with blocks [2,3] x [4,5]
let blockdims = BlockDims::new(vec![
    BlockDim::new(vec![2, 3]),  // dim 0: block sizes 2, 3
    BlockDim::new(vec![4, 5]),  // dim 1: block sizes 4, 5
]);

// Only blocks (0,0) and (1,1) are non-zero
let blocks = vec![
    Block::new(&[0, 0]),  // 2x4 block
    Block::new(&[1, 1]),  // 3x5 block
];

let tensor: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);

assert_eq!(tensor.shape(), &[5, 9]);
assert_eq!(tensor.nnzblocks(), 2);
assert!(tensor.isblocknz(&Block::new(&[0, 0])));
assert!(!tensor.isblocknz(&Block::new(&[0, 1])));  // Zero block
```

### Julia (via C API)

```julia
using NDTensorsRS

# Create tensors
a = TensorF64(2, 3)
fill!(a, 1.0)
b = TensorF64(3, 4)
fill!(b, 1.0)

# Tensor contraction
c = contract(a, (1, -1), b, (-1, 2))

# Access data (zero-copy)
data = unsafe_wrap(Array, unsafe_data(c), length(c))
```

## Key Challenges

- Zero-copy data sharing between Julia and Rust
- Automatic differentiation integration (ChainRules.jl)
- Performance parity with pure Julia implementation

## Future: CUDA.jl Integration

To support GPU tensors via CUDA.jl, the C API needs to accept CUDA device pointers directly from Julia.

### CUDA.jl Pointer Access

CUDA.jl provides the following methods to obtain device pointers from `CuArray`:

```julia
# Get CuPtr{T} (CUDA device pointer) from CuArray
ptr = pointer(cuarray)                           # CuPtr{T}
ptr = Base.unsafe_convert(CuPtr{T}, cuarray)     # CuPtr{T}

# Wrap existing CuPtr as CuArray (zero-copy)
cuarray = unsafe_wrap(CuArray, ptr, dims; own=false)
```

### Required C API Extensions

For GPU support, ndtensors-rs needs new C API functions that:

1. **Accept CUDA device pointers**: Create tensors from `CuPtr` without copying data to CPU
2. **Return CUDA device pointers**: Allow Julia to wrap results as `CuArray`
3. **Specify memory location**: Distinguish between CPU (`Ptr`) and GPU (`CuPtr`) memory

```rust
// Proposed C API for GPU tensors
#[no_mangle]
pub extern "C" fn ndt_tensor_f64_from_cuda_ptr(
    cuda_ptr: *mut c_double,  // CuPtr from Julia
    len: size_t,
    shape: *const size_t,
    ndim: size_t,
    status: *mut StatusCode,
) -> *mut ndt_tensor_f64_cuda;

#[no_mangle]
pub extern "C" fn ndt_tensor_f64_cuda_ptr(
    tensor: *const ndt_tensor_f64_cuda,
) -> *mut c_double;  // Returns CuPtr for Julia to wrap
```

### Julia Integration Pattern

```julia
using CUDA, NDTensorsRS

# Create CuArray in Julia
cu_data = CUDA.rand(Float64, 2, 3)

# Pass device pointer to Rust (zero-copy)
ptr = pointer(cu_data)  # CuPtr{Float64}
rust_tensor = NDTensorsRS.from_cuda_ptr(ptr, size(cu_data))

# Operations run on GPU via Rust
result = contract(rust_tensor, ...)

# Get result back as CuArray (zero-copy)
result_ptr = NDTensorsRS.cuda_ptr(result)
result_cuarray = unsafe_wrap(CuArray, result_ptr, size(result))
```

### Implementation Notes

- Rust side needs CUDA runtime bindings (e.g., `cudarc` crate) to work with device memory
- Memory ownership must be clearly defined (Julia owns CuArray, Rust borrows the pointer)
- Synchronization between CUDA streams may be required before/after Rust operations
- Consider using `cuBLAS` for GPU GEMM operations instead of faer (CPU-only)

## Design Document

See [docs/design.md](docs/design.md) for technical details.

## Citation

This is a port of **NDTensors.jl** from **ITensors.jl**. Please cite:

> M. Fishman, S. R. White, E. M. Stoudenmire, "The ITensor Software Library for Tensor Network Calculations", SciPost Phys. Codebases 4 (2022), arXiv:2007.14822

## License

Apache License 2.0 (see [LICENSE-APACHE](LICENSE-APACHE))
