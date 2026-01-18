# ndtensors-rs Design Document

## Overview

ndtensors-rs is an unofficial experimental Rust port of [NDTensors.jl](https://github.com/ITensor/ITensors.jl/tree/main/NDTensors), exploring whether it can be ported to Rust while maintaining API compatibility.

## Primary Goal

**Technical feasibility study**: Can ITensors.jl's backend (NDTensors.jl) be swapped with a Rust implementation?

```
Current:
  ITensors.jl → NDTensors.jl (Julia)

Target:
  ITensors.jl → NDTensors.jl (Julia wrapper) → ndtensors-rs (Rust via C API)
```

### Success Criteria

1. **API compatibility**: NDTensors.jl-compatible Julia interface that internally calls ndtensors-rs
2. **Zero-copy interop**: Minimize data copying between Julia and Rust
3. **Performance parity**: No significant regression compared to pure Julia
4. **AD integration**: ChainRules.jl compatibility for seamless Zygote/Enzyme support

### Why Rust?

**1. Faster precompilation → Vibe coding optimization**

ITensors.jl's long precompile/compile times slow down trial-and-error development cycles:

```
Current (Julia):
  Edit code → Wait for precompile (minutes) → Test → Repeat

Target (Rust backend):
  Edit code → Instant load (precompiled binary) → Test → Repeat
```

Libraries depending on ITensors.jl suffer compounding precompile times. A Rust backend eliminates JIT overhead for core tensor operations, enabling rapid iteration.

**2. Maintainability via Rust's type system**

- Compile-time error detection vs runtime failures
- Explicit ownership/borrowing prevents memory bugs
- Trait-based polymorphism is explicit and predictable
- Refactoring is safer with compiler guarantees

**3. Shared backend with tensor4all-rs**

ndtensors-rs can share core infrastructure with tensor4all-rs (TCI, Quantics TT), avoiding duplicate implementations.

**4. External BLAS/MPI injection**

Following the pattern in `sparse-ir-rs`, external libraries can be injected via C function pointers:

```rust
// Julia/Python can inject their BLAS implementation
pub type GemmFn = extern "C" fn(
    transa: c_char, transb: c_char,
    m: c_int, n: c_int, k: c_int,
    alpha: *const f64, a: *const f64, lda: c_int,
    b: *const f64, ldb: c_int,
    beta: *const f64, c: *mut f64, ldc: c_int,
);

pub fn set_gemm_fn(f: GemmFn);
```

Benefits:
- Use same BLAS as host language (OpenBLAS, MKL, etc.)
- Future MPI injection for distributed computing (ITensorParallel.jl compatibility)

> **Note**: NDTensors.jl itself has no MPI dependency. MPI support is via separate [ITensorParallel.jl](https://github.com/ITensor/ITensorParallel.jl). The injection pattern enables future distributed memory support without core library changes.

### Expected Benefits (if successful)

- **Faster startup**: No Julia JIT compilation for core tensor operations
- **Vibe coding friendly**: Rapid trial-and-error for ITensors.jl-dependent libraries
- **Multi-language**: Same backend usable from Python, C++, Rust
- **Ecosystem sharing**: One optimized implementation for all languages
- **Long-term maintainability**: Rust's strict type system catches bugs early
- **Shared infrastructure**: Common backend with tensor4all-rs

---

## NDTensors.jl Architecture Reference

This section details NDTensors.jl's internal architecture to guide ndtensors-rs implementation for a smooth migration path.

### Type Hierarchy

#### NDTensors.jl Structure

```
TensorStorage{ElT} <: AbstractVector{ElT}
├── Dense{ElT, DataT<:AbstractVector}    # Dense vector storage
├── BlockSparse{...}                      # Sparse block structure
├── Diag{ElT, DataT<:AbstractVector}     # Diagonal storage
└── EmptyStorage                          # Trivial storage

Tensor{ElT, N, StoreT<:TensorStorage, IndsT}
├── DenseTensor   = Tensor where StoreT<:Dense
├── BlockSparseTensor = Tensor where StoreT<:BlockSparse
└── DiagTensor    = Tensor where StoreT<:Diag
```

#### Key Design: Polymorphic Storage

```julia
# NDTensors.jl: Storage is a TYPE PARAMETER
struct Tensor{ElT, N, StoreT, IndsT} <: AbstractArray{ElT, N}
    storage::StoreT    # Polymorphic! Dense, BlockSparse, Diag, etc.
    inds::IndsT        # Index/dimension tuple
end
```

**Critical**: The same `Tensor` type wraps different storage types. Operations dispatch based on storage type.

#### Current ndtensors-rs (Needs Refactoring)

```rust
// PROBLEM: Storage type is hardcoded
pub struct Tensor<T: Scalar> {
    storage: Dense<T>,  // ← Fixed to Dense, not polymorphic
    shape: Vec<usize>,
    strides: Vec<usize>,
}
```

#### Target ndtensors-rs Structure

```rust
// Option A: Trait-based polymorphism
pub trait TensorStorage<T>: Clone {
    fn len(&self) -> usize;
    fn as_slice(&self) -> &[T];
    fn as_mut_slice(&mut self) -> &mut [T];
}

pub struct Dense<T> { data: Vec<T> }
pub struct Diag<T> { data: Vec<T> }
// Future: pub struct BlockSparse<T> { ... }

impl<T: Scalar> TensorStorage<T> for Dense<T> { ... }
impl<T: Scalar> TensorStorage<T> for Diag<T> { ... }

pub struct Tensor<T: Scalar, S: TensorStorage<T>> {
    storage: S,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

// Type aliases matching NDTensors.jl
pub type DenseTensor<T> = Tensor<T, Dense<T>>;
pub type DiagTensor<T> = Tensor<T, Diag<T>>;

// Option B: Enum-based (simpler but less extensible)
pub enum Storage<T> {
    Dense(Dense<T>),
    Diag(Diag<T>),
}

pub struct Tensor<T: Scalar> {
    storage: Storage<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}
```

### Dense Storage

#### NDTensors.jl

```julia
# src/dense/dense.jl
struct Dense{ElT, DataT<:AbstractVector} <: TensorStorage{ElT}
    data::DataT
end

# Key: Storage is always a flat vector, NOT multi-dimensional
data(S::Dense) = S.data
```

#### ndtensors-rs (Current - OK)

```rust
pub struct Dense<T: Scalar> {
    data: Vec<T>,  // Flat vector, column-major order
}
```

This matches NDTensors.jl. The shape comes from the Tensor wrapper, not storage.

### Memory Copy Behavior (AliasStyle)

NDTensors.jl uses `AliasStyle` to control whether data is copied during tensor construction:

```julia
# Two construction patterns:
tensor(storage, inds)   # AllowAlias - NO copy (zero-copy)
Tensor(storage, inds)   # NeverAlias - COPIES data

# AliasStyle types:
AllowAlias()   # Store reference directly (zero-copy)
NeverAlias()   # Copy storage before storing
```

#### Source References

**Dense constructor** (`dense/dense.jl:65-72`):
```julia
function Dense(data::AbstractVector)
  return Dense{eltype(data)}(data)  # NO copy
end

function Dense(data::DataT) where {DataT<:AbstractArray{<:Any,N}} where {N}
  return Dense(vec(data))  # vec() returns a view, NO copy
end
```

**Tensor constructor** (`tensor/tensor.jl:86-88`):
```julia
tensor(args...; kwargs...) = Tensor(AllowAlias(), args...; kwargs...)  # NO copy
Tensor(storage::TensorStorage, inds::Tuple) = Tensor(NeverAlias(), storage, inds)  # COPIES
```

#### Copy Behavior Summary

| Function | Copies Data? | Notes |
|----------|--------------|-------|
| `tensor(storage, inds)` | **NO** | Uses AllowAlias |
| `Tensor(storage, inds)` | **YES** | Uses NeverAlias (default) |
| `Dense(vec)` | **NO** | Direct storage |
| `Dense(matrix)` | **NO** | `vec()` returns view |

#### ndtensors-rs Implications

For C API interop with Julia:
1. **Julia → Rust**: Julia can pass pointer to its Array, Rust wraps without copy
2. **Rust → Julia**: Rust exposes pointer, Julia wraps with `unsafe_wrap`

The key is that `Dense` storage is just a flat vector wrapper - no implicit copying.

### Dispatch Hierarchy

#### permutedims Dispatch Chain

NDTensors.jl uses a 5-level dispatch hierarchy:

```
Level 1: High-level API
    permutedims(tensor::Tensor, perm)
        → allocate output
        → call permutedims!!(output, tensor, perm)

Level 2: Abstract Array Wrapper
    permutedims!!(B::AbstractArray, A::AbstractArray, perm)
        → call permutedims!(expose(B), expose(A), perm)

Level 3: Expose Layer (type stripping)
    expose(object) → Exposed{UnwrappedType, typeof(object)}
    # Enables dispatch on underlying array type

Level 4: Array-specific implementation
    permutedims!(Edest::Exposed{<:Array}, Esrc::Exposed{<:Array}, perm)
        → @strided a_dest .= permutedims(a_src, perm)
        # Uses Strided.jl for efficient permutation

Level 5: DenseTensor specialization
    permutedims!(R::DenseTensor, T::DenseTensor, perm)
        → array(R), array(T)  # Reshape storage to N-dim
        → call Level 4
```

#### contract Dispatch Chain

```
Level 1: High-level API
    contract(tensor1, labels1, tensor2, labels2)
        → compute output labels
        → allocate output_tensor
        → call contract!!(output, ...)

Level 2: Label/Index computation
    ContractionProperties struct:
        - ai, bi, ci: index labels
        - AtoB, AtoC, BtoC: index mappings
        - permuteA, permuteB, permuteC: flags
        - dleft, dmid, dright: GEMM dimensions
        - PA, PB, PC: permutation sequences

Level 3: DenseTensor specialization
    contract!(R::DenseTensor, labelsR, T1, labelsT1, T2, labelsT2, α, β)
        → scalar case: _contract_scalar!()
        → TBLIS case: contract!(Val(:TBLIS), ...)
        → outer product: outer!() + permute
        → general: compute_contraction_properties!() → _contract!()

Level 4: Array-level GEMM contraction
    _contract!(CT, AT, BT, props, α, β)
        → Reshape A to (dleft, dmid)
        → Reshape B to (dmid, dright)
        → Reshape C to (dleft, dright)
        → GEMM: mul!!(CM, AM, BM, α, β)
        → Permute result if needed

Level 5: Backend selection
    mul!!(C, A, B, α, β) → _gemm!(backend, ...)
        → GemmBackend{:BLAS}: BLAS.gemm!()
        → GemmBackend{:Generic}: LinearAlgebra.mul!()
        → GemmBackend{:Octavian}: Octavian.matmul!()
```

### ContractionProperties

The core data structure for contraction optimization:

```julia
# NDTensors.jl: src/tensoroperations/contraction_logic.jl
mutable struct ContractionProperties{NA, NB, NC}
    # Index labels for A, B, C
    ai::NTuple{NA, Int}
    bi::NTuple{NB, Int}
    ci::NTuple{NC, Int}

    # Index mappings
    AtoB::NTuple{NA, Int}  # Which B index each A contracts with
    AtoC::NTuple{NA, Int}  # Which C index each A maps to
    BtoC::NTuple{NB, Int}  # Which C index each B maps to

    # Permutation decisions
    permuteA::Bool
    permuteB::Bool
    permuteC::Bool

    # GEMM dimensions
    dleft::Int   # Uncontracted dims from A
    dmid::Int    # Contracted dims
    dright::Int  # Uncontracted dims from B

    # Permutation sequences
    PA::NTuple{NA, Int}
    PB::NTuple{NB, Int}
    PC::NTuple{NC, Int}

    ncont::Int  # Number of contracted indices
end
```

#### ndtensors-rs Equivalent

```rust
pub struct ContractionProperties {
    // Index mappings
    a_to_c: Vec<Option<usize>>,  // None = contracted
    b_to_c: Vec<Option<usize>>,
    contracted_pairs: Vec<(usize, usize)>,  // (a_idx, b_idx)

    // Permutation decisions
    permute_a: bool,
    permute_b: bool,
    permute_c: bool,

    // GEMM dimensions
    dleft: usize,   // product of uncontracted A dims
    dmid: usize,    // product of contracted dims
    dright: usize,  // product of uncontracted B dims

    // Permutations (if needed)
    perm_a: Option<Vec<usize>>,
    perm_b: Option<Vec<usize>>,
    perm_c: Option<Vec<usize>>,
}
```

### Backend Abstraction

#### NDTensors.jl Pattern

```julia
# Runtime backend state
const _using_tblis = Ref(false)
const gemm_backend = Ref(:Auto)  # :Auto, :BLAS, :Generic, :Octavian

enable_tblis() = (_using_tblis[] = true)
disable_tblis() = (_using_tblis[] = false)

# Dispatch via Val type
contract!(Val(:TBLIS), R, ...)  # TBLIS-specific
_gemm!(GemmBackend{:BLAS}, ...)  # BLAS.gemm!
_gemm!(GemmBackend{:Generic}, ...)  # LinearAlgebra.mul!

# Weak dependencies (optional backends)
[weakdeps]
TBLIS = "..."
Octavian = "..."

[extensions]
NDTensorsTBLISExt = "TBLIS"
NDTensorsOctavianExt = "Octavian"
```

#### ndtensors-rs Pattern

```rust
// Backend trait
pub trait ContractionBackend {
    fn contract<T: Scalar>(
        c: &mut Tensor<T>, a: &Tensor<T>, b: &Tensor<T>,
        props: &ContractionProperties, alpha: T, beta: T,
    );
}

// Implementations
pub struct GenericBackend;  // Naive loop-based
pub struct GemmBackend;     // Reshape to GEMM (uses faer)
pub struct TblisBackend;    // TBLIS (future, cargo feature)

impl ContractionBackend for GenericBackend { ... }
impl ContractionBackend for GemmBackend { ... }

// Runtime selection
thread_local! {
    static CONTRACTION_BACKEND: RefCell<Box<dyn ContractionBackend>> =
        RefCell::new(Box::new(GemmBackend));
}

pub fn set_contraction_backend(backend: impl ContractionBackend + 'static) {
    CONTRACTION_BACKEND.with(|b| *b.borrow_mut() = Box::new(backend));
}

// Cargo features for optional backends
// Cargo.toml:
// [features]
// tblis = ["dep:tblis-sys"]
```

### File Organization

#### NDTensors.jl

```
NDTensors/src/
├── lib/
│   ├── BackendSelection/     # Algorithm/Backend types
│   └── Expose/               # Type stripping wrapper
├── abstractarray/
│   ├── permutedims.jl       # Generic permutedims wrapper
│   └── tensoralgebra/
│       └── contract.jl      # Array-level _contract!
├── tensorstorage/
│   └── tensorstorage.jl     # TensorStorage abstract type
├── tensor/
│   └── tensor.jl            # Tensor{ElT,N,StoreT,IndsT}
├── dense/
│   ├── dense.jl             # Dense storage
│   ├── densetensor.jl       # DenseTensor methods
│   └── tensoralgebra/
│       └── contract.jl      # Dense-specific contract!
├── diag/
│   └── ...
├── blocksparse/
│   └── ...
└── tensoroperations/
    ├── generic_tensor_operations.jl  # High-level dispatch
    └── contraction_logic.jl          # ContractionProperties
```

#### Target ndtensors-rs

```
crates/ndtensors/src/
├── lib.rs
├── scalar.rs                 # Scalar trait
├── error.rs                  # Error types
├── storage/
│   ├── mod.rs               # TensorStorage trait
│   ├── dense.rs             # Dense<T>
│   └── diag.rs              # Diag<T> (future)
├── tensor/
│   ├── mod.rs               # Tensor<T, S>
│   └── dense_tensor.rs      # DenseTensor methods
├── operations/
│   ├── mod.rs
│   ├── permutedims.rs       # permutedims dispatch
│   └── contract/
│       ├── mod.rs           # High-level contract API
│       ├── properties.rs    # ContractionProperties
│       ├── generic.rs       # Naive loop implementation
│       └── gemm.rs          # Reshape-to-GEMM (faer)
└── backend/
    ├── mod.rs               # Backend trait
    ├── generic.rs           # GenericBackend
    └── gemm.rs              # GemmBackend
```

### Key Differences

| Aspect | NDTensors.jl | ndtensors-rs (Target) |
|--------|--------------|----------------------|
| Polymorphism | Parametric types | Rust generics/traits |
| Extensions | Julia weak deps | Cargo features |
| Runtime state | Ref{Bool} | thread_local! RefCell |
| GEMM | BLAS.gemm! | faer::matmul |
| Type stripping | Expose module | Direct trait dispatch |

---

## Core Design Decisions

### 1. Reference-Counted Storage

Tensor data is managed via `Rc<Vec<T>>` (or `Arc<Vec<T>>` for multi-threaded scenarios):

```rust
pub struct Tensor<T> {
    data: Rc<Vec<T>>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}
```

**Rationale**:
- `clone()` is O(1) - only increments reference count
- AD can save tensors for backward pass without copying
- Copy-on-Write (CoW) via `Rc::make_mut()` for mutations

### 2. Separation from Index System

Unlike tensor4all-rs's `TensorDynLen` which embeds `Index` objects, ndtensors-rs stores only dimensions:

```rust
// ndtensors-rs: low-level, index-agnostic
pub struct Tensor<T> {
    data: Rc<Vec<T>>,
    shape: Vec<usize>,  // just dimensions
}

// Higher-level layer (ITensor-like) manages indices
pub struct IndexedTensor<T> {
    tensor: Tensor<T>,
    indices: Vec<Index>,
}
```

This mirrors NDTensors.jl's design where `IndsT` is a generic parameter that can be simple integers.

### 3. Storage Types

```rust
pub enum Storage<T> {
    Dense(Rc<Vec<T>>),
    Diag(Rc<Vec<T>>),      // diagonal elements only
    // Future: BlockSparse for quantum number symmetries
}
```

### 4. Automatic Differentiation

Following libtorch's design, ndtensors-rs supports multiple AD modes:

#### AD Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| Forward | Value only, no gradient tracking | Inference, evaluation |
| Forward + Reverse | Standard backprop (1st order) | Training, optimization |
| Forward + Forward | Forward-mode AD (JVP) | Directional derivatives |
| Forward + Forward + Reverse | Higher-order (Hessian-vector) | 2nd order optimization |

```rust
/// AD mode selection
pub enum ADMode {
    NoGrad,           // Forward only
    Reverse,          // Reverse-mode (backprop)
    Forward,          // Forward-mode (JVP)
    ForwardReverse,   // Hessian-vector products
}

pub struct Tensor<T> {
    data: Rc<Vec<T>>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    // AD state
    grad: Option<Rc<Tensor<T>>>,         // Reverse-mode gradient
    tangent: Option<Rc<Tensor<T>>>,      // Forward-mode tangent (JVP)
    grad_fn: Option<Box<dyn GradFn<T>>>, // Backward function
    requires_grad: bool,
}
```

#### Reverse-Mode AD (Backpropagation)

Standard tape-based reverse-mode for computing gradients:

```rust
// Forward: build computation graph
let a = Tensor::new(data_a).requires_grad();
let b = Tensor::new(data_b).requires_grad();
let c = a.matmul(&b);  // records MatMulBackward

// Backward: traverse graph in reverse
c.backward();
let grad_a = a.grad();  // ∂c/∂a
let grad_b = b.grad();  // ∂c/∂b
```

#### Forward-Mode AD (JVP)

For directional derivatives without building a tape:

```rust
// Set tangent vectors (direction of differentiation)
let a = Tensor::new(data_a).with_tangent(v_a);
let b = Tensor::new(data_b).with_tangent(v_b);

// Forward pass propagates tangents
let c = a.matmul(&b);
// c.tangent = a.tangent @ b + a @ b.tangent (JVP rule)

let jvp = c.tangent();  // directional derivative
```

#### Higher-Order AD

Combine forward and reverse for Hessian-vector products:

```rust
// Hessian-vector product: H @ v
let a = Tensor::new(data).requires_grad();
let v = Tensor::new(direction);

// Forward pass with tangent
let a_with_tangent = a.with_tangent(v);
let loss = compute_loss(&a_with_tangent);

// Backward on the tangent gives Hessian-vector product
loss.tangent().backward();
let hvp = a.grad();  // H @ v
```

#### Backward Functions

```rust
trait GradFn<T> {
    /// Reverse-mode: compute VJP (vector-Jacobian product)
    fn backward(&self, grad_output: &Tensor<T>) -> Vec<Tensor<T>>;

    /// Forward-mode: compute JVP (Jacobian-vector product)
    fn forward_ad(&self, tangents: &[&Tensor<T>]) -> Tensor<T>;
}

struct MatMulBackward<T> {
    saved_a: Rc<Vec<T>>,  // O(1) save via Rc::clone
    saved_b: Rc<Vec<T>>,
    shape_a: Vec<usize>,
    shape_b: Vec<usize>,
}

impl<T> GradFn<T> for MatMulBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> Vec<Tensor<T>> {
        // VJP: ∂L/∂A = G @ B^T, ∂L/∂B = A^T @ G
        vec![
            grad_output.matmul(&self.saved_b.transpose()),
            self.saved_a.transpose().matmul(grad_output),
        ]
    }

    fn forward_ad(&self, tangents: &[&Tensor<T>]) -> Tensor<T> {
        // JVP: d(A @ B) = dA @ B + A @ dB
        let da = tangents[0];
        let db = tangents[1];
        da.matmul(&self.saved_b) + self.saved_a.matmul(db)
    }
}
```

---

## FFI / Zero-Copy Interop

### Challenge: Reference Counting Across Language Boundaries

When exporting a tensor to Julia/Python:
- Julia/Python doesn't know about Rust's `Rc`
- If Rust side drops all references, data is freed while Julia still uses it
- If Julia modifies data, Rust's saved references (for AD) become invalid

### Current Implementation: Opaque Pointer Pattern

Following the pattern established in `sparse-ir-rs`, we use an opaque pointer approach for the C API:

```rust
/// Opaque tensor type for f64
#[repr(C)]
pub struct ndt_tensor_f64 {
    _private: *mut std::ffi::c_void,  // Box<Tensor<f64>>
}
```

**Key Design Decisions**:

1. **Rust owns the tensor**: Tensor objects are created and owned by Rust (`Box::new`). Julia holds an opaque pointer.

2. **Explicit lifecycle management**: Julia must call `ndt_tensor_f64_release()` to free the tensor. We use Julia's finalizer to ensure cleanup.

3. **Status codes for error handling**: All fallible operations return status codes, allowing proper error propagation across FFI boundary.

4. **Panic safety**: All C API functions use `catch_unwind` to convert Rust panics into error codes.

### Zero-Copy Data Access

When Rust creates a tensor and Julia needs to read/write its data:

```rust
// Rust side: return pointer to internal data
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_data(tensor: *const ndt_tensor_f64) -> *const c_double {
    // Returns pointer directly into Rust's Vec<f64>
    (*tensor).inner().data().as_ptr()
}
```

```julia
# Julia side: wrap pointer as Array (zero-copy)
data_ptr = unsafe_data(t)
arr = unsafe_wrap(Array, data_ptr, length(t))  # No copy!
```

**When data is copied**:
- Creating tensor from Julia Array → data copied to Rust
- Converting Rust tensor to Julia Array → data copied to Julia

**When zero-copy**:
- Rust creates tensor, Julia reads via `unsafe_wrap` → no copy
- Rust creates tensor, Julia writes via pointer → no copy

### Solution: Export Policy (Future)

```rust
impl<T: Clone> Tensor<T> {
    /// Strict mode: error if shared (detects bugs)
    pub fn export_exclusive(&self) -> Result<*const T, ExportError> {
        if Rc::strong_count(&self.data) > 1 {
            Err(ExportError::SharedReference)
        } else {
            Ok(self.data.as_ptr())
        }
    }

    /// Safe mode: copy if shared
    pub fn export_or_copy(&mut self) -> *const T {
        Rc::make_mut(&mut self.data);  // CoW if shared
        self.data.as_ptr()
    }
}
```

**Usage Guidelines**:
- Development/debugging: use `export_exclusive()` to catch unintended sharing
- Production: use `export_or_copy()` for safety

### DLPack Support (Future)

For interop with NumPy, PyTorch, Julia's DLPack.jl:

```rust
pub fn to_dlpack(&self) -> DLManagedTensor {
    // Increment Rc, pass to external
    // External calls deleter when done
}
```

---

## Operations

### Contraction

Label-based contraction (NDTensors.jl style):

```rust
// contract(A, [i, j], B, [j, k]) -> C with indices [i, k]
pub fn contract<T>(
    a: &Tensor<T>, labels_a: &[usize],
    b: &Tensor<T>, labels_b: &[usize],
) -> Tensor<T>;
```

### Decompositions

```rust
pub fn svd<T>(tensor: &Tensor<T>, left_inds: &[usize])
    -> (Tensor<T>, Tensor<T>, Tensor<T>);  // U, S, V

pub fn qr<T>(tensor: &Tensor<T>, left_inds: &[usize])
    -> (Tensor<T>, Tensor<T>);  // Q, R
```

### AD-Enabled Operations

```rust
impl TensorWithGrad<f64> {
    pub fn matmul(&self, other: &Self) -> Self {
        // 1. Forward computation
        let result = self.data.matmul(&other.data);

        // 2. Save for backward (O(1) via Rc::clone)
        let backward = MatMulBackward {
            saved_a: Rc::clone(&self.data.data),
            saved_b: Rc::clone(&other.data.data),
        };

        // 3. Return with grad_fn attached
        Self {
            data: result,
            grad_fn: Some(Box::new(backward)),
            requires_grad: self.requires_grad || other.requires_grad,
            grad: None,
        }
    }
}
```

---

## ChainRules.jl Integration

For seamless AD interop with Julia's ecosystem (Zygote.jl, Enzyme.jl, etc.), ndtensors-rs can expose ChainRules-compatible interfaces.

### The C-API Boundary Problem

**Key insight: C-API calls break automatic differentiation tracking.**

```
Julia AD (Zygote, etc.):
  x (tracked) → [C API boundary] → Rust computation → [C API boundary] → y (untracked!)
                     ↑
              AD tape/graph is cut here
```

**Why AD doesn't work across C-API:**

1. **Tape-based AD (Zygote, ReverseDiff)**: The computation graph only records Julia operations. Rust operations inside the C call are invisible to the tape.

2. **Source-to-source AD (Enzyme)**: Enzyme analyzes LLVM IR, but pre-compiled Rust binaries are opaque - it cannot see inside `ccall`.

**Solution: Manual rrule/frule definitions**

This is the same approach used by:
- PyTorch: `torch.autograd.Function` with custom `forward`/`backward`
- JAX: `jax.custom_vjp` / `jax.custom_jvp`
- Current ITensors.jl: ChainRules extension for ITensor operations

The pattern:
1. **Rust provides**: forward computation + separate VJP/JVP functions
2. **Julia provides**: ChainRules definitions that wire them together
3. **Result**: Julia's AD framework sees the operation as differentiable

**Implication**: Rust does NOT need internal tape/autograd machinery. The host language (Julia/Python) handles the computation graph; Rust just provides the mathematical derivatives.

### Architecture

```
Julia AD Ecosystem
       │
       ▼
┌─────────────────┐
│  ChainRules.jl  │  ← defines rrule/frule protocol
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ndtensors-rs    │
│ Julia binding   │  ← implements rrule/frule using C API
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ndtensors-rs    │
│ Rust core       │  ← actual VJP/JVP computation
└─────────────────┘
```

### Julia-side Implementation

```julia
using ChainRulesCore
using NDTensors  # Julia binding for ndtensors-rs

# Forward rule (JVP)
function ChainRulesCore.frule((_, ΔA, ΔB), ::typeof(contract), A, B)
    C = contract(A, B)
    # Call Rust's forward_ad
    ΔC = ccall(:ndtensor_contract_jvp, ..., A, B, ΔA, ΔB)
    return C, ΔC
end

# Reverse rule (VJP)
function ChainRulesCore.rrule(::typeof(contract), A, B)
    C = contract(A, B)
    function contract_pullback(ΔC)
        # Call Rust's backward
        ΔA, ΔB = ccall(:ndtensor_contract_vjp, ..., A, B, ΔC)
        return NoTangent(), ΔA, ΔB
    end
    return C, contract_pullback
end
```

### C API for AD

```rust
/// VJP for contraction (used by ChainRules.rrule)
#[no_mangle]
pub extern "C" fn ndtensor_contract_vjp(
    a: *const Tensor,
    b: *const Tensor,
    grad_c: *const Tensor,
    grad_a_out: *mut Tensor,
    grad_b_out: *mut Tensor,
) -> i32;

/// JVP for contraction (used by ChainRules.frule)
#[no_mangle]
pub extern "C" fn ndtensor_contract_jvp(
    a: *const Tensor,
    b: *const Tensor,
    tangent_a: *const Tensor,
    tangent_b: *const Tensor,
    tangent_c_out: *mut Tensor,
) -> i32;
```

### Benefits

- **Zygote.jl compatibility**: Automatic reverse-mode AD for Julia code using ndtensors-rs
- **Enzyme.jl compatibility**: Works with Enzyme's source-to-source AD
- **Composability**: ndtensors-rs operations compose with other Julia AD-enabled code
- **Single implementation**: AD rules implemented once in Rust, used everywhere

---

## Native Rust AD Options

When using ndtensors-rs as a pure Rust crate (not via C-API), there are additional AD options.

### Option 1: Manual VJP/JVP Functions (Recommended)

Always provide explicit derivative functions:

```rust
// Core primitives - always available
pub fn contract_vjp<T: Scalar>(
    a: &DenseTensor<T>, b: &DenseTensor<T>, grad_c: &DenseTensor<T>
) -> (DenseTensor<T>, DenseTensor<T>);

pub fn contract_jvp<T: Scalar>(
    a: &DenseTensor<T>, b: &DenseTensor<T>,
    tangent_a: &DenseTensor<T>, tangent_b: &DenseTensor<T>
) -> DenseTensor<T>;
```

These can be used by any AD framework (host language or Rust).

### Option 2: Tape-based Autograd (Cargo Feature)

PyTorch-style tape-based reverse-mode AD:

```rust
#[cfg(feature = "autograd")]
pub struct TrackedTensor<T> {
    tensor: DenseTensor<T>,
    grad: Option<DenseTensor<T>>,
    grad_fn: Option<Box<dyn GradFn<T>>>,
    requires_grad: bool,
}
```

**Pros**: Pure Rust, no external dependencies, stable
**Cons**: Manual backward implementation for each operation

### Option 3: Enzyme (Experimental, Nightly Only)

[Enzyme](https://enzyme.mit.edu/) performs source-to-source AD at LLVM IR level.

```rust
#![feature(autodiff)]
use std::autodiff::autodiff;

#[autodiff(d_contract, Reverse, Duplicated, Duplicated, Active)]
fn contract_scalar(a: &DenseTensor<f64>, b: &DenseTensor<f64>) -> f64 {
    // ...
}
// d_contract is auto-generated
```

**Status (as of late 2025)**:
- Requires Rust nightly with `#![feature(autodiff)]`
- Requires `lto = "fat"` in Cargo.toml (slower compilation)
- [rlibs now supported](https://github.com/rust-lang/rust/pull/129176) - can differentiate through dependencies
- [GSoC 2025](https://blog.karanjanthe.me/posts/enzyme-autodiff-rust-gsoc/) improved TypeTree handling

**Constraints**:
- Complex types (`Vec<T>`, custom structs) may cause type inference failures
- Not production-ready yet

### AD Strategy Summary

```
┌─────────────────────────────────────────────────────┐
│  ndtensors-rs AD Architecture                       │
│                                                      │
│  Layer 1: VJP/JVP primitives (always available)     │
│    contract_vjp(), permutedims_vjp(), svd_vjp()...  │
│                                                      │
│  Layer 2: Host language integration (via C-API)     │
│    - Julia: ChainRules.jl rrule/frule               │
│    - Python: JAX custom_vjp / PyTorch Function      │
│                                                      │
│  Layer 3: Native Rust AD (cargo features)           │
│    - feature = "autograd" → tape-based (stable)     │
│    - feature = "enzyme"   → Enzyme (nightly only)   │
└─────────────────────────────────────────────────────┘
```

**Recommendation**: Build on VJP/JVP primitives. Host language AD delegation is the primary path. Native Rust AD is optional for pure-Rust users.

---

## C API Design

The C API follows these principles:

### Status Codes

```rust
pub const NDT_SUCCESS: StatusCode = 0;
pub const NDT_INVALID_ARGUMENT: StatusCode = -1;
pub const NDT_SHAPE_MISMATCH: StatusCode = -2;
pub const NDT_INDEX_OUT_OF_BOUNDS: StatusCode = -3;
pub const NDT_INTERNAL_ERROR: StatusCode = -4;
```

### Function Naming Convention

- `ndt_tensor_f64_*` - operations on f64 tensors
- Future: `ndt_tensor_c64_*` - operations on c64 tensors

### Lifecycle Functions

```rust
// Creation
ndt_tensor_f64_zeros(shape, ndim, status) -> *mut ndt_tensor_f64
ndt_tensor_f64_from_data(data, len, shape, ndim, status) -> *mut ndt_tensor_f64

// Destruction
ndt_tensor_f64_release(tensor)

// Copy
ndt_tensor_f64_clone(src) -> *mut ndt_tensor_f64
```

---

## Implementation Notes

### Memory Layout

Both Faer and NDTensors.jl use **column-major** storage order:

- **Faer**: Confirmed in `faer/src/mat/mod.rs`
- **NDTensors.jl**: Uses `Base.size_to_strides(1, dims...)` (standard Julia column-major)
- **ndtensors-rs**: Follows the same convention

This ensures zero-copy interop between Rust tensors and Julia arrays.

### Scalar Types

We use Faer's `ComplexField` trait as the foundation for scalar types:

```rust
pub trait Scalar: ComplexField + Copy + Debug + Default + 'static {
    type Real: Scalar;
    fn zero() -> Self;
    fn one() -> Self;
}

impl Scalar for f64 { ... }
impl Scalar for c64 { ... }  // faer::c64 (Complex64)
```

---

## Project Structure

```
ndtensors-rs/
├── Cargo.toml              # workspace root
├── crates/
│   ├── ndtensors/          # Core tensor library
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── scalar.rs   # Scalar trait (wraps faer ComplexField)
│   │       ├── error.rs
│   │       ├── strides.rs
│   │       ├── storage/
│   │       │   ├── mod.rs
│   │       │   └── dense.rs
│   │       └── tensor.rs
│   └── ndtensors-capi/     # C API for FFI
│       └── src/lib.rs
├── julia/
│   └── NDTensorsRS/        # Julia bindings
│       ├── Project.toml
│       ├── deps/build.jl   # Calls cargo build
│       ├── src/NDTensorsRS.jl
│       └── test/runtests.jl
└── docs/
    └── design.md
```

---

## Migration Checklist

### Phase 1: Storage Abstraction
- [ ] Create `TensorStorage` trait
- [ ] Refactor `Tensor` to be generic over storage
- [ ] Add type aliases: `DenseTensor<T>`, etc.

### Phase 2: Contraction Refactoring
- [ ] Implement `ContractionProperties`
- [ ] Separate label computation from contraction execution
- [ ] Implement GEMM-based contraction path
- [ ] Keep naive implementation as fallback

### Phase 3: Backend Abstraction
- [ ] Create `ContractionBackend` trait
- [ ] Implement `GemmBackend` using faer
- [ ] Add runtime backend selection
- [ ] Add cargo feature for TBLIS (future)

### Phase 4: permutedims Optimization
- [ ] Evaluate rayon for parallel permutation
- [ ] Consider blocked algorithms for cache efficiency
- [ ] Profile against Strided.jl performance

---

## Future Work

- [ ] GPU backend abstraction
- [ ] BlockSparse storage for quantum number symmetries
- [ ] DLPack interop
- [ ] Fermionic sign tracking
- [ ] ChainRules.jl integration
- [ ] Python autograd interop (JAX custom_vjp, PyTorch autograd.Function)

---

## References

- [NDTensors.jl](https://github.com/ITensor/ITensors.jl/tree/main/NDTensors)
- [ITensors.jl paper](https://arxiv.org/abs/2007.14822)
- [sparse-ir-rs](https://github.com/tensor4all/sparse-ir-rs) - C API pattern reference
- PyTorch autograd design

### NDTensors.jl Source References

- BackendSelection: `lib/BackendSelection/`
- Contraction logic: `tensoroperations/contraction_logic.jl`
- Dense contract: `dense/tensoralgebra/contract.jl`
