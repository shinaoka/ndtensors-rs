# NDTensors.jl Architecture Reference

This document details NDTensors.jl's internal architecture to guide ndtensors-rs implementation for a smooth migration path.

## Overview

NDTensors.jl uses a multi-layered architecture with clear separation between:
1. **Storage layer** - Raw data containers (Dense, BlockSparse, Diag)
2. **Tensor layer** - Universal tensor wrapper with indices
3. **Operation layer** - Algorithm dispatch (permutedims, contract)
4. **Backend layer** - Optimized implementations (BLAS, TBLIS, Strided.jl)

## Type Hierarchy

### NDTensors.jl Structure

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

### Key Design: Polymorphic Storage

```julia
# NDTensors.jl: Storage is a TYPE PARAMETER
struct Tensor{ElT, N, StoreT, IndsT} <: AbstractArray{ElT, N}
    storage::StoreT    # Polymorphic! Dense, BlockSparse, Diag, etc.
    inds::IndsT        # Index/dimension tuple
end
```

**Critical**: The same `Tensor` type wraps different storage types. Operations dispatch based on storage type.

### Current ndtensors-rs (Needs Refactoring)

```rust
// PROBLEM: Storage type is hardcoded
pub struct Tensor<T: Scalar> {
    storage: Dense<T>,  // ← Fixed to Dense, not polymorphic
    shape: Vec<usize>,
    strides: Vec<usize>,
}
```

### Target ndtensors-rs Structure

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

## Dense Storage

### NDTensors.jl

```julia
# src/dense/dense.jl
struct Dense{ElT, DataT<:AbstractVector} <: TensorStorage{ElT}
    data::DataT
end

# Key: Storage is always a flat vector, NOT multi-dimensional
data(S::Dense) = S.data
```

### ndtensors-rs (Current - OK)

```rust
pub struct Dense<T: Scalar> {
    data: Vec<T>,  // Flat vector, column-major order
}
```

This matches NDTensors.jl. The shape comes from the Tensor wrapper, not storage.

## Dispatch Hierarchy

### permutedims Dispatch Chain

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

### contract Dispatch Chain

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

## ContractionProperties

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

### ndtensors-rs Equivalent

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

## Backend Abstraction

### NDTensors.jl Pattern

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

### ndtensors-rs Pattern

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

## File Organization

### NDTensors.jl

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

### Target ndtensors-rs

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

## Key Differences

| Aspect | NDTensors.jl | ndtensors-rs (Target) |
|--------|--------------|----------------------|
| Polymorphism | Parametric types | Rust generics/traits |
| Extensions | Julia weak deps | Cargo features |
| Runtime state | Ref{Bool} | thread_local! RefCell |
| GEMM | BLAS.gemm! | faer::matmul |
| Type stripping | Expose module | Direct trait dispatch |

## References

- NDTensors.jl source: `extern/ITensors.jl/NDTensors/src/`
- BackendSelection: `lib/BackendSelection/`
- Contraction logic: `tensoroperations/contraction_logic.jl`
- Dense contract: `dense/tensoralgebra/contract.jl`
