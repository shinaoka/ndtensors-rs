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

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Language Bindings                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │  Julia  │  │ Python  │  │   C++   │  │  Rust   │    │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │
│       │            │            │            │          │
│       └────────────┴─────┬──────┴────────────┘          │
│                          │                              │
│                    ┌─────┴─────┐                        │
│                    │   C API   │                        │
│                    └─────┬─────┘                        │
└──────────────────────────┼──────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────┐
│                   ndtensors-rs                          │
│                          │                              │
│  ┌───────────────────────┴────────────────────────┐    │
│  │              Core Tensor Layer                  │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────┐    │    │
│  │  │ Tensor  │  │ Storage │  │   Autograd  │    │    │
│  │  └─────────┘  └─────────┘  └─────────────┘    │    │
│  └────────────────────────────────────────────────┘    │
│                          │                              │
│  ┌───────────────────────┴────────────────────────┐    │
│  │              Backend Abstraction                │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────┐    │    │
│  │  │   CPU   │  │   GPU   │  │  libtorch   │    │    │
│  │  └─────────┘  └─────────┘  └─────────────┘    │    │
│  └────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

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

## FFI / Zero-Copy Interop

### Challenge: Reference Counting Across Language Boundaries

When exporting a tensor to Julia/Python:
- Julia/Python doesn't know about Rust's `Rc`
- If Rust side drops all references, data is freed while Julia still uses it
- If Julia modifies data, Rust's saved references (for AD) become invalid

### Solution: Export Policy

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

## Differences from NDTensors.jl

| Aspect | NDTensors.jl | ndtensors-rs |
|--------|--------------|--------------|
| Language | Julia | Rust + C API |
| Memory | Julia GC | Rc/Arc + CoW |
| Index type | Generic `IndsT` | Simple `Vec<usize>` |
| AD | External (Zygote) | Built-in tape-based |
| BlockSparse | Full support | Future work |

## ChainRules.jl Integration

For seamless AD interop with Julia's ecosystem (Zygote.jl, Enzyme.jl, etc.), ndtensors-rs can expose ChainRules-compatible interfaces.

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

## Future Work

- [ ] GPU backend abstraction
- [ ] BlockSparse storage for quantum number symmetries
- [ ] DLPack interop
- [ ] Fermionic sign tracking
- [ ] ChainRules.jl integration
- [ ] Python autograd interop (JAX custom_vjp, PyTorch autograd.Function)

## References

- [NDTensors.jl](https://github.com/ITensor/ITensors.jl/tree/main/NDTensors)
- [ITensors.jl paper](https://arxiv.org/abs/2007.14822)
- PyTorch autograd design
