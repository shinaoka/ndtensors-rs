//! Storage types for tensor data.
//!
//! Following NDTensors.jl's storage hierarchy:
//!
//! ```text
//! TensorStorage<ElT> (trait)
//! ├── Dense<ElT, D>           - Contiguous array storage (generic over DataBuffer)
//! ├── Diag<ElT, D>            - Diagonal storage
//! ├── BlockSparse<ElT, D>     - Block sparse storage
//! ├── DiagBlockSparse<ElT, D> - Diagonal block sparse storage
//! ├── EmptyStorage<ElT>       - Empty (zero-element) storage
//! └── Combiner                - Index combining/uncombining operations
//! ```
//!
//! ## Backend Abstraction
//!
//! The `DataBuffer` trait provides backend abstraction:
//!
//! ```text
//! DataBuffer<T> (trait)
//! ├── CpuBuffer<T>   - CPU backend (Vec<T>)
//! ├── CudaBuffer<T>  - CUDA GPU backend (future)
//! └── MetalBuffer<T> - Apple Metal backend (future)
//! ```

pub mod blocksparse;
pub mod buffer;
mod combiner;
mod dense;
mod diag;
mod empty;

use crate::scalar::Scalar;

pub use buffer::{CpuBuffer, DataBuffer};
pub use combiner::Combiner;
pub use dense::{CpuDense, Dense};
pub use diag::{CpuDiag, Diag};
pub use empty::{EmptyNumberStorage, EmptyStorage};

/// Trait for tensor storage types.
///
/// This mirrors NDTensors.jl's `TensorStorage{ElT} <: AbstractVector{ElT}`.
/// Storage is always a flat vector; shape/strides come from the Tensor wrapper.
pub trait TensorStorage<ElT: Scalar>: Clone + std::fmt::Debug {
    /// Create storage with given length, zero-initialized.
    fn zeros(len: usize) -> Self;

    /// Create storage from existing vector.
    fn from_vec(data: Vec<ElT>) -> Self;

    /// Length of storage (number of elements).
    fn len(&self) -> usize;

    /// Check if storage is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get immutable slice of data.
    fn as_slice(&self) -> &[ElT];

    /// Get mutable slice of data.
    fn as_mut_slice(&mut self) -> &mut [ElT];

    /// Get raw pointer (for FFI).
    fn as_ptr(&self) -> *const ElT {
        self.as_slice().as_ptr()
    }

    /// Get mutable raw pointer (for FFI).
    fn as_mut_ptr(&mut self) -> *mut ElT {
        self.as_mut_slice().as_mut_ptr()
    }
}

// Implement TensorStorage for Dense<ElT, D>
impl<ElT: Scalar, D: DataBuffer<ElT>> TensorStorage<ElT> for Dense<ElT, D> {
    fn zeros(len: usize) -> Self {
        Dense::zeros(len)
    }

    fn from_vec(data: Vec<ElT>) -> Self {
        Dense::from_vec(data)
    }

    fn len(&self) -> usize {
        Dense::len(self)
    }

    fn as_slice(&self) -> &[ElT] {
        Dense::as_slice(self)
    }

    fn as_mut_slice(&mut self) -> &mut [ElT] {
        Dense::as_mut_slice(self)
    }
}

// Implement TensorStorage for Diag<ElT, D>
impl<ElT: Scalar, D: DataBuffer<ElT>> TensorStorage<ElT> for Diag<ElT, D> {
    fn zeros(len: usize) -> Self {
        Diag::zeros(len)
    }

    fn from_vec(data: Vec<ElT>) -> Self {
        Diag::from_vec(data)
    }

    fn len(&self) -> usize {
        Diag::len(self)
    }

    fn as_slice(&self) -> &[ElT] {
        Diag::as_slice(self)
    }

    fn as_mut_slice(&mut self) -> &mut [ElT] {
        Diag::as_mut_slice(self)
    }
}
