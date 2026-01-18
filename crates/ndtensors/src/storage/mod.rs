//! Storage types for tensor data.
//!
//! Following NDTensors.jl's storage hierarchy:
//!
//! ```text
//! TensorStorage<T> (trait)
//! ├── Dense<T>       - Contiguous array storage
//! ├── Diag<T>        - Diagonal storage (future)
//! └── BlockSparse<T> - Block sparse storage (future)
//! ```

mod dense;

use crate::scalar::Scalar;

pub use dense::Dense;

/// Trait for tensor storage types.
///
/// This mirrors NDTensors.jl's `TensorStorage{ElT} <: AbstractVector{ElT}`.
/// Storage is always a flat vector; shape/strides come from the Tensor wrapper.
pub trait TensorStorage<T: Scalar>: Clone + std::fmt::Debug {
    /// Create storage with given length, zero-initialized.
    fn zeros(len: usize) -> Self;

    /// Create storage from existing vector.
    fn from_vec(data: Vec<T>) -> Self;

    /// Length of storage (number of elements).
    fn len(&self) -> usize;

    /// Check if storage is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get immutable slice of data.
    fn as_slice(&self) -> &[T];

    /// Get mutable slice of data.
    fn as_mut_slice(&mut self) -> &mut [T];

    /// Get raw pointer (for FFI).
    fn as_ptr(&self) -> *const T {
        self.as_slice().as_ptr()
    }

    /// Get mutable raw pointer (for FFI).
    fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_slice().as_mut_ptr()
    }
}

// Implement TensorStorage for Dense
impl<T: Scalar> TensorStorage<T> for Dense<T> {
    fn zeros(len: usize) -> Self {
        Dense::zeros(len)
    }

    fn from_vec(data: Vec<T>) -> Self {
        Dense::from_vec(data)
    }

    fn len(&self) -> usize {
        Dense::len(self)
    }

    fn as_slice(&self) -> &[T] {
        Dense::as_slice(self)
    }

    fn as_mut_slice(&mut self) -> &mut [T] {
        Dense::as_mut_slice(self)
    }
}
