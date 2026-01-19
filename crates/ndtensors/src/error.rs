//! Error types for ndtensors.

use thiserror::Error;

/// Errors that can occur in tensor operations.
#[derive(Debug, Error)]
pub enum TensorError {
    /// Shape mismatch between data length and expected size.
    #[error("shape mismatch: expected {expected} elements, got {actual}")]
    ShapeMismatch { expected: usize, actual: usize },

    /// Index out of bounds.
    #[error("index out of bounds: index {index} is out of range for dimension {dim_size}")]
    IndexOutOfBounds { index: usize, dim_size: usize },

    /// Wrong number of indices provided.
    #[error("wrong number of indices: expected {expected}, got {actual}")]
    WrongNumberOfIndices { expected: usize, actual: usize },

    /// Invalid permutation.
    #[error("invalid permutation {perm:?} for tensor with {ndim} dimensions")]
    InvalidPermutation { perm: Vec<usize>, ndim: usize },

    /// SVD computation error.
    #[error("SVD error: {message}")]
    SvdError { message: String },
}
