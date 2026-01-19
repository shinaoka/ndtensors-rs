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

    /// Operation requires specific tensor rank.
    #[error("expected tensor of rank {expected}, got rank {actual}")]
    RankMismatch { expected: usize, actual: usize },

    /// Slice range out of bounds.
    #[error("slice range {start}..{end} out of bounds for dimension {dim} with size {size}")]
    SliceOutOfBounds {
        start: usize,
        end: usize,
        dim: usize,
        size: usize,
    },

    /// Polar decomposition error.
    #[error("polar decomposition error: {message}")]
    PolarError { message: String },

    /// Matrix exponential error.
    #[error("matrix exponential error: {message}")]
    MatrixExpError { message: String },

    /// Matrix must be square.
    #[error("matrix must be square: got {rows}x{cols}")]
    NotSquareMatrix { rows: usize, cols: usize },

    /// Block not found in block-sparse tensor.
    #[error("block {block:?} not found in tensor")]
    BlockNotFound { block: Vec<usize> },
}
