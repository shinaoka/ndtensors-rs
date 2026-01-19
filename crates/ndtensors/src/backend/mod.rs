//! Backend abstraction for tensor operations.
//!
//! This module provides traits and implementations for different computational backends,
//! following NDTensors.jl's backend selection pattern.
//!
//! # Backends
//!
//! - `GenericBackend`: Naive loop-based implementation (always available)
//! - Future: `HpttBackend`: High-Performance Tensor Transpose (cargo feature)
//! - Future: `RayonBackend`: Parallel implementation using rayon
//!
//! # faer Integration
//!
//! The `faer_interop` module provides zero-copy conversion between tensors and
//! faer's matrix types for high-performance linear algebra operations.

mod faer_interop;
mod generic;
mod permutation;

pub use faer_interop::{
    AsFaerMat, CpuDenseTensor, faer_mat_from_tensor, reshape_to_matrix, tensor_from_faer_mat,
};
pub use generic::GenericBackend;
pub use permutation::PermutationBackend;
