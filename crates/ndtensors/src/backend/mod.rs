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

mod generic;
mod permutation;

pub use generic::GenericBackend;
pub use permutation::PermutationBackend;
