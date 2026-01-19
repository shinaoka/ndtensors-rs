//! Tensor operations.
//!
//! This module provides high-level tensor operations, following NDTensors.jl's
//! dispatch hierarchy:
//!
//! ```text
//! Level 1: High-level API (permutedims, contract)
//!     → allocate output
//!     → call in-place version
//!
//! Level 2: In-place API (permutedims!, contract!)
//!     → dispatch to backend
//!
//! Level 3: Backend implementation (Generic, HPTT, etc.)
//! ```
//!
//! BlockSparseTensor operations are in the `blocksparse` submodule.

pub mod blocksparse;
mod convert;
mod copy;
mod diag;
mod elementwise;
mod norm;
mod outer;
mod permutedims;
mod slice;

pub use convert::to_nested_vec_2d;
pub use copy::copy_into;
pub use diag::{diag, diag_from_vec, diag_nd};
pub use elementwise::{
    apply, apply_binary, apply_inplace, conj, conj_inplace, imag, real, scale, scale_inplace,
};
pub use norm::{norm, norm_sqr};
pub use outer::{outer, outer_into};
pub use permutedims::{permutedims, permutedims_into, permutedims_with};
pub use slice::slice;
