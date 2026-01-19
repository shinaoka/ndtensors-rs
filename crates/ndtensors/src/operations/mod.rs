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

mod outer;
mod permutedims;

pub use outer::{outer, outer_into};
pub use permutedims::{permutedims, permutedims_into};
