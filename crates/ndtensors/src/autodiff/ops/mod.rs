//! Tracked tensor operations with automatic differentiation.
//!
//! This module provides tracked versions of tensor operations that
//! record the computation graph for backward pass.

mod contract;

pub use contract::tracked_contract;
