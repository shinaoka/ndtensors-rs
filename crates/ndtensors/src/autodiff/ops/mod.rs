//! Tracked tensor operations with automatic differentiation.
//!
//! This module provides:
//! - Tracked operations for backward-mode AD (reverse-mode)
//! - Dual operations for forward-mode AD (JVP)

mod contract;
mod dual_contract;

pub use contract::tracked_contract;
pub use dual_contract::dual_contract;
