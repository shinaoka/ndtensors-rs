//! Tensor contraction operations.
//!
//! This module provides tensor contraction using label-based indexing,
//! following NDTensors.jl's convention:
//! - Negative labels indicate contracted indices (summed over)
//! - Positive labels indicate uncontracted indices (appear in output)
//!
//! # Implementations
//!
//! - `naive`: Loop-based implementation (fallback)
//! - `gemm`: GEMM-based implementation using faer (optimized)
//!
//! # Example
//!
//! ```
//! use ndtensors::{Tensor, contract};
//!
//! // Matrix multiplication: C[i,k] = A[i,j] * B[j,k]
//! let a = Tensor::<f64>::ones(&[2, 3]);
//! let b = Tensor::<f64>::ones(&[3, 4]);
//!
//! // labels: A[1,-1], B[-1,2] -> C[1,2]
//! let c = contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();
//! assert_eq!(c.shape(), &[2, 4]);
//! ```

mod gemm;
mod naive;
mod properties;

pub use naive::{contract, contract_vjp};
pub use properties::ContractionProperties;

// Re-export GEMM-based contraction for explicit use
pub use gemm::contract_gemm;
