//! Tensor decomposition operations.
//!
//! This module provides matrix decompositions for tensors by reshaping them
//! to 2D matrices, applying the decomposition using faer, and reshaping
//! the results back to tensor form.
//!
//! # Available Decompositions
//!
//! - [`svd`] / [`svd_truncated`]: Singular Value Decomposition
//! - [`qr`]: QR Decomposition
//! - [`polar`]: Polar Decomposition (A = U * P)
//! - [`matrix_exp`]: Matrix Exponential
//!
//! # Design
//!
//! Following NDTensors.jl's approach, decompositions work by:
//! 1. Specifying which tensor indices become "left" (row) dimensions
//! 2. Specifying which tensor indices become "right" (column) dimensions
//! 3. Permuting and reshaping to a 2D matrix
//! 4. Applying the matrix decomposition
//! 5. Reshaping results back to tensor form
//!
//! # Example
//!
//! ```
//! use ndtensors::Tensor;
//! use ndtensors::decomposition::{svd, qr};
//!
//! // 3D tensor
//! let t = Tensor::<f64>::ones(&[2, 3, 4]);
//!
//! // SVD with left indices [0,1] and right index [2]
//! // Reshapes to 6x4 matrix, computes SVD, reshapes back
//! let svd_result = svd(&t, &[0, 1], &[2]).unwrap();
//!
//! // QR with same indices
//! let qr_result = qr(&t, &[0, 1], &[2]).unwrap();
//! ```

mod exp;
mod polar;
mod qr;
mod svd;
mod util;

pub use exp::matrix_exp;
pub use polar::{PolarResult, polar};
pub use qr::{QrResult, qr};
pub use svd::{SvdResult, svd, svd_truncated};
pub use util::{PermuteReshapeResult, permute_reshape};
