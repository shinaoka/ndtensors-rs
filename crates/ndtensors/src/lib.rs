//! ndtensors - Rust port of NDTensors.jl
//!
//! This crate provides n-dimensional tensor operations, designed to be
//! compatible with ITensors.jl's NDTensors backend.
//!
//! # Example
//!
//! ```
//! use ndtensors::{Tensor, c64};
//!
//! // Create a 2x3 zero-initialized tensor
//! let mut t: Tensor<f64> = Tensor::zeros(&[2, 3]);
//!
//! // Set and get elements
//! t.set(&[0, 1], 5.0).unwrap();
//! assert_eq!(t.get(&[0, 1]), Some(&5.0));
//!
//! // Create from data (column-major order)
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let t2 = Tensor::from_vec(data, &[2, 3]).unwrap();
//! ```

pub mod error;
pub mod scalar;
pub mod storage;
pub mod strides;
pub mod tensor;

pub use error::TensorError;
pub use scalar::{Scalar, c64};
pub use storage::Dense;
pub use tensor::Tensor;
