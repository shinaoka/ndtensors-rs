//! ndtensors - Rust port of NDTensors.jl
//!
//! This crate provides n-dimensional tensor operations, designed to be
//! compatible with ITensors.jl's NDTensors backend.
//!
//! # Architecture
//!
//! Following NDTensors.jl's dispatch hierarchy:
//!
//! ```text
//! Level 1: High-level API (operations module)
//!     → permutedims, contract
//!
//! Level 2: In-place API
//!     → permutedims_into, contract (with output)
//!
//! Level 3: Backend implementation (backend module)
//!     → GenericBackend (naive loops)
//!     → Future: HpttBackend, RayonBackend
//! ```
//!
//! # Example
//!
//! ```
//! use ndtensors::{DenseTensor, Tensor, c64};
//!
//! // Create a 2x3 zero-initialized tensor
//! let mut t: DenseTensor<f64> = Tensor::zeros(&[2, 3]);
//!
//! // Set and get elements
//! t.set(&[0, 1], 5.0).unwrap();
//! assert_eq!(t.get(&[0, 1]), Some(&5.0));
//!
//! // Create from data (column-major order)
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let t2: DenseTensor<f64> = Tensor::from_vec(data, &[2, 3]).unwrap();
//! ```

pub mod backend;
pub mod contract;
pub mod error;
pub mod operations;
pub mod scalar;
pub mod storage;
pub mod strides;
pub mod tensor;

pub use contract::contract;
pub use error::TensorError;
pub use scalar::{Scalar, c64};
pub use storage::{Dense, TensorStorage};
pub use tensor::{DenseTensor, Tensor};
