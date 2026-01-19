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
//!
//! # Permutation and Contraction
//!
//! ```
//! use ndtensors::{Tensor, contract};
//!
//! // Create tensors
//! let a = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
//! let b = Tensor::<f64>::ones(&[3, 4]);
//!
//! // Permute dimensions (transpose)
//! let a_t = a.permutedims(&[1, 0]).unwrap();
//! assert_eq!(a_t.shape(), &[3, 2]);
//!
//! // Tensor contraction: C[i,k] = A[i,j] * B[j,k]
//! // Negative labels indicate contracted indices
//! let c = contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();
//! assert_eq!(c.shape(), &[2, 4]);
//! ```

pub mod backend;
pub mod blocksparse_tensor;
pub mod contract;
pub mod decomposition;
pub mod error;
pub mod operations;
pub mod random;
pub mod scalar;
pub mod storage;
pub mod strides;
pub mod tensor;

pub use blocksparse_tensor::{BlockSparseTensor, CpuBlockSparseTensor};
pub use contract::{contract, contract_vjp};
pub use error::TensorError;
pub use scalar::{Scalar, c64};
pub use storage::{CpuBuffer, CpuDense, DataBuffer, Dense, TensorStorage};
pub use tensor::{DenseTensor, Tensor};
