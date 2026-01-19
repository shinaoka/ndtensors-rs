//! Native Rust autodiff module for reverse-mode automatic differentiation.
//!
//! This module provides tape-based reverse-mode AD for tensor operations,
//! following a PyTorch-style API with a thread-local computation graph.
//!
//! # Architecture
//!
//! ```text
//! TrackedTensor<f64>  ──registers in──►  ComputationGraph (thread_local)
//!        │                                      │
//!        ▼                                      ▼
//!   DenseTensor<f64>                    Vec<Node<f64>>
//!                                              │
//!                                              ▼
//!                                ContractBackward (GradFn trait)
//!                                              │
//!                                         SavedTensor (Rc)
//! ```
//!
//! # Example
//!
//! ```ignore
//! use ndtensors::autodiff::{TrackedTensor, tracked_contract, backward, clear_graph};
//! use ndtensors::Tensor;
//!
//! // Clear the graph for a fresh computation
//! clear_graph();
//!
//! // Create leaf tensors that require gradients
//! let a = TrackedTensor::leaf(Tensor::ones(&[2, 3]));
//! let b = TrackedTensor::leaf(Tensor::ones(&[3, 4]));
//!
//! // Forward pass: C = A @ B
//! let c = tracked_contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();
//!
//! // Sum to scalar for loss
//! let ones = TrackedTensor::new(Tensor::ones(&[2, 4]));
//! let loss = tracked_contract(&c, &[-1, -2], &ones, &[-1, -2]).unwrap();
//!
//! // Backward pass
//! let grads = backward(&loss).unwrap();
//!
//! // Access gradients
//! let grad_a = grads.get(a.node_id().unwrap()).unwrap();
//! let grad_b = grads.get(b.node_id().unwrap()).unwrap();
//! ```
//!
//! # Key Types
//!
//! - [`TrackedTensor`]: Tensor with gradient tracking
//! - [`backward`]: Execute backward pass from scalar loss
//! - [`Gradients`]: Container for accumulated gradients
//! - [`tracked_contract`]: Tracked contraction operation
//!
//! # Design Notes
//!
//! - Thread-local computation graph (no `Arc`, uses `Rc`)
//! - Type-erased tensor storage via `AnyStorage` trait
//! - Gradient accumulation for multiple paths to same node

mod any_storage;
mod backward;
mod gradients;
mod graph;
mod ops;
mod saved_tensor;
mod tensor;

pub use any_storage::AnyStorage;
pub use backward::backward;
pub use gradients::Gradients;
pub use graph::{ComputationGraph, GradFn, NodeId, NodeRef, clear_graph_f64, with_graph_f64};
pub use ops::tracked_contract;
pub use saved_tensor::{SavePolicy, SavedTensor};
pub use tensor::{TrackedTensor, clear_graph};
