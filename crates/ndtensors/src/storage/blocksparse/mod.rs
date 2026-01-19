//! BlockSparse storage types for sparse block tensors.
//!
//! This module provides the core types for block-sparse tensor storage,
//! mirroring NDTensors.jl's BlockSparse implementation.
//!
//! # Overview
//!
//! Block-sparse tensors store only non-zero blocks, which is efficient
//! for tensors with block-structured sparsity patterns (common in
//! quantum physics and tensor network applications).
//!
//! ## Core Types
//!
//! - [`Block`] - Block coordinates with precomputed hash
//! - [`BlockDim`] - Block sizes for a single dimension
//! - [`BlockDims`] - Block dimensions for all tensor dimensions
//! - [`BlockOffsets`] - Mapping from blocks to storage offsets
//!
//! # Example
//!
//! ```
//! use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims, BlockOffsets};
//!
//! // Define block structure: 2 blocks in dim 0, 3 blocks in dim 1
//! let dims = BlockDims::new(vec![
//!     BlockDim::new(vec![2, 3]),    // dim 0: blocks of size 2, 3
//!     BlockDim::new(vec![4, 5, 6]), // dim 1: blocks of size 4, 5, 6
//! ]);
//!
//! // Only store blocks (0,0), (0,2), (1,1)
//! let blocks = vec![
//!     Block::new(&[0, 0]),
//!     Block::new(&[0, 2]),
//!     Block::new(&[1, 1]),
//! ];
//!
//! let offsets = BlockOffsets::from_blocks(&blocks, &dims);
//! println!("Total non-zeros: {}", offsets.total_nnz());
//! ```

mod block;
mod block_dim;
mod block_offsets;

pub use block::Block;
pub use block_dim::{BlockDim, BlockDims};
pub use block_offsets::BlockOffsets;
