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
//! - [`BlockSparse`] - Block-sparse storage container
//! - [`DiagBlockSparse`] - Diagonal block-sparse storage container
//!
//! # Example
//!
//! ```
//! use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims, BlockSparse};
//!
//! // Define block structure: 2 blocks in dim 0, 2 blocks in dim 1
//! let blockdims = BlockDims::new(vec![
//!     BlockDim::new(vec![2, 3]),  // dim 0: blocks of size 2, 3
//!     BlockDim::new(vec![4, 5]),  // dim 1: blocks of size 4, 5
//! ]);
//!
//! // Only store blocks (0,0) and (1,1)
//! let blocks = vec![
//!     Block::new(&[0, 0]),
//!     Block::new(&[1, 1]),
//! ];
//!
//! let storage: BlockSparse<f64> = BlockSparse::zeros(blocks, blockdims);
//! println!("Non-zero blocks: {}", storage.nnzblocks());
//! println!("Total non-zeros: {}", storage.nnz());
//! ```

mod block;
mod block_dim;
mod block_offsets;
mod diagblocksparse;
mod storage;

pub use block::Block;
pub use block_dim::{BlockDim, BlockDims};
pub use block_offsets::BlockOffsets;
pub use diagblocksparse::{CpuDiagBlockSparse, DiagBlockSparse};
pub use storage::{BlockSparse, CpuBlockSparse};
