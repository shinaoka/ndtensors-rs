//! DiagBlockSparse storage for diagonal block-sparse tensors.
//!
//! This module provides the `DiagBlockSparse` storage type, which stores only
//! diagonal elements within non-zero blocks. This mirrors NDTensors.jl's
//! `DiagBlockSparse{ElT, VecT, N}` type.
//!
//! # Overview
//!
//! For a block-sparse tensor where only diagonal elements (i.e., elements
//! where all indices within each block are equal) are non-zero, DiagBlockSparse
//! provides more efficient storage than BlockSparse by only storing the
//! diagonal elements.
//!
//! For a block with shape [d0, d1, ..., dn], only min(d0, d1, ..., dn)
//! diagonal elements are stored.

use std::marker::PhantomData;

use crate::scalar::Scalar;
use crate::storage::buffer::{CpuBuffer, DataBuffer};

use super::block::Block;
use super::block_dim::BlockDims;
use super::block_offsets::BlockOffsets;

/// Diagonal block-sparse storage for tensors.
///
/// Stores only diagonal elements within non-zero blocks, which is efficient
/// for tensors with diagonal block structure (common in quantum physics for
/// identity and diagonal operators).
///
/// # Type Parameters
///
/// - `ElT`: Element type (e.g., `f64`, `c64`)
/// - `D`: Data buffer type (e.g., `CpuBuffer<ElT>`)
///
/// # NDTensors.jl Equivalence
///
/// This corresponds to NDTensors.jl's `DiagBlockSparse{ElT, VecT, N}`.
/// The Julia version supports two variants:
/// - Non-uniform: `VecT <: AbstractVector{ElT}` - different diagonal values
/// - Uniform: `VecT <: Number` - single scalar for all diagonal elements
///
/// Currently, only the non-uniform variant is implemented.
///
/// # Example
///
/// ```
/// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims, DiagBlockSparse};
///
/// // Define block structure: 2x2 blocks
/// let blockdims = BlockDims::new(vec![
///     BlockDim::new(vec![2, 3]),  // dim 0: blocks of size 2, 3
///     BlockDim::new(vec![2, 3]),  // dim 1: blocks of size 2, 3
/// ]);
///
/// // Create diagonal storage with blocks (0,0) and (1,1)
/// let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];
/// let storage: DiagBlockSparse<f64> = DiagBlockSparse::zeros(blocks, blockdims);
///
/// // Only diagonal elements are stored
/// // Block (0,0) has shape [2,2], so 2 diagonal elements
/// // Block (1,1) has shape [3,3], so 3 diagonal elements
/// assert_eq!(storage.nnzblocks(), 2);
/// assert_eq!(storage.nnz(), 2 + 3);
/// ```
#[derive(Clone, Debug)]
pub struct DiagBlockSparse<ElT: Scalar, D: DataBuffer<ElT> = CpuBuffer<ElT>> {
    /// Flat storage for all diagonal elements
    data: D,
    /// Mapping from block coordinates to offsets in data
    blockoffsets: BlockOffsets,
    /// Block dimensions for each tensor dimension
    blockdims: BlockDims,
    /// Phantom data for element type
    _phantom: PhantomData<ElT>,
}

/// Type alias for CPU-backed DiagBlockSparse storage.
pub type CpuDiagBlockSparse<ElT> = DiagBlockSparse<ElT, CpuBuffer<ElT>>;

impl<ElT: Scalar, D: DataBuffer<ElT>> DiagBlockSparse<ElT, D> {
    /// Compute the number of diagonal elements in a block given its shape.
    ///
    /// For a block with shape [d0, d1, ..., dn], this returns min(d0, d1, ..., dn).
    fn diag_size_from_shape(shape: &[usize]) -> usize {
        shape.iter().copied().min().unwrap_or(0)
    }

    /// Compute the total number of diagonal elements for the given blocks.
    fn total_diag_elements(blocks: &[Block], blockdims: &BlockDims) -> usize {
        blocks
            .iter()
            .map(|b| Self::diag_size_from_shape(&blockdims.block_shape(b.coords())))
            .sum()
    }

    /// Create a new DiagBlockSparse storage with the given blocks initialized to zero.
    ///
    /// # Arguments
    ///
    /// * `blocks` - List of non-zero block coordinates
    /// * `blockdims` - Block dimensions for each tensor dimension
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims, DiagBlockSparse};
    ///
    /// let blockdims = BlockDims::new(vec![
    ///     BlockDim::new(vec![2, 3]),
    ///     BlockDim::new(vec![2, 3]),
    /// ]);
    /// let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];
    /// let storage: DiagBlockSparse<f64> = DiagBlockSparse::zeros(blocks, blockdims);
    /// ```
    pub fn zeros(blocks: Vec<Block>, blockdims: BlockDims) -> Self {
        let blockoffsets = Self::compute_diag_blockoffsets(&blocks, &blockdims);
        let total_nnz = Self::total_diag_elements(&blocks, &blockdims);
        let data = D::zeros(total_nnz);
        Self {
            data,
            blockoffsets,
            blockdims,
            _phantom: PhantomData,
        }
    }

    /// Create a new DiagBlockSparse storage from existing data.
    ///
    /// # Arguments
    ///
    /// * `data` - Flat data buffer containing all diagonal elements
    /// * `blockoffsets` - Mapping from blocks to offsets
    /// * `blockdims` - Block dimensions for each tensor dimension
    ///
    /// # Panics
    ///
    /// Panics if data length doesn't match total diagonal elements.
    pub fn from_data(data: D, blockoffsets: BlockOffsets, blockdims: BlockDims) -> Self {
        // Verify data length matches expected total
        let expected_nnz: usize = blockoffsets
            .iter()
            .map(|(block, _)| Self::diag_size_from_shape(&blockdims.block_shape(block.coords())))
            .sum();
        assert_eq!(
            data.len(),
            expected_nnz,
            "data length ({}) must match total diagonal elements ({})",
            data.len(),
            expected_nnz
        );
        Self {
            data,
            blockoffsets,
            blockdims,
            _phantom: PhantomData,
        }
    }

    /// Create a new DiagBlockSparse storage from a Vec.
    ///
    /// # Arguments
    ///
    /// * `data` - Vec containing all diagonal element data
    /// * `blockoffsets` - Mapping from blocks to offsets
    /// * `blockdims` - Block dimensions for each tensor dimension
    pub fn from_vec(data: Vec<ElT>, blockoffsets: BlockOffsets, blockdims: BlockDims) -> Self {
        Self::from_data(D::from_vec(data), blockoffsets, blockdims)
    }

    /// Create a uniform DiagBlockSparse where all diagonal elements have the same value.
    ///
    /// # Arguments
    ///
    /// * `value` - The uniform value for all diagonal elements
    /// * `blocks` - List of non-zero block coordinates
    /// * `blockdims` - Block dimensions for each tensor dimension
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims, DiagBlockSparse};
    ///
    /// let blockdims = BlockDims::new(vec![
    ///     BlockDim::new(vec![2, 3]),
    ///     BlockDim::new(vec![2, 3]),
    /// ]);
    /// let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];
    /// let storage: DiagBlockSparse<f64> = DiagBlockSparse::uniform(1.0, blocks, blockdims);
    ///
    /// // All diagonal elements are 1.0
    /// assert!(storage.as_slice().iter().all(|&v| v == 1.0));
    /// ```
    pub fn uniform(value: ElT, blocks: Vec<Block>, blockdims: BlockDims) -> Self {
        let blockoffsets = Self::compute_diag_blockoffsets(&blocks, &blockdims);
        let total_nnz = Self::total_diag_elements(&blocks, &blockdims);
        let data = D::from_vec(vec![value; total_nnz]);
        Self {
            data,
            blockoffsets,
            blockdims,
            _phantom: PhantomData,
        }
    }

    /// Create an identity-like DiagBlockSparse where all diagonal elements are 1.
    ///
    /// # Arguments
    ///
    /// * `blocks` - List of non-zero block coordinates
    /// * `blockdims` - Block dimensions for each tensor dimension
    pub fn identity(blocks: Vec<Block>, blockdims: BlockDims) -> Self {
        Self::uniform(ElT::one(), blocks, blockdims)
    }

    /// Compute block offsets for diagonal storage.
    fn compute_diag_blockoffsets(blocks: &[Block], blockdims: &BlockDims) -> BlockOffsets {
        let mut offset = 0;
        let pairs: Vec<(Block, usize)> = blocks
            .iter()
            .map(|block| {
                let current_offset = offset;
                let diag_size = Self::diag_size_from_shape(&blockdims.block_shape(block.coords()));
                offset += diag_size;
                (block.clone(), current_offset)
            })
            .collect();
        let total_nnz = offset;
        BlockOffsets::from_iter(pairs, total_nnz)
    }

    /// Get the number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.blockdims.ndims()
    }

    /// Get the dense shape of the tensor.
    ///
    /// This is the shape the tensor would have if fully dense.
    #[inline]
    pub fn shape(&self) -> Vec<usize> {
        self.blockdims.dense_shape()
    }

    /// Get the number of non-zero (diagonal) elements.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// Get the number of non-zero blocks.
    #[inline]
    pub fn nnzblocks(&self) -> usize {
        self.blockoffsets.nnzblocks()
    }

    /// Get the block dimensions.
    #[inline]
    pub fn blockdims(&self) -> &BlockDims {
        &self.blockdims
    }

    /// Get the block offsets.
    #[inline]
    pub fn blockoffsets(&self) -> &BlockOffsets {
        &self.blockoffsets
    }

    /// Check if a block is non-zero (present in storage).
    ///
    /// # Arguments
    ///
    /// * `block` - Block coordinates to check
    #[inline]
    pub fn isblocknz(&self, block: &Block) -> bool {
        self.blockoffsets.contains(block)
    }

    /// Get the offset of a block's diagonal in the flat storage.
    ///
    /// Returns `None` if the block is not present.
    #[inline]
    pub fn block_offset(&self, block: &Block) -> Option<usize> {
        self.blockoffsets.get(block)
    }

    /// Get the diagonal size of a block (number of diagonal elements).
    ///
    /// For a block with shape [d0, d1, ..., dn], this is min(d0, d1, ..., dn).
    #[inline]
    pub fn diag_size(&self, block: &Block) -> usize {
        Self::diag_size_from_shape(&self.blockdims.block_shape(block.coords()))
    }

    /// Get the full block shape (for compatibility with BlockSparse).
    #[inline]
    pub fn block_shape(&self, block: &Block) -> Vec<usize> {
        self.blockdims.block_shape(block.coords())
    }

    /// Get an immutable view of a block's diagonal data as a slice.
    ///
    /// Returns `None` if the block is not present (structurally zero).
    ///
    /// # Arguments
    ///
    /// * `block` - Block coordinates
    pub fn blockview(&self, block: &Block) -> Option<&[ElT]> {
        let offset = self.blockoffsets.get(block)?;
        let size = self.diag_size(block);
        Some(&self.data.as_slice()[offset..offset + size])
    }

    /// Get a mutable view of a block's diagonal data as a slice.
    ///
    /// Returns `None` if the block is not present (structurally zero).
    ///
    /// # Arguments
    ///
    /// * `block` - Block coordinates
    pub fn blockview_mut(&mut self, block: &Block) -> Option<&mut [ElT]> {
        let offset = self.blockoffsets.get(block)?;
        let size = self.diag_size(block);
        Some(&mut self.data.as_mut_slice()[offset..offset + size])
    }

    /// Get the underlying data buffer.
    #[inline]
    pub fn data(&self) -> &D {
        &self.data
    }

    /// Get the underlying data as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[ElT] {
        self.data.as_slice()
    }

    /// Get the underlying data as a mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [ElT] {
        self.data.as_mut_slice()
    }

    /// Iterate over all non-zero blocks with their diagonal data.
    ///
    /// Returns an iterator of (Block, &[ElT]) pairs.
    pub fn iter_blocks(&self) -> impl Iterator<Item = (&Block, &[ElT])> {
        self.blockoffsets.iter().map(move |(block, &offset)| {
            let size = self.diag_size(block);
            let data = &self.data.as_slice()[offset..offset + size];
            (block, data)
        })
    }

    /// Check if the block coordinates are valid for this storage.
    #[inline]
    pub fn is_valid_block(&self, block: &Block) -> bool {
        self.blockdims.is_valid_block(block.coords())
    }

    /// Get an iterator over the non-zero blocks.
    pub fn nzblocks(&self) -> impl Iterator<Item = &Block> {
        self.blockoffsets.blocks()
    }
}

impl<ElT: Scalar, D: DataBuffer<ElT>> PartialEq for DiagBlockSparse<ElT, D> {
    fn eq(&self, other: &Self) -> bool {
        // Check block structure
        if self.blockdims != other.blockdims {
            return false;
        }
        if self.blockoffsets.nnzblocks() != other.blockoffsets.nnzblocks() {
            return false;
        }

        // Check that all blocks match
        for block in self.blockoffsets.blocks() {
            match (self.blockview(block), other.blockview(block)) {
                (Some(a), Some(b)) => {
                    if a != b {
                        return false;
                    }
                }
                (None, None) => {}
                _ => return false,
            }
        }
        true
    }
}

impl<ElT: Scalar, D: DataBuffer<ElT>> std::fmt::Display for DiagBlockSparse<ElT, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DiagBlockSparse(shape={:?}, nnzblocks={}, nnz={})",
            self.shape(),
            self.nnzblocks(),
            self.nnz()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::blocksparse::BlockDim;

    fn create_test_blockdims() -> BlockDims {
        // Square blocks for diagonal storage
        BlockDims::new(vec![
            BlockDim::new(vec![2, 3]), // dim 0: blocks of size 2, 3
            BlockDim::new(vec![2, 3]), // dim 1: blocks of size 2, 3
        ])
    }

    #[test]
    fn test_diagblocksparse_zeros() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let storage: CpuDiagBlockSparse<f64> = DiagBlockSparse::zeros(blocks, blockdims);

        assert_eq!(storage.ndim(), 2);
        assert_eq!(storage.shape(), vec![5, 5]); // 2+3, 2+3
        assert_eq!(storage.nnzblocks(), 2);
        // Block (0,0): shape [2,2], diag size = 2
        // Block (1,1): shape [3,3], diag size = 3
        assert_eq!(storage.nnz(), 2 + 3);

        // All elements should be zero
        for &val in storage.as_slice() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_diagblocksparse_from_vec() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];
        let blockoffsets = DiagBlockSparse::<f64>::compute_diag_blockoffsets(&blocks, &blockdims);

        // Block (0,0) has shape [2,2], so diag size = 2
        let data: Vec<f64> = vec![1.0, 2.0];
        let storage: CpuDiagBlockSparse<f64> =
            DiagBlockSparse::from_vec(data.clone(), blockoffsets, blockdims);

        assert_eq!(storage.nnz(), 2);
        assert_eq!(storage.as_slice(), &data[..]);
    }

    #[test]
    fn test_diagblocksparse_uniform() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let storage: CpuDiagBlockSparse<f64> = DiagBlockSparse::uniform(1.0, blocks, blockdims);

        assert_eq!(storage.nnz(), 5);
        assert!(storage.as_slice().iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_diagblocksparse_identity() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let storage: CpuDiagBlockSparse<f64> = DiagBlockSparse::identity(blocks, blockdims);

        assert_eq!(storage.nnz(), 5);
        assert!(storage.as_slice().iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_diagblocksparse_isblocknz() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let storage: CpuDiagBlockSparse<f64> = DiagBlockSparse::zeros(blocks, blockdims);

        assert!(storage.isblocknz(&Block::new(&[0, 0])));
        assert!(storage.isblocknz(&Block::new(&[1, 1])));
        assert!(!storage.isblocknz(&Block::new(&[0, 1])));
        assert!(!storage.isblocknz(&Block::new(&[1, 0])));
    }

    #[test]
    fn test_diagblocksparse_block_offset() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let storage: CpuDiagBlockSparse<f64> = DiagBlockSparse::zeros(blocks, blockdims);

        assert_eq!(storage.block_offset(&Block::new(&[0, 0])), Some(0));
        // Block (0,0) has 2 diagonal elements, so (1,1) starts at offset 2
        assert_eq!(storage.block_offset(&Block::new(&[1, 1])), Some(2));
        assert_eq!(storage.block_offset(&Block::new(&[0, 1])), None);
    }

    #[test]
    fn test_diagblocksparse_diag_size() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];

        let storage: CpuDiagBlockSparse<f64> = DiagBlockSparse::zeros(blocks, blockdims);

        // Block (0, 0): shape [2, 2], diag size = 2
        assert_eq!(storage.diag_size(&Block::new(&[0, 0])), 2);
        // Block (0, 1): shape [2, 3], diag size = 2
        assert_eq!(storage.diag_size(&Block::new(&[0, 1])), 2);
        // Block (1, 0): shape [3, 2], diag size = 2
        assert_eq!(storage.diag_size(&Block::new(&[1, 0])), 2);
        // Block (1, 1): shape [3, 3], diag size = 3
        assert_eq!(storage.diag_size(&Block::new(&[1, 1])), 3);
    }

    #[test]
    fn test_diagblocksparse_blockview() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];
        let blockoffsets = DiagBlockSparse::<f64>::compute_diag_blockoffsets(&blocks, &blockdims);

        // Create with specific data
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // 2 + 3 elements
        let storage: CpuDiagBlockSparse<f64> =
            DiagBlockSparse::from_vec(data, blockoffsets, blockdims);

        let view0 = storage.blockview(&Block::new(&[0, 0])).unwrap();
        assert_eq!(view0, &[1.0, 2.0]);

        let view1 = storage.blockview(&Block::new(&[1, 1])).unwrap();
        assert_eq!(view1, &[3.0, 4.0, 5.0]);

        // Non-existent block
        assert!(storage.blockview(&Block::new(&[0, 1])).is_none());
    }

    #[test]
    fn test_diagblocksparse_blockview_mut() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];

        let mut storage: CpuDiagBlockSparse<f64> = DiagBlockSparse::zeros(blocks, blockdims);

        // Modify block data
        {
            let view = storage.blockview_mut(&Block::new(&[0, 0])).unwrap();
            for (i, v) in view.iter_mut().enumerate() {
                *v = i as f64;
            }
        }

        // Verify modification
        let view = storage.blockview(&Block::new(&[0, 0])).unwrap();
        assert_eq!(view, &[0.0, 1.0]);
    }

    #[test]
    fn test_diagblocksparse_iter_blocks() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let storage: CpuDiagBlockSparse<f64> = DiagBlockSparse::identity(blocks.clone(), blockdims);

        let collected: Vec<_> = storage.iter_blocks().collect();
        assert_eq!(collected.len(), 2);

        // Check that we got both blocks
        let block_coords: Vec<_> = collected.iter().map(|(b, _)| b.coords().to_vec()).collect();
        assert!(block_coords.contains(&vec![0, 0]));
        assert!(block_coords.contains(&vec![1, 1]));
    }

    #[test]
    fn test_diagblocksparse_display() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let storage: CpuDiagBlockSparse<f64> = DiagBlockSparse::zeros(blocks, blockdims);
        let display = format!("{}", storage);

        assert!(display.contains("DiagBlockSparse"));
        assert!(display.contains("nnzblocks=2"));
        assert!(display.contains("nnz=5"));
    }

    #[test]
    fn test_diagblocksparse_equality() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];

        let storage1: CpuDiagBlockSparse<f64> =
            DiagBlockSparse::identity(blocks.clone(), blockdims.clone());
        let storage2: CpuDiagBlockSparse<f64> = DiagBlockSparse::identity(blocks, blockdims);

        assert_eq!(storage1, storage2);
    }

    #[test]
    fn test_diagblocksparse_complex() {
        use crate::scalar::c64;

        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];
        let blockoffsets = DiagBlockSparse::<c64>::compute_diag_blockoffsets(&blocks, &blockdims);

        let data: Vec<c64> = vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)];
        let storage: CpuDiagBlockSparse<c64> =
            DiagBlockSparse::from_vec(data, blockoffsets, blockdims);

        assert_eq!(storage.nnz(), 2);

        let view = storage.blockview(&Block::new(&[0, 0])).unwrap();
        assert_eq!(view[0], c64::new(1.0, 2.0));
        assert_eq!(view[1], c64::new(3.0, 4.0));
    }

    #[test]
    fn test_diagblocksparse_rectangular_blocks() {
        // Test with rectangular (non-square) blocks
        let blockdims = BlockDims::new(vec![
            BlockDim::new(vec![2, 4]), // dim 0
            BlockDim::new(vec![3, 5]), // dim 1
        ]);
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let storage: CpuDiagBlockSparse<f64> = DiagBlockSparse::zeros(blocks, blockdims);

        // Block (0,0): shape [2,3], diag size = min(2,3) = 2
        // Block (1,1): shape [4,5], diag size = min(4,5) = 4
        assert_eq!(storage.diag_size(&Block::new(&[0, 0])), 2);
        assert_eq!(storage.diag_size(&Block::new(&[1, 1])), 4);
        assert_eq!(storage.nnz(), 2 + 4);
    }

    #[test]
    fn test_diagblocksparse_nzblocks() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let storage: CpuDiagBlockSparse<f64> = DiagBlockSparse::zeros(blocks, blockdims);

        let nzblocks: Vec<_> = storage.nzblocks().collect();
        assert_eq!(nzblocks.len(), 2);
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn test_diagblocksparse_from_data_wrong_size() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];
        let blockoffsets = DiagBlockSparse::<f64>::compute_diag_blockoffsets(&blocks, &blockdims);

        // Wrong size data (should be 2, not 5)
        let data: Vec<f64> = vec![0.0; 5];
        let _storage: CpuDiagBlockSparse<f64> =
            DiagBlockSparse::from_vec(data, blockoffsets, blockdims);
    }
}
