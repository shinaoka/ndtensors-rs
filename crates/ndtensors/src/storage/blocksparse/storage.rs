//! BlockSparse storage for block-sparse tensors.
//!
//! This module provides the `BlockSparse` storage type, which stores only
//! non-zero blocks of a tensor. This mirrors NDTensors.jl's
//! `BlockSparse{ElT, VecT, N}` type.

use std::marker::PhantomData;

use crate::scalar::Scalar;
use crate::storage::buffer::{CpuBuffer, DataBuffer};

use super::block::Block;
use super::block_dim::BlockDims;
use super::block_offsets::BlockOffsets;

/// Block-sparse storage for tensors.
///
/// Stores only non-zero blocks, which is efficient for tensors with
/// block-structured sparsity patterns (common in quantum physics and
/// tensor network applications).
///
/// # Type Parameters
///
/// - `ElT`: Element type (e.g., `f64`, `c64`)
/// - `D`: Data buffer type (e.g., `CpuBuffer<ElT>`)
///
/// # Example
///
/// ```
/// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims, BlockSparse};
///
/// // Define block structure: 2x2 blocks
/// let blockdims = BlockDims::new(vec![
///     BlockDim::new(vec![2, 3]),  // dim 0: blocks of size 2, 3
///     BlockDim::new(vec![4, 5]),  // dim 1: blocks of size 4, 5
/// ]);
///
/// // Create storage with blocks (0,0) and (1,1)
/// let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];
/// let storage: BlockSparse<f64> = BlockSparse::zeros(blocks, blockdims);
///
/// assert_eq!(storage.nnzblocks(), 2);
/// assert_eq!(storage.nnz(), 8 + 15); // 2*4 + 3*5
/// ```
#[derive(Clone, Debug)]
pub struct BlockSparse<ElT: Scalar, D: DataBuffer<ElT> = CpuBuffer<ElT>> {
    /// Flat storage for all non-zero blocks
    data: D,
    /// Mapping from block coordinates to offsets in data
    blockoffsets: BlockOffsets,
    /// Block dimensions for each tensor dimension
    blockdims: BlockDims,
    /// Phantom data for element type
    _phantom: PhantomData<ElT>,
}

/// Type alias for CPU-backed BlockSparse storage.
pub type CpuBlockSparse<ElT> = BlockSparse<ElT, CpuBuffer<ElT>>;

impl<ElT: Scalar, D: DataBuffer<ElT>> BlockSparse<ElT, D> {
    /// Create a new BlockSparse storage with the given blocks initialized to zero.
    ///
    /// # Arguments
    ///
    /// * `blocks` - List of non-zero block coordinates
    /// * `blockdims` - Block dimensions for each tensor dimension
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims, BlockSparse};
    ///
    /// let blockdims = BlockDims::new(vec![
    ///     BlockDim::new(vec![2, 3]),
    ///     BlockDim::new(vec![4, 5]),
    /// ]);
    /// let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];
    /// let storage: BlockSparse<f64> = BlockSparse::zeros(blocks, blockdims);
    /// ```
    pub fn zeros(blocks: Vec<Block>, blockdims: BlockDims) -> Self {
        let blockoffsets = BlockOffsets::from_blocks(&blocks, &blockdims);
        let data = D::zeros(blockoffsets.total_nnz());
        Self {
            data,
            blockoffsets,
            blockdims,
            _phantom: PhantomData,
        }
    }

    /// Create a new BlockSparse storage from existing data.
    ///
    /// # Arguments
    ///
    /// * `data` - Flat data buffer containing all non-zero blocks
    /// * `blockoffsets` - Mapping from blocks to offsets
    /// * `blockdims` - Block dimensions for each tensor dimension
    ///
    /// # Panics
    ///
    /// Panics if data length doesn't match total_nnz from blockoffsets.
    pub fn from_data(data: D, blockoffsets: BlockOffsets, blockdims: BlockDims) -> Self {
        assert_eq!(
            data.len(),
            blockoffsets.total_nnz(),
            "data length ({}) must match total_nnz ({})",
            data.len(),
            blockoffsets.total_nnz()
        );
        Self {
            data,
            blockoffsets,
            blockdims,
            _phantom: PhantomData,
        }
    }

    /// Create a new BlockSparse storage from a Vec.
    ///
    /// # Arguments
    ///
    /// * `data` - Vec containing all non-zero block data
    /// * `blockoffsets` - Mapping from blocks to offsets
    /// * `blockdims` - Block dimensions for each tensor dimension
    pub fn from_vec(data: Vec<ElT>, blockoffsets: BlockOffsets, blockdims: BlockDims) -> Self {
        Self::from_data(D::from_vec(data), blockoffsets, blockdims)
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

    /// Get the number of non-zero elements.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.blockoffsets.total_nnz()
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

    /// Get the offset of a block in the flat storage.
    ///
    /// Returns `None` if the block is not present.
    #[inline]
    pub fn block_offset(&self, block: &Block) -> Option<usize> {
        self.blockoffsets.get(block)
    }

    /// Get the size of a block (number of elements).
    ///
    /// This works for any valid block coordinates, whether or not the block is present.
    #[inline]
    pub fn block_size(&self, block: &Block) -> usize {
        self.blockdims.block_size(block.coords())
    }

    /// Get the shape of a block.
    ///
    /// This works for any valid block coordinates, whether or not the block is present.
    #[inline]
    pub fn block_shape(&self, block: &Block) -> Vec<usize> {
        self.blockdims.block_shape(block.coords())
    }

    /// Get an immutable view of a block's data as a slice.
    ///
    /// Returns `None` if the block is not present (structurally zero).
    ///
    /// # Arguments
    ///
    /// * `block` - Block coordinates
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims, BlockSparse};
    ///
    /// let blockdims = BlockDims::new(vec![
    ///     BlockDim::new(vec![2, 3]),
    ///     BlockDim::new(vec![4, 5]),
    /// ]);
    /// let blocks = vec![Block::new(&[0, 0])];
    /// let storage: BlockSparse<f64> = BlockSparse::zeros(blocks, blockdims);
    ///
    /// let view = storage.blockview(&Block::new(&[0, 0]));
    /// assert!(view.is_some());
    /// assert_eq!(view.unwrap().len(), 8); // 2 * 4
    ///
    /// let view_missing = storage.blockview(&Block::new(&[1, 1]));
    /// assert!(view_missing.is_none());
    /// ```
    pub fn blockview(&self, block: &Block) -> Option<&[ElT]> {
        let offset = self.blockoffsets.get(block)?;
        let size = self.blockdims.block_size(block.coords());
        Some(&self.data.as_slice()[offset..offset + size])
    }

    /// Get a mutable view of a block's data as a slice.
    ///
    /// Returns `None` if the block is not present (structurally zero).
    ///
    /// # Arguments
    ///
    /// * `block` - Block coordinates
    pub fn blockview_mut(&mut self, block: &Block) -> Option<&mut [ElT]> {
        let offset = self.blockoffsets.get(block)?;
        let size = self.blockdims.block_size(block.coords());
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

    /// Iterate over all non-zero blocks with their data.
    ///
    /// Returns an iterator of (Block, &[ElT]) pairs.
    pub fn iter_blocks(&self) -> impl Iterator<Item = (&Block, &[ElT])> {
        self.blockoffsets.iter().map(move |(block, &offset)| {
            let size = self.blockdims.block_size(block.coords());
            let data = &self.data.as_slice()[offset..offset + size];
            (block, data)
        })
    }

    /// Check if the block coordinates are valid for this storage.
    #[inline]
    pub fn is_valid_block(&self, block: &Block) -> bool {
        self.blockdims.is_valid_block(block.coords())
    }
}

impl<ElT: Scalar, D: DataBuffer<ElT>> PartialEq for BlockSparse<ElT, D> {
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

impl<ElT: Scalar, D: DataBuffer<ElT>> std::fmt::Display for BlockSparse<ElT, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BlockSparse(shape={:?}, nnzblocks={}, nnz={})",
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
        BlockDims::new(vec![
            BlockDim::new(vec![2, 3]), // dim 0: blocks of size 2, 3
            BlockDim::new(vec![4, 5]), // dim 1: blocks of size 4, 5
        ])
    }

    #[test]
    fn test_blocksparse_zeros() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let storage: CpuBlockSparse<f64> = BlockSparse::zeros(blocks, blockdims);

        assert_eq!(storage.ndim(), 2);
        assert_eq!(storage.shape(), vec![5, 9]); // 2+3, 4+5
        assert_eq!(storage.nnzblocks(), 2);
        assert_eq!(storage.nnz(), 8 + 15); // 2*4 + 3*5

        // All elements should be zero
        for &val in storage.as_slice() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_blocksparse_from_vec() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];
        let blockoffsets = BlockOffsets::from_blocks(&blocks, &blockdims);

        let data: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let storage: CpuBlockSparse<f64> =
            BlockSparse::from_vec(data.clone(), blockoffsets, blockdims);

        assert_eq!(storage.nnz(), 8);
        assert_eq!(storage.as_slice(), &data[..]);
    }

    #[test]
    fn test_blocksparse_isblocknz() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let storage: CpuBlockSparse<f64> = BlockSparse::zeros(blocks, blockdims);

        assert!(storage.isblocknz(&Block::new(&[0, 0])));
        assert!(storage.isblocknz(&Block::new(&[1, 1])));
        assert!(!storage.isblocknz(&Block::new(&[0, 1])));
        assert!(!storage.isblocknz(&Block::new(&[1, 0])));
    }

    #[test]
    fn test_blocksparse_block_offset() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let storage: CpuBlockSparse<f64> = BlockSparse::zeros(blocks, blockdims);

        assert_eq!(storage.block_offset(&Block::new(&[0, 0])), Some(0));
        assert_eq!(storage.block_offset(&Block::new(&[1, 1])), Some(8)); // After first block
        assert_eq!(storage.block_offset(&Block::new(&[0, 1])), None);
    }

    #[test]
    fn test_blocksparse_block_size() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];

        let storage: CpuBlockSparse<f64> = BlockSparse::zeros(blocks, blockdims);

        // Block (0, 0): 2 * 4 = 8
        assert_eq!(storage.block_size(&Block::new(&[0, 0])), 8);
        // Block (0, 1): 2 * 5 = 10
        assert_eq!(storage.block_size(&Block::new(&[0, 1])), 10);
        // Block (1, 0): 3 * 4 = 12
        assert_eq!(storage.block_size(&Block::new(&[1, 0])), 12);
        // Block (1, 1): 3 * 5 = 15
        assert_eq!(storage.block_size(&Block::new(&[1, 1])), 15);
    }

    #[test]
    fn test_blocksparse_block_shape() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];

        let storage: CpuBlockSparse<f64> = BlockSparse::zeros(blocks, blockdims);

        assert_eq!(storage.block_shape(&Block::new(&[0, 0])), vec![2, 4]);
        assert_eq!(storage.block_shape(&Block::new(&[1, 1])), vec![3, 5]);
    }

    #[test]
    fn test_blocksparse_blockview() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];
        let blockoffsets = BlockOffsets::from_blocks(&blocks, &blockdims);

        // Create with specific data
        let mut data: Vec<f64> = vec![0.0; blockoffsets.total_nnz()];
        // Fill first block with 1.0
        for val in data.iter_mut().take(8) {
            *val = 1.0;
        }
        // Fill second block with 2.0
        for val in data.iter_mut().skip(8).take(15) {
            *val = 2.0;
        }

        let storage: CpuBlockSparse<f64> = BlockSparse::from_vec(data, blockoffsets, blockdims);

        let view0 = storage.blockview(&Block::new(&[0, 0])).unwrap();
        assert_eq!(view0.len(), 8);
        assert!(view0.iter().all(|&v| v == 1.0));

        let view1 = storage.blockview(&Block::new(&[1, 1])).unwrap();
        assert_eq!(view1.len(), 15);
        assert!(view1.iter().all(|&v| v == 2.0));

        // Non-existent block
        assert!(storage.blockview(&Block::new(&[0, 1])).is_none());
    }

    #[test]
    fn test_blocksparse_blockview_mut() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];

        let mut storage: CpuBlockSparse<f64> = BlockSparse::zeros(blocks, blockdims);

        // Modify block data
        {
            let view = storage.blockview_mut(&Block::new(&[0, 0])).unwrap();
            for (i, v) in view.iter_mut().enumerate() {
                *v = i as f64;
            }
        }

        // Verify modification
        let view = storage.blockview(&Block::new(&[0, 0])).unwrap();
        for (i, &v) in view.iter().enumerate() {
            assert_eq!(v, i as f64);
        }
    }

    #[test]
    fn test_blocksparse_iter_blocks() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let storage: CpuBlockSparse<f64> = BlockSparse::zeros(blocks.clone(), blockdims);

        let collected: Vec<_> = storage.iter_blocks().collect();
        assert_eq!(collected.len(), 2);

        // Check that we got both blocks
        let block_coords: Vec<_> = collected.iter().map(|(b, _)| b.coords().to_vec()).collect();
        assert!(block_coords.contains(&vec![0, 0]));
        assert!(block_coords.contains(&vec![1, 1]));
    }

    #[test]
    fn test_blocksparse_is_valid_block() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];

        let storage: CpuBlockSparse<f64> = BlockSparse::zeros(blocks, blockdims);

        assert!(storage.is_valid_block(&Block::new(&[0, 0])));
        assert!(storage.is_valid_block(&Block::new(&[0, 1])));
        assert!(storage.is_valid_block(&Block::new(&[1, 0])));
        assert!(storage.is_valid_block(&Block::new(&[1, 1])));
        assert!(!storage.is_valid_block(&Block::new(&[2, 0]))); // Out of bounds
        assert!(!storage.is_valid_block(&Block::new(&[0, 2]))); // Out of bounds
    }

    #[test]
    fn test_blocksparse_display() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let storage: CpuBlockSparse<f64> = BlockSparse::zeros(blocks, blockdims);
        let display = format!("{}", storage);

        assert!(display.contains("BlockSparse"));
        assert!(display.contains("nnzblocks=2"));
        assert!(display.contains("nnz=23"));
    }

    #[test]
    fn test_blocksparse_equality() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];

        let storage1: CpuBlockSparse<f64> = BlockSparse::zeros(blocks.clone(), blockdims.clone());
        let storage2: CpuBlockSparse<f64> = BlockSparse::zeros(blocks, blockdims);

        assert_eq!(storage1, storage2);
    }

    #[test]
    fn test_blocksparse_complex() {
        use crate::scalar::c64;

        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];
        let blockoffsets = BlockOffsets::from_blocks(&blocks, &blockdims);

        let data: Vec<c64> = (0..8).map(|i| c64::new(i as f64, -(i as f64))).collect();
        let storage: CpuBlockSparse<c64> = BlockSparse::from_vec(data, blockoffsets, blockdims);

        assert_eq!(storage.nnz(), 8);

        let view = storage.blockview(&Block::new(&[0, 0])).unwrap();
        assert_eq!(view[0], c64::new(0.0, 0.0));
        assert_eq!(view[1], c64::new(1.0, -1.0));
    }

    #[test]
    #[should_panic(expected = "data length")]
    fn test_blocksparse_from_data_wrong_size() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];
        let blockoffsets = BlockOffsets::from_blocks(&blocks, &blockdims);

        // Wrong size data
        let data: Vec<f64> = vec![0.0; 5]; // Should be 8
        let _storage: CpuBlockSparse<f64> = BlockSparse::from_vec(data, blockoffsets, blockdims);
    }
}
