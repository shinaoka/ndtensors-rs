//! BlockOffsets for mapping blocks to storage offsets.
//!
//! Provides O(1) lookup from block coordinates to data offsets in the flat storage.

use std::collections::HashMap;

use super::block::Block;
use super::block_dim::BlockDims;

/// Maps blocks to their offsets in flat storage.
///
/// Uses HashMap with Block's precomputed hash for O(1) lookup.
///
/// # Example
/// ```
/// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims, BlockOffsets};
///
/// let dims = BlockDims::new(vec![
///     BlockDim::new(vec![2, 3]),
///     BlockDim::new(vec![4, 5]),
/// ]);
///
/// // Create offsets for blocks (0,0), (0,1), (1,1)
/// let blocks = vec![
///     Block::new(&[0, 0]),
///     Block::new(&[0, 1]),
///     Block::new(&[1, 1]),
/// ];
/// let offsets = BlockOffsets::from_blocks(&blocks, &dims);
///
/// assert_eq!(offsets.nnzblocks(), 3);
/// assert_eq!(offsets.get(&Block::new(&[0, 0])), Some(0));
/// ```
#[derive(Clone, Debug)]
pub struct BlockOffsets {
    /// Map from Block to offset in flat storage
    offsets: HashMap<Block, usize>,
    /// Total number of non-zero elements
    total_nnz: usize,
}

impl BlockOffsets {
    /// Create an empty BlockOffsets.
    pub fn new() -> Self {
        Self {
            offsets: HashMap::new(),
            total_nnz: 0,
        }
    }

    /// Create BlockOffsets from a list of blocks and block dimensions.
    ///
    /// Blocks are stored in the order given, with offsets computed sequentially.
    ///
    /// # Arguments
    /// * `blocks` - List of blocks to include
    /// * `dims` - Block dimensions for computing block sizes
    ///
    /// # Example
    /// ```
    /// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims, BlockOffsets};
    ///
    /// let dims = BlockDims::new(vec![
    ///     BlockDim::new(vec![2, 3]),
    ///     BlockDim::new(vec![4, 5]),
    /// ]);
    /// let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];
    /// let offsets = BlockOffsets::from_blocks(&blocks, &dims);
    ///
    /// // Block (0,0) has size 2*4 = 8, starts at 0
    /// assert_eq!(offsets.get(&Block::new(&[0, 0])), Some(0));
    /// // Block (1,1) has size 3*5 = 15, starts at 8
    /// assert_eq!(offsets.get(&Block::new(&[1, 1])), Some(8));
    /// assert_eq!(offsets.total_nnz(), 23); // 8 + 15
    /// ```
    pub fn from_blocks(blocks: &[Block], dims: &BlockDims) -> Self {
        let mut offsets = HashMap::with_capacity(blocks.len());
        let mut current_offset = 0;

        for block in blocks {
            offsets.insert(block.clone(), current_offset);
            let block_size = dims.block_size(block.coords());
            current_offset += block_size;
        }

        Self {
            offsets,
            total_nnz: current_offset,
        }
    }

    /// Create BlockOffsets with explicit offsets.
    ///
    /// # Arguments
    /// * `block_offsets` - Iterator of (Block, offset) pairs
    /// * `total_nnz` - Total number of non-zero elements
    pub fn from_iter<I>(block_offsets: I, total_nnz: usize) -> Self
    where
        I: IntoIterator<Item = (Block, usize)>,
    {
        Self {
            offsets: block_offsets.into_iter().collect(),
            total_nnz,
        }
    }

    /// Get the offset for a block.
    ///
    /// Returns `None` if the block is not present.
    #[inline]
    pub fn get(&self, block: &Block) -> Option<usize> {
        self.offsets.get(block).copied()
    }

    /// Check if a block is present.
    #[inline]
    pub fn contains(&self, block: &Block) -> bool {
        self.offsets.contains_key(block)
    }

    /// Get the number of non-zero blocks.
    #[inline]
    pub fn nnzblocks(&self) -> usize {
        self.offsets.len()
    }

    /// Get the total number of non-zero elements.
    #[inline]
    pub fn total_nnz(&self) -> usize {
        self.total_nnz
    }

    /// Check if there are no blocks.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// Iterate over all (block, offset) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&Block, &usize)> {
        self.offsets.iter()
    }

    /// Iterate over all blocks.
    pub fn blocks(&self) -> impl Iterator<Item = &Block> {
        self.offsets.keys()
    }

    /// Get all blocks as a sorted vector.
    pub fn sorted_blocks(&self) -> Vec<Block> {
        let mut blocks: Vec<_> = self.offsets.keys().cloned().collect();
        blocks.sort();
        blocks
    }

    /// Insert a block with its offset.
    ///
    /// Returns the previous offset if the block was already present.
    pub fn insert(&mut self, block: Block, offset: usize) -> Option<usize> {
        self.offsets.insert(block, offset)
    }

    /// Set the total nnz value.
    pub fn set_total_nnz(&mut self, total_nnz: usize) {
        self.total_nnz = total_nnz;
    }

    /// Get the size of a block given its position in storage.
    ///
    /// This requires knowing the offset of the next block or total_nnz.
    pub fn block_size_at(&self, block: &Block, dims: &BlockDims) -> Option<usize> {
        if self.contains(block) {
            Some(dims.block_size(block.coords()))
        } else {
            None
        }
    }

    /// Create a permuted version of BlockOffsets.
    ///
    /// # Arguments
    /// * `perm` - Permutation to apply to block coordinates
    /// * `dims` - Original block dimensions
    ///
    /// # Returns
    /// New BlockOffsets with permuted blocks and recomputed offsets.
    pub fn permute(&self, perm: &[usize], dims: &BlockDims) -> (Self, BlockDims) {
        // Permute the dimensions
        let new_dim_vec: Vec<_> = perm.iter().map(|&i| dims.dim(i).clone()).collect();
        let new_dims = BlockDims::new(new_dim_vec);

        // Permute each block and collect
        let permuted_blocks: Vec<Block> = self.blocks().map(|b| b.permute(perm)).collect();

        // Sort blocks and recompute offsets
        let mut sorted_blocks = permuted_blocks;
        sorted_blocks.sort();

        let new_offsets = Self::from_blocks(&sorted_blocks, &new_dims);
        (new_offsets, new_dims)
    }
}

impl Default for BlockOffsets {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for BlockOffsets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BlockOffsets(nnzblocks={}, total_nnz={})",
            self.nnzblocks(),
            self.total_nnz
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::blocksparse::BlockDim;

    fn create_test_dims() -> BlockDims {
        BlockDims::new(vec![
            BlockDim::new(vec![2, 3]), // blocks of size 2, 3
            BlockDim::new(vec![4, 5]), // blocks of size 4, 5
        ])
    }

    #[test]
    fn test_block_offsets_empty() {
        let offsets = BlockOffsets::new();
        assert_eq!(offsets.nnzblocks(), 0);
        assert_eq!(offsets.total_nnz(), 0);
        assert!(offsets.is_empty());
    }

    #[test]
    fn test_block_offsets_from_blocks() {
        let dims = create_test_dims();
        let blocks = vec![
            Block::new(&[0, 0]), // size: 2 * 4 = 8
            Block::new(&[0, 1]), // size: 2 * 5 = 10
            Block::new(&[1, 0]), // size: 3 * 4 = 12
        ];

        let offsets = BlockOffsets::from_blocks(&blocks, &dims);

        assert_eq!(offsets.nnzblocks(), 3);
        assert_eq!(offsets.total_nnz(), 30); // 8 + 10 + 12

        assert_eq!(offsets.get(&Block::new(&[0, 0])), Some(0));
        assert_eq!(offsets.get(&Block::new(&[0, 1])), Some(8));
        assert_eq!(offsets.get(&Block::new(&[1, 0])), Some(18));
    }

    #[test]
    fn test_block_offsets_contains() {
        let dims = create_test_dims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let offsets = BlockOffsets::from_blocks(&blocks, &dims);

        assert!(offsets.contains(&Block::new(&[0, 0])));
        assert!(offsets.contains(&Block::new(&[1, 1])));
        assert!(!offsets.contains(&Block::new(&[0, 1])));
        assert!(!offsets.contains(&Block::new(&[1, 0])));
    }

    #[test]
    fn test_block_offsets_iteration() {
        let dims = create_test_dims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let offsets = BlockOffsets::from_blocks(&blocks, &dims);

        let collected: Vec<_> = offsets.blocks().collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn test_block_offsets_sorted_blocks() {
        let dims = create_test_dims();
        // Insert in non-sorted order
        let blocks = vec![
            Block::new(&[1, 1]),
            Block::new(&[0, 0]),
            Block::new(&[1, 0]),
        ];

        let offsets = BlockOffsets::from_blocks(&blocks, &dims);
        let sorted = offsets.sorted_blocks();

        assert_eq!(sorted[0], Block::new(&[0, 0]));
        assert_eq!(sorted[1], Block::new(&[1, 0]));
        assert_eq!(sorted[2], Block::new(&[1, 1]));
    }

    #[test]
    fn test_block_offsets_insert() {
        let mut offsets = BlockOffsets::new();

        assert!(offsets.insert(Block::new(&[0, 0]), 0).is_none());
        assert!(offsets.insert(Block::new(&[1, 1]), 10).is_none());

        // Inserting same block returns previous value
        assert_eq!(offsets.insert(Block::new(&[0, 0]), 5), Some(0));

        assert_eq!(offsets.get(&Block::new(&[0, 0])), Some(5));
    }

    #[test]
    fn test_block_offsets_block_size_at() {
        let dims = create_test_dims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let offsets = BlockOffsets::from_blocks(&blocks, &dims);

        // Block (0, 0): 2 * 4 = 8
        assert_eq!(offsets.block_size_at(&Block::new(&[0, 0]), &dims), Some(8));
        // Block (1, 1): 3 * 5 = 15
        assert_eq!(offsets.block_size_at(&Block::new(&[1, 1]), &dims), Some(15));
        // Non-existent block
        assert_eq!(offsets.block_size_at(&Block::new(&[0, 1]), &dims), None);
    }

    #[test]
    fn test_block_offsets_display() {
        let dims = create_test_dims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let offsets = BlockOffsets::from_blocks(&blocks, &dims);
        let display = format!("{}", offsets);

        assert!(display.contains("nnzblocks=2"));
        assert!(display.contains("total_nnz=23")); // 8 + 15
    }

    #[test]
    fn test_block_offsets_permute() {
        let dims = BlockDims::new(vec![
            BlockDim::new(vec![2, 3]), // dim 0
            BlockDim::new(vec![4, 5]), // dim 1
        ]);

        let blocks = vec![
            Block::new(&[0, 0]), // size: 2 * 4 = 8
            Block::new(&[1, 1]), // size: 3 * 5 = 15
        ];

        let offsets = BlockOffsets::from_blocks(&blocks, &dims);

        // Permute: swap dimensions
        let (permuted_offsets, permuted_dims) = offsets.permute(&[1, 0], &dims);

        // After permutation: (0, 0) -> (0, 0), (1, 1) -> (1, 1)
        // Dimensions swapped: dim 0 is now [4, 5], dim 1 is now [2, 3]
        assert_eq!(permuted_dims.dim(0).block_sizes(), &[4, 5]);
        assert_eq!(permuted_dims.dim(1).block_sizes(), &[2, 3]);

        // Block (0, 0) permuted is still (0, 0) with size 4 * 2 = 8
        assert!(permuted_offsets.contains(&Block::new(&[0, 0])));
        // Block (1, 1) permuted is still (1, 1) with size 5 * 3 = 15
        assert!(permuted_offsets.contains(&Block::new(&[1, 1])));
    }

    #[test]
    fn test_block_offsets_from_iter() {
        let pairs = vec![(Block::new(&[0, 0]), 0), (Block::new(&[1, 1]), 8)];

        let offsets = BlockOffsets::from_iter(pairs, 23);

        assert_eq!(offsets.nnzblocks(), 2);
        assert_eq!(offsets.total_nnz(), 23);
        assert_eq!(offsets.get(&Block::new(&[0, 0])), Some(0));
        assert_eq!(offsets.get(&Block::new(&[1, 1])), Some(8));
    }
}
