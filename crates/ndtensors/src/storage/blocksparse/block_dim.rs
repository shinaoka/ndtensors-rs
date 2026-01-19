//! BlockDim type for block sizes in each dimension.
//!
//! A BlockDim represents the block structure of one dimension of a block-sparse tensor.
//! It mirrors NDTensors.jl's `BlockDim = Vector{Int}` type.

/// Block dimension structure for a single axis.
///
/// Stores block sizes and precomputes cumulative offsets for fast lookup.
///
/// # Example
/// ```
/// use ndtensors::storage::blocksparse::BlockDim;
///
/// // Create a dimension with blocks of sizes [2, 3, 4]
/// let dim = BlockDim::new(vec![2, 3, 4]);
/// assert_eq!(dim.nblocks(), 3);
/// assert_eq!(dim.total_size(), 9);  // 2 + 3 + 4
/// assert_eq!(dim.block_size(0), 2);
/// assert_eq!(dim.block_size(1), 3);
/// assert_eq!(dim.block_size(2), 4);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockDim {
    /// Size of each block
    block_sizes: Vec<usize>,
    /// Cumulative offsets: cumulative[i] = sum of block_sizes[0..i]
    cumulative: Vec<usize>,
    /// Total size (sum of all block sizes)
    total_size: usize,
}

impl BlockDim {
    /// Create a new BlockDim from block sizes.
    ///
    /// # Arguments
    /// * `block_sizes` - Vector of block sizes
    ///
    /// # Example
    /// ```
    /// use ndtensors::storage::blocksparse::BlockDim;
    ///
    /// let dim = BlockDim::new(vec![2, 3, 4]);
    /// assert_eq!(dim.nblocks(), 3);
    /// ```
    pub fn new(block_sizes: Vec<usize>) -> Self {
        let mut cumulative = Vec::with_capacity(block_sizes.len() + 1);
        cumulative.push(0);

        let mut total = 0usize;
        for &size in &block_sizes {
            total += size;
            cumulative.push(total);
        }

        Self {
            block_sizes,
            cumulative,
            total_size: total,
        }
    }

    /// Create a BlockDim with uniform block sizes.
    ///
    /// # Arguments
    /// * `nblocks` - Number of blocks
    /// * `block_size` - Size of each block
    ///
    /// # Example
    /// ```
    /// use ndtensors::storage::blocksparse::BlockDim;
    ///
    /// let dim = BlockDim::uniform(3, 4);  // 3 blocks of size 4
    /// assert_eq!(dim.nblocks(), 3);
    /// assert_eq!(dim.total_size(), 12);
    /// ```
    pub fn uniform(nblocks: usize, block_size: usize) -> Self {
        Self::new(vec![block_size; nblocks])
    }

    /// Get the number of blocks.
    #[inline]
    pub fn nblocks(&self) -> usize {
        self.block_sizes.len()
    }

    /// Get the total size (sum of all block sizes).
    #[inline]
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Get the size of a specific block.
    ///
    /// # Panics
    /// Panics if block_index is out of bounds.
    #[inline]
    pub fn block_size(&self, block_index: usize) -> usize {
        self.block_sizes[block_index]
    }

    /// Get the offset of a block (index where the block starts in dense storage).
    ///
    /// # Panics
    /// Panics if block_index is out of bounds.
    #[inline]
    pub fn block_offset(&self, block_index: usize) -> usize {
        self.cumulative[block_index]
    }

    /// Get all block sizes.
    #[inline]
    pub fn block_sizes(&self) -> &[usize] {
        &self.block_sizes
    }

    /// Get cumulative offsets.
    #[inline]
    pub fn cumulative_offsets(&self) -> &[usize] {
        &self.cumulative
    }

    /// Find which block contains a given index (in dense coordinates).
    ///
    /// Returns `(block_index, offset_within_block)`.
    ///
    /// # Panics
    /// Panics if index is >= total_size.
    ///
    /// # Example
    /// ```
    /// use ndtensors::storage::blocksparse::BlockDim;
    ///
    /// let dim = BlockDim::new(vec![2, 3, 4]);
    /// // Index 0, 1 are in block 0
    /// assert_eq!(dim.find_block(0), (0, 0));
    /// assert_eq!(dim.find_block(1), (0, 1));
    /// // Index 2, 3, 4 are in block 1
    /// assert_eq!(dim.find_block(2), (1, 0));
    /// assert_eq!(dim.find_block(4), (1, 2));
    /// // Index 5, 6, 7, 8 are in block 2
    /// assert_eq!(dim.find_block(5), (2, 0));
    /// ```
    pub fn find_block(&self, index: usize) -> (usize, usize) {
        assert!(
            index < self.total_size,
            "index {} out of bounds for BlockDim with total_size {}",
            index,
            self.total_size
        );

        // Binary search for the block
        let block_index = match self.cumulative[1..].binary_search(&index) {
            Ok(i) => i + 1, // Exactly at a boundary, belongs to next block
            Err(i) => i,    // Falls within block i
        };

        let offset_within_block = index - self.cumulative[block_index];
        (block_index, offset_within_block)
    }

    /// Check if block index is valid.
    #[inline]
    pub fn is_valid_block(&self, block_index: usize) -> bool {
        block_index < self.nblocks()
    }

    /// Permute block sizes according to a permutation.
    pub fn permute_blocks(&self, perm: &[usize]) -> Self {
        assert_eq!(
            perm.len(),
            self.nblocks(),
            "permutation length must match number of blocks"
        );
        let new_sizes: Vec<usize> = perm.iter().map(|&i| self.block_sizes[i]).collect();
        Self::new(new_sizes)
    }
}

impl std::fmt::Display for BlockDim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BlockDim({:?})", self.block_sizes)
    }
}

impl From<Vec<usize>> for BlockDim {
    fn from(block_sizes: Vec<usize>) -> Self {
        Self::new(block_sizes)
    }
}

impl<const N: usize> From<[usize; N]> for BlockDim {
    fn from(block_sizes: [usize; N]) -> Self {
        Self::new(block_sizes.to_vec())
    }
}

/// Block dimensions for a multi-dimensional block-sparse tensor.
///
/// This is a collection of `BlockDim` for each dimension.
///
/// # Example
/// ```
/// use ndtensors::storage::blocksparse::{BlockDim, BlockDims};
///
/// let dims = BlockDims::new(vec![
///     BlockDim::new(vec![2, 3]),    // Dimension 0: 2 blocks
///     BlockDim::new(vec![4, 5, 6]), // Dimension 1: 3 blocks
/// ]);
/// assert_eq!(dims.ndims(), 2);
/// assert_eq!(dims.nblocks().collect::<Vec<_>>(), vec![2, 3]);
/// assert_eq!(dims.dense_shape(), vec![5, 15]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockDims {
    dims: Vec<BlockDim>,
}

impl BlockDims {
    /// Create BlockDims from a vector of BlockDim.
    pub fn new(dims: Vec<BlockDim>) -> Self {
        Self { dims }
    }

    /// Get the number of dimensions.
    #[inline]
    pub fn ndims(&self) -> usize {
        self.dims.len()
    }

    /// Get the BlockDim for a specific dimension.
    #[inline]
    pub fn dim(&self, i: usize) -> &BlockDim {
        &self.dims[i]
    }

    /// Get the number of blocks in each dimension.
    pub fn nblocks(&self) -> impl Iterator<Item = usize> + '_ {
        self.dims.iter().map(|d| d.nblocks())
    }

    /// Get the total number of possible blocks (product of nblocks in each dim).
    pub fn total_nblocks(&self) -> usize {
        self.dims.iter().map(|d| d.nblocks()).product()
    }

    /// Get the dense shape (total size in each dimension).
    pub fn dense_shape(&self) -> Vec<usize> {
        self.dims.iter().map(|d| d.total_size()).collect()
    }

    /// Get the total dense size (product of all dimensions).
    pub fn dense_size(&self) -> usize {
        self.dims.iter().map(|d| d.total_size()).product()
    }

    /// Get the size of a specific block.
    ///
    /// # Arguments
    /// * `block_coords` - Block coordinates in each dimension
    ///
    /// # Returns
    /// Total number of elements in the block.
    pub fn block_size(&self, block_coords: &[usize]) -> usize {
        assert_eq!(
            block_coords.len(),
            self.ndims(),
            "block_coords length must match number of dimensions"
        );
        block_coords
            .iter()
            .zip(&self.dims)
            .map(|(&coord, dim)| dim.block_size(coord))
            .product()
    }

    /// Get the shape of a specific block.
    pub fn block_shape(&self, block_coords: &[usize]) -> Vec<usize> {
        assert_eq!(
            block_coords.len(),
            self.ndims(),
            "block_coords length must match number of dimensions"
        );
        block_coords
            .iter()
            .zip(&self.dims)
            .map(|(&coord, dim)| dim.block_size(coord))
            .collect()
    }

    /// Check if block coordinates are valid.
    pub fn is_valid_block(&self, block_coords: &[usize]) -> bool {
        if block_coords.len() != self.ndims() {
            return false;
        }
        block_coords
            .iter()
            .zip(&self.dims)
            .all(|(&coord, dim)| dim.is_valid_block(coord))
    }

    /// Get all BlockDims.
    #[inline]
    pub fn as_slice(&self) -> &[BlockDim] {
        &self.dims
    }
}

impl std::ops::Index<usize> for BlockDims {
    type Output = BlockDim;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.dims[index]
    }
}

impl IntoIterator for BlockDims {
    type Item = BlockDim;
    type IntoIter = std::vec::IntoIter<BlockDim>;

    fn into_iter(self) -> Self::IntoIter {
        self.dims.into_iter()
    }
}

impl<'a> IntoIterator for &'a BlockDims {
    type Item = &'a BlockDim;
    type IntoIter = std::slice::Iter<'a, BlockDim>;

    fn into_iter(self) -> Self::IntoIter {
        self.dims.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_block_dim_creation() {
        let dim = BlockDim::new(vec![2, 3, 4]);
        assert_eq!(dim.nblocks(), 3);
        assert_eq!(dim.total_size(), 9);
        assert_eq!(dim.block_sizes(), &[2, 3, 4]);
    }

    #[test]
    fn test_block_dim_uniform() {
        let dim = BlockDim::uniform(4, 5);
        assert_eq!(dim.nblocks(), 4);
        assert_eq!(dim.total_size(), 20);
        assert_eq!(dim.block_sizes(), &[5, 5, 5, 5]);
    }

    #[test]
    fn test_block_dim_offsets() {
        let dim = BlockDim::new(vec![2, 3, 4]);
        assert_eq!(dim.block_offset(0), 0);
        assert_eq!(dim.block_offset(1), 2);
        assert_eq!(dim.block_offset(2), 5);
        assert_eq!(dim.cumulative_offsets(), &[0, 2, 5, 9]);
    }

    #[test]
    fn test_block_dim_find_block() {
        let dim = BlockDim::new(vec![2, 3, 4]);

        // Block 0: indices 0, 1
        assert_eq!(dim.find_block(0), (0, 0));
        assert_eq!(dim.find_block(1), (0, 1));

        // Block 1: indices 2, 3, 4
        assert_eq!(dim.find_block(2), (1, 0));
        assert_eq!(dim.find_block(3), (1, 1));
        assert_eq!(dim.find_block(4), (1, 2));

        // Block 2: indices 5, 6, 7, 8
        assert_eq!(dim.find_block(5), (2, 0));
        assert_eq!(dim.find_block(6), (2, 1));
        assert_eq!(dim.find_block(7), (2, 2));
        assert_eq!(dim.find_block(8), (2, 3));
    }

    #[test]
    fn test_block_dim_permute() {
        let dim = BlockDim::new(vec![2, 3, 4]);
        let permuted = dim.permute_blocks(&[2, 0, 1]);
        assert_eq!(permuted.block_sizes(), &[4, 2, 3]);
    }

    #[test]
    fn test_block_dim_display() {
        let dim = BlockDim::new(vec![2, 3, 4]);
        assert_eq!(format!("{}", dim), "BlockDim([2, 3, 4])");
    }

    #[test]
    fn test_block_dims_creation() {
        let dims = BlockDims::new(vec![
            BlockDim::new(vec![2, 3]),
            BlockDim::new(vec![4, 5, 6]),
        ]);
        assert_eq!(dims.ndims(), 2);
        assert_eq!(dims.nblocks().collect::<Vec<_>>(), vec![2, 3]);
        assert_eq!(dims.total_nblocks(), 6);
    }

    #[test]
    fn test_block_dims_dense_shape() {
        let dims = BlockDims::new(vec![
            BlockDim::new(vec![2, 3]),
            BlockDim::new(vec![4, 5, 6]),
        ]);
        assert_eq!(dims.dense_shape(), vec![5, 15]);
        assert_eq!(dims.dense_size(), 75);
    }

    #[test]
    fn test_block_dims_block_size() {
        let dims = BlockDims::new(vec![
            BlockDim::new(vec![2, 3]),    // blocks of size 2, 3
            BlockDim::new(vec![4, 5, 6]), // blocks of size 4, 5, 6
        ]);

        // Block (0, 0): 2 * 4 = 8
        assert_eq!(dims.block_size(&[0, 0]), 8);
        // Block (0, 1): 2 * 5 = 10
        assert_eq!(dims.block_size(&[0, 1]), 10);
        // Block (1, 2): 3 * 6 = 18
        assert_eq!(dims.block_size(&[1, 2]), 18);
    }

    #[test]
    fn test_block_dims_block_shape() {
        let dims = BlockDims::new(vec![
            BlockDim::new(vec![2, 3]),
            BlockDim::new(vec![4, 5, 6]),
        ]);

        assert_eq!(dims.block_shape(&[0, 0]), vec![2, 4]);
        assert_eq!(dims.block_shape(&[1, 2]), vec![3, 6]);
    }

    #[test]
    fn test_block_dims_is_valid_block() {
        let dims = BlockDims::new(vec![
            BlockDim::new(vec![2, 3]),
            BlockDim::new(vec![4, 5, 6]),
        ]);

        assert!(dims.is_valid_block(&[0, 0]));
        assert!(dims.is_valid_block(&[1, 2]));
        assert!(!dims.is_valid_block(&[2, 0])); // Out of bounds in dim 0
        assert!(!dims.is_valid_block(&[0, 3])); // Out of bounds in dim 1
        assert!(!dims.is_valid_block(&[0])); // Wrong number of dimensions
    }

    #[test]
    fn test_block_dims_indexing() {
        let dims = BlockDims::new(vec![
            BlockDim::new(vec![2, 3]),
            BlockDim::new(vec![4, 5, 6]),
        ]);

        assert_eq!(dims[0].block_sizes(), &[2, 3]);
        assert_eq!(dims[1].block_sizes(), &[4, 5, 6]);
    }
}
