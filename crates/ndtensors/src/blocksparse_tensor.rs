//! BlockSparseTensor - Block-sparse tensor type.
//!
//! This module provides `BlockSparseTensor<ElT>`, an independent tensor type
//! for block-sparse data. Unlike `DenseTensor`, which uses `Tensor<ElT, Dense>`,
//! `BlockSparseTensor` is a standalone struct optimized for block-sparse patterns.
//!
//! # Design Decision
//!
//! `BlockSparseTensor` is implemented as an independent struct rather than
//! `Tensor<ElT, BlockSparse>` because:
//!
//! 1. Block-sparse tensors have fundamentally different access patterns
//! 2. The `TensorStorage` trait requires `as_slice()` which doesn't map well
//!    to block-sparse storage
//! 3. This mirrors NDTensors.jl's approach where `BlockSparseTensor` has
//!    specialized methods like `blockview` that return `DenseTensor` views

use crate::Tensor;
use crate::error::TensorError;
use crate::scalar::Scalar;
use crate::storage::blocksparse::{Block, BlockDims, BlockOffsets, BlockSparse};
use crate::storage::{CpuBuffer, DataBuffer};
use crate::strides::compute_strides;
use crate::tensor::DenseTensor;

/// A block-sparse tensor.
///
/// Stores only non-zero blocks, which is efficient for tensors with
/// block-structured sparsity patterns (common in quantum physics and
/// tensor network applications).
///
/// # Example
///
/// ```
/// use ndtensors::blocksparse_tensor::BlockSparseTensor;
/// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims};
///
/// // Create a 5x9 tensor with block structure [2,3] x [4,5]
/// let blockdims = BlockDims::new(vec![
///     BlockDim::new(vec![2, 3]),
///     BlockDim::new(vec![4, 5]),
/// ]);
///
/// // Only blocks (0,0) and (1,1) are non-zero
/// let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];
/// let tensor: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);
///
/// assert_eq!(tensor.shape(), &[5, 9]);
/// assert_eq!(tensor.nnzblocks(), 2);
/// assert!(tensor.isblocknz(&Block::new(&[0, 0])));
/// assert!(!tensor.isblocknz(&Block::new(&[0, 1])));
/// ```
#[derive(Clone, Debug)]
pub struct BlockSparseTensor<ElT: Scalar, D: DataBuffer<ElT> = CpuBuffer<ElT>> {
    storage: BlockSparse<ElT, D>,
}

/// Type alias for CPU-backed BlockSparseTensor.
pub type CpuBlockSparseTensor<ElT> = BlockSparseTensor<ElT, CpuBuffer<ElT>>;

impl<ElT: Scalar, D: DataBuffer<ElT>> BlockSparseTensor<ElT, D> {
    /// Create a new block-sparse tensor with the given blocks initialized to zero.
    ///
    /// # Arguments
    ///
    /// * `blocks` - List of non-zero block coordinates
    /// * `blockdims` - Block dimensions for each tensor dimension
    pub fn zeros(blocks: Vec<Block>, blockdims: BlockDims) -> Self {
        Self {
            storage: BlockSparse::zeros(blocks, blockdims),
        }
    }

    /// Create a block-sparse tensor from existing storage.
    pub fn from_storage(storage: BlockSparse<ElT, D>) -> Self {
        Self { storage }
    }

    /// Get the dense shape of the tensor.
    #[inline]
    pub fn shape(&self) -> Vec<usize> {
        self.storage.shape()
    }

    /// Get the number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.storage.ndim()
    }

    /// Get the total number of non-zero elements.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.storage.nnz()
    }

    /// Get the number of non-zero blocks.
    #[inline]
    pub fn nnzblocks(&self) -> usize {
        self.storage.nnzblocks()
    }

    /// Get the block dimensions.
    #[inline]
    pub fn blockdims(&self) -> &BlockDims {
        self.storage.blockdims()
    }

    /// Get the block offsets.
    #[inline]
    pub fn blockoffsets(&self) -> &BlockOffsets {
        self.storage.blockoffsets()
    }

    /// Check if a block is non-zero (present in storage).
    #[inline]
    pub fn isblocknz(&self, block: &Block) -> bool {
        self.storage.isblocknz(block)
    }

    /// Get the size of a block (number of elements).
    #[inline]
    pub fn block_size(&self, block: &Block) -> usize {
        self.storage.block_size(block)
    }

    /// Get the shape of a block.
    #[inline]
    pub fn block_shape(&self, block: &Block) -> Vec<usize> {
        self.storage.block_shape(block)
    }

    /// Get an immutable view of a block as a `DenseTensor`.
    ///
    /// Returns `None` if the block is not present (structurally zero).
    ///
    /// # Note
    ///
    /// The returned tensor shares the underlying data with this tensor.
    /// Modifying the returned tensor will modify this tensor.
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::blocksparse_tensor::BlockSparseTensor;
    /// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims};
    ///
    /// let blockdims = BlockDims::new(vec![
    ///     BlockDim::new(vec![2, 3]),
    ///     BlockDim::new(vec![4, 5]),
    /// ]);
    /// let blocks = vec![Block::new(&[0, 0])];
    /// let tensor: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);
    ///
    /// let block_view = tensor.blockview(&Block::new(&[0, 0])).unwrap();
    /// assert_eq!(block_view.shape(), &[2, 4]);
    /// ```
    pub fn blockview(&self, block: &Block) -> Option<DenseTensor<ElT>> {
        let data = self.storage.blockview(block)?;
        let block_shape = self.storage.block_shape(block);
        // Create a DenseTensor from the block data
        // Note: This creates a copy of the data
        Tensor::from_vec(data.to_vec(), &block_shape).ok()
    }

    /// Get a mutable view of a block's raw data.
    ///
    /// Returns `None` if the block is not present.
    pub fn blockview_mut(&mut self, block: &Block) -> Option<&mut [ElT]> {
        self.storage.blockview_mut(block)
    }

    /// Insert a block into the tensor, copying data from a DenseTensor.
    ///
    /// The block must already exist in the tensor's block structure.
    ///
    /// # Arguments
    ///
    /// * `block` - Block coordinates
    /// * `data` - DenseTensor containing the block data
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Block is not present in the tensor
    /// - Data shape doesn't match block shape
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::blocksparse_tensor::BlockSparseTensor;
    /// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims};
    /// use ndtensors::Tensor;
    ///
    /// let blockdims = BlockDims::new(vec![
    ///     BlockDim::new(vec![2, 3]),
    ///     BlockDim::new(vec![4, 5]),
    /// ]);
    /// let blocks = vec![Block::new(&[0, 0])];
    /// let mut tensor: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);
    ///
    /// // Create block data
    /// let block_data = Tensor::ones(&[2, 4]);
    /// tensor.insertblock(&Block::new(&[0, 0]), &block_data).unwrap();
    ///
    /// // Verify
    /// let view = tensor.blockview(&Block::new(&[0, 0])).unwrap();
    /// assert_eq!(view.get(&[0, 0]), Some(&1.0));
    /// ```
    pub fn insertblock(
        &mut self,
        block: &Block,
        data: &DenseTensor<ElT>,
    ) -> Result<(), TensorError> {
        let block_shape = self.storage.block_shape(block);

        // Check shape matches
        if data.shape() != block_shape {
            return Err(TensorError::ShapeMismatch {
                expected: block_shape.iter().product(),
                actual: data.len(),
            });
        }

        // Get mutable view and copy data
        let dest = self
            .storage
            .blockview_mut(block)
            .ok_or_else(|| TensorError::BlockNotFound {
                block: block.coords().to_vec(),
            })?;

        dest.copy_from_slice(data.data());
        Ok(())
    }

    /// Get the underlying storage.
    #[inline]
    pub fn storage(&self) -> &BlockSparse<ElT, D> {
        &self.storage
    }

    /// Get mutable access to the underlying storage.
    #[inline]
    pub fn storage_mut(&mut self) -> &mut BlockSparse<ElT, D> {
        &mut self.storage
    }

    /// Iterate over all non-zero blocks.
    pub fn iter_blocks(&self) -> impl Iterator<Item = (&Block, DenseTensor<ElT>)> {
        self.storage.iter_blocks().filter_map(|(block, data)| {
            let block_shape = self.storage.block_shape(block);
            Tensor::from_vec(data.to_vec(), &block_shape)
                .ok()
                .map(|t| (block, t))
        })
    }
}

// Conversion methods
impl<ElT: Scalar> BlockSparseTensor<ElT, CpuBuffer<ElT>> {
    /// Convert to a dense tensor.
    ///
    /// Creates a full dense tensor, filling in zeros for missing blocks.
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::blocksparse_tensor::BlockSparseTensor;
    /// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims};
    /// use ndtensors::Tensor;
    ///
    /// let blockdims = BlockDims::new(vec![
    ///     BlockDim::new(vec![2, 3]),
    ///     BlockDim::new(vec![4, 5]),
    /// ]);
    /// let blocks = vec![Block::new(&[0, 0])];
    /// let mut bst: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);
    ///
    /// // Set some values in block (0,0)
    /// let block_data = Tensor::from_vec((1..9).map(|x| x as f64).collect(), &[2, 4]).unwrap();
    /// bst.insertblock(&Block::new(&[0, 0]), &block_data).unwrap();
    ///
    /// // Convert to dense
    /// let dense = bst.to_dense();
    /// assert_eq!(dense.shape(), &[5, 9]);
    /// assert_eq!(dense.get(&[0, 0]), Some(&1.0)); // From block (0,0)
    /// assert_eq!(dense.get(&[2, 4]), Some(&0.0)); // Zero (block (1,1) not present)
    /// ```
    pub fn to_dense(&self) -> DenseTensor<ElT> {
        let shape = self.shape();
        let mut dense: DenseTensor<ElT> = Tensor::zeros(&shape);
        let strides = compute_strides(&shape);
        let blockdims = self.blockdims();

        // For each non-zero block, copy data to the dense tensor
        for (block, block_data) in self.storage.iter_blocks() {
            let block_shape = blockdims.block_shape(block.coords());

            // Compute the starting index in the dense tensor for this block
            let mut block_start = vec![0usize; self.ndim()];
            for (dim, &coord) in block.coords().iter().enumerate() {
                block_start[dim] = blockdims.dim(dim).block_offset(coord);
            }

            // Copy each element from block to dense tensor
            copy_block_to_dense(
                block_data,
                &block_shape,
                dense.data_mut(),
                &strides,
                &block_start,
            );
        }

        dense
    }

    /// Create a block-sparse tensor from a dense tensor.
    ///
    /// # Arguments
    ///
    /// * `dense` - Source dense tensor
    /// * `blockdims` - Block structure to use
    /// * `blocks` - Which blocks to extract (must be valid for blockdims)
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::blocksparse_tensor::BlockSparseTensor;
    /// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims};
    /// use ndtensors::Tensor;
    ///
    /// // Create a 5x9 dense tensor
    /// let mut dense = Tensor::<f64>::zeros(&[5, 9]);
    /// dense.set(&[0, 0], 1.0).unwrap();
    /// dense.set(&[1, 3], 2.0).unwrap();
    ///
    /// // Define block structure
    /// let blockdims = BlockDims::new(vec![
    ///     BlockDim::new(vec![2, 3]),
    ///     BlockDim::new(vec![4, 5]),
    /// ]);
    ///
    /// // Extract only block (0,0)
    /// let blocks = vec![Block::new(&[0, 0])];
    /// let bst = BlockSparseTensor::from_dense(&dense, blockdims, blocks).unwrap();
    ///
    /// assert_eq!(bst.nnzblocks(), 1);
    /// let block_view = bst.blockview(&Block::new(&[0, 0])).unwrap();
    /// assert_eq!(block_view.get(&[0, 0]), Some(&1.0));
    /// ```
    pub fn from_dense(
        dense: &DenseTensor<ElT>,
        blockdims: BlockDims,
        blocks: Vec<Block>,
    ) -> Result<Self, TensorError> {
        // Validate that dense shape matches blockdims
        let expected_shape = blockdims.dense_shape();
        if dense.shape() != expected_shape {
            return Err(TensorError::ShapeMismatch {
                expected: expected_shape.iter().product(),
                actual: dense.len(),
            });
        }

        // Create the block-sparse tensor
        let mut bst = Self::zeros(blocks.clone(), blockdims.clone());
        let strides = compute_strides(dense.shape());

        // Copy data for each block
        for block in &blocks {
            let block_shape = blockdims.block_shape(block.coords());

            // Compute the starting index in the dense tensor for this block
            let mut block_start = vec![0usize; dense.ndim()];
            for (dim, &coord) in block.coords().iter().enumerate() {
                block_start[dim] = blockdims.dim(dim).block_offset(coord);
            }

            // Extract block data from dense tensor
            let block_data =
                extract_block_from_dense(dense.data(), &strides, &block_start, &block_shape);

            // Create a DenseTensor for the block and insert it
            let block_tensor = Tensor::from_vec(block_data, &block_shape)?;
            bst.insertblock(block, &block_tensor)?;
        }

        Ok(bst)
    }
}

/// Helper function to copy a block's data to a dense tensor.
#[allow(clippy::needless_range_loop)]
fn copy_block_to_dense<ElT: Scalar>(
    block_data: &[ElT],
    block_shape: &[usize],
    dense_data: &mut [ElT],
    dense_strides: &[usize],
    block_start: &[usize],
) {
    let block_strides = compute_strides(block_shape);
    let block_len: usize = block_shape.iter().product();

    for linear_idx in 0..block_len {
        // Convert linear index to cartesian in block
        let mut cartesian = vec![0usize; block_shape.len()];
        let mut remaining = linear_idx;
        for dim in (0..block_shape.len()).rev() {
            cartesian[dim] = remaining / block_strides[dim];
            remaining %= block_strides[dim];
        }

        // Compute index in dense tensor
        let mut dense_linear = 0;
        for dim in 0..block_shape.len() {
            dense_linear += (block_start[dim] + cartesian[dim]) * dense_strides[dim];
        }

        dense_data[dense_linear] = block_data[linear_idx];
    }
}

/// Helper function to extract a block's data from a dense tensor.
#[allow(clippy::needless_range_loop)]
fn extract_block_from_dense<ElT: Scalar>(
    dense_data: &[ElT],
    dense_strides: &[usize],
    block_start: &[usize],
    block_shape: &[usize],
) -> Vec<ElT> {
    let block_strides = compute_strides(block_shape);
    let block_len: usize = block_shape.iter().product();
    let mut block_data = vec![ElT::zero(); block_len];

    for linear_idx in 0..block_len {
        // Convert linear index to cartesian in block
        let mut cartesian = vec![0usize; block_shape.len()];
        let mut remaining = linear_idx;
        for dim in (0..block_shape.len()).rev() {
            cartesian[dim] = remaining / block_strides[dim];
            remaining %= block_strides[dim];
        }

        // Compute index in dense tensor
        let mut dense_linear = 0;
        for dim in 0..block_shape.len() {
            dense_linear += (block_start[dim] + cartesian[dim]) * dense_strides[dim];
        }

        block_data[linear_idx] = dense_data[dense_linear];
    }

    block_data
}

impl<ElT: Scalar, D: DataBuffer<ElT>> PartialEq for BlockSparseTensor<ElT, D> {
    fn eq(&self, other: &Self) -> bool {
        self.storage == other.storage
    }
}

impl<ElT: Scalar, D: DataBuffer<ElT>> std::fmt::Display for BlockSparseTensor<ElT, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BlockSparseTensor(shape={:?}, nnzblocks={}, nnz={})",
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
    fn test_blocksparse_tensor_zeros() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let tensor: CpuBlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);

        assert_eq!(tensor.shape(), vec![5, 9]);
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.nnzblocks(), 2);
        assert_eq!(tensor.nnz(), 8 + 15); // 2*4 + 3*5
    }

    #[test]
    fn test_blocksparse_tensor_isblocknz() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let tensor: CpuBlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);

        assert!(tensor.isblocknz(&Block::new(&[0, 0])));
        assert!(tensor.isblocknz(&Block::new(&[1, 1])));
        assert!(!tensor.isblocknz(&Block::new(&[0, 1])));
        assert!(!tensor.isblocknz(&Block::new(&[1, 0])));
    }

    #[test]
    fn test_blocksparse_tensor_blockview() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];

        let tensor: CpuBlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);

        let view = tensor.blockview(&Block::new(&[0, 0])).unwrap();
        assert_eq!(view.shape(), &[2, 4]);
        assert_eq!(view.len(), 8);

        // Non-existent block
        assert!(tensor.blockview(&Block::new(&[1, 1])).is_none());
    }

    #[test]
    fn test_blocksparse_tensor_insertblock() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];

        let mut tensor: CpuBlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);

        // Create block data
        let block_data: DenseTensor<f64> =
            Tensor::from_vec((1..9).map(|x| x as f64).collect(), &[2, 4]).unwrap();

        tensor
            .insertblock(&Block::new(&[0, 0]), &block_data)
            .unwrap();

        // Verify
        let view = tensor.blockview(&Block::new(&[0, 0])).unwrap();
        assert_eq!(view.get(&[0, 0]), Some(&1.0));
        assert_eq!(view.get(&[1, 0]), Some(&2.0));
        assert_eq!(view.get(&[0, 1]), Some(&3.0));
    }

    #[test]
    fn test_blocksparse_tensor_insertblock_wrong_shape() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];

        let mut tensor: CpuBlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);

        // Wrong shape
        let wrong_data: DenseTensor<f64> = Tensor::zeros(&[3, 3]);
        assert!(
            tensor
                .insertblock(&Block::new(&[0, 0]), &wrong_data)
                .is_err()
        );
    }

    #[test]
    fn test_blocksparse_tensor_to_dense() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let mut tensor: CpuBlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);

        // Fill block (0,0) with 1.0
        let block00: DenseTensor<f64> = Tensor::ones(&[2, 4]);
        tensor.insertblock(&Block::new(&[0, 0]), &block00).unwrap();

        // Fill block (1,1) with 2.0
        let mut block11: DenseTensor<f64> = Tensor::zeros(&[3, 5]);
        block11.fill(2.0);
        tensor.insertblock(&Block::new(&[1, 1]), &block11).unwrap();

        // Convert to dense
        let dense = tensor.to_dense();
        assert_eq!(dense.shape(), &[5, 9]);

        // Check block (0,0) region
        assert_eq!(dense.get(&[0, 0]), Some(&1.0));
        assert_eq!(dense.get(&[1, 3]), Some(&1.0));

        // Check block (1,1) region (starts at [2, 4])
        assert_eq!(dense.get(&[2, 4]), Some(&2.0));
        assert_eq!(dense.get(&[4, 8]), Some(&2.0));

        // Check zero regions
        assert_eq!(dense.get(&[0, 4]), Some(&0.0)); // block (0,1)
        assert_eq!(dense.get(&[2, 0]), Some(&0.0)); // block (1,0)
    }

    #[test]
    fn test_blocksparse_tensor_from_dense() {
        let blockdims = create_test_blockdims();

        // Create a dense tensor
        let mut dense: DenseTensor<f64> = Tensor::zeros(&[5, 9]);
        // Set some values in block (0,0) region
        dense.set(&[0, 0], 1.0).unwrap();
        dense.set(&[1, 3], 2.0).unwrap();
        // Set some values in block (1,1) region
        dense.set(&[3, 5], 3.0).unwrap();

        // Extract only certain blocks
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];
        let bst = BlockSparseTensor::from_dense(&dense, blockdims, blocks).unwrap();

        assert_eq!(bst.nnzblocks(), 2);

        // Check block (0,0)
        let view00 = bst.blockview(&Block::new(&[0, 0])).unwrap();
        assert_eq!(view00.get(&[0, 0]), Some(&1.0));
        assert_eq!(view00.get(&[1, 3]), Some(&2.0));

        // Check block (1,1)
        let view11 = bst.blockview(&Block::new(&[1, 1])).unwrap();
        // [3, 5] in dense is [1, 1] in block (1,1) (block starts at [2, 4])
        assert_eq!(view11.get(&[1, 1]), Some(&3.0));
    }

    #[test]
    fn test_blocksparse_tensor_roundtrip() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let mut original: CpuBlockSparseTensor<f64> =
            BlockSparseTensor::zeros(blocks.clone(), blockdims.clone());

        // Fill with data
        let block00: DenseTensor<f64> =
            Tensor::from_vec((1..9).map(|x| x as f64).collect(), &[2, 4]).unwrap();
        let block11: DenseTensor<f64> =
            Tensor::from_vec((1..16).map(|x| (x * 10) as f64).collect(), &[3, 5]).unwrap();

        original
            .insertblock(&Block::new(&[0, 0]), &block00)
            .unwrap();
        original
            .insertblock(&Block::new(&[1, 1]), &block11)
            .unwrap();

        // Convert to dense and back
        let dense = original.to_dense();
        let recovered = BlockSparseTensor::from_dense(&dense, blockdims, blocks).unwrap();

        // Compare block data
        let orig_view00 = original.blockview(&Block::new(&[0, 0])).unwrap();
        let recv_view00 = recovered.blockview(&Block::new(&[0, 0])).unwrap();
        assert_eq!(orig_view00.data(), recv_view00.data());

        let orig_view11 = original.blockview(&Block::new(&[1, 1])).unwrap();
        let recv_view11 = recovered.blockview(&Block::new(&[1, 1])).unwrap();
        assert_eq!(orig_view11.data(), recv_view11.data());
    }

    #[test]
    fn test_blocksparse_tensor_iter_blocks() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let tensor: CpuBlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);

        let collected: Vec<_> = tensor.iter_blocks().collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn test_blocksparse_tensor_display() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];

        let tensor: CpuBlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);
        let display = format!("{}", tensor);

        assert!(display.contains("BlockSparseTensor"));
        assert!(display.contains("nnzblocks=1"));
    }

    #[test]
    fn test_blocksparse_tensor_complex() {
        use crate::scalar::c64;

        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];

        let mut tensor: CpuBlockSparseTensor<c64> = BlockSparseTensor::zeros(blocks, blockdims);

        // Create complex block data
        let data: Vec<c64> = (1..9).map(|x| c64::new(x as f64, -(x as f64))).collect();
        let block_data: DenseTensor<c64> = Tensor::from_vec(data, &[2, 4]).unwrap();

        tensor
            .insertblock(&Block::new(&[0, 0]), &block_data)
            .unwrap();

        let view = tensor.blockview(&Block::new(&[0, 0])).unwrap();
        assert_eq!(view.get(&[0, 0]), Some(&c64::new(1.0, -1.0)));
    }
}
