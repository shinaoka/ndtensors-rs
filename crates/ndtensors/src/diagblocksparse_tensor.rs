//! DiagBlockSparseTensor - Diagonal block-sparse tensor type.
//!
//! This module provides `DiagBlockSparseTensor<ElT>`, an independent tensor type
//! for diagonal block-sparse data. This mirrors NDTensors.jl's approach where
//! DiagBlockSparse tensors store only diagonal elements within non-zero blocks.
//!
//! # Design Decision
//!
//! `DiagBlockSparseTensor` stores only diagonal elements (where all indices
//! within a block are equal). This is efficient for identity-like operators
//! and diagonal matrices in tensor network applications.

use crate::Tensor;
use crate::error::TensorError;
use crate::scalar::Scalar;
use crate::storage::blocksparse::{Block, BlockDims, BlockOffsets, DiagBlockSparse};
use crate::storage::{CpuBuffer, DataBuffer, Diag};
use crate::strides::compute_strides;
use crate::tensor::DenseTensor;

/// A diagonal block-sparse tensor.
///
/// Stores only diagonal elements within non-zero blocks. For a block with
/// shape [d0, d1, ..., dn], only min(d0, d1, ..., dn) diagonal elements
/// are stored.
///
/// # Example
///
/// ```
/// use ndtensors::diagblocksparse_tensor::DiagBlockSparseTensor;
/// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims};
///
/// // Create a 5x5 tensor with block structure [2,3] x [2,3]
/// let blockdims = BlockDims::new(vec![
///     BlockDim::new(vec![2, 3]),
///     BlockDim::new(vec![2, 3]),
/// ]);
///
/// // Only diagonal blocks (0,0) and (1,1) are non-zero
/// let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];
/// let tensor: DiagBlockSparseTensor<f64> = DiagBlockSparseTensor::identity(blocks, blockdims);
///
/// assert_eq!(tensor.shape(), &[5, 5]);
/// assert_eq!(tensor.nnzblocks(), 2);
/// // Block (0,0) has shape [2,2], so 2 diagonal elements
/// // Block (1,1) has shape [3,3], so 3 diagonal elements
/// assert_eq!(tensor.nnz(), 5);
/// ```
#[derive(Clone, Debug)]
pub struct DiagBlockSparseTensor<ElT: Scalar, D: DataBuffer<ElT> = CpuBuffer<ElT>> {
    storage: DiagBlockSparse<ElT, D>,
}

/// Type alias for CPU-backed DiagBlockSparseTensor.
pub type CpuDiagBlockSparseTensor<ElT> = DiagBlockSparseTensor<ElT, CpuBuffer<ElT>>;

impl<ElT: Scalar, D: DataBuffer<ElT>> DiagBlockSparseTensor<ElT, D> {
    /// Create a new diagonal block-sparse tensor with the given blocks initialized to zero.
    ///
    /// # Arguments
    ///
    /// * `blocks` - List of non-zero block coordinates
    /// * `blockdims` - Block dimensions for each tensor dimension
    pub fn zeros(blocks: Vec<Block>, blockdims: BlockDims) -> Self {
        Self {
            storage: DiagBlockSparse::zeros(blocks, blockdims),
        }
    }

    /// Create a diagonal block-sparse tensor from existing storage.
    pub fn from_storage(storage: DiagBlockSparse<ElT, D>) -> Self {
        Self { storage }
    }

    /// Create a uniform diagonal block-sparse tensor where all diagonal elements have the same value.
    ///
    /// # Arguments
    ///
    /// * `value` - The uniform value for all diagonal elements
    /// * `blocks` - List of non-zero block coordinates
    /// * `blockdims` - Block dimensions for each tensor dimension
    pub fn uniform(value: ElT, blocks: Vec<Block>, blockdims: BlockDims) -> Self {
        Self {
            storage: DiagBlockSparse::uniform(value, blocks, blockdims),
        }
    }

    /// Create an identity-like diagonal block-sparse tensor where all diagonal elements are 1.
    ///
    /// # Arguments
    ///
    /// * `blocks` - List of non-zero block coordinates
    /// * `blockdims` - Block dimensions for each tensor dimension
    pub fn identity(blocks: Vec<Block>, blockdims: BlockDims) -> Self {
        Self {
            storage: DiagBlockSparse::identity(blocks, blockdims),
        }
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

    /// Get the total number of non-zero (diagonal) elements.
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

    /// Get the diagonal size of a block.
    ///
    /// For a block with shape [d0, d1, ..., dn], this is min(d0, d1, ..., dn).
    #[inline]
    pub fn diag_size(&self, block: &Block) -> usize {
        self.storage.diag_size(block)
    }

    /// Get the full shape of a block.
    #[inline]
    pub fn block_shape(&self, block: &Block) -> Vec<usize> {
        self.storage.block_shape(block)
    }

    /// Get an immutable view of a block's diagonal data.
    ///
    /// Returns `None` if the block is not present (structurally zero).
    pub fn blockview(&self, block: &Block) -> Option<&[ElT]> {
        self.storage.blockview(block)
    }

    /// Get a mutable view of a block's diagonal data.
    ///
    /// Returns `None` if the block is not present.
    pub fn blockview_mut(&mut self, block: &Block) -> Option<&mut [ElT]> {
        self.storage.blockview_mut(block)
    }

    /// Get a block's diagonal as a Diag storage.
    ///
    /// Returns `None` if the block is not present.
    pub fn block_diag(&self, block: &Block) -> Option<Diag<ElT>> {
        let data = self.storage.blockview(block)?;
        Some(Diag::from_vec(data.to_vec()))
    }

    /// Get the underlying storage.
    #[inline]
    pub fn storage(&self) -> &DiagBlockSparse<ElT, D> {
        &self.storage
    }

    /// Get mutable access to the underlying storage.
    #[inline]
    pub fn storage_mut(&mut self) -> &mut DiagBlockSparse<ElT, D> {
        &mut self.storage
    }

    /// Get the underlying data as a slice.
    #[inline]
    pub fn data(&self) -> &[ElT] {
        self.storage.as_slice()
    }

    /// Get the underlying data as a mutable slice.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [ElT] {
        self.storage.as_mut_slice()
    }

    /// Iterate over all non-zero blocks with their diagonal data.
    pub fn iter_blocks(&self) -> impl Iterator<Item = (&Block, &[ElT])> {
        self.storage.iter_blocks()
    }

    /// Get an iterator over the non-zero blocks.
    pub fn nzblocks(&self) -> impl Iterator<Item = &Block> {
        self.storage.nzblocks()
    }

    /// Get an element by its indices.
    ///
    /// Returns the diagonal value if all indices are equal and the block exists,
    /// otherwise returns zero.
    ///
    /// # Arguments
    ///
    /// * `indices` - The multi-dimensional indices
    ///
    /// # Returns
    ///
    /// * `Some(value)` if the element exists (on diagonal of existing block)
    /// * `None` if indices are out of bounds
    /// * Zero if off-diagonal or block doesn't exist
    pub fn get(&self, indices: &[usize]) -> Option<ElT> {
        if indices.len() != self.ndim() {
            return None;
        }

        // Check bounds
        let blockdims = self.blockdims();
        for (dim, &idx) in indices.iter().enumerate() {
            if idx >= blockdims.dim(dim).total_size() {
                return None;
            }
        }

        // Check if this is a diagonal element (all indices equal)
        let all_equal = indices.iter().skip(1).all(|&i| i == indices[0]);

        if !all_equal {
            // Off-diagonal element is always zero
            return Some(ElT::zero());
        }

        // Find which block this index belongs to
        let mut block_coords = Vec::with_capacity(self.ndim());
        let mut local_index = 0usize;

        for (dim, &idx) in indices.iter().enumerate() {
            let (block_idx, local_idx) = blockdims.dim(dim).find_block(idx);
            block_coords.push(block_idx);
            if dim == 0 {
                local_index = local_idx;
            } else if local_idx != local_index {
                // Not on the block diagonal
                return Some(ElT::zero());
            }
        }

        let block = Block::new(&block_coords);

        // Check if all block coordinates are equal (only diagonal blocks have data)
        let all_block_coords_equal = block_coords.iter().skip(1).all(|&c| c == block_coords[0]);
        if !all_block_coords_equal {
            // Not a diagonal block
            return Some(ElT::zero());
        }

        // Get the value from the block's diagonal
        if let Some(diag_data) = self.storage.blockview(&block) {
            diag_data.get(local_index).copied()
        } else {
            // Block not present
            Some(ElT::zero())
        }
    }

    /// Set an element by its indices.
    ///
    /// Only diagonal elements can be set. Attempting to set off-diagonal
    /// elements will return an error.
    ///
    /// # Arguments
    ///
    /// * `indices` - The multi-dimensional indices
    /// * `value` - The value to set
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Indices are out of bounds
    /// - Attempting to set off-diagonal element
    /// - Block does not exist
    pub fn set(&mut self, indices: &[usize], value: ElT) -> Result<(), TensorError> {
        if indices.len() != self.ndim() {
            return Err(TensorError::DimensionMismatch {
                expected: self.ndim(),
                actual: indices.len(),
            });
        }

        // Check bounds
        let blockdims = self.blockdims().clone();
        for (dim, &idx) in indices.iter().enumerate() {
            if idx >= blockdims.dim(dim).total_size() {
                return Err(TensorError::IndexOutOfBounds {
                    index: idx,
                    dim_size: blockdims.dim(dim).total_size(),
                });
            }
        }

        // Check if this is a diagonal element
        let all_equal = indices.iter().skip(1).all(|&i| i == indices[0]);

        if !all_equal {
            return Err(TensorError::InvalidOperation(
                "Cannot set off-diagonal element in DiagBlockSparseTensor".to_string(),
            ));
        }

        // Find which block this index belongs to
        let mut block_coords = Vec::with_capacity(self.ndim());
        let mut local_index = 0usize;

        for (dim, &idx) in indices.iter().enumerate() {
            let (block_idx, local_idx) = blockdims.dim(dim).find_block(idx);
            block_coords.push(block_idx);
            if dim == 0 {
                local_index = local_idx;
            } else if local_idx != local_index {
                return Err(TensorError::InvalidOperation(
                    "Index not on block diagonal".to_string(),
                ));
            }
        }

        let block = Block::new(&block_coords);

        // Check if all block coordinates are equal
        let all_block_coords_equal = block_coords.iter().skip(1).all(|&c| c == block_coords[0]);
        if !all_block_coords_equal {
            return Err(TensorError::InvalidOperation(
                "Cannot set element in non-diagonal block".to_string(),
            ));
        }

        // Set the value in the block's diagonal
        let diag_data = self
            .storage
            .blockview_mut(&block)
            .ok_or(TensorError::BlockNotFound {
                block: block_coords,
            })?;

        if local_index >= diag_data.len() {
            return Err(TensorError::IndexOutOfBounds {
                index: local_index,
                dim_size: diag_data.len(),
            });
        }

        diag_data[local_index] = value;
        Ok(())
    }
}

// Conversion methods
impl<ElT: Scalar> DiagBlockSparseTensor<ElT, CpuBuffer<ElT>> {
    /// Convert to a dense tensor.
    ///
    /// Creates a full dense tensor, filling in zeros for missing and off-diagonal elements.
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::diagblocksparse_tensor::DiagBlockSparseTensor;
    /// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims};
    ///
    /// let blockdims = BlockDims::new(vec![
    ///     BlockDim::new(vec![2, 3]),
    ///     BlockDim::new(vec![2, 3]),
    /// ]);
    /// let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];
    /// let dbst: DiagBlockSparseTensor<f64> = DiagBlockSparseTensor::identity(blocks, blockdims);
    ///
    /// // Convert to dense
    /// let dense = dbst.to_dense();
    /// assert_eq!(dense.shape(), &[5, 5]);
    /// // Diagonal elements are 1.0
    /// assert_eq!(dense.get(&[0, 0]), Some(&1.0));
    /// assert_eq!(dense.get(&[1, 1]), Some(&1.0));
    /// assert_eq!(dense.get(&[2, 2]), Some(&1.0));
    /// // Off-diagonal elements are 0.0
    /// assert_eq!(dense.get(&[0, 1]), Some(&0.0));
    /// ```
    pub fn to_dense(&self) -> DenseTensor<ElT> {
        let shape = self.shape();
        let mut dense: DenseTensor<ElT> = Tensor::zeros(&shape);
        let strides = compute_strides(&shape);
        let blockdims = self.blockdims();

        // For each non-zero block, copy diagonal data to the dense tensor
        for (block, diag_data) in self.storage.iter_blocks() {
            // Compute the starting index in the dense tensor for this block
            let mut block_start = vec![0usize; self.ndim()];
            for (dim, &coord) in block.coords().iter().enumerate() {
                block_start[dim] = blockdims.dim(dim).block_offset(coord);
            }

            // Copy each diagonal element to the dense tensor
            for (i, &val) in diag_data.iter().enumerate() {
                // Compute the linear index in the dense tensor
                let mut linear_idx = 0;
                for (dim, &stride) in strides.iter().enumerate() {
                    linear_idx += (block_start[dim] + i) * stride;
                }

                dense.data_mut()[linear_idx] = val;
            }
        }

        dense
    }

    /// Compute the squared Frobenius norm.
    ///
    /// For a diagonal block-sparse tensor, this is the sum of squares of diagonal elements.
    pub fn norm_sqr(&self) -> <ElT as Scalar>::Real {
        let mut sum = <ElT as Scalar>::Real::zero();
        for &val in self.data() {
            sum = sum + val.abs_sqr();
        }
        sum
    }

    /// Compute the Frobenius norm.
    pub fn norm(&self) -> <ElT as Scalar>::Real {
        use crate::scalar::RealScalar;
        RealScalar::sqrt(self.norm_sqr())
    }
}

impl<ElT: Scalar, D: DataBuffer<ElT>> PartialEq for DiagBlockSparseTensor<ElT, D> {
    fn eq(&self, other: &Self) -> bool {
        self.storage == other.storage
    }
}

impl<ElT: Scalar, D: DataBuffer<ElT>> std::fmt::Display for DiagBlockSparseTensor<ElT, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DiagBlockSparseTensor(shape={:?}, nnzblocks={}, nnz={})",
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
        BlockDims::new(vec![BlockDim::new(vec![2, 3]), BlockDim::new(vec![2, 3])])
    }

    #[test]
    fn test_diagblocksparse_tensor_zeros() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let tensor: CpuDiagBlockSparseTensor<f64> = DiagBlockSparseTensor::zeros(blocks, blockdims);

        assert_eq!(tensor.shape(), &[5, 5]);
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.nnzblocks(), 2);
        assert_eq!(tensor.nnz(), 5); // 2 + 3

        // All elements should be zero
        for &val in tensor.data() {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_diagblocksparse_tensor_identity() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let tensor: CpuDiagBlockSparseTensor<f64> =
            DiagBlockSparseTensor::identity(blocks, blockdims);

        assert_eq!(tensor.nnz(), 5);
        assert!(tensor.data().iter().all(|&v| v == 1.0));
    }

    #[test]
    fn test_diagblocksparse_tensor_isblocknz() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let tensor: CpuDiagBlockSparseTensor<f64> = DiagBlockSparseTensor::zeros(blocks, blockdims);

        assert!(tensor.isblocknz(&Block::new(&[0, 0])));
        assert!(tensor.isblocknz(&Block::new(&[1, 1])));
        assert!(!tensor.isblocknz(&Block::new(&[0, 1])));
        assert!(!tensor.isblocknz(&Block::new(&[1, 0])));
    }

    #[test]
    fn test_diagblocksparse_tensor_blockview() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let tensor: CpuDiagBlockSparseTensor<f64> =
            DiagBlockSparseTensor::identity(blocks, blockdims);

        let view0 = tensor.blockview(&Block::new(&[0, 0])).unwrap();
        assert_eq!(view0, &[1.0, 1.0]);

        let view1 = tensor.blockview(&Block::new(&[1, 1])).unwrap();
        assert_eq!(view1, &[1.0, 1.0, 1.0]);

        assert!(tensor.blockview(&Block::new(&[0, 1])).is_none());
    }

    #[test]
    fn test_diagblocksparse_tensor_blockview_mut() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];

        let mut tensor: CpuDiagBlockSparseTensor<f64> =
            DiagBlockSparseTensor::zeros(blocks, blockdims);

        {
            let view = tensor.blockview_mut(&Block::new(&[0, 0])).unwrap();
            view[0] = 1.0;
            view[1] = 2.0;
        }

        let view = tensor.blockview(&Block::new(&[0, 0])).unwrap();
        assert_eq!(view, &[1.0, 2.0]);
    }

    #[test]
    fn test_diagblocksparse_tensor_to_dense() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let tensor: CpuDiagBlockSparseTensor<f64> =
            DiagBlockSparseTensor::identity(blocks, blockdims);

        let dense = tensor.to_dense();
        assert_eq!(dense.shape(), &[5, 5]);

        // Check diagonal elements
        assert_eq!(dense.get(&[0, 0]), Some(&1.0));
        assert_eq!(dense.get(&[1, 1]), Some(&1.0));
        assert_eq!(dense.get(&[2, 2]), Some(&1.0));
        assert_eq!(dense.get(&[3, 3]), Some(&1.0));
        assert_eq!(dense.get(&[4, 4]), Some(&1.0));

        // Check off-diagonal elements
        assert_eq!(dense.get(&[0, 1]), Some(&0.0));
        assert_eq!(dense.get(&[1, 0]), Some(&0.0));
        assert_eq!(dense.get(&[2, 3]), Some(&0.0));
    }

    #[test]
    fn test_diagblocksparse_tensor_get() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let tensor: CpuDiagBlockSparseTensor<f64> =
            DiagBlockSparseTensor::identity(blocks, blockdims);

        // Diagonal elements
        assert_eq!(tensor.get(&[0, 0]), Some(1.0));
        assert_eq!(tensor.get(&[1, 1]), Some(1.0));
        assert_eq!(tensor.get(&[2, 2]), Some(1.0));

        // Off-diagonal elements are zero
        assert_eq!(tensor.get(&[0, 1]), Some(0.0));
        assert_eq!(tensor.get(&[1, 0]), Some(0.0));
    }

    #[test]
    fn test_diagblocksparse_tensor_set() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let mut tensor: CpuDiagBlockSparseTensor<f64> =
            DiagBlockSparseTensor::zeros(blocks, blockdims);

        // Set diagonal elements
        tensor.set(&[0, 0], 1.0).unwrap();
        tensor.set(&[1, 1], 2.0).unwrap();
        tensor.set(&[2, 2], 3.0).unwrap();

        assert_eq!(tensor.get(&[0, 0]), Some(1.0));
        assert_eq!(tensor.get(&[1, 1]), Some(2.0));
        assert_eq!(tensor.get(&[2, 2]), Some(3.0));
    }

    #[test]
    fn test_diagblocksparse_tensor_set_offdiag_fails() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];

        let mut tensor: CpuDiagBlockSparseTensor<f64> =
            DiagBlockSparseTensor::zeros(blocks, blockdims);

        // Setting off-diagonal should fail
        assert!(tensor.set(&[0, 1], 1.0).is_err());
    }

    #[test]
    fn test_diagblocksparse_tensor_norm() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let tensor: CpuDiagBlockSparseTensor<f64> =
            DiagBlockSparseTensor::identity(blocks, blockdims);

        // All 5 diagonal elements are 1.0, so norm_sqr = 5, norm = sqrt(5)
        assert_eq!(tensor.norm_sqr(), 5.0);
        assert!((tensor.norm() - 5.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_diagblocksparse_tensor_complex() {
        use crate::scalar::c64;

        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0])];

        let mut tensor: CpuDiagBlockSparseTensor<c64> =
            DiagBlockSparseTensor::zeros(blocks, blockdims);

        tensor.set(&[0, 0], c64::new(1.0, 2.0)).unwrap();
        tensor.set(&[1, 1], c64::new(3.0, 4.0)).unwrap();

        assert_eq!(tensor.get(&[0, 0]), Some(c64::new(1.0, 2.0)));
        assert_eq!(tensor.get(&[1, 1]), Some(c64::new(3.0, 4.0)));

        // norm_sqr = |1+2i|^2 + |3+4i|^2 = 5 + 25 = 30
        assert!((tensor.norm_sqr() - 30.0).abs() < 1e-10);
    }

    #[test]
    fn test_diagblocksparse_tensor_display() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let tensor: CpuDiagBlockSparseTensor<f64> = DiagBlockSparseTensor::zeros(blocks, blockdims);
        let display = format!("{}", tensor);

        assert!(display.contains("DiagBlockSparseTensor"));
        assert!(display.contains("nnzblocks=2"));
        assert!(display.contains("nnz=5"));
    }

    #[test]
    fn test_diagblocksparse_tensor_iter_blocks() {
        let blockdims = create_test_blockdims();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let tensor: CpuDiagBlockSparseTensor<f64> =
            DiagBlockSparseTensor::identity(blocks, blockdims);

        let collected: Vec<_> = tensor.iter_blocks().collect();
        assert_eq!(collected.len(), 2);

        let block_coords: Vec<_> = collected.iter().map(|(b, _)| b.coords().to_vec()).collect();
        assert!(block_coords.contains(&vec![0, 0]));
        assert!(block_coords.contains(&vec![1, 1]));
    }
}
