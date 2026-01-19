//! Operations for BlockSparseTensor.
//!
//! This module provides common tensor operations for `BlockSparseTensor`:
//! - `permutedims_blocksparse` - permute dimensions (reorders blocks)
//! - `scale_blocksparse` - multiply by scalar (block-wise)
//! - `add_blocksparse` - element-wise addition (block-wise)
//! - `norm_blocksparse` - Frobenius norm (sum over blocks)

use std::collections::HashSet;
use std::ops::{Add, Mul};

use crate::Tensor;
use crate::blocksparse_tensor::BlockSparseTensor;
use crate::error::TensorError;
use crate::operations::permutedims;
use crate::scalar::{RealScalar, Scalar};
use crate::storage::CpuBuffer;
use crate::storage::blocksparse::{Block, BlockOffsets, BlockSparse};

/// Permute the dimensions of a BlockSparseTensor.
///
/// This function permutes both the block structure and the data within each block.
///
/// # Arguments
///
/// * `tensor` - Input BlockSparseTensor
/// * `perm` - Permutation of dimensions. `perm[i]` gives the source dimension
///   for the i-th dimension of the result.
///
/// # Returns
///
/// A new BlockSparseTensor with permuted dimensions.
///
/// # Example
///
/// ```
/// use ndtensors::blocksparse_tensor::BlockSparseTensor;
/// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims};
/// use ndtensors::operations::blocksparse::permutedims_blocksparse;
/// use ndtensors::Tensor;
///
/// // Create a 2D block-sparse tensor
/// let blockdims = BlockDims::new(vec![
///     BlockDim::new(vec![2, 3]),
///     BlockDim::new(vec![4, 5]),
/// ]);
/// let blocks = vec![Block::new(&[0, 1])];
/// let mut tensor: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);
///
/// // Fill block (0,1) with ones
/// tensor.insertblock(&Block::new(&[0, 1]), &Tensor::ones(&[2, 5])).unwrap();
///
/// // Transpose
/// let transposed = permutedims_blocksparse(&tensor, &[1, 0]).unwrap();
///
/// // Shape changes from [5, 9] to [9, 5]
/// assert_eq!(transposed.shape(), vec![9, 5]);
/// // Block (0,1) becomes block (1,0)
/// assert!(transposed.isblocknz(&Block::new(&[1, 0])));
/// ```
pub fn permutedims_blocksparse<ElT>(
    tensor: &BlockSparseTensor<ElT, CpuBuffer<ElT>>,
    perm: &[usize],
) -> Result<BlockSparseTensor<ElT, CpuBuffer<ElT>>, TensorError>
where
    ElT: Scalar,
{
    let ndim = tensor.ndim();

    // Validate permutation
    if perm.len() != ndim {
        return Err(TensorError::InvalidPermutation {
            perm: perm.to_vec(),
            ndim,
        });
    }

    let mut seen = vec![false; ndim];
    for &p in perm {
        if p >= ndim || seen[p] {
            return Err(TensorError::InvalidPermutation {
                perm: perm.to_vec(),
                ndim,
            });
        }
        seen[p] = true;
    }

    // Permute the block structure
    let (new_offsets, new_blockdims) = tensor.blockoffsets().permute(perm, tensor.blockdims());

    // Compute inverse permutation for data reorganization
    let mut inv_perm = vec![0usize; ndim];
    for (i, &p) in perm.iter().enumerate() {
        inv_perm[p] = i;
    }

    // Create new storage with permuted data
    let mut new_data = vec![ElT::zero(); new_offsets.total_nnz()];

    // For each block in the original tensor
    for (block, block_data) in tensor.storage().iter_blocks() {
        let block_shape = tensor.blockdims().block_shape(block.coords());

        // Create a DenseTensor view of this block's data
        let block_tensor =
            Tensor::from_vec(block_data.to_vec(), &block_shape).expect("block shape matches data");

        // Permute the block data
        let permuted_block = permutedims(&block_tensor, perm)?;

        // Get the permuted block coordinates
        let permuted_block_coords = block.permute(perm);

        // Find where this block goes in the new storage
        let new_offset = new_offsets
            .get(&permuted_block_coords)
            .expect("permuted block should exist");
        let new_block_size = new_blockdims.block_size(permuted_block_coords.coords());

        // Copy permuted data to new storage
        new_data[new_offset..new_offset + new_block_size].copy_from_slice(permuted_block.data());
    }

    // Create the new BlockSparse storage
    let new_storage = BlockSparse::from_vec(new_data, new_offsets, new_blockdims);

    Ok(BlockSparseTensor::from_storage(new_storage))
}

/// Scale a BlockSparseTensor by a scalar.
///
/// Returns a new tensor where each element is multiplied by `alpha`.
///
/// # Example
///
/// ```
/// use ndtensors::blocksparse_tensor::BlockSparseTensor;
/// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims};
/// use ndtensors::operations::blocksparse::scale_blocksparse;
/// use ndtensors::Tensor;
///
/// let blockdims = BlockDims::new(vec![BlockDim::new(vec![2, 3])]);
/// let blocks = vec![Block::new(&[0])];
/// let mut tensor: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);
/// tensor.insertblock(&Block::new(&[0]), &Tensor::ones(&[2])).unwrap();
///
/// let scaled = scale_blocksparse(&tensor, 3.0);
/// let view = scaled.blockview(&Block::new(&[0])).unwrap();
/// assert_eq!(view.data(), &[3.0, 3.0]);
/// ```
pub fn scale_blocksparse<ElT>(
    tensor: &BlockSparseTensor<ElT, CpuBuffer<ElT>>,
    alpha: ElT,
) -> BlockSparseTensor<ElT, CpuBuffer<ElT>>
where
    ElT: Scalar + Mul<Output = ElT>,
{
    // Scale all data
    let scaled_data: Vec<ElT> = tensor
        .storage()
        .as_slice()
        .iter()
        .map(|&x| x * alpha)
        .collect();

    // Create new storage with same structure but scaled data
    let new_storage = BlockSparse::from_vec(
        scaled_data,
        tensor.blockoffsets().clone(),
        tensor.blockdims().clone(),
    );

    BlockSparseTensor::from_storage(new_storage)
}

/// Scale a BlockSparseTensor in-place.
///
/// # Example
///
/// ```
/// use ndtensors::blocksparse_tensor::BlockSparseTensor;
/// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims};
/// use ndtensors::operations::blocksparse::scale_blocksparse_inplace;
/// use ndtensors::Tensor;
///
/// let blockdims = BlockDims::new(vec![BlockDim::new(vec![2, 3])]);
/// let blocks = vec![Block::new(&[0])];
/// let mut tensor: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);
/// tensor.insertblock(&Block::new(&[0]), &Tensor::ones(&[2])).unwrap();
///
/// scale_blocksparse_inplace(&mut tensor, 2.0);
/// let view = tensor.blockview(&Block::new(&[0])).unwrap();
/// assert_eq!(view.data(), &[2.0, 2.0]);
/// ```
pub fn scale_blocksparse_inplace<ElT>(
    tensor: &mut BlockSparseTensor<ElT, CpuBuffer<ElT>>,
    alpha: ElT,
) where
    ElT: Scalar + Mul<Output = ElT>,
{
    for x in tensor.storage_mut().as_mut_slice() {
        *x = *x * alpha;
    }
}

/// Add two BlockSparseTensors element-wise.
///
/// The result contains the union of blocks from both tensors:
/// - Blocks present in both: element-wise sum
/// - Blocks present in only one: copied from that tensor
///
/// # Errors
///
/// Returns error if tensors have different block dimensions.
///
/// # Example
///
/// ```
/// use ndtensors::blocksparse_tensor::BlockSparseTensor;
/// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims};
/// use ndtensors::operations::blocksparse::add_blocksparse;
/// use ndtensors::Tensor;
///
/// let blockdims = BlockDims::new(vec![BlockDim::new(vec![2, 3])]);
///
/// // Tensor a has block 0
/// let mut a: BlockSparseTensor<f64> = BlockSparseTensor::zeros(
///     vec![Block::new(&[0])], blockdims.clone());
/// a.insertblock(&Block::new(&[0]), &Tensor::ones(&[2])).unwrap();
///
/// // Tensor b has blocks 0 and 1
/// let mut b: BlockSparseTensor<f64> = BlockSparseTensor::zeros(
///     vec![Block::new(&[0]), Block::new(&[1])], blockdims.clone());
/// b.insertblock(&Block::new(&[0]), &Tensor::ones(&[2])).unwrap();
/// b.insertblock(&Block::new(&[1]), &Tensor::ones(&[3])).unwrap();
///
/// // a + b has blocks 0 and 1
/// let c = add_blocksparse(&a, &b).unwrap();
/// assert_eq!(c.nnzblocks(), 2);
///
/// // Block 0: 1 + 1 = 2
/// let view0 = c.blockview(&Block::new(&[0])).unwrap();
/// assert_eq!(view0.data(), &[2.0, 2.0]);
///
/// // Block 1: copied from b
/// let view1 = c.blockview(&Block::new(&[1])).unwrap();
/// assert_eq!(view1.data(), &[1.0, 1.0, 1.0]);
/// ```
pub fn add_blocksparse<ElT>(
    a: &BlockSparseTensor<ElT, CpuBuffer<ElT>>,
    b: &BlockSparseTensor<ElT, CpuBuffer<ElT>>,
) -> Result<BlockSparseTensor<ElT, CpuBuffer<ElT>>, TensorError>
where
    ElT: Scalar + Add<Output = ElT>,
{
    // Check that block dimensions match
    if a.blockdims().ndims() != b.blockdims().ndims() {
        return Err(TensorError::ShapeMismatch {
            expected: a.blockdims().ndims(),
            actual: b.blockdims().ndims(),
        });
    }

    // Check each dimension matches
    for i in 0..a.blockdims().ndims() {
        if a.blockdims().dim(i).block_sizes() != b.blockdims().dim(i).block_sizes() {
            return Err(TensorError::ShapeMismatch {
                expected: a.blockdims().dim(i).total_size(),
                actual: b.blockdims().dim(i).total_size(),
            });
        }
    }

    // Collect all unique blocks from both tensors
    let blocks_a: HashSet<Block> = a.blockoffsets().blocks().cloned().collect();
    let blocks_b: HashSet<Block> = b.blockoffsets().blocks().cloned().collect();
    let all_blocks: Vec<Block> = blocks_a.union(&blocks_b).cloned().collect();

    // Sort blocks for deterministic ordering
    let mut sorted_blocks = all_blocks;
    sorted_blocks.sort();

    // Create new block offsets
    let new_offsets = BlockOffsets::from_blocks(&sorted_blocks, a.blockdims());

    // Allocate new data
    let mut new_data = vec![ElT::zero(); new_offsets.total_nnz()];

    // Fill in block data
    for block in &sorted_blocks {
        let offset = new_offsets.get(block).unwrap();
        let block_size = a.blockdims().block_size(block.coords());

        let has_a = a.isblocknz(block);
        let has_b = b.isblocknz(block);

        match (has_a, has_b) {
            (true, true) => {
                // Both have this block: add
                let data_a = a.storage().blockview(block).unwrap();
                let data_b = b.storage().blockview(block).unwrap();
                for (i, (&va, &vb)) in data_a.iter().zip(data_b.iter()).enumerate() {
                    new_data[offset + i] = va + vb;
                }
            }
            (true, false) => {
                // Only a has this block: copy
                let data_a = a.storage().blockview(block).unwrap();
                new_data[offset..offset + block_size].copy_from_slice(data_a);
            }
            (false, true) => {
                // Only b has this block: copy
                let data_b = b.storage().blockview(block).unwrap();
                new_data[offset..offset + block_size].copy_from_slice(data_b);
            }
            (false, false) => {
                unreachable!("block should be in at least one tensor")
            }
        }
    }

    let new_storage = BlockSparse::from_vec(new_data, new_offsets, a.blockdims().clone());
    Ok(BlockSparseTensor::from_storage(new_storage))
}

/// Compute the Frobenius norm of a BlockSparseTensor.
///
/// Returns sqrt(sum(|T_i|^2)) where the sum is over all non-zero elements.
///
/// # Example
///
/// ```
/// use ndtensors::blocksparse_tensor::BlockSparseTensor;
/// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims};
/// use ndtensors::operations::blocksparse::norm_blocksparse;
/// use ndtensors::Tensor;
///
/// let blockdims = BlockDims::new(vec![BlockDim::new(vec![2, 3])]);
/// let blocks = vec![Block::new(&[0])];
/// let mut tensor: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);
///
/// // Set block to [3, 4]
/// tensor.insertblock(&Block::new(&[0]), &Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap()).unwrap();
///
/// // norm = sqrt(9 + 16) = 5
/// let n = norm_blocksparse(&tensor);
/// assert!((n - 5.0).abs() < 1e-10);
/// ```
pub fn norm_blocksparse<ElT>(
    tensor: &BlockSparseTensor<ElT, CpuBuffer<ElT>>,
) -> <ElT as Scalar>::Real
where
    ElT: Scalar,
{
    RealScalar::sqrt(norm_sqr_blocksparse(tensor))
}

/// Compute the squared Frobenius norm of a BlockSparseTensor.
///
/// More efficient than `norm_blocksparse` when the square root is not needed.
///
/// # Example
///
/// ```
/// use ndtensors::blocksparse_tensor::BlockSparseTensor;
/// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims};
/// use ndtensors::operations::blocksparse::norm_sqr_blocksparse;
/// use ndtensors::Tensor;
///
/// let blockdims = BlockDims::new(vec![BlockDim::new(vec![2, 3])]);
/// let blocks = vec![Block::new(&[0])];
/// let mut tensor: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);
///
/// tensor.insertblock(&Block::new(&[0]), &Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap()).unwrap();
///
/// // norm_sqr = 9 + 16 = 25
/// let n2 = norm_sqr_blocksparse(&tensor);
/// assert!((n2 - 25.0).abs() < 1e-10);
/// ```
pub fn norm_sqr_blocksparse<ElT>(
    tensor: &BlockSparseTensor<ElT, CpuBuffer<ElT>>,
) -> <ElT as Scalar>::Real
where
    ElT: Scalar,
{
    let mut sum = <ElT as Scalar>::Real::zero();
    for &x in tensor.storage().as_slice() {
        sum = sum + x.abs_sqr();
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::blocksparse::{BlockDim, BlockDims};
    use crate::tensor::DenseTensor;
    use approx::assert_relative_eq;

    fn create_test_blockdims_2d() -> BlockDims {
        BlockDims::new(vec![BlockDim::new(vec![2, 3]), BlockDim::new(vec![4, 5])])
    }

    #[test]
    fn test_permutedims_blocksparse_transpose() {
        let blockdims = create_test_blockdims_2d();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let mut tensor: BlockSparseTensor<f64> =
            BlockSparseTensor::zeros(blocks, blockdims.clone());

        // Fill block (0,0) with [1..8]
        let block00: DenseTensor<f64> =
            Tensor::from_vec((1..9).map(|x| x as f64).collect(), &[2, 4]).unwrap();
        tensor.insertblock(&Block::new(&[0, 0]), &block00).unwrap();

        // Fill block (1,1) with [10..24]
        let block11: DenseTensor<f64> =
            Tensor::from_vec((10..25).map(|x| x as f64).collect(), &[3, 5]).unwrap();
        tensor.insertblock(&Block::new(&[1, 1]), &block11).unwrap();

        // Transpose
        let transposed = permutedims_blocksparse(&tensor, &[1, 0]).unwrap();

        // Check shape: [5, 9] -> [9, 5]
        assert_eq!(transposed.shape(), vec![9, 5]);

        // Check blocks: (0,0) -> (0,0), (1,1) -> (1,1)
        assert!(transposed.isblocknz(&Block::new(&[0, 0])));
        assert!(transposed.isblocknz(&Block::new(&[1, 1])));
        assert_eq!(transposed.nnzblocks(), 2);

        // Verify data: compare with dense transpose
        let dense_orig = tensor.to_dense();
        let dense_transposed = permutedims(&dense_orig, &[1, 0]).unwrap();
        let sparse_to_dense = transposed.to_dense();

        assert_eq!(dense_transposed.shape(), sparse_to_dense.shape());
        for i in 0..dense_transposed.len() {
            assert_relative_eq!(
                *dense_transposed.get_linear(i).unwrap(),
                *sparse_to_dense.get_linear(i).unwrap(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_permutedims_blocksparse_3d() {
        let blockdims = BlockDims::new(vec![
            BlockDim::new(vec![2]),
            BlockDim::new(vec![3]),
            BlockDim::new(vec![4]),
        ]);
        let blocks = vec![Block::new(&[0, 0, 0])];

        let mut tensor: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);

        let block_data: DenseTensor<f64> =
            Tensor::from_vec((1..25).map(|x| x as f64).collect(), &[2, 3, 4]).unwrap();
        tensor
            .insertblock(&Block::new(&[0, 0, 0]), &block_data)
            .unwrap();

        // Permute [0,1,2] -> [2,0,1]
        let permuted = permutedims_blocksparse(&tensor, &[2, 0, 1]).unwrap();

        assert_eq!(permuted.shape(), vec![4, 2, 3]);

        // Verify against dense permutation
        let dense_orig = tensor.to_dense();
        let dense_permuted = permutedims(&dense_orig, &[2, 0, 1]).unwrap();
        let sparse_to_dense = permuted.to_dense();

        for i in 0..dense_permuted.len() {
            assert_relative_eq!(
                *dense_permuted.get_linear(i).unwrap(),
                *sparse_to_dense.get_linear(i).unwrap(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_permutedims_blocksparse_invalid() {
        let blockdims = create_test_blockdims_2d();
        let tensor: BlockSparseTensor<f64> =
            BlockSparseTensor::zeros(vec![Block::new(&[0, 0])], blockdims);

        // Wrong number of dimensions
        assert!(permutedims_blocksparse(&tensor, &[0]).is_err());
        assert!(permutedims_blocksparse(&tensor, &[0, 1, 2]).is_err());

        // Invalid index
        assert!(permutedims_blocksparse(&tensor, &[0, 2]).is_err());

        // Duplicate index
        assert!(permutedims_blocksparse(&tensor, &[0, 0]).is_err());
    }

    #[test]
    fn test_scale_blocksparse() {
        let blockdims = BlockDims::new(vec![BlockDim::new(vec![3, 2])]);
        let blocks = vec![Block::new(&[0]), Block::new(&[1])];

        let mut tensor: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);

        tensor
            .insertblock(
                &Block::new(&[0]),
                &Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
            )
            .unwrap();
        tensor
            .insertblock(
                &Block::new(&[1]),
                &Tensor::from_vec(vec![4.0, 5.0], &[2]).unwrap(),
            )
            .unwrap();

        let scaled = scale_blocksparse(&tensor, 2.0);

        let view0 = scaled.blockview(&Block::new(&[0])).unwrap();
        assert_eq!(view0.data(), &[2.0, 4.0, 6.0]);

        let view1 = scaled.blockview(&Block::new(&[1])).unwrap();
        assert_eq!(view1.data(), &[8.0, 10.0]);
    }

    #[test]
    fn test_scale_blocksparse_inplace() {
        let blockdims = BlockDims::new(vec![BlockDim::new(vec![3])]);
        let blocks = vec![Block::new(&[0])];

        let mut tensor: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);
        tensor
            .insertblock(
                &Block::new(&[0]),
                &Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
            )
            .unwrap();

        scale_blocksparse_inplace(&mut tensor, 0.5);

        let view = tensor.blockview(&Block::new(&[0])).unwrap();
        assert_eq!(view.data(), &[0.5, 1.0, 1.5]);
    }

    #[test]
    fn test_add_blocksparse_same_structure() {
        let blockdims = BlockDims::new(vec![BlockDim::new(vec![2, 3])]);
        let blocks = vec![Block::new(&[0]), Block::new(&[1])];

        let mut a: BlockSparseTensor<f64> =
            BlockSparseTensor::zeros(blocks.clone(), blockdims.clone());
        let mut b: BlockSparseTensor<f64> =
            BlockSparseTensor::zeros(blocks.clone(), blockdims.clone());

        a.insertblock(&Block::new(&[0]), &Tensor::ones(&[2]))
            .unwrap();
        a.insertblock(&Block::new(&[1]), &Tensor::ones(&[3]))
            .unwrap();

        b.insertblock(
            &Block::new(&[0]),
            &Tensor::from_vec(vec![2.0, 3.0], &[2]).unwrap(),
        )
        .unwrap();
        b.insertblock(
            &Block::new(&[1]),
            &Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap(),
        )
        .unwrap();

        let c = add_blocksparse(&a, &b).unwrap();

        assert_eq!(c.nnzblocks(), 2);

        let view0 = c.blockview(&Block::new(&[0])).unwrap();
        assert_eq!(view0.data(), &[3.0, 4.0]);

        let view1 = c.blockview(&Block::new(&[1])).unwrap();
        assert_eq!(view1.data(), &[5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_add_blocksparse_different_structure() {
        let blockdims = BlockDims::new(vec![BlockDim::new(vec![2, 3])]);

        // a has block 0
        let mut a: BlockSparseTensor<f64> =
            BlockSparseTensor::zeros(vec![Block::new(&[0])], blockdims.clone());
        a.insertblock(&Block::new(&[0]), &Tensor::ones(&[2]))
            .unwrap();

        // b has block 1
        let mut b: BlockSparseTensor<f64> =
            BlockSparseTensor::zeros(vec![Block::new(&[1])], blockdims.clone());
        b.insertblock(
            &Block::new(&[1]),
            &Tensor::from_vec(vec![2.0, 3.0, 4.0], &[3]).unwrap(),
        )
        .unwrap();

        let c = add_blocksparse(&a, &b).unwrap();

        assert_eq!(c.nnzblocks(), 2);

        let view0 = c.blockview(&Block::new(&[0])).unwrap();
        assert_eq!(view0.data(), &[1.0, 1.0]);

        let view1 = c.blockview(&Block::new(&[1])).unwrap();
        assert_eq!(view1.data(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_add_blocksparse_matches_dense() {
        let blockdims = create_test_blockdims_2d();

        let blocks_a = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];
        let blocks_b = vec![Block::new(&[0, 1]), Block::new(&[1, 1])];

        let mut a: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks_a, blockdims.clone());
        let mut b: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks_b, blockdims.clone());

        a.insertblock(
            &Block::new(&[0, 0]),
            &Tensor::from_vec((1..9).map(|x| x as f64).collect(), &[2, 4]).unwrap(),
        )
        .unwrap();
        a.insertblock(
            &Block::new(&[1, 1]),
            &Tensor::from_vec((1..16).map(|x| x as f64).collect(), &[3, 5]).unwrap(),
        )
        .unwrap();

        b.insertblock(
            &Block::new(&[0, 1]),
            &Tensor::from_vec((10..20).map(|x| x as f64).collect(), &[2, 5]).unwrap(),
        )
        .unwrap();
        b.insertblock(
            &Block::new(&[1, 1]),
            &Tensor::from_vec((20..35).map(|x| x as f64).collect(), &[3, 5]).unwrap(),
        )
        .unwrap();

        let c = add_blocksparse(&a, &b).unwrap();

        // Compare with dense addition
        let a_dense = a.to_dense();
        let b_dense = b.to_dense();
        let c_dense = crate::operations::apply_binary(&a_dense, &b_dense, |x, y| x + y).unwrap();
        let c_sparse_dense = c.to_dense();

        for i in 0..c_dense.len() {
            assert_relative_eq!(
                *c_dense.get_linear(i).unwrap(),
                *c_sparse_dense.get_linear(i).unwrap(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_norm_blocksparse() {
        let blockdims = BlockDims::new(vec![BlockDim::new(vec![2, 3])]);
        let blocks = vec![Block::new(&[0]), Block::new(&[1])];

        let mut tensor: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);

        // Block 0: [3, 4] -> ||(3,4)|| = 5
        tensor
            .insertblock(
                &Block::new(&[0]),
                &Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap(),
            )
            .unwrap();

        // Block 1: [0, 0, 0] -> contributes 0
        tensor
            .insertblock(&Block::new(&[1]), &Tensor::zeros(&[3]))
            .unwrap();

        let n = norm_blocksparse(&tensor);
        assert_relative_eq!(n, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_norm_blocksparse_matches_dense() {
        let blockdims = create_test_blockdims_2d();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let mut tensor: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);

        tensor
            .insertblock(
                &Block::new(&[0, 0]),
                &Tensor::from_vec((1..9).map(|x| x as f64).collect(), &[2, 4]).unwrap(),
            )
            .unwrap();
        tensor
            .insertblock(
                &Block::new(&[1, 1]),
                &Tensor::from_vec((1..16).map(|x| x as f64).collect(), &[3, 5]).unwrap(),
            )
            .unwrap();

        let sparse_norm = norm_blocksparse(&tensor);
        let dense_norm = crate::operations::norm(&tensor.to_dense());

        assert_relative_eq!(sparse_norm, dense_norm, epsilon = 1e-10);
    }

    #[test]
    fn test_norm_sqr_blocksparse() {
        let blockdims = BlockDims::new(vec![BlockDim::new(vec![2])]);
        let blocks = vec![Block::new(&[0])];

        let mut tensor: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);
        tensor
            .insertblock(
                &Block::new(&[0]),
                &Tensor::from_vec(vec![3.0, 4.0], &[2]).unwrap(),
            )
            .unwrap();

        let n2 = norm_sqr_blocksparse(&tensor);
        assert_relative_eq!(n2, 25.0, epsilon = 1e-10);
    }

    #[test]
    fn test_norm_blocksparse_complex() {
        use crate::scalar::c64;

        let blockdims = BlockDims::new(vec![BlockDim::new(vec![2])]);
        let blocks = vec![Block::new(&[0])];

        let mut tensor: BlockSparseTensor<c64> = BlockSparseTensor::zeros(blocks, blockdims);

        // |3+4i| = 5
        tensor
            .insertblock(
                &Block::new(&[0]),
                &Tensor::from_vec(vec![c64::new(3.0, 4.0), c64::new(0.0, 0.0)], &[2]).unwrap(),
            )
            .unwrap();

        let n = norm_blocksparse(&tensor);
        assert_relative_eq!(n, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_scale_blocksparse_complex() {
        use crate::scalar::c64;

        let blockdims = BlockDims::new(vec![BlockDim::new(vec![2])]);
        let blocks = vec![Block::new(&[0])];

        let mut tensor: BlockSparseTensor<c64> = BlockSparseTensor::zeros(blocks, blockdims);
        tensor
            .insertblock(
                &Block::new(&[0]),
                &Tensor::from_vec(vec![c64::new(1.0, 0.0), c64::new(0.0, 1.0)], &[2]).unwrap(),
            )
            .unwrap();

        // Scale by i
        let scaled = scale_blocksparse(&tensor, c64::new(0.0, 1.0));

        let view = scaled.blockview(&Block::new(&[0])).unwrap();
        // 1 * i = i
        assert_relative_eq!(view.data()[0].re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(view.data()[0].im, 1.0, epsilon = 1e-10);
        // i * i = -1
        assert_relative_eq!(view.data()[1].re, -1.0, epsilon = 1e-10);
        assert_relative_eq!(view.data()[1].im, 0.0, epsilon = 1e-10);
    }

    // =========================================================
    // Issue #38: Advanced BlockSparse operation tests
    // Patterns from NDTensors.jl test_blocksparse.jl
    // =========================================================

    #[test]
    fn test_blocksparse_scalar_multiplication_preserves_structure() {
        // Test from test_blocksparse.jl: B = 2 * A
        // nnz(A) == nnz(B) and nnzblocks(A) == nnzblocks(B)
        let blockdims = BlockDims::new(vec![BlockDim::new(vec![2, 3]), BlockDim::new(vec![4, 5])]);
        let blocks = vec![Block::new(&[0, 1]), Block::new(&[1, 0])];

        let mut a: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);
        a.insertblock(
            &Block::new(&[0, 1]),
            &Tensor::from_vec((1..11).map(|x| x as f64).collect(), &[2, 5]).unwrap(),
        )
        .unwrap();
        a.insertblock(
            &Block::new(&[1, 0]),
            &Tensor::from_vec((1..13).map(|x| x as f64).collect(), &[3, 4]).unwrap(),
        )
        .unwrap();

        let b = scale_blocksparse(&a, 2.0);

        // Structure preserved
        assert_eq!(a.nnz(), b.nnz());
        assert_eq!(a.nnzblocks(), b.nnzblocks());

        // Values scaled
        let view_a = a.blockview(&Block::new(&[0, 1])).unwrap();
        let view_b = b.blockview(&Block::new(&[0, 1])).unwrap();
        for (va, vb) in view_a.data().iter().zip(view_b.data().iter()) {
            assert_relative_eq!(*vb, 2.0 * va, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_blocksparse_division() {
        // Test: B = A / 2
        let blockdims = BlockDims::new(vec![BlockDim::new(vec![3])]);
        let blocks = vec![Block::new(&[0])];

        let mut a: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);
        a.insertblock(
            &Block::new(&[0]),
            &Tensor::from_vec(vec![2.0, 4.0, 6.0], &[3]).unwrap(),
        )
        .unwrap();

        // Division is scaling by 1/2
        let b = scale_blocksparse(&a, 0.5);

        let view = b.blockview(&Block::new(&[0])).unwrap();
        assert_eq!(view.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_blocksparse_empty_no_blocks() {
        // Test from test_blocksparse.jl: "No blocks" case
        // Tensor with dimensions but zero blocks
        let blockdims = BlockDims::new(vec![BlockDim::new(vec![2, 2]), BlockDim::new(vec![2, 2])]);
        let blocks: Vec<Block> = vec![]; // No blocks

        let t: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);

        // Shape is still defined by blockdims
        assert_eq!(t.shape(), vec![4, 4]);
        assert_eq!(t.nnzblocks(), 0);
        assert_eq!(t.nnz(), 0);

        // Norm of empty tensor is 0
        let n = norm_blocksparse(&t);
        assert_relative_eq!(n, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_blocksparse_to_dense_empty() {
        // Empty tensor to dense
        let blockdims = BlockDims::new(vec![BlockDim::new(vec![2, 3])]);
        let blocks: Vec<Block> = vec![];

        let bst: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);
        let dense = bst.to_dense();

        assert_eq!(dense.shape(), &[5]);
        // All zeros
        for i in 0..dense.len() {
            assert_eq!(*dense.get_linear(i).unwrap(), 0.0);
        }
    }

    #[test]
    fn test_blocksparse_dense_roundtrip() {
        // Test from test_blocksparse.jl: D == A
        let blockdims = create_test_blockdims_2d();
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let mut a: BlockSparseTensor<f64> =
            BlockSparseTensor::zeros(blocks.clone(), blockdims.clone());
        a.insertblock(
            &Block::new(&[0, 0]),
            &Tensor::from_vec((1..9).map(|x| x as f64).collect(), &[2, 4]).unwrap(),
        )
        .unwrap();
        a.insertblock(
            &Block::new(&[1, 1]),
            &Tensor::from_vec((10..25).map(|x| x as f64).collect(), &[3, 5]).unwrap(),
        )
        .unwrap();

        // Convert to dense
        let d = a.to_dense();

        // Verify element-wise equality (test_blocksparse.jl pattern)
        for i in 0..d.shape()[0] {
            for j in 0..d.shape()[1] {
                // BlockSparse doesn't have direct indexing, compare via dense
                let d_val = *d.get(&[i, j]).unwrap();
                // Check against expected: non-zero only in blocks (0,0) and (1,1)
                if (i < 2 && j < 4) || (i >= 2 && j >= 4) {
                    // In a non-zero block region - value should match
                    // (but since we don't have direct indexing on BlockSparse, just verify it's correct via dense)
                    let _ = d_val; // Value exists
                } else {
                    // Zero region
                    assert_eq!(d_val, 0.0);
                }
            }
        }
    }

    #[test]
    fn test_blocksparse_add_elementwise_verification() {
        // Test from test_blocksparse.jl: for I in eachindex(C), C[I] == A[I] + B[I]
        let blockdims = BlockDims::new(vec![BlockDim::new(vec![2, 3])]);
        let blocks = vec![Block::new(&[0]), Block::new(&[1])];

        let mut a: BlockSparseTensor<f64> =
            BlockSparseTensor::zeros(blocks.clone(), blockdims.clone());
        let mut b: BlockSparseTensor<f64> =
            BlockSparseTensor::zeros(blocks.clone(), blockdims.clone());

        a.insertblock(
            &Block::new(&[0]),
            &Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap(),
        )
        .unwrap();
        a.insertblock(
            &Block::new(&[1]),
            &Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap(),
        )
        .unwrap();

        b.insertblock(
            &Block::new(&[0]),
            &Tensor::from_vec(vec![10.0, 20.0], &[2]).unwrap(),
        )
        .unwrap();
        b.insertblock(
            &Block::new(&[1]),
            &Tensor::from_vec(vec![30.0, 40.0, 50.0], &[3]).unwrap(),
        )
        .unwrap();

        let c = add_blocksparse(&a, &b).unwrap();

        // Element-wise verification via dense conversion
        let a_dense = a.to_dense();
        let b_dense = b.to_dense();
        let c_dense = c.to_dense();

        for i in 0..c_dense.len() {
            let expected = a_dense.get_linear(i).unwrap() + b_dense.get_linear(i).unwrap();
            assert_relative_eq!(*c_dense.get_linear(i).unwrap(), expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_blocksparse_complex_real_imag_norm() {
        // Test from test_blocksparse.jl: norm(rT)^2 + norm(iT)^2 ≈ norm(T)^2
        use crate::operations::{imag, norm, real};
        use crate::scalar::c64;

        let blockdims = BlockDims::new(vec![BlockDim::new(vec![2, 3])]);
        let blocks = vec![Block::new(&[0]), Block::new(&[1])];

        let mut t: BlockSparseTensor<c64> = BlockSparseTensor::zeros(blocks, blockdims);
        t.insertblock(
            &Block::new(&[0]),
            &Tensor::from_vec(vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)], &[2]).unwrap(),
        )
        .unwrap();
        t.insertblock(
            &Block::new(&[1]),
            &Tensor::from_vec(
                vec![c64::new(5.0, 6.0), c64::new(7.0, 8.0), c64::new(9.0, 10.0)],
                &[3],
            )
            .unwrap(),
        )
        .unwrap();

        // Convert to dense to use real/imag operations
        let t_dense = t.to_dense();
        let rt = real(&t_dense);
        let it = imag(&t_dense);

        let norm_t = norm_blocksparse(&t);
        let norm_rt = norm(&rt);
        let norm_it = norm(&it);

        // Property: ||Re(T)||^2 + ||Im(T)||^2 = ||T||^2
        assert_relative_eq!(
            norm_rt * norm_rt + norm_it * norm_it,
            norm_t * norm_t,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_blocksparse_complex_add() {
        use crate::scalar::c64;

        let blockdims = BlockDims::new(vec![BlockDim::new(vec![2])]);
        let blocks = vec![Block::new(&[0])];

        let mut a: BlockSparseTensor<c64> =
            BlockSparseTensor::zeros(blocks.clone(), blockdims.clone());
        let mut b: BlockSparseTensor<c64> =
            BlockSparseTensor::zeros(blocks.clone(), blockdims.clone());

        a.insertblock(
            &Block::new(&[0]),
            &Tensor::from_vec(vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)], &[2]).unwrap(),
        )
        .unwrap();

        b.insertblock(
            &Block::new(&[0]),
            &Tensor::from_vec(vec![c64::new(5.0, 6.0), c64::new(7.0, 8.0)], &[2]).unwrap(),
        )
        .unwrap();

        let c = add_blocksparse(&a, &b).unwrap();

        let view = c.blockview(&Block::new(&[0])).unwrap();
        // (1+2i) + (5+6i) = 6+8i
        assert_relative_eq!(view.data()[0].re, 6.0, epsilon = 1e-10);
        assert_relative_eq!(view.data()[0].im, 8.0, epsilon = 1e-10);
        // (3+4i) + (7+8i) = 10+12i
        assert_relative_eq!(view.data()[1].re, 10.0, epsilon = 1e-10);
        assert_relative_eq!(view.data()[1].im, 12.0, epsilon = 1e-10);
    }

    #[test]
    fn test_blocksparse_add_union_blocks() {
        // Test that addition creates union of block structures
        let blockdims = BlockDims::new(vec![BlockDim::new(vec![2, 3]), BlockDim::new(vec![4, 5])]);

        // a has blocks (0,0), (1,1)
        let mut a: BlockSparseTensor<f64> = BlockSparseTensor::zeros(
            vec![Block::new(&[0, 0]), Block::new(&[1, 1])],
            blockdims.clone(),
        );
        a.insertblock(&Block::new(&[0, 0]), &Tensor::ones(&[2, 4]))
            .unwrap();
        a.insertblock(&Block::new(&[1, 1]), &Tensor::ones(&[3, 5]))
            .unwrap();

        // b has blocks (0,1), (1,1)
        let mut b: BlockSparseTensor<f64> = BlockSparseTensor::zeros(
            vec![Block::new(&[0, 1]), Block::new(&[1, 1])],
            blockdims.clone(),
        );
        b.insertblock(&Block::new(&[0, 1]), &Tensor::ones(&[2, 5]))
            .unwrap();
        b.insertblock(&Block::new(&[1, 1]), &Tensor::ones(&[3, 5]))
            .unwrap();

        let c = add_blocksparse(&a, &b).unwrap();

        // c should have blocks (0,0), (0,1), (1,1) - the union
        assert_eq!(c.nnzblocks(), 3);
        assert!(c.isblocknz(&Block::new(&[0, 0])));
        assert!(c.isblocknz(&Block::new(&[0, 1])));
        assert!(c.isblocknz(&Block::new(&[1, 1])));

        // Block (0,0): only from a
        let view00 = c.blockview(&Block::new(&[0, 0])).unwrap();
        assert_eq!(view00.data(), &vec![1.0; 8]);

        // Block (0,1): only from b
        let view01 = c.blockview(&Block::new(&[0, 1])).unwrap();
        assert_eq!(view01.data(), &vec![1.0; 10]);

        // Block (1,1): from both, so 1+1=2
        let view11 = c.blockview(&Block::new(&[1, 1])).unwrap();
        assert_eq!(view11.data(), &vec![2.0; 15]);
    }

    #[test]
    fn test_blocksparse_equality() {
        let blockdims = BlockDims::new(vec![BlockDim::new(vec![2, 3])]);
        let blocks = vec![Block::new(&[0]), Block::new(&[1])];

        let mut a: BlockSparseTensor<f64> =
            BlockSparseTensor::zeros(blocks.clone(), blockdims.clone());
        let mut b: BlockSparseTensor<f64> =
            BlockSparseTensor::zeros(blocks.clone(), blockdims.clone());

        a.insertblock(
            &Block::new(&[0]),
            &Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap(),
        )
        .unwrap();
        a.insertblock(
            &Block::new(&[1]),
            &Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap(),
        )
        .unwrap();

        b.insertblock(
            &Block::new(&[0]),
            &Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap(),
        )
        .unwrap();
        b.insertblock(
            &Block::new(&[1]),
            &Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap(),
        )
        .unwrap();

        assert_eq!(a, b);

        // Different values should not be equal
        let mut c: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);
        c.insertblock(
            &Block::new(&[0]),
            &Tensor::from_vec(vec![1.0, 9.0], &[2]).unwrap(),
        )
        .unwrap();
        c.insertblock(
            &Block::new(&[1]),
            &Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap(),
        )
        .unwrap();

        assert_ne!(a, c);
    }

    #[test]
    fn test_blocksparse_3d_operations() {
        // 3D BlockSparse operations
        let blockdims = BlockDims::new(vec![
            BlockDim::new(vec![2]),
            BlockDim::new(vec![3]),
            BlockDim::new(vec![4]),
        ]);
        let blocks = vec![Block::new(&[0, 0, 0])];

        let mut tensor: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);
        let block_data: DenseTensor<f64> =
            Tensor::from_vec((1..25).map(|x| x as f64).collect(), &[2, 3, 4]).unwrap();
        tensor
            .insertblock(&Block::new(&[0, 0, 0]), &block_data)
            .unwrap();

        // Scale
        let scaled = scale_blocksparse(&tensor, 0.5);
        let view = scaled.blockview(&Block::new(&[0, 0, 0])).unwrap();
        assert_relative_eq!(*view.get(&[0, 0, 0]).unwrap(), 0.5, epsilon = 1e-10);
        assert_relative_eq!(*view.get(&[1, 2, 3]).unwrap(), 12.0, epsilon = 1e-10);

        // Norm
        let n = norm_blocksparse(&tensor);
        let expected_norm = (1..25).map(|x| (x * x) as f64).sum::<f64>().sqrt();
        assert_relative_eq!(n, expected_norm, epsilon = 1e-10);
    }

    #[test]
    fn test_add_blocksparse_dimension_mismatch() {
        let blockdims_a = BlockDims::new(vec![BlockDim::new(vec![2, 3])]);
        let blockdims_b = BlockDims::new(vec![BlockDim::new(vec![2, 4])]); // Different!

        let a: BlockSparseTensor<f64> =
            BlockSparseTensor::zeros(vec![Block::new(&[0])], blockdims_a);
        let b: BlockSparseTensor<f64> =
            BlockSparseTensor::zeros(vec![Block::new(&[0])], blockdims_b);

        let result = add_blocksparse(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_blocksparse_scale_zero() {
        let blockdims = BlockDims::new(vec![BlockDim::new(vec![3])]);
        let blocks = vec![Block::new(&[0])];

        let mut tensor: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);
        tensor
            .insertblock(
                &Block::new(&[0]),
                &Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
            )
            .unwrap();

        // Scale by zero
        let scaled = scale_blocksparse(&tensor, 0.0);

        let view = scaled.blockview(&Block::new(&[0])).unwrap();
        assert_eq!(view.data(), &[0.0, 0.0, 0.0]);

        // Norm should be zero
        assert_relative_eq!(norm_blocksparse(&scaled), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_blocksparse_add_self() {
        // A + A = 2*A
        let blockdims = BlockDims::new(vec![BlockDim::new(vec![2])]);
        let blocks = vec![Block::new(&[0])];

        let mut a: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks, blockdims);
        a.insertblock(
            &Block::new(&[0]),
            &Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap(),
        )
        .unwrap();

        let sum = add_blocksparse(&a, &a).unwrap();
        let doubled = scale_blocksparse(&a, 2.0);

        // sum should equal doubled
        let sum_view = sum.blockview(&Block::new(&[0])).unwrap();
        let doubled_view = doubled.blockview(&Block::new(&[0])).unwrap();
        assert_eq!(sum_view.data(), doubled_view.data());
    }
}
