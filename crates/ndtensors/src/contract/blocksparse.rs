//! Block-sparse tensor contraction.
//!
//! This module provides tensor contraction for `BlockSparseTensor` using
//! block-wise GEMM operations. The algorithm:
//!
//! 1. Determine output block structure from input block sparsity patterns
//! 2. For each output block, find contributing (block_a, block_b) pairs
//! 3. Contract each pair using GEMM
//! 4. Accumulate results when multiple pairs contribute
//!
//! # Block Contraction Properties
//!
//! For a contraction `C[i,k] = sum_j A[i,j] * B[j,k]`, at the block level:
//! - `C` block `(i_b, k_b)` = `sum_{j_b} A[i_b, j_b] * B[j_b, k_b]`
//! - The sum is only over `j_b` where both `A[i_b, j_b]` and `B[j_b, k_b]` are non-zero

use std::collections::HashMap;
use std::ops::{Add, Mul};

use crate::blocksparse_tensor::BlockSparseTensor;
use crate::contract::contract_gemm;
use crate::contract::properties::ContractionProperties;
use crate::error::TensorError;
use crate::scalar::Scalar;
use crate::storage::blocksparse::{Block, BlockDim, BlockDims};
use crate::storage::{CpuBuffer, DataBuffer};

/// Type alias for block contribution mapping.
/// Maps output blocks to lists of contributing (block_a, block_b) pairs.
type BlockContributions = HashMap<Block, Vec<(Block, Block)>>;

/// Contract two block-sparse tensors.
///
/// This function contracts two `BlockSparseTensor`s using block-wise GEMM
/// operations. Only blocks that can contribute to non-zero output blocks
/// are computed.
///
/// # Arguments
///
/// * `a` - First block-sparse tensor
/// * `labels_a` - Labels for each dimension of `a`
/// * `b` - Second block-sparse tensor
/// * `labels_b` - Labels for each dimension of `b`
///
/// # Returns
///
/// A new `BlockSparseTensor` with the contracted result.
///
/// # Example
///
/// ```
/// use ndtensors::blocksparse_tensor::BlockSparseTensor;
/// use ndtensors::storage::blocksparse::{Block, BlockDim, BlockDims};
/// use ndtensors::contract::blocksparse::contract_blocksparse;
/// use ndtensors::Tensor;
///
/// // Create block-sparse tensors for matrix multiplication
/// // A: 5x6 with blocks [2,3] x [2,4]
/// let blockdims_a = BlockDims::new(vec![
///     BlockDim::new(vec![2, 3]),
///     BlockDim::new(vec![2, 4]),
/// ]);
/// let blocks_a = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];
/// let mut a: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks_a, blockdims_a);
///
/// // B: 6x8 with blocks [2,4] x [3,5]
/// let blockdims_b = BlockDims::new(vec![
///     BlockDim::new(vec![2, 4]),
///     BlockDim::new(vec![3, 5]),
/// ]);
/// let blocks_b = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];
/// let mut b: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks_b, blockdims_b);
///
/// // Fill with ones
/// a.insertblock(&Block::new(&[0, 0]), &Tensor::ones(&[2, 2])).unwrap();
/// a.insertblock(&Block::new(&[1, 1]), &Tensor::ones(&[3, 4])).unwrap();
/// b.insertblock(&Block::new(&[0, 0]), &Tensor::ones(&[2, 3])).unwrap();
/// b.insertblock(&Block::new(&[1, 1]), &Tensor::ones(&[4, 5])).unwrap();
///
/// // Contract: C[i,k] = A[i,j] * B[j,k]
/// let c = contract_blocksparse(&a, &[1, -1], &b, &[-1, 2]).unwrap();
///
/// // Output has shape [5, 8] with blocks (0,0) and (1,1)
/// assert_eq!(c.shape(), vec![5, 8]);
/// assert_eq!(c.nnzblocks(), 2);
/// ```
pub fn contract_blocksparse<ElT>(
    a: &BlockSparseTensor<ElT, CpuBuffer<ElT>>,
    labels_a: &[i32],
    b: &BlockSparseTensor<ElT, CpuBuffer<ElT>>,
    labels_b: &[i32],
) -> Result<BlockSparseTensor<ElT, CpuBuffer<ElT>>, TensorError>
where
    ElT: Scalar + Add<Output = ElT> + Mul<Output = ElT>,
{
    // Validate label lengths
    if labels_a.len() != a.ndim() {
        return Err(TensorError::WrongNumberOfIndices {
            expected: a.ndim(),
            actual: labels_a.len(),
        });
    }
    if labels_b.len() != b.ndim() {
        return Err(TensorError::WrongNumberOfIndices {
            expected: b.ndim(),
            actual: labels_b.len(),
        });
    }

    // Compute contraction properties
    let props = ContractionProperties::compute(labels_a, &a.shape(), labels_b, &b.shape());

    // Validate dimension compatibility for contracted indices (at block level)
    for &(idx_a, idx_b) in &props.contracted_pairs {
        if a.blockdims().dim(idx_a).nblocks() != b.blockdims().dim(idx_b).nblocks() {
            return Err(TensorError::ShapeMismatch {
                expected: a.blockdims().dim(idx_a).nblocks(),
                actual: b.blockdims().dim(idx_b).nblocks(),
            });
        }
        // Also check that block sizes match
        for block_idx in 0..a.blockdims().dim(idx_a).nblocks() {
            if a.blockdims().dim(idx_a).block_size(block_idx)
                != b.blockdims().dim(idx_b).block_size(block_idx)
            {
                return Err(TensorError::ShapeMismatch {
                    expected: a.blockdims().dim(idx_a).block_size(block_idx),
                    actual: b.blockdims().dim(idx_b).block_size(block_idx),
                });
            }
        }
    }

    // Compute output block dimensions
    let output_blockdims = compute_output_blockdims(a.blockdims(), b.blockdims(), &props);

    // Find all contributing block pairs and output blocks
    let (output_blocks, block_contributions) = compute_block_contributions(a, b, &props)?;

    // Create output tensor
    let mut c = BlockSparseTensor::zeros(output_blocks, output_blockdims);

    // Contract each block pair and accumulate
    for (output_block, contributing_pairs) in block_contributions {
        // Initialize output block to zero (already done by zeros())
        let mut first = true;

        for (block_a, block_b) in contributing_pairs {
            // Get dense views of input blocks
            let dense_a = a
                .blockview(&block_a)
                .ok_or_else(|| TensorError::BlockNotFound {
                    block: block_a.coords().to_vec(),
                })?;
            let dense_b = b
                .blockview(&block_b)
                .ok_or_else(|| TensorError::BlockNotFound {
                    block: block_b.coords().to_vec(),
                })?;

            // Contract the dense blocks
            let contracted = contract_gemm(&dense_a, labels_a, &dense_b, labels_b)?;

            if first {
                // First contribution: just copy
                c.insertblock(&output_block, &contracted)?;
                first = false;
            } else {
                // Accumulate: add to existing block
                let dest =
                    c.blockview_mut(&output_block)
                        .ok_or_else(|| TensorError::BlockNotFound {
                            block: output_block.coords().to_vec(),
                        })?;
                for (d, s) in dest.iter_mut().zip(contracted.data().iter()) {
                    *d = *d + *s;
                }
            }
        }
    }

    Ok(c)
}

/// Compute output block dimensions from input block dimensions and contraction properties.
fn compute_output_blockdims(
    blockdims_a: &BlockDims,
    blockdims_b: &BlockDims,
    props: &ContractionProperties,
) -> BlockDims {
    let mut output_dims = Vec::new();

    // Add uncontracted dimensions from A
    for &idx in &props.uncontracted_a {
        output_dims.push(blockdims_a.dim(idx).clone());
    }

    // Add uncontracted dimensions from B
    for &idx in &props.uncontracted_b {
        output_dims.push(blockdims_b.dim(idx).clone());
    }

    // Handle scalar output case
    if output_dims.is_empty() {
        output_dims.push(BlockDim::new(vec![1]));
    }

    BlockDims::new(output_dims)
}

/// Compute which output blocks are non-zero and their contributing input block pairs.
///
/// Returns:
/// - List of non-zero output blocks
/// - Map from output block to list of (block_a, block_b) pairs that contribute
fn compute_block_contributions<ElT, D>(
    a: &BlockSparseTensor<ElT, D>,
    b: &BlockSparseTensor<ElT, D>,
    props: &ContractionProperties,
) -> Result<(Vec<Block>, BlockContributions), TensorError>
where
    ElT: Scalar,
    D: DataBuffer<ElT>,
{
    let mut contributions: HashMap<Block, Vec<(Block, Block)>> = HashMap::new();

    // For each non-zero block in A
    for (block_a, _) in a.iter_blocks() {
        // For each non-zero block in B
        for (block_b, _) in b.iter_blocks() {
            // Check if contracted indices match
            let mut contracted_match = true;
            for &(idx_a, idx_b) in &props.contracted_pairs {
                if block_a.coords()[idx_a] != block_b.coords()[idx_b] {
                    contracted_match = false;
                    break;
                }
            }

            if contracted_match {
                // Compute output block coordinates
                let mut output_coords = Vec::new();

                // Uncontracted dimensions from A
                for &idx in &props.uncontracted_a {
                    output_coords.push(block_a.coords()[idx]);
                }

                // Uncontracted dimensions from B
                for &idx in &props.uncontracted_b {
                    output_coords.push(block_b.coords()[idx]);
                }

                // Handle scalar output case
                if output_coords.is_empty() {
                    output_coords.push(0);
                }

                let output_block = Block::new(&output_coords);

                // Add this pair to the contributions
                contributions
                    .entry(output_block)
                    .or_default()
                    .push((block_a.clone(), block_b.clone()));
            }
        }
    }

    // Extract output blocks as a sorted list (for deterministic ordering)
    let mut output_blocks: Vec<Block> = contributions.keys().cloned().collect();
    output_blocks.sort_by_key(|b| b.coords().to_vec());

    Ok((output_blocks, contributions))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use crate::tensor::DenseTensor;
    use approx::assert_relative_eq;

    fn create_matrix_blockdims(
        block_sizes_dim0: Vec<usize>,
        block_sizes_dim1: Vec<usize>,
    ) -> BlockDims {
        BlockDims::new(vec![
            BlockDim::new(block_sizes_dim0),
            BlockDim::new(block_sizes_dim1),
        ])
    }

    #[test]
    fn test_contract_blocksparse_simple_matmul() {
        // Simple 2x2 block matrix multiplication
        // A: 4x4 with 2x2 blocks, blocks (0,0) and (1,1) non-zero
        // B: 4x4 with 2x2 blocks, blocks (0,0) and (1,1) non-zero
        let blockdims = create_matrix_blockdims(vec![2, 2], vec![2, 2]);

        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let mut a: BlockSparseTensor<f64> =
            BlockSparseTensor::zeros(blocks.clone(), blockdims.clone());
        let mut b: BlockSparseTensor<f64> =
            BlockSparseTensor::zeros(blocks.clone(), blockdims.clone());

        // Fill A with identity-like blocks
        let eye2: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        a.insertblock(&Block::new(&[0, 0]), &eye2).unwrap();
        a.insertblock(&Block::new(&[1, 1]), &eye2).unwrap();

        // Fill B with ones
        let ones2: DenseTensor<f64> = Tensor::ones(&[2, 2]);
        b.insertblock(&Block::new(&[0, 0]), &ones2).unwrap();
        b.insertblock(&Block::new(&[1, 1]), &ones2).unwrap();

        // Contract: C = A * B
        let c = contract_blocksparse(&a, &[1, -1], &b, &[-1, 2]).unwrap();

        assert_eq!(c.shape(), vec![4, 4]);
        assert_eq!(c.nnzblocks(), 2);

        // Check results: C[i,j] = A[i,k] * B[k,j]
        // Since A is identity-like, C should equal B for non-zero blocks
        let c_00 = c.blockview(&Block::new(&[0, 0])).unwrap();
        let c_11 = c.blockview(&Block::new(&[1, 1])).unwrap();

        for i in 0..4 {
            assert_relative_eq!(c_00.data()[i], ones2.data()[i], epsilon = 1e-10);
            assert_relative_eq!(c_11.data()[i], ones2.data()[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_contract_blocksparse_multiple_contributions() {
        // Test case where multiple block pairs contribute to the same output block
        // A: 4x6 with blocks [2,2] x [2,2,2]
        // B: 6x4 with blocks [2,2,2] x [2,2]
        // Blocks: A has (0,0), (0,1), (0,2); B has (0,0), (1,0), (2,0)
        // This means C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]

        let blockdims_a = BlockDims::new(vec![
            BlockDim::new(vec![2, 2]),
            BlockDim::new(vec![2, 2, 2]),
        ]);
        let blockdims_b = BlockDims::new(vec![
            BlockDim::new(vec![2, 2, 2]),
            BlockDim::new(vec![2, 2]),
        ]);

        let blocks_a = vec![
            Block::new(&[0, 0]),
            Block::new(&[0, 1]),
            Block::new(&[0, 2]),
        ];
        let blocks_b = vec![
            Block::new(&[0, 0]),
            Block::new(&[1, 0]),
            Block::new(&[2, 0]),
        ];

        let mut a: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks_a, blockdims_a);
        let mut b: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks_b, blockdims_b);

        // Fill A blocks with ones
        let ones: DenseTensor<f64> = Tensor::ones(&[2, 2]);
        a.insertblock(&Block::new(&[0, 0]), &ones).unwrap();
        a.insertblock(&Block::new(&[0, 1]), &ones).unwrap();
        a.insertblock(&Block::new(&[0, 2]), &ones).unwrap();

        // Fill B blocks with ones
        b.insertblock(&Block::new(&[0, 0]), &ones).unwrap();
        b.insertblock(&Block::new(&[1, 0]), &ones).unwrap();
        b.insertblock(&Block::new(&[2, 0]), &ones).unwrap();

        // Contract
        let c = contract_blocksparse(&a, &[1, -1], &b, &[-1, 2]).unwrap();

        assert_eq!(c.shape(), vec![4, 4]);
        assert_eq!(c.nnzblocks(), 1); // Only block (0,0) is non-zero

        // C[0,0] = 3 * (ones * ones) = 3 * 2*ones = all 6s
        let c_00 = c.blockview(&Block::new(&[0, 0])).unwrap();
        for &val in c_00.data() {
            assert_relative_eq!(val, 6.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_contract_blocksparse_matches_dense() {
        // Create block-sparse tensors
        let blockdims_a = create_matrix_blockdims(vec![2, 3], vec![4, 2]);
        let blockdims_b = create_matrix_blockdims(vec![4, 2], vec![3, 4]);

        let blocks_a = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];
        let blocks_b = vec![Block::new(&[0, 1]), Block::new(&[1, 0])];

        let mut a_sparse: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks_a, blockdims_a);
        let mut b_sparse: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks_b, blockdims_b);

        // Fill with random-ish data
        let a00: DenseTensor<f64> =
            Tensor::from_vec((1..9).map(|x| x as f64).collect(), &[2, 4]).unwrap();
        let a11: DenseTensor<f64> =
            Tensor::from_vec((1..7).map(|x| (x * 2) as f64).collect(), &[3, 2]).unwrap();
        a_sparse.insertblock(&Block::new(&[0, 0]), &a00).unwrap();
        a_sparse.insertblock(&Block::new(&[1, 1]), &a11).unwrap();

        let b01: DenseTensor<f64> =
            Tensor::from_vec((1..17).map(|x| (x + 10) as f64).collect(), &[4, 4]).unwrap();
        let b10: DenseTensor<f64> =
            Tensor::from_vec((1..7).map(|x| (x * 3) as f64).collect(), &[2, 3]).unwrap();
        b_sparse.insertblock(&Block::new(&[0, 1]), &b01).unwrap();
        b_sparse.insertblock(&Block::new(&[1, 0]), &b10).unwrap();

        // Contract using block-sparse
        let c_sparse = contract_blocksparse(&a_sparse, &[1, -1], &b_sparse, &[-1, 2]).unwrap();

        // Convert to dense and contract using dense
        let a_dense = a_sparse.to_dense();
        let b_dense = b_sparse.to_dense();
        let c_dense = contract_gemm(&a_dense, &[1, -1], &b_dense, &[-1, 2]).unwrap();

        // Convert block-sparse result to dense for comparison
        let c_sparse_dense = c_sparse.to_dense();

        // Compare
        assert_eq!(c_sparse_dense.shape(), c_dense.shape());
        for i in 0..c_dense.len() {
            assert_relative_eq!(
                *c_sparse_dense.get_linear(i).unwrap(),
                *c_dense.get_linear(i).unwrap(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_contract_blocksparse_inner_product() {
        // Inner product (full contraction to scalar)
        let blockdims = BlockDims::new(vec![BlockDim::new(vec![3, 2])]);
        let blocks = vec![Block::new(&[0]), Block::new(&[1])];

        let mut a: BlockSparseTensor<f64> =
            BlockSparseTensor::zeros(blocks.clone(), blockdims.clone());
        let mut b: BlockSparseTensor<f64> =
            BlockSparseTensor::zeros(blocks.clone(), blockdims.clone());

        let a0: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let a1: DenseTensor<f64> = Tensor::from_vec(vec![4.0, 5.0], &[2]).unwrap();
        a.insertblock(&Block::new(&[0]), &a0).unwrap();
        a.insertblock(&Block::new(&[1]), &a1).unwrap();

        let b0: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3]).unwrap();
        let b1: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 1.0], &[2]).unwrap();
        b.insertblock(&Block::new(&[0]), &b0).unwrap();
        b.insertblock(&Block::new(&[1]), &b1).unwrap();

        // Contract: scalar = sum_i A[i] * B[i]
        let c = contract_blocksparse(&a, &[-1], &b, &[-1]).unwrap();

        // Result should be 1+2+3+4+5 = 15
        let c_dense = c.to_dense();
        assert_relative_eq!(*c_dense.get_linear(0).unwrap(), 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_contract_blocksparse_complex() {
        use crate::scalar::c64;

        let blockdims = create_matrix_blockdims(vec![2, 2], vec![2, 2]);
        let blocks = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let mut a: BlockSparseTensor<c64> =
            BlockSparseTensor::zeros(blocks.clone(), blockdims.clone());
        let mut b: BlockSparseTensor<c64> =
            BlockSparseTensor::zeros(blocks.clone(), blockdims.clone());

        // Fill with complex data
        let a_data: Vec<c64> = (1..5).map(|x| c64::new(x as f64, 0.0)).collect();
        let b_data: Vec<c64> = (1..5).map(|x| c64::new(0.0, x as f64)).collect();

        let a00: DenseTensor<c64> = Tensor::from_vec(a_data.clone(), &[2, 2]).unwrap();
        let a11: DenseTensor<c64> = Tensor::from_vec(a_data, &[2, 2]).unwrap();
        a.insertblock(&Block::new(&[0, 0]), &a00).unwrap();
        a.insertblock(&Block::new(&[1, 1]), &a11).unwrap();

        let b00: DenseTensor<c64> = Tensor::from_vec(b_data.clone(), &[2, 2]).unwrap();
        let b11: DenseTensor<c64> = Tensor::from_vec(b_data, &[2, 2]).unwrap();
        b.insertblock(&Block::new(&[0, 0]), &b00).unwrap();
        b.insertblock(&Block::new(&[1, 1]), &b11).unwrap();

        // Contract and compare with dense
        let c_sparse = contract_blocksparse(&a, &[1, -1], &b, &[-1, 2]).unwrap();

        let a_dense = a.to_dense();
        let b_dense = b.to_dense();
        let c_dense = contract_gemm(&a_dense, &[1, -1], &b_dense, &[-1, 2]).unwrap();

        let c_sparse_dense = c_sparse.to_dense();

        for i in 0..c_dense.len() {
            let s = c_sparse_dense.get_linear(i).unwrap();
            let d = c_dense.get_linear(i).unwrap();
            assert_relative_eq!(s.re, d.re, epsilon = 1e-10);
            assert_relative_eq!(s.im, d.im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_contract_blocksparse_dimension_mismatch() {
        // Test that dimension mismatch is caught
        let blockdims_a = create_matrix_blockdims(vec![2, 3], vec![4, 2]);
        let blockdims_b = create_matrix_blockdims(vec![3, 3], vec![2, 2]); // Wrong block count

        let blocks_a = vec![Block::new(&[0, 0])];
        let blocks_b = vec![Block::new(&[0, 0])];

        let a: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks_a, blockdims_a);
        let b: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks_b, blockdims_b);

        let result = contract_blocksparse(&a, &[1, -1], &b, &[-1, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_contract_blocksparse_3d_tensor() {
        // Test 3D tensor contraction: C[i,l] = A[i,j,k] * B[k,l]
        let blockdims_a = BlockDims::new(vec![
            BlockDim::new(vec![2, 2]), // i
            BlockDim::new(vec![3]),    // j
            BlockDim::new(vec![2, 2]), // k
        ]);
        let blockdims_b = BlockDims::new(vec![
            BlockDim::new(vec![2, 2]), // k
            BlockDim::new(vec![2, 3]), // l
        ]);

        let blocks_a = vec![Block::new(&[0, 0, 0]), Block::new(&[1, 0, 1])];
        let blocks_b = vec![Block::new(&[0, 0]), Block::new(&[1, 1])];

        let mut a: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks_a, blockdims_a);
        let mut b: BlockSparseTensor<f64> = BlockSparseTensor::zeros(blocks_b, blockdims_b);

        // Fill with ones
        a.insertblock(&Block::new(&[0, 0, 0]), &Tensor::ones(&[2, 3, 2]))
            .unwrap();
        a.insertblock(&Block::new(&[1, 0, 1]), &Tensor::ones(&[2, 3, 2]))
            .unwrap();
        b.insertblock(&Block::new(&[0, 0]), &Tensor::ones(&[2, 2]))
            .unwrap();
        b.insertblock(&Block::new(&[1, 1]), &Tensor::ones(&[2, 3]))
            .unwrap();

        // Contract: C[i,j,l] = A[i,j,k] * B[k,l]
        let c = contract_blocksparse(&a, &[1, 2, -1], &b, &[-1, 3]).unwrap();

        // Compare with dense
        let a_dense = a.to_dense();
        let b_dense = b.to_dense();
        let c_dense = contract_gemm(&a_dense, &[1, 2, -1], &b_dense, &[-1, 3]).unwrap();

        let c_sparse_dense = c.to_dense();

        assert_eq!(c_sparse_dense.shape(), c_dense.shape());
        for i in 0..c_dense.len() {
            assert_relative_eq!(
                *c_sparse_dense.get_linear(i).unwrap(),
                *c_dense.get_linear(i).unwrap(),
                epsilon = 1e-10
            );
        }
    }
}
