//! Utility functions for tensor decomposition.
//!
//! This module provides helper functions for preparing tensors for matrix decompositions.

use crate::error::TensorError;
use crate::operations::permutedims;
use crate::scalar::Scalar;
use crate::tensor::DenseTensor;

/// Result of permute_reshape operation.
pub struct PermuteReshapeResult<ElT: Scalar> {
    /// The reshaped tensor as a 2D matrix.
    pub matrix: DenseTensor<ElT>,
    /// Number of rows (product of left dimensions).
    pub nrows: usize,
    /// Number of columns (product of right dimensions).
    pub ncols: usize,
    /// Original shape of the tensor.
    pub original_shape: Vec<usize>,
    /// Left dimensions in the original shape.
    pub left_dims: Vec<usize>,
    /// Right dimensions in the original shape.
    pub right_dims: Vec<usize>,
}

/// Permute and reshape a tensor for matrix decomposition.
///
/// This function reorders the tensor dimensions so that the specified left indices
/// come first, followed by the right indices. The result is then reshaped into a
/// 2D matrix suitable for SVD or QR decomposition.
///
/// # Arguments
///
/// * `tensor` - The input tensor
/// * `left_inds` - Indices to place on the left (become rows)
/// * `right_inds` - Indices to place on the right (become columns)
///
/// # Returns
///
/// A `PermuteReshapeResult` containing the reshaped matrix and dimension information.
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::decomposition::permute_reshape;
///
/// // 2x3x4 tensor -> 6x4 matrix (left_inds=[0,1], right_inds=[2])
/// let t = Tensor::<f64>::ones(&[2, 3, 4]);
/// let result = permute_reshape(&t, &[0, 1], &[2]).unwrap();
///
/// assert_eq!(result.nrows, 6);  // 2 * 3
/// assert_eq!(result.ncols, 4);
/// ```
pub fn permute_reshape<ElT: Scalar>(
    tensor: &DenseTensor<ElT>,
    left_inds: &[usize],
    right_inds: &[usize],
) -> Result<PermuteReshapeResult<ElT>, TensorError> {
    let ndim = tensor.ndim();
    let shape = tensor.shape();

    // Validate indices
    let mut all_inds: Vec<usize> = left_inds.iter().chain(right_inds.iter()).copied().collect();
    all_inds.sort();
    all_inds.dedup();

    if all_inds.len() != ndim {
        return Err(TensorError::InvalidPermutation {
            perm: all_inds,
            ndim,
        });
    }

    for &i in &all_inds {
        if i >= ndim {
            return Err(TensorError::InvalidPermutation {
                perm: vec![i],
                ndim,
            });
        }
    }

    // Build permutation: left_inds followed by right_inds
    let perm: Vec<usize> = left_inds.iter().chain(right_inds.iter()).copied().collect();

    // Compute dimensions
    let left_dims: Vec<usize> = left_inds.iter().map(|&i| shape[i]).collect();
    let right_dims: Vec<usize> = right_inds.iter().map(|&i| shape[i]).collect();

    let nrows: usize = left_dims.iter().product::<usize>().max(1);
    let ncols: usize = right_dims.iter().product::<usize>().max(1);

    // Check if permutation is identity (no permute needed)
    let is_identity = perm.iter().enumerate().all(|(i, &p)| i == p);

    let matrix = if is_identity {
        // No permutation needed, just reshape
        DenseTensor::from_vec(tensor.data().to_vec(), &[nrows, ncols])?
    } else {
        // Permute then reshape
        let permuted = permutedims(tensor, &perm)?;
        DenseTensor::from_vec(permuted.data().to_vec(), &[nrows, ncols])?
    };

    Ok(PermuteReshapeResult {
        matrix,
        nrows,
        ncols,
        original_shape: shape.to_vec(),
        left_dims,
        right_dims,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_permute_reshape_identity() {
        // No permutation needed
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let result = permute_reshape(&t, &[0], &[1]).unwrap();

        assert_eq!(result.nrows, 2);
        assert_eq!(result.ncols, 3);
        assert_eq!(result.matrix.shape(), &[2, 3]);

        // Data should be unchanged
        for i in 0..6 {
            assert_relative_eq!(
                *result.matrix.get_linear(i).unwrap(),
                *t.get_linear(i).unwrap()
            );
        }
    }

    #[test]
    fn test_permute_reshape_transpose() {
        // Transpose: left=[1], right=[0]
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let result = permute_reshape(&t, &[1], &[0]).unwrap();

        assert_eq!(result.nrows, 3);
        assert_eq!(result.ncols, 2);
        assert_eq!(result.matrix.shape(), &[3, 2]);
    }

    #[test]
    fn test_permute_reshape_3d() {
        // 2x3x4 -> 6x4 (left=[0,1], right=[2])
        let t = DenseTensor::<f64>::ones(&[2, 3, 4]);
        let result = permute_reshape(&t, &[0, 1], &[2]).unwrap();

        assert_eq!(result.nrows, 6);
        assert_eq!(result.ncols, 4);
        assert_eq!(result.left_dims, vec![2, 3]);
        assert_eq!(result.right_dims, vec![4]);
    }

    #[test]
    fn test_permute_reshape_3d_permuted() {
        // 2x3x4 -> 8x3 (left=[0,2], right=[1])
        let t = DenseTensor::<f64>::ones(&[2, 3, 4]);
        let result = permute_reshape(&t, &[0, 2], &[1]).unwrap();

        assert_eq!(result.nrows, 8); // 2 * 4
        assert_eq!(result.ncols, 3);
        assert_eq!(result.left_dims, vec![2, 4]);
        assert_eq!(result.right_dims, vec![3]);
    }

    #[test]
    fn test_permute_reshape_invalid_indices() {
        let t = DenseTensor::<f64>::ones(&[2, 3]);

        // Missing index
        let result = permute_reshape(&t, &[0], &[]);
        assert!(result.is_err());

        // Duplicate index
        let result = permute_reshape(&t, &[0, 0], &[1]);
        assert!(result.is_err());

        // Out of bounds
        let result = permute_reshape(&t, &[0], &[5]);
        assert!(result.is_err());
    }
}
