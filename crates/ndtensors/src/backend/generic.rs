//! Generic (naive loop-based) backend implementation.

use crate::backend::PermutationBackend;
use crate::scalar::Scalar;
use crate::strides::{cartesian_to_linear, linear_to_cartesian};
use crate::tensor::DenseTensor;

/// Generic backend using naive loop-based implementations.
///
/// This backend is always available and serves as a fallback.
/// It's suitable for small tensors and debugging.
pub struct GenericBackend;

impl PermutationBackend for GenericBackend {
    fn permute_into<ElT: Scalar>(
        dest: &mut DenseTensor<ElT>,
        src: &DenseTensor<ElT>,
        perm: &[usize],
    ) {
        let old_shape = src.shape();
        // Copy strides to avoid borrow conflict with data_mut()
        let new_strides: Vec<usize> = dest.strides().to_vec();

        // Iterate over all elements in source
        let total = src.len();
        for linear_old in 0..total {
            // Convert to old cartesian indices
            let old_indices = linear_to_cartesian(linear_old, old_shape);

            // Permute to new indices: new_indices[i] = old_indices[perm[i]]
            let new_indices: Vec<usize> = perm.iter().map(|&p| old_indices[p]).collect();

            // Convert to new linear index
            let linear_new = cartesian_to_linear(&new_indices, &new_strides);

            // Copy value
            dest.data_mut()[linear_new] = src.data()[linear_old];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generic_permute_transpose() {
        // 2x3 matrix
        let src: DenseTensor<f64> =
            DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Create destination with transposed shape
        let mut dest: DenseTensor<f64> = DenseTensor::zeros(&[3, 2]);

        GenericBackend::permute_into(&mut dest, &src, &[1, 0]);

        // Verify: src[i,j] == dest[j,i]
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(src.get(&[i, j]), dest.get(&[j, i]));
            }
        }
    }

    #[test]
    fn test_generic_permute_3d() {
        // 2x3x4 tensor
        let mut src: DenseTensor<f64> = DenseTensor::zeros(&[2, 3, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    src.set(&[i, j, k], (i * 100 + j * 10 + k) as f64).unwrap();
                }
            }
        }

        // Permute [0,1,2] -> [2,0,1]: shape 2x3x4 -> 4x2x3
        let mut dest: DenseTensor<f64> = DenseTensor::zeros(&[4, 2, 3]);
        GenericBackend::permute_into(&mut dest, &src, &[2, 0, 1]);

        // Verify: src[i,j,k] == dest[k,i,j]
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(src.get(&[i, j, k]), dest.get(&[k, i, j]));
                }
            }
        }
    }
}
