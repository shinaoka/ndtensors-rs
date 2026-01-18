//! Permutation operations for tensors.
//!
//! Following NDTensors.jl's dispatch hierarchy:
//!
//! ```text
//! permutedims(tensor, perm)           # DenseTensor specialization
//!     → validate permutation
//!     → allocate output with permuted shape
//!     → permutedims_into(output, tensor, perm)
//!
//! permutedims_into(dest, src, perm)   # Low-level backend dispatch
//!     → dispatch to backend (GenericBackend by default)
//! ```

use crate::backend::{GenericBackend, PermutationBackend};
use crate::error::TensorError;
use crate::scalar::Scalar;
use crate::tensor::DenseTensor;

/// Permute the dimensions of a DenseTensor, returning a new DenseTensor.
///
/// This is the DenseTensor specialization, following NDTensors.jl's dispatch pattern.
///
/// # Arguments
///
/// * `tensor` - Input DenseTensor
/// * `perm` - Permutation of dimensions. `perm[i]` gives the source dimension
///   for the i-th dimension of the result.
///
/// # Errors
///
/// Returns error if `perm` is not a valid permutation of `0..ndim`.
///
/// # Examples
///
/// ```
/// use ndtensors::DenseTensor;
/// use ndtensors::operations::permutedims;
///
/// // Create a 2x3 tensor
/// let t: DenseTensor<f64> = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
///
/// // Transpose (swap dimensions 0 and 1)
/// let t2 = permutedims(&t, &[1, 0]).unwrap();
/// assert_eq!(t2.shape(), &[3, 2]);
///
/// // t[i,j] == t2[j,i]
/// assert_eq!(t.get(&[0, 0]), t2.get(&[0, 0]));
/// assert_eq!(t.get(&[1, 0]), t2.get(&[0, 1]));
/// ```
pub fn permutedims<T: Scalar>(
    tensor: &DenseTensor<T>,
    perm: &[usize],
) -> Result<DenseTensor<T>, TensorError> {
    // Validate permutation
    validate_permutation(perm, tensor.ndim())?;

    // Compute new shape
    let new_shape: Vec<usize> = perm.iter().map(|&p| tensor.shape()[p]).collect();

    // Create output tensor
    let mut result = DenseTensor::zeros(&new_shape);

    // Dispatch to backend
    permutedims_into(&mut result, tensor, perm);

    Ok(result)
}

/// Permute DenseTensor dimensions into an existing output tensor (in-place).
///
/// This is the low-level function that dispatches to the backend.
/// Currently uses `GenericBackend` (naive loops).
///
/// # Arguments
///
/// * `dest` - Output DenseTensor (must have correct permuted shape)
/// * `src` - Input DenseTensor
/// * `perm` - Permutation of dimensions
///
/// # Panics
///
/// Panics if dest shape doesn't match the permuted src shape.
///
/// # Examples
///
/// ```
/// use ndtensors::DenseTensor;
/// use ndtensors::operations::permutedims_into;
///
/// let src: DenseTensor<f64> = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
/// let mut dest: DenseTensor<f64> = DenseTensor::zeros(&[3, 2]);
///
/// permutedims_into(&mut dest, &src, &[1, 0]);
/// assert_eq!(src.get(&[1, 2]), dest.get(&[2, 1]));
/// ```
pub fn permutedims_into<T: Scalar>(
    dest: &mut DenseTensor<T>,
    src: &DenseTensor<T>,
    perm: &[usize],
) {
    // TODO: Add backend selection (thread_local or parameter)
    // For now, always use GenericBackend
    GenericBackend::permute_into(dest, src, perm);
}

/// Validate that perm is a valid permutation of 0..ndim.
fn validate_permutation(perm: &[usize], ndim: usize) -> Result<(), TensorError> {
    if perm.len() != ndim {
        return Err(TensorError::InvalidPermutation {
            perm: perm.to_vec(),
            ndim,
        });
    }

    let mut seen = vec![false; ndim];
    for &p in perm {
        if p >= ndim {
            return Err(TensorError::InvalidPermutation {
                perm: perm.to_vec(),
                ndim,
            });
        }
        if seen[p] {
            return Err(TensorError::InvalidPermutation {
                perm: perm.to_vec(),
                ndim,
            });
        }
        seen[p] = true;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::c64;

    fn test_permutedims_transpose_generic<T: Scalar + From<f64>>() {
        let data: Vec<T> = (1..=6).map(|x| T::from(x as f64)).collect();
        let t: DenseTensor<T> = DenseTensor::from_vec(data, &[2, 3]).unwrap();

        let t2 = permutedims(&t, &[1, 0]).unwrap();
        assert_eq!(t2.shape(), &[3, 2]);

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(t.get(&[i, j]), t2.get(&[j, i]));
            }
        }
    }

    #[test]
    fn test_permutedims_transpose_f64() {
        test_permutedims_transpose_generic::<f64>();
    }

    #[test]
    fn test_permutedims_transpose_c64() {
        test_permutedims_transpose_generic::<c64>();
    }

    #[test]
    fn test_permutedims_3d() {
        let mut t: DenseTensor<f64> = DenseTensor::zeros(&[2, 3, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    t.set(&[i, j, k], (i * 100 + j * 10 + k) as f64).unwrap();
                }
            }
        }

        // Permute [0,1,2] -> [2,0,1]: shape 2x3x4 -> 4x2x3
        let t2 = permutedims(&t, &[2, 0, 1]).unwrap();
        assert_eq!(t2.shape(), &[4, 2, 3]);

        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(t.get(&[i, j, k]), t2.get(&[k, i, j]));
                }
            }
        }
    }

    #[test]
    fn test_permutedims_identity() {
        let t: DenseTensor<f64> =
            DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let t2 = permutedims(&t, &[0, 1]).unwrap();
        assert_eq!(t2.shape(), &[2, 3]);
        assert_eq!(t.data(), t2.data());
    }

    #[test]
    fn test_permutedims_invalid() {
        let t: DenseTensor<f64> = DenseTensor::zeros(&[2, 3]);

        // Wrong number of dimensions
        assert!(permutedims(&t, &[0]).is_err());
        assert!(permutedims(&t, &[0, 1, 2]).is_err());

        // Invalid index
        assert!(permutedims(&t, &[0, 2]).is_err());

        // Duplicate index
        assert!(permutedims(&t, &[0, 0]).is_err());
    }

    #[test]
    fn test_permutedims_into() {
        let src: DenseTensor<f64> =
            DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let mut dest: DenseTensor<f64> = DenseTensor::zeros(&[3, 2]);

        permutedims_into(&mut dest, &src, &[1, 0]);

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(src.get(&[i, j]), dest.get(&[j, i]));
            }
        }
    }
}
