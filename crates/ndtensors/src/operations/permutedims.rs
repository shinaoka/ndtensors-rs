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
pub fn permutedims<ElT: Scalar>(
    tensor: &DenseTensor<ElT>,
    perm: &[usize],
) -> Result<DenseTensor<ElT>, TensorError> {
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
pub fn permutedims_into<ElT: Scalar>(
    dest: &mut DenseTensor<ElT>,
    src: &DenseTensor<ElT>,
    perm: &[usize],
) {
    // TODO: Add backend selection (thread_local or parameter)
    // For now, always use GenericBackend
    GenericBackend::permute_into(dest, src, perm);
}

/// Permute dimensions with a combining function.
///
/// Like `permutedims`, but applies a function that combines the existing
/// result value with the permuted source value.
///
/// This is useful for operations like accumulating permuted tensors:
/// `result[perm_indices] = f(result[perm_indices], src[indices])`
///
/// # Arguments
///
/// * `result` - Output tensor (must have correct permuted shape)
/// * `src` - Input tensor
/// * `perm` - Permutation of dimensions
/// * `f` - Binary function to combine result and source values
///
/// # Errors
///
/// Returns error if `perm` is not a valid permutation of `0..ndim`.
///
/// # Example
///
/// ```
/// use ndtensors::DenseTensor;
/// use ndtensors::operations::permutedims_with;
///
/// let src: DenseTensor<f64> = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
/// let mut result: DenseTensor<f64> = DenseTensor::ones(&[2, 2]);
///
/// // Add transposed tensor to result
/// permutedims_with(&mut result, &src, &[1, 0], |r, s| r + s).unwrap();
///
/// // result[i,j] = 1.0 + src[j,i]
/// assert_eq!(*result.get(&[0, 0]).unwrap(), 2.0); // 1 + src[0,0] = 1 + 1
/// assert_eq!(*result.get(&[0, 1]).unwrap(), 3.0); // 1 + src[1,0] = 1 + 2
/// ```
pub fn permutedims_with<ElT: Scalar, F>(
    result: &mut DenseTensor<ElT>,
    src: &DenseTensor<ElT>,
    perm: &[usize],
    f: F,
) -> Result<(), TensorError>
where
    F: Fn(ElT, ElT) -> ElT,
{
    // Validate permutation
    validate_permutation(perm, src.ndim())?;

    // Validate shapes match
    let expected_shape: Vec<usize> = perm.iter().map(|&p| src.shape()[p]).collect();
    if result.shape() != expected_shape {
        return Err(TensorError::ShapeMismatch {
            expected: expected_shape.iter().product(),
            actual: result.len(),
        });
    }

    // Compute inverse permutation for mapping dest indices to src indices
    let ndim = perm.len();
    let mut inv_perm = vec![0usize; ndim];
    for (i, &p) in perm.iter().enumerate() {
        inv_perm[p] = i;
    }

    // Iterate over all indices
    let src_shape = src.shape();
    let mut src_indices = vec![0usize; ndim];

    let total_elements = src.len();
    for _ in 0..total_elements {
        // Compute destination indices from source indices
        let dest_indices: Vec<usize> = inv_perm.iter().map(|&i| src_indices[i]).collect();

        // Apply function
        let src_val = *src.get(&src_indices).unwrap();
        let dest_val = *result.get(&dest_indices).unwrap();
        result.set(&dest_indices, f(dest_val, src_val)).unwrap();

        // Increment source indices (column-major)
        for d in 0..ndim {
            src_indices[d] += 1;
            if src_indices[d] < src_shape[d] {
                break;
            }
            src_indices[d] = 0;
        }
    }

    Ok(())
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

    #[test]
    fn test_permutedims_with_add() {
        let src: DenseTensor<f64> =
            DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let mut result: DenseTensor<f64> = DenseTensor::ones(&[2, 2]);

        // Add transposed tensor to result
        permutedims_with(&mut result, &src, &[1, 0], |r, s| r + s).unwrap();

        // result[i,j] = 1.0 + src[j,i]
        assert_eq!(*result.get(&[0, 0]).unwrap(), 2.0); // 1 + src[0,0] = 1 + 1
        assert_eq!(*result.get(&[0, 1]).unwrap(), 3.0); // 1 + src[1,0] = 1 + 2
        assert_eq!(*result.get(&[1, 0]).unwrap(), 4.0); // 1 + src[0,1] = 1 + 3
        assert_eq!(*result.get(&[1, 1]).unwrap(), 5.0); // 1 + src[1,1] = 1 + 4
    }

    #[test]
    fn test_permutedims_with_multiply() {
        let src: DenseTensor<f64> =
            DenseTensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]).unwrap();
        let mut result: DenseTensor<f64> =
            DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        // Multiply by transposed tensor
        permutedims_with(&mut result, &src, &[1, 0], |r, s| r * s).unwrap();

        // result[i,j] = result[i,j] * src[j,i]
        assert_eq!(*result.get(&[0, 0]).unwrap(), 2.0); // 1 * 2
        assert_eq!(*result.get(&[0, 1]).unwrap(), 9.0); // 3 * 3
        assert_eq!(*result.get(&[1, 0]).unwrap(), 8.0); // 2 * 4
        assert_eq!(*result.get(&[1, 1]).unwrap(), 20.0); // 4 * 5
    }

    #[test]
    fn test_permutedims_with_identity_perm() {
        let src: DenseTensor<f64> =
            DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let mut result: DenseTensor<f64> = DenseTensor::zeros(&[2, 2]);

        // Copy with identity permutation
        permutedims_with(&mut result, &src, &[0, 1], |_r, s| s).unwrap();

        assert_eq!(result.data(), src.data());
    }

    #[test]
    fn test_permutedims_with_invalid_perm() {
        let src: DenseTensor<f64> = DenseTensor::zeros(&[2, 3]);
        let mut result: DenseTensor<f64> = DenseTensor::zeros(&[3, 2]);

        // Invalid permutation
        assert!(permutedims_with(&mut result, &src, &[0], |r, s| r + s).is_err());
        assert!(permutedims_with(&mut result, &src, &[0, 0], |r, s| r + s).is_err());
    }
}
