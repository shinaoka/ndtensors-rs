//! Tensor contraction operations.
//!
//! Uses label-based contraction following NDTensors.jl convention:
//! - Negative labels indicate contracted indices
//! - Positive labels indicate uncontracted (output) indices

use crate::error::TensorError;
use crate::scalar::Scalar;
use crate::strides::{cartesian_to_linear, linear_to_cartesian};
use crate::tensor::Tensor;
use std::ops::{Add, Mul};

/// Contract two tensors using label-based contraction.
///
/// Labels are integers where:
/// - Negative values indicate contracted indices (matched between tensors)
/// - Positive values indicate uncontracted indices (appear in output)
///
/// # Arguments
///
/// * `a` - First tensor
/// * `labels_a` - Labels for each dimension of `a`
/// * `b` - Second tensor
/// * `labels_b` - Labels for each dimension of `b`
///
/// # Returns
///
/// Result tensor with dimensions corresponding to positive labels (from A then B)
///
/// # Examples
///
/// ```
/// use ndtensors::{Tensor, contract};
///
/// // Matrix multiplication: C[i,k] = A[i,j] * B[j,k]
/// // A is 2x3, B is 3x4, C is 2x4
/// let a = Tensor::<f64>::ones(&[2, 3]);
/// let b = Tensor::<f64>::ones(&[3, 4]);
///
/// // labels: A[1,-1], B[-1,2] -> C[1,2]
/// // -1 is contracted, 1 and 2 are output dimensions
/// let c = contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();
/// assert_eq!(c.shape(), &[2, 4]);
/// ```
pub fn contract<T: Scalar + Add<Output = T> + Mul<Output = T>>(
    a: &Tensor<T>,
    labels_a: &[i32],
    b: &Tensor<T>,
    labels_b: &[i32],
) -> Result<Tensor<T>, TensorError> {
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

    // Find contracted pairs (negative labels that appear in both)
    let mut contracted_pairs: Vec<(usize, usize)> = Vec::new();
    for (i, &la) in labels_a.iter().enumerate() {
        if la < 0 {
            for (j, &lb) in labels_b.iter().enumerate() {
                if la == lb {
                    // Check dimension compatibility
                    if a.shape()[i] != b.shape()[j] {
                        return Err(TensorError::ShapeMismatch {
                            expected: a.shape()[i],
                            actual: b.shape()[j],
                        });
                    }
                    contracted_pairs.push((i, j));
                    break;
                }
            }
        }
    }

    // Determine output dimensions
    // Uncontracted from A (positive labels)
    let uncontracted_a: Vec<usize> = labels_a
        .iter()
        .enumerate()
        .filter(|&(_, l)| *l > 0)
        .map(|(i, _)| i)
        .collect();

    // Uncontracted from B (positive labels)
    let uncontracted_b: Vec<usize> = labels_b
        .iter()
        .enumerate()
        .filter(|&(_, l)| *l > 0)
        .map(|(i, _)| i)
        .collect();

    // Build output shape
    let mut output_shape: Vec<usize> = Vec::new();
    for &i in &uncontracted_a {
        output_shape.push(a.shape()[i]);
    }
    for &j in &uncontracted_b {
        output_shape.push(b.shape()[j]);
    }

    // Handle scalar output (all indices contracted)
    if output_shape.is_empty() {
        output_shape.push(1);
    }

    let mut result = Tensor::<T>::zeros(&output_shape);

    // Get contracted dimension sizes
    let contracted_dims: Vec<usize> = contracted_pairs
        .iter()
        .map(|&(i, _)| a.shape()[i])
        .collect();

    // Total iterations over contracted indices
    let contracted_total: usize = contracted_dims.iter().product::<usize>().max(1);

    // Iterate over output indices
    let output_total = result.len();

    for out_linear in 0..output_total {
        let out_indices = linear_to_cartesian(out_linear, &output_shape);

        let mut sum = T::zero();

        // Iterate over contracted indices
        for contracted_linear in 0..contracted_total {
            let contracted_indices = linear_to_cartesian(contracted_linear, &contracted_dims);

            // Build A indices
            let mut a_indices = vec![0usize; a.ndim()];
            let mut out_idx = 0;
            for i in 0..a.ndim() {
                if labels_a[i] > 0 {
                    a_indices[i] = out_indices[out_idx];
                    out_idx += 1;
                } else {
                    // Find which contracted pair this is
                    if let Some(cp_idx) = contracted_pairs.iter().position(|&(ai, _)| ai == i) {
                        a_indices[i] = contracted_indices[cp_idx];
                    }
                }
            }

            // Build B indices
            let mut b_indices = vec![0usize; b.ndim()];
            for j in 0..b.ndim() {
                if labels_b[j] > 0 {
                    b_indices[j] = out_indices[out_idx];
                    out_idx += 1;
                } else {
                    // Find which contracted pair this is
                    if let Some(cp_idx) = contracted_pairs.iter().position(|&(_, bj)| bj == j) {
                        b_indices[j] = contracted_indices[cp_idx];
                    }
                }
            }

            // Get values and accumulate
            let a_linear = cartesian_to_linear(&a_indices, a.strides());
            let b_linear = cartesian_to_linear(&b_indices, b.strides());

            let a_val = *a.get_linear(a_linear).unwrap();
            let b_val = *b.get_linear(b_linear).unwrap();

            sum = sum + a_val * b_val;
        }

        result.data_mut()[out_linear] = sum;
    }

    // Handle scalar case - return shape [] instead of [1]
    if output_shape == [1] && uncontracted_a.is_empty() && uncontracted_b.is_empty() {
        // Keep as [1] for simplicity in FFI
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_matrix_multiply() {
        // A is 2x3, B is 3x4 -> C is 2x4
        // A[i,j] * B[j,k] -> C[i,k]
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = Tensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
        )
        .unwrap();

        // labels: A[1,-1], B[-1,2]
        let c = contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();
        assert_eq!(c.shape(), &[2, 4]);

        // Verify: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
        //                = 1*1 + 3*2 + 5*3 = 1 + 6 + 15 = 22
        assert_relative_eq!(*c.get(&[0, 0]).unwrap(), 22.0);
    }

    #[test]
    fn test_inner_product() {
        // Vector inner product: a[i] * b[i] -> scalar
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();

        // labels: A[-1], B[-1] -> scalar
        let c = contract(&a, &[-1], &b, &[-1]).unwrap();
        assert_eq!(c.len(), 1);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_relative_eq!(*c.get_linear(0).unwrap(), 32.0);
    }

    #[test]
    fn test_outer_product() {
        // Outer product: a[i] * b[j] -> C[i,j]
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let b = Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap();

        // labels: A[1], B[2] -> C[1,2] (no contraction)
        let c = contract(&a, &[1], &b, &[2]).unwrap();
        assert_eq!(c.shape(), &[2, 3]);

        // C[i,j] = a[i] * b[j]
        assert_relative_eq!(*c.get(&[0, 0]).unwrap(), 3.0); // 1*3
        assert_relative_eq!(*c.get(&[1, 0]).unwrap(), 6.0); // 2*3
        assert_relative_eq!(*c.get(&[0, 1]).unwrap(), 4.0); // 1*4
        assert_relative_eq!(*c.get(&[1, 2]).unwrap(), 10.0); // 2*5
    }

    #[test]
    fn test_tensor_contraction_3d() {
        // A[i,j,k] * B[k,l] -> C[i,j,l]
        let a = Tensor::<f64>::ones(&[2, 3, 4]);
        let b = Tensor::<f64>::ones(&[4, 5]);

        let c = contract(&a, &[1, 2, -1], &b, &[-1, 3]).unwrap();
        assert_eq!(c.shape(), &[2, 3, 5]);

        // Each element should be 4 (sum over k dimension of size 4)
        assert_relative_eq!(*c.get(&[0, 0, 0]).unwrap(), 4.0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let a = Tensor::<f64>::ones(&[2, 3]);
        let b = Tensor::<f64>::ones(&[4, 5]);

        // Try to contract dimensions of different sizes
        let result = contract(&a, &[1, -1], &b, &[-1, 2]);
        assert!(result.is_err());
    }
}
