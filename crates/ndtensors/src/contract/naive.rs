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
pub fn contract<ElT: Scalar + Add<Output = ElT> + Mul<Output = ElT>>(
    a: &Tensor<ElT>,
    labels_a: &[i32],
    b: &Tensor<ElT>,
    labels_b: &[i32],
) -> Result<Tensor<ElT>, TensorError> {
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

    let mut result = Tensor::<ElT>::zeros(&output_shape);

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

        let mut sum = ElT::zero();

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

/// Compute the vector-Jacobian product (VJP) for tensor contraction.
///
/// Given the forward pass `c = contract(a, labels_a, b, labels_b)` and the gradient
/// of the loss with respect to `c` (`grad_output`), this computes the gradients
/// with respect to `a` and `b`.
///
/// # Arguments
///
/// * `a` - First tensor from forward pass
/// * `labels_a` - Labels for each dimension of `a`
/// * `b` - Second tensor from forward pass
/// * `labels_b` - Labels for each dimension of `b`
/// * `grad_output` - Gradient of loss with respect to output
///
/// # Returns
///
/// Tuple of (grad_a, grad_b) - gradients with respect to inputs
///
/// # Examples
///
/// ```
/// use ndtensors::{Tensor, contract, contract_vjp};
///
/// // Matrix multiplication: C[i,k] = A[i,j] * B[j,k]
/// let a = Tensor::<f64>::ones(&[2, 3]);
/// let b = Tensor::<f64>::ones(&[3, 4]);
/// let grad_c = Tensor::<f64>::ones(&[2, 4]);
///
/// // VJP: grad_A = grad_C @ B^T, grad_B = A^T @ grad_C
/// let (grad_a, grad_b) = contract_vjp(&a, &[1, -1], &b, &[-1, 2], &grad_c).unwrap();
/// assert_eq!(grad_a.shape(), &[2, 3]);
/// assert_eq!(grad_b.shape(), &[3, 4]);
/// ```
pub fn contract_vjp<ElT: Scalar + Add<Output = ElT> + Mul<Output = ElT>>(
    a: &Tensor<ElT>,
    labels_a: &[i32],
    b: &Tensor<ElT>,
    labels_b: &[i32],
    grad_output: &Tensor<ElT>,
) -> Result<(Tensor<ElT>, Tensor<ElT>), TensorError> {
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

    // VJP for contract(A, labels_a, B, labels_b) = C
    //
    // Original contraction:
    // - A has dimensions with labels_a (positive = free, negative = contracted with B)
    // - B has dimensions with labels_b (positive = free, negative = contracted with A)
    // - C has dimensions: [A's positive labels in order] ++ [B's positive labels in order]
    //
    // VJP rules:
    // - grad_A = contract(grad_C, B) where we contract over B's free indices
    // - grad_B = contract(A, grad_C) where we contract over A's free indices

    // Find max absolute label to create unique new labels
    let max_label = labels_a
        .iter()
        .chain(labels_b.iter())
        .map(|&l| l.abs())
        .max()
        .unwrap_or(0);

    // Check for scalar output case (all indices contracted)
    let a_has_positive = labels_a.iter().any(|&l| l > 0);
    let b_has_positive = labels_b.iter().any(|&l| l > 0);
    let is_scalar_output = !a_has_positive && !b_has_positive;

    // Compute grad_a
    let grad_a = {
        // We want: grad_A[i1, i2, ...] = sum over B's free indices of grad_C * B
        //
        // grad_C has shape corresponding to: [A's positive] ++ [B's positive]
        // B has shape corresponding to: [B's labels]
        //
        // For grad_A:
        // - A's positive labels in grad_C → remain as output
        // - B's positive labels in grad_C → contract with corresponding B dimensions
        // - B's negative labels → become output (these are the contracted indices)

        let mut label_counter = max_label + 1;

        let b_positive_count = labels_b.iter().filter(|&&l| l > 0).count();

        let mut grad_c_labels = Vec::new();

        // First part: A's positive labels - use unique positive labels for output
        for &l in labels_a.iter().filter(|&&l| l > 0) {
            grad_c_labels.push(l); // Keep original positive label
        }

        // Second part: B's positive labels - make them negative to contract with B
        let mut b_pos_contract_labels = Vec::new();
        for _ in 0..b_positive_count {
            let neg_label = -label_counter;
            grad_c_labels.push(neg_label);
            b_pos_contract_labels.push(neg_label);
            label_counter += 1;
        }

        // Track what positive labels we assign to B's negative dimensions
        let mut b_neg_to_pos = std::collections::HashMap::new();

        // Handle scalar output - need to treat grad_output specially
        if is_scalar_output {
            // Scalar output: grad_A = grad_output[0] * B
            // This is a simple scalar multiplication, not a contraction
            // grad_output is shape [1], B has same shape as A
            let scalar = *grad_output
                .get_linear(0)
                .ok_or(TensorError::IndexOutOfBounds {
                    index: 0,
                    dim_size: grad_output.len(),
                })?;

            // Result is scalar * B, but we need it in A's dimension order
            // Since all labels are negative, A and B have matching contracted indices
            // B's shape corresponds to those indices

            // Create result with B's values scaled by grad_output scalar
            let result_data: Vec<ElT> = b.data().iter().map(|&v| v * scalar).collect();

            // B and A might have different dimension orderings for the same contracted indices
            // Build mapping from B's order to A's order
            let b_result = Tensor::from_vec(result_data.clone(), b.shape())?;

            // Map B's negative labels to their positions
            let b_label_to_idx: std::collections::HashMap<i32, usize> =
                labels_b.iter().enumerate().map(|(i, &l)| (l, i)).collect();

            // Build permutation from B's order to A's order
            let perm: Vec<usize> = labels_a
                .iter()
                .map(|&la| *b_label_to_idx.get(&la).expect("Label not found in B"))
                .collect();

            // Check if permutation is identity
            let is_identity = perm.iter().enumerate().all(|(i, &p)| i == p);
            if is_identity {
                // No permutation needed, but reshape to A's shape
                Tensor::from_vec(result_data, a.shape())?
            } else {
                b_result.permutedims(&perm)?
            }
        } else {
            // Create labels for B
            // B's positive labels → contract with grad_C (use corresponding negative labels)
            // B's negative labels → become positive in output (contracted with A in forward)
            let mut b_labels_new = Vec::new();
            let mut b_pos_idx = 0;

            for &l in labels_b {
                if l < 0 {
                    // Make it positive with a new unique label
                    let pos_label = label_counter;
                    b_labels_new.push(pos_label);
                    b_neg_to_pos.insert(l, pos_label);
                    label_counter += 1;
                } else {
                    // Use the corresponding contraction label
                    b_labels_new.push(b_pos_contract_labels[b_pos_idx]);
                    b_pos_idx += 1;
                }
            }

            // Contract grad_C with B
            let result = contract(grad_output, &grad_c_labels, b, &b_labels_new)?;

            // Result has labels: [A's positive] ++ [new positive for B's negative, in B's order]
            // We need to permute to match A's label order

            // Build the label order of the result
            let mut result_label_order = Vec::new();
            for &l in labels_a {
                if l > 0 {
                    result_label_order.push(l);
                }
            }
            for &l in labels_b {
                if l < 0 {
                    result_label_order.push(b_neg_to_pos[&l]);
                }
            }

            // Build the target label order (A's shape)
            // For A's positive labels: use the label itself
            // For A's negative labels: find the corresponding positive label we assigned
            let perm: Vec<usize> = labels_a
                .iter()
                .map(|&la| {
                    if la > 0 {
                        result_label_order.iter().position(|&r| r == la).unwrap()
                    } else {
                        // A's negative label should match B's negative label
                        // B's negative label got assigned a positive label via b_neg_to_pos
                        let pos = b_neg_to_pos[&la];
                        result_label_order.iter().position(|&r| r == pos).unwrap()
                    }
                })
                .collect();

            result.permutedims(&perm)?
        }
    };

    // Compute grad_b
    let grad_b = {
        let mut label_counter = max_label + 1;

        let mut a_neg_to_pos = std::collections::HashMap::new();

        if is_scalar_output {
            // Scalar output: grad_B = grad_output[0] * A
            // This is a simple scalar multiplication, not a contraction
            let scalar = *grad_output
                .get_linear(0)
                .ok_or(TensorError::IndexOutOfBounds {
                    index: 0,
                    dim_size: grad_output.len(),
                })?;

            // Create result with A's values scaled by grad_output scalar
            let result_data: Vec<ElT> = a.data().iter().map(|&v| v * scalar).collect();

            // A and B might have different dimension orderings for the same contracted indices
            let a_result = Tensor::from_vec(result_data.clone(), a.shape())?;

            // Map A's negative labels to their positions
            let a_label_to_idx: std::collections::HashMap<i32, usize> =
                labels_a.iter().enumerate().map(|(i, &l)| (l, i)).collect();

            // Build permutation from A's order to B's order
            let perm: Vec<usize> = labels_b
                .iter()
                .map(|&lb| *a_label_to_idx.get(&lb).expect("Label not found in A"))
                .collect();

            // Check if permutation is identity
            let is_identity = perm.iter().enumerate().all(|(i, &p)| i == p);
            if is_identity {
                Tensor::from_vec(result_data, b.shape())?
            } else {
                a_result.permutedims(&perm)?
            }
        } else {
            let a_positive_count = labels_a.iter().filter(|&&l| l > 0).count();

            // Create labels for A
            // A's positive labels → contract with grad_C
            // A's negative labels → become positive in output
            let mut a_labels_new = Vec::new();
            let mut a_pos_contract_labels = Vec::new();

            for &l in labels_a {
                if l > 0 {
                    let neg_label = -label_counter;
                    a_labels_new.push(neg_label);
                    a_pos_contract_labels.push(neg_label);
                    label_counter += 1;
                } else {
                    let pos_label = label_counter;
                    a_labels_new.push(pos_label);
                    a_neg_to_pos.insert(l, pos_label);
                    label_counter += 1;
                }
            }

            // Create labels for grad_C
            // First part: A's positive labels → contract with A
            // Second part: B's positive labels → remain as output
            let mut grad_c_labels = Vec::new();
            for (a_pos_idx, _) in (0..a_positive_count).enumerate() {
                grad_c_labels.push(a_pos_contract_labels[a_pos_idx]);
            }
            for &l in labels_b.iter().filter(|&&l| l > 0) {
                grad_c_labels.push(l);
            }

            // Contract A with grad_C
            let result = contract(a, &a_labels_new, grad_output, &grad_c_labels)?;

            // Result has labels: [new positive for A's negative, in A's order] ++ [B's positive]
            let mut result_label_order = Vec::new();
            for &l in labels_a {
                if l < 0 {
                    result_label_order.push(a_neg_to_pos[&l]);
                }
            }
            for &l in labels_b {
                if l > 0 {
                    result_label_order.push(l);
                }
            }

            // Build permutation for B's shape
            let perm: Vec<usize> = labels_b
                .iter()
                .map(|&lb| {
                    if lb > 0 {
                        result_label_order.iter().position(|&r| r == lb).unwrap()
                    } else {
                        let pos = a_neg_to_pos[&lb];
                        result_label_order.iter().position(|&r| r == pos).unwrap()
                    }
                })
                .collect();

            result.permutedims(&perm)?
        }
    };

    Ok((grad_a, grad_b))
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

    #[test]
    fn test_contract_vjp_matrix_multiply() {
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

        // grad_c is 2x4
        let grad_c = Tensor::<f64>::ones(&[2, 4]);

        let (grad_a, grad_b) = contract_vjp(&a, &[1, -1], &b, &[-1, 2], &grad_c).unwrap();

        // grad_a should be 2x3, grad_b should be 3x4
        assert_eq!(grad_a.shape(), &[2, 3]);
        assert_eq!(grad_b.shape(), &[3, 4]);

        // grad_a = grad_c @ B^T
        // grad_c is 2x4 (all ones), B^T is 4x3
        // grad_a[i,j] = sum_k grad_c[i,k] * B[j,k] = sum_k B[j,k]
        // For j=0: sum of row 0 of B = 1+4+7+10 = 22
        // For j=1: sum of row 1 of B = 2+5+8+11 = 26
        // For j=2: sum of row 2 of B = 3+6+9+12 = 30
        assert_relative_eq!(*grad_a.get(&[0, 0]).unwrap(), 22.0);
        assert_relative_eq!(*grad_a.get(&[0, 1]).unwrap(), 26.0);
        assert_relative_eq!(*grad_a.get(&[0, 2]).unwrap(), 30.0);
        assert_relative_eq!(*grad_a.get(&[1, 0]).unwrap(), 22.0);
        assert_relative_eq!(*grad_a.get(&[1, 1]).unwrap(), 26.0);
        assert_relative_eq!(*grad_a.get(&[1, 2]).unwrap(), 30.0);

        // grad_b = A^T @ grad_c
        // A^T is 3x2, grad_c is 2x4
        // grad_b[j,k] = sum_i A[i,j] * grad_c[i,k] = sum_i A[i,j] (since grad_c is all ones)
        // For j=0: A[0,0] + A[1,0] = 1 + 2 = 3
        // For j=1: A[0,1] + A[1,1] = 3 + 4 = 7
        // For j=2: A[0,2] + A[1,2] = 5 + 6 = 11
        assert_relative_eq!(*grad_b.get(&[0, 0]).unwrap(), 3.0);
        assert_relative_eq!(*grad_b.get(&[1, 0]).unwrap(), 7.0);
        assert_relative_eq!(*grad_b.get(&[2, 0]).unwrap(), 11.0);
    }

    #[test]
    fn test_contract_vjp_inner_product() {
        // Vector inner product: a[i] * b[i] -> scalar
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();

        // grad_output is a scalar (represented as [1])
        let grad_output = Tensor::from_vec(vec![1.0], &[1]).unwrap();

        let (grad_a, grad_b) = contract_vjp(&a, &[-1], &b, &[-1], &grad_output).unwrap();

        // grad_a = grad_output * b
        assert_eq!(grad_a.shape(), &[3]);
        assert_relative_eq!(*grad_a.get(&[0]).unwrap(), 4.0);
        assert_relative_eq!(*grad_a.get(&[1]).unwrap(), 5.0);
        assert_relative_eq!(*grad_a.get(&[2]).unwrap(), 6.0);

        // grad_b = grad_output * a
        assert_eq!(grad_b.shape(), &[3]);
        assert_relative_eq!(*grad_b.get(&[0]).unwrap(), 1.0);
        assert_relative_eq!(*grad_b.get(&[1]).unwrap(), 2.0);
        assert_relative_eq!(*grad_b.get(&[2]).unwrap(), 3.0);
    }

    #[test]
    fn test_contract_vjp_outer_product() {
        // Outer product: a[i] * b[j] -> C[i,j]
        let a = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let b = Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap();

        // grad_c is 2x3
        let grad_c = Tensor::<f64>::ones(&[2, 3]);

        let (grad_a, grad_b) = contract_vjp(&a, &[1], &b, &[2], &grad_c).unwrap();

        // grad_a[i] = sum_j grad_c[i,j] * b[j] = sum_j b[j] = 3+4+5 = 12
        assert_eq!(grad_a.shape(), &[2]);
        assert_relative_eq!(*grad_a.get(&[0]).unwrap(), 12.0);
        assert_relative_eq!(*grad_a.get(&[1]).unwrap(), 12.0);

        // grad_b[j] = sum_i grad_c[i,j] * a[i] = sum_i a[i] = 1+2 = 3
        assert_eq!(grad_b.shape(), &[3]);
        assert_relative_eq!(*grad_b.get(&[0]).unwrap(), 3.0);
        assert_relative_eq!(*grad_b.get(&[1]).unwrap(), 3.0);
        assert_relative_eq!(*grad_b.get(&[2]).unwrap(), 3.0);
    }
}
