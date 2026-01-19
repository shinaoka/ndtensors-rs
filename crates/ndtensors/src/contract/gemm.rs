//! GEMM-based tensor contraction using faer.
//!
//! This module provides optimized tensor contraction by reshaping tensors
//! to matrices and using faer's high-performance matrix multiplication.

use std::ops::{Add, Mul};

use faer::linalg::matmul::matmul;
use faer::{Accum, Par};

use crate::backend::AsFaerMat;
use crate::contract::properties::ContractionProperties;
use crate::error::TensorError;
use crate::operations::permutedims;
use crate::scalar::Scalar;
use crate::tensor::DenseTensor;

/// Contract two tensors using GEMM-based optimization.
///
/// This function reshapes tensors to matrices and uses faer's matmul
/// for high-performance contraction.
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
/// Result tensor with dimensions corresponding to positive labels
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::contract::contract_gemm;
///
/// // Matrix multiplication: C[i,k] = A[i,j] * B[j,k]
/// let a = Tensor::<f64>::ones(&[2, 3]);
/// let b = Tensor::<f64>::ones(&[3, 4]);
///
/// let c = contract_gemm(&a, &[1, -1], &b, &[-1, 2]).unwrap();
/// assert_eq!(c.shape(), &[2, 4]);
/// ```
pub fn contract_gemm<ElT: Scalar + Add<Output = ElT> + Mul<Output = ElT>>(
    a: &DenseTensor<ElT>,
    labels_a: &[i32],
    b: &DenseTensor<ElT>,
    labels_b: &[i32],
) -> Result<DenseTensor<ElT>, TensorError> {
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

    // Validate dimension compatibility for contracted indices
    for (i, &la) in labels_a.iter().enumerate() {
        if la < 0 {
            for (j, &lb) in labels_b.iter().enumerate() {
                if la == lb && a.shape()[i] != b.shape()[j] {
                    return Err(TensorError::ShapeMismatch {
                        expected: a.shape()[i],
                        actual: b.shape()[j],
                    });
                }
            }
        }
    }

    // Compute contraction properties
    let props = ContractionProperties::compute(labels_a, a.shape(), labels_b, b.shape());

    // Handle special cases
    if props.is_full_contraction() {
        return contract_full(a, b, &props);
    }

    if props.is_outer_product() {
        return contract_outer(a, b, &props);
    }

    // General GEMM-based contraction
    contract_gemm_general(a, b, &props)
}

/// Full contraction (scalar result).
fn contract_full<ElT: Scalar + Add<Output = ElT> + Mul<Output = ElT>>(
    a: &DenseTensor<ElT>,
    b: &DenseTensor<ElT>,
    props: &ContractionProperties,
) -> Result<DenseTensor<ElT>, TensorError> {
    // For full contraction, we compute dot product
    // Permute if needed, then reshape to vectors and compute dot product

    let a_work = if props.permute_a {
        permutedims(a, &props.perm_a)?
    } else {
        a.clone()
    };

    let b_work = if props.permute_b {
        permutedims(b, &props.perm_b)?
    } else {
        b.clone()
    };

    // Compute dot product using loop (for full generality)
    let mut sum = ElT::zero();
    for (&av, &bv) in a_work.data().iter().zip(b_work.data().iter()) {
        sum = sum + av * bv;
    }

    DenseTensor::from_vec(vec![sum], &[1])
}

/// Outer product (no contracted indices).
fn contract_outer<ElT: Scalar + Add<Output = ElT> + Mul<Output = ElT>>(
    a: &DenseTensor<ElT>,
    b: &DenseTensor<ElT>,
    props: &ContractionProperties,
) -> Result<DenseTensor<ElT>, TensorError> {
    // Outer product: C[i,j] = A[i] * B[j]
    // Use GEMM: C(m,n) = A(m,1) * B(1,n)

    let m = props.dleft;
    let n = props.dright;

    // Create output tensor
    let mut output_shape: Vec<usize> = props.uncontracted_a.iter().map(|&i| a.shape()[i]).collect();
    output_shape.extend(props.uncontracted_b.iter().map(|&j| b.shape()[j]));

    let mut c = DenseTensor::<ElT>::zeros(&[m, n]);

    // Perform outer product using GEMM: C = A * B^T where A is (m,1) and B is (n,1)
    let a_mat = a.as_faer_mat(m, 1);
    let b_mat = b.as_faer_mat(n, 1);
    let mut c_mat = c.as_faer_mat_mut(m, n);

    // C = alpha * A * B^T + beta * C
    // With beta = Replace, alpha = 1
    matmul(
        c_mat.as_mut(),
        Accum::Replace,
        a_mat,
        b_mat.transpose(),
        ElT::one(),
        Par::Seq,
    );

    // Reshape to output shape
    let result = DenseTensor::from_vec(c.data().to_vec(), &output_shape)?;

    // Apply output permutation if needed
    if props.permute_c {
        permutedims(&result, &props.perm_c)
    } else {
        Ok(result)
    }
}

/// General GEMM-based contraction.
fn contract_gemm_general<ElT: Scalar + Add<Output = ElT> + Mul<Output = ElT>>(
    a: &DenseTensor<ElT>,
    b: &DenseTensor<ElT>,
    props: &ContractionProperties,
) -> Result<DenseTensor<ElT>, TensorError> {
    // Step 1: Permute A if needed to get [uncontracted..., contracted...] order
    let a_work = if props.permute_a {
        permutedims(a, &props.perm_a)?
    } else {
        a.clone()
    };

    // Step 2: Permute B if needed to get [contracted..., uncontracted...] order
    let b_work = if props.permute_b {
        permutedims(b, &props.perm_b)?
    } else {
        b.clone()
    };

    // Step 3: Reshape to matrices
    // A: (dleft, dmid)
    // B: (dmid, dright)
    // C: (dleft, dright)
    let m = props.dleft;
    let k = props.dmid;
    let n = props.dright;

    let a_mat = a_work.as_faer_mat(m, k);
    let b_mat = b_work.as_faer_mat(k, n);

    // Step 4: Allocate output and perform GEMM
    let mut c = DenseTensor::<ElT>::zeros(&[m, n]);
    let mut c_mat = c.as_faer_mat_mut(m, n);

    // C = alpha * A * B + beta * C
    // With beta = Replace, alpha = 1
    matmul(
        c_mat.as_mut(),
        Accum::Replace,
        a_mat,
        b_mat,
        ElT::one(),
        Par::Seq,
    );

    // Step 5: Compute output shape
    let mut output_shape: Vec<usize> = props.uncontracted_a.iter().map(|&i| a.shape()[i]).collect();
    output_shape.extend(props.uncontracted_b.iter().map(|&j| b.shape()[j]));

    // Handle scalar case
    if output_shape.is_empty() {
        output_shape.push(1);
    }

    // Reshape C to output shape
    let result = DenseTensor::from_vec(c.data().to_vec(), &output_shape)?;

    // Step 6: Apply output permutation if needed
    if props.permute_c {
        permutedims(&result, &props.perm_c)
    } else {
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::contract;
    use approx::assert_relative_eq;

    #[test]
    fn test_gemm_matrix_multiply() {
        let a = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = DenseTensor::from_vec(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
        )
        .unwrap();

        let c_gemm = contract_gemm(&a, &[1, -1], &b, &[-1, 2]).unwrap();
        let c_naive = contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();

        assert_eq!(c_gemm.shape(), &[2, 4]);
        assert_eq!(c_gemm.shape(), c_naive.shape());

        // Compare results
        for i in 0..c_gemm.len() {
            assert_relative_eq!(
                *c_gemm.get_linear(i).unwrap(),
                *c_naive.get_linear(i).unwrap(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_gemm_inner_product() {
        let a = DenseTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = DenseTensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();

        let c_gemm = contract_gemm(&a, &[-1], &b, &[-1]).unwrap();
        let c_naive = contract(&a, &[-1], &b, &[-1]).unwrap();

        // 1*4 + 2*5 + 3*6 = 32
        assert_relative_eq!(*c_gemm.get_linear(0).unwrap(), 32.0);
        assert_relative_eq!(
            *c_gemm.get_linear(0).unwrap(),
            *c_naive.get_linear(0).unwrap()
        );
    }

    #[test]
    fn test_gemm_outer_product() {
        let a = DenseTensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let b = DenseTensor::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap();

        let c_gemm = contract_gemm(&a, &[1], &b, &[2]).unwrap();
        let c_naive = contract(&a, &[1], &b, &[2]).unwrap();

        assert_eq!(c_gemm.shape(), &[2, 3]);

        // Compare results
        for i in 0..c_gemm.len() {
            assert_relative_eq!(
                *c_gemm.get_linear(i).unwrap(),
                *c_naive.get_linear(i).unwrap(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_gemm_3d_tensor() {
        let a = DenseTensor::<f64>::ones(&[2, 3, 4]);
        let b = DenseTensor::<f64>::ones(&[4, 5]);

        let c_gemm = contract_gemm(&a, &[1, 2, -1], &b, &[-1, 3]).unwrap();
        let c_naive = contract(&a, &[1, 2, -1], &b, &[-1, 3]).unwrap();

        assert_eq!(c_gemm.shape(), &[2, 3, 5]);

        // Each element should be 4 (sum over k dimension of size 4)
        for i in 0..c_gemm.len() {
            assert_relative_eq!(
                *c_gemm.get_linear(i).unwrap(),
                *c_naive.get_linear(i).unwrap(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_gemm_dimension_mismatch() {
        let a = DenseTensor::<f64>::ones(&[2, 3]);
        let b = DenseTensor::<f64>::ones(&[4, 5]);

        let result = contract_gemm(&a, &[1, -1], &b, &[-1, 2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_gemm_complex() {
        use crate::scalar::c64;

        let a = DenseTensor::from_vec(
            vec![
                c64::new(1.0, 0.0),
                c64::new(2.0, 0.0),
                c64::new(3.0, 0.0),
                c64::new(4.0, 0.0),
            ],
            &[2, 2],
        )
        .unwrap();

        let b = DenseTensor::from_vec(
            vec![
                c64::new(1.0, 0.0),
                c64::new(0.0, 1.0),
                c64::new(0.0, -1.0),
                c64::new(1.0, 0.0),
            ],
            &[2, 2],
        )
        .unwrap();

        let c_gemm = contract_gemm(&a, &[1, -1], &b, &[-1, 2]).unwrap();
        let c_naive = contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();

        assert_eq!(c_gemm.shape(), &[2, 2]);

        for i in 0..c_gemm.len() {
            let g = c_gemm.get_linear(i).unwrap();
            let n = c_naive.get_linear(i).unwrap();
            assert_relative_eq!(g.re, n.re, epsilon = 1e-10);
            assert_relative_eq!(g.im, n.im, epsilon = 1e-10);
        }
    }
}
