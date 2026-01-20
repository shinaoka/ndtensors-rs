//! Eigenvalue decomposition for tensors.
//!
//! This module provides eigenvalue decomposition for tensors by reshaping them
//! to square matrices, computing the eigendecomposition using faer, and reshaping
//! the results back to tensor form.
//!
//! Two variants are provided:
//! - `eigen_hermitian`: For Hermitian (or symmetric) matrices, returns real eigenvalues
//! - `eigen`: For general matrices, returns complex eigenvalues and eigenvectors

use faer::c64;
use faer::linalg::solvers::EvdError;

use crate::backend::AsFaerMat;
use crate::decomposition::util::{PermuteReshapeResult, permute_reshape};
use crate::error::TensorError;
use crate::scalar::{RealScalar, Scalar};
use crate::tensor::DenseTensor;

/// Compute the eigendecomposition of a Hermitian (or symmetric) tensor.
///
/// The tensor is reshaped into a square matrix by grouping the specified left indices.
/// The remaining indices are automatically treated as right indices.
/// For Hermitian matrices, eigenvalues are real and returned in nondecreasing order.
///
/// # Arguments
///
/// * `tensor` - The input tensor (must reshape to a square matrix)
/// * `left_inds` - Indices to place on the left (rows)
///
/// # Returns
///
/// A tuple `(eigenvalues, eigenvectors)` where:
/// - `eigenvalues`: Real eigenvalues with shape `[n]` (sorted in nondecreasing order)
/// - `eigenvectors`: Eigenvectors as columns with shape `[n, n]`
///
/// The eigenvectors satisfy `A * v_i = λ_i * v_i` where `v_i` is the i-th column.
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::decomposition::eigen_hermitian;
///
/// // Symmetric 2x2 matrix
/// let t = Tensor::<f64>::from_vec(vec![4.0, 1.0, 1.0, 3.0], &[2, 2]).unwrap();
/// let (eigenvalues, eigenvectors) = eigen_hermitian(&t, &[0]).unwrap();
///
/// assert_eq!(eigenvalues.shape(), &[2]);
/// assert_eq!(eigenvectors.shape(), &[2, 2]);
/// ```
pub fn eigen_hermitian<ElT: Scalar>(
    tensor: &DenseTensor<ElT>,
    left_inds: &[usize],
) -> Result<(DenseTensor<<ElT as Scalar>::Real>, DenseTensor<ElT>), TensorError> {
    // Validate that the tensor reshapes to a square matrix
    let (matrix, n) = validate_square_for_eigen(tensor, left_inds)?;

    // Convert to faer matrix (zero-copy)
    let mat = matrix.as_faer_mat(n, n);

    // Compute Hermitian eigendecomposition
    let evd = mat
        .self_adjoint_eigen(faer::Side::Lower)
        .map_err(|e: EvdError| TensorError::EigenError {
            message: format!("Hermitian eigendecomposition failed: {:?}", e),
        })?;

    // Extract eigenvalues (real, sorted in nondecreasing order)
    let eigenvalues_diag = evd.S();
    let mut eigenvalues_vec: Vec<<ElT as Scalar>::Real> = Vec::with_capacity(n);
    for i in 0..n {
        let lambda = eigenvalues_diag[i];
        eigenvalues_vec.push(lambda.real_part());
    }
    let eigenvalues = DenseTensor::from_vec(eigenvalues_vec, &[n])?;

    // Extract eigenvectors (stored as columns in U)
    let u_mat = evd.U();
    let mut eigenvectors_data: Vec<ElT> = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            eigenvectors_data.push(u_mat[(i, j)]);
        }
    }
    let eigenvectors = DenseTensor::from_vec(eigenvectors_data, &[n, n])?;

    Ok((eigenvalues, eigenvectors))
}

/// Compute the eigendecomposition of a general (non-Hermitian) tensor.
///
/// The tensor is reshaped into a square matrix by grouping the specified left indices.
/// For general matrices, eigenvalues and eigenvectors are complex-valued.
///
/// # Arguments
///
/// * `tensor` - The input tensor (must reshape to a square matrix)
/// * `left_inds` - Indices to place on the left (rows)
///
/// # Returns
///
/// A tuple `(eigenvalues, eigenvectors)` where:
/// - `eigenvalues`: Complex eigenvalues with shape `[n]`
/// - `eigenvectors`: Complex eigenvectors as columns with shape `[n, n]`
///
/// The eigenvectors satisfy `A * v_i = λ_i * v_i` where `v_i` is the i-th column.
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::decomposition::eigen;
///
/// // General 2x2 matrix
/// let t = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
/// let (eigenvalues, eigenvectors) = eigen(&t, &[0]).unwrap();
///
/// assert_eq!(eigenvalues.shape(), &[2]);
/// assert_eq!(eigenvectors.shape(), &[2, 2]);
/// ```
pub fn eigen<ElT: Scalar>(
    tensor: &DenseTensor<ElT>,
    left_inds: &[usize],
) -> Result<(DenseTensor<c64>, DenseTensor<c64>), TensorError> {
    // Validate that the tensor reshapes to a square matrix
    let (matrix, n) = validate_square_for_eigen(tensor, left_inds)?;

    // Convert to complex using Scalar trait methods
    let complex_data: Vec<c64> = matrix
        .data()
        .iter()
        .map(|x| c64::new(x.real_part().to_f64(), x.imag_part().to_f64()))
        .collect();
    let complex_matrix = DenseTensor::from_vec(complex_data, matrix.shape())?;

    // Convert to faer matrix (zero-copy)
    let mat = complex_matrix.as_faer_mat(n, n);

    // Compute general eigendecomposition
    let evd = mat.eigen().map_err(|e: EvdError| TensorError::EigenError {
        message: format!("eigendecomposition failed: {:?}", e),
    })?;

    // Extract eigenvalues (complex)
    let eigenvalues_diag = evd.S();
    let mut eigenvalues_vec: Vec<c64> = Vec::with_capacity(n);
    for i in 0..n {
        eigenvalues_vec.push(eigenvalues_diag[i]);
    }
    let eigenvalues = DenseTensor::from_vec(eigenvalues_vec, &[n])?;

    // Extract eigenvectors (complex columns)
    let u_mat = evd.U();
    let mut eigenvectors_data: Vec<c64> = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            eigenvectors_data.push(u_mat[(i, j)]);
        }
    }
    let eigenvectors = DenseTensor::from_vec(eigenvectors_data, &[n, n])?;

    Ok((eigenvalues, eigenvectors))
}

/// Validate that a tensor reshapes to a square matrix for eigendecomposition.
///
/// Returns the reshaped matrix and its dimension n (for an n×n matrix).
fn validate_square_for_eigen<ElT: Scalar>(
    tensor: &DenseTensor<ElT>,
    left_inds: &[usize],
) -> Result<(DenseTensor<ElT>, usize), TensorError> {
    // Create right_inds as complement of left_inds
    let right_inds: Vec<usize> = (0..tensor.ndim())
        .filter(|i| !left_inds.contains(i))
        .collect();

    // Reshape tensor to matrix
    let PermuteReshapeResult {
        matrix,
        nrows,
        ncols,
        ..
    } = permute_reshape(tensor, left_inds, &right_inds)?;

    // Verify square matrix
    if nrows != ncols {
        return Err(TensorError::NotSquareMatrix {
            rows: nrows,
            cols: ncols,
        });
    }

    Ok((matrix, nrows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::c64;

    #[test]
    fn test_eigen_hermitian_f64_2d() {
        // Symmetric 2x2 matrix with known eigenvalues
        // [4  1]
        // [1  3]
        // Eigenvalues: λ₁ ≈ 2.382, λ₂ ≈ 4.618
        let t = DenseTensor::from_vec(vec![4.0, 1.0, 1.0, 3.0], &[2, 2]).unwrap();
        let (eigenvalues, eigenvectors) = eigen_hermitian(&t, &[0]).unwrap();

        assert_eq!(eigenvalues.shape(), &[2]);
        assert_eq!(eigenvectors.shape(), &[2, 2]);

        // Verify eigenvalues are real and sorted
        let lambda_0 = *eigenvalues.get(&[0]).unwrap();
        let lambda_1 = *eigenvalues.get(&[1]).unwrap();
        assert!(lambda_0 <= lambda_1, "eigenvalues should be sorted");

        // Verify reconstruction: A = V * diag(λ) * V^T
        verify_hermitian_reconstruction(&t, &eigenvalues, &eigenvectors);
    }

    #[test]
    fn test_eigen_hermitian_c64_2d() {
        // Hermitian 2x2 matrix
        // [2+0i    1+1i]
        // [1-1i    3+0i]
        let t = DenseTensor::from_vec(
            vec![
                c64::new(2.0, 0.0),
                c64::new(1.0, 1.0),
                c64::new(1.0, -1.0),
                c64::new(3.0, 0.0),
            ],
            &[2, 2],
        )
        .unwrap();

        let (eigenvalues, eigenvectors) = eigen_hermitian(&t, &[0]).unwrap();

        assert_eq!(eigenvalues.shape(), &[2]);
        assert_eq!(eigenvectors.shape(), &[2, 2]);

        // Eigenvalues should be real for Hermitian matrix
        let lambda_0 = *eigenvalues.get(&[0]).unwrap();
        let lambda_1 = *eigenvalues.get(&[1]).unwrap();
        assert!(lambda_0 <= lambda_1);

        // Verify reconstruction: A = V * diag(λ) * V^H
        verify_hermitian_reconstruction(&t, &eigenvalues, &eigenvectors);
    }

    #[test]
    fn test_eigen_general_f64_2d() {
        // Non-symmetric 2x2 matrix
        // [1  2]
        // [3  4]
        let t = DenseTensor::from_vec(vec![1.0, 3.0, 2.0, 4.0], &[2, 2]).unwrap();
        let (eigenvalues, eigenvectors) = eigen(&t, &[0]).unwrap();

        assert_eq!(eigenvalues.shape(), &[2]);
        assert_eq!(eigenvectors.shape(), &[2, 2]);

        // Verify A * V = V * diag(λ)
        verify_general_reconstruction(&t, &eigenvalues, &eigenvectors);
    }

    #[test]
    fn test_eigen_general_c64_2d() {
        // Complex non-Hermitian matrix
        let t = DenseTensor::from_vec(
            vec![
                c64::new(1.0, 0.5),
                c64::new(2.0, 0.0),
                c64::new(0.0, 1.0),
                c64::new(3.0, -0.5),
            ],
            &[2, 2],
        )
        .unwrap();

        let (eigenvalues, eigenvectors) = eigen(&t, &[0]).unwrap();

        assert_eq!(eigenvalues.shape(), &[2]);
        assert_eq!(eigenvectors.shape(), &[2, 2]);

        // Verify A * V = V * diag(λ)
        verify_general_reconstruction(&t, &eigenvalues, &eigenvectors);
    }

    #[test]
    fn test_eigen_not_square() {
        // Non-square matrix should fail
        // With left_inds=[0], we get rows=2 (from dim 0) and cols=3 (from dim 1)
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let result = eigen_hermitian(&t, &[0]);
        assert!(result.is_err());
        match result {
            Err(TensorError::NotSquareMatrix { rows, cols }) => {
                assert_eq!(rows, 2);
                assert_eq!(cols, 3);
            }
            _ => panic!("expected NotSquareMatrix error"),
        }
    }

    #[test]
    fn test_eigen_identity_matrix() {
        // Identity matrix should have all eigenvalues = 1
        let t = DenseTensor::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let (eigenvalues, eigenvectors) = eigen_hermitian(&t, &[0]).unwrap();

        for i in 0..2 {
            let lambda = *eigenvalues.get(&[i]).unwrap();
            let diff = (lambda - 1.0).abs();
            assert!(
                diff < 1e-10,
                "eigenvalue {} should be 1.0, got {}",
                i,
                lambda
            );
        }

        // Eigenvectors should form an orthonormal basis
        verify_orthonormal(&eigenvectors);
    }

    // Helper function to verify reconstruction for Hermitian case
    fn verify_hermitian_reconstruction<
        ElT: Scalar + std::ops::Mul<Output = ElT> + std::ops::Add<Output = ElT>,
    >(
        original: &DenseTensor<ElT>,
        eigenvalues: &DenseTensor<<ElT as Scalar>::Real>,
        eigenvectors: &DenseTensor<ElT>,
    ) {
        use faer_traits::math_utils::from_f64;

        let n = eigenvalues.len();

        // Compute A = V * diag(λ) * V^H
        for i in 0..n {
            for j in 0..n {
                let mut sum = ElT::zero();
                for k in 0..n {
                    let v_ik = *eigenvectors.get(&[i, k]).unwrap();
                    let v_jk = *eigenvectors.get(&[j, k]).unwrap();
                    let lambda_k = eigenvalues.get(&[k]).unwrap().to_f64();
                    // v_ik * λ_k * conj(v_jk)
                    sum = sum + v_ik * from_f64::<ElT>(lambda_k) * v_jk.conjugate();
                }

                let original_val = *original.get(&[i, j]).unwrap();
                // Compute |sum - original_val|^2 and compare
                let diff = sum + from_f64::<ElT>(-1.0) * original_val;
                let diff_abs: f64 = diff.abs_sqr().to_f64().sqrt();

                assert!(
                    diff_abs < 1e-9,
                    "reconstruction error at ({}, {}): {}",
                    i,
                    j,
                    diff_abs
                );
            }
        }
    }

    // Helper function to verify reconstruction for general case
    fn verify_general_reconstruction<ElT: Scalar>(
        original: &DenseTensor<ElT>,
        eigenvalues: &DenseTensor<c64>,
        eigenvectors: &DenseTensor<c64>,
    ) {
        let n = eigenvalues.len();

        // Convert original to complex using Scalar trait
        let complex_data: Vec<c64> = original
            .data()
            .iter()
            .map(|x| c64::new(x.real_part().to_f64(), x.imag_part().to_f64()))
            .collect();
        let original_c64 = DenseTensor::from_vec(complex_data, original.shape()).unwrap();

        // Verify A * v_k = λ_k * v_k for each eigenvector
        for k in 0..n {
            let lambda_k = *eigenvalues.get(&[k]).unwrap();

            for i in 0..n {
                // Compute (A * v_k)_i
                let mut av_i = c64::zero();
                for j in 0..n {
                    let a_ij = *original_c64.get(&[i, j]).unwrap();
                    let v_jk = *eigenvectors.get(&[j, k]).unwrap();
                    av_i += a_ij * v_jk;
                }

                // Compute (λ_k * v_k)_i
                let v_ik = *eigenvectors.get(&[i, k]).unwrap();
                let lambda_v_i = lambda_k * v_ik;

                let diff_val = av_i + (-lambda_v_i);
                let diff_abs = diff_val.abs_sqr().sqrt();
                assert!(
                    diff_abs < 1e-9,
                    "eigenvalue equation error for eigenvector {} at position {}: {}",
                    k,
                    i,
                    diff_abs
                );
            }
        }
    }

    // Helper function to verify orthonormality of eigenvectors
    fn verify_orthonormal<
        ElT: Scalar + std::ops::Mul<Output = ElT> + std::ops::Add<Output = ElT>,
    >(
        eigenvectors: &DenseTensor<ElT>,
    ) {
        use faer_traits::math_utils::from_f64;

        let n = eigenvectors.shape()[1];

        for j in 0..n {
            for col_k in 0..n {
                let mut dot = ElT::zero();
                for i in 0..n {
                    let v_ij = *eigenvectors.get(&[i, j]).unwrap();
                    let v_ik = *eigenvectors.get(&[i, col_k]).unwrap();
                    // conj(v_ij) * v_ik
                    dot = dot + v_ij.conjugate() * v_ik;
                }

                let expected = if j == col_k { ElT::one() } else { ElT::zero() };
                let diff = dot + from_f64::<ElT>(-1.0) * expected;
                let diff_abs: f64 = diff.abs_sqr().to_f64().sqrt();

                assert!(
                    diff_abs < 1e-10,
                    "orthonormality error at ({}, {}): {}",
                    j,
                    col_k,
                    diff_abs
                );
            }
        }
    }
}
