//! QR Decomposition for tensors.
//!
//! This module provides QR decomposition for tensors by reshaping them
//! to matrices, computing the QR using faer, and reshaping the results.

use faer::linalg::solvers::Qr;

use crate::backend::AsFaerMat;
use crate::decomposition::util::{PermuteReshapeResult, permute_reshape};
use crate::error::TensorError;
use crate::scalar::Scalar;
use crate::tensor::DenseTensor;

/// Result of QR decomposition.
#[derive(Debug, Clone)]
pub struct QrResult<ElT: Scalar> {
    /// Orthogonal/unitary matrix Q reshaped to tensor form.
    /// Shape: [...left_dims..., k] where k = min(m, n)
    pub q: DenseTensor<ElT>,

    /// Upper triangular matrix R reshaped to tensor form.
    /// Shape: [k, ...right_dims...] where k = min(m, n)
    pub r: DenseTensor<ElT>,

    /// The rank k = min(m, n)
    pub rank: usize,
}

/// Compute the thin QR decomposition of a tensor.
///
/// The tensor is reshaped into a matrix by grouping the specified left indices
/// into rows and right indices into columns. The QR decomposition is then computed
/// and the results are reshaped back to tensor form.
///
/// For a matrix A of shape (m, n), the thin QR produces:
/// - Q: (m, k) where k = min(m, n), with orthonormal columns
/// - R: (k, n), upper triangular
///
/// Such that A = Q @ R.
///
/// # Arguments
///
/// * `tensor` - The input tensor
/// * `left_inds` - Indices to place on the left (become Q's row dimensions)
/// * `right_inds` - Indices to place on the right (become R's column dimensions)
///
/// # Returns
///
/// `QrResult` containing:
/// - `q`: Orthogonal factor with shape [...left_dims..., k]
/// - `r`: Upper triangular factor with shape [k, ...right_dims...]
/// - `rank`: k = min(m, n)
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::decomposition::qr;
///
/// let t = Tensor::<f64>::ones(&[4, 3]);
/// let result = qr(&t, &[0], &[1]).unwrap();
///
/// // Q has shape [4, 3], R has shape [3, 3]
/// assert_eq!(result.q.shape(), &[4, 3]);
/// assert_eq!(result.r.shape(), &[3, 3]);
/// ```
pub fn qr<ElT: Scalar>(
    tensor: &DenseTensor<ElT>,
    left_inds: &[usize],
    right_inds: &[usize],
) -> Result<QrResult<ElT>, TensorError> {
    // Permute and reshape to matrix
    let PermuteReshapeResult {
        matrix,
        nrows,
        ncols,
        left_dims,
        right_dims,
        ..
    } = permute_reshape(tensor, left_inds, right_inds)?;

    let m = nrows;
    let n = ncols;
    let k = m.min(n);

    // Convert to faer matrix
    let mat = matrix.as_faer_mat(m, n);

    // Compute QR decomposition
    let qr_result: Qr<ElT> = Qr::new(mat);

    // Get thin Q (m x k)
    let q_mat = qr_result.compute_thin_Q();

    // Get thin R (k x n)
    let r_mat = qr_result.thin_R();

    // Extract Q data (column-major)
    let mut q_data = Vec::with_capacity(m * k);
    for j in 0..k {
        for i in 0..m {
            q_data.push(q_mat[(i, j)]);
        }
    }

    // Extract R data (column-major)
    let mut r_data = Vec::with_capacity(k * n);
    for j in 0..n {
        for i in 0..k {
            r_data.push(r_mat[(i, j)]);
        }
    }

    // Reshape Q to [...left_dims..., k]
    let mut q_shape = left_dims;
    q_shape.push(k);
    let q = DenseTensor::from_vec(q_data, &q_shape)?;

    // Reshape R to [k, ...right_dims...]
    let mut r_shape = vec![k];
    r_shape.extend(right_dims);
    let r = DenseTensor::from_vec(r_data, &r_shape)?;

    Ok(QrResult { q, r, rank: k })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer_traits::ComplexField;
    use faer_traits::math_utils::{abs, conj, from_f64, neg};
    use std::ops::{Add, Mul};

    fn reconstruct<ElT: Scalar + Add<Output = ElT> + Mul<Output = ElT>>(
        result: &QrResult<ElT>,
        original_shape: &[usize],
        left_inds: &[usize],
        right_inds: &[usize],
    ) -> DenseTensor<ElT> {
        use crate::contract::contract;

        // Contract Q with R
        let q_ndim = result.q.ndim();
        let r_ndim = result.r.ndim();

        // Labels for Q: [1, 2, ..., q_ndim-1, -1]
        let mut q_labels: Vec<i32> = (1..q_ndim as i32).collect();
        q_labels.push(-1);

        // Labels for R: [-1, q_ndim, q_ndim+1, ...]
        let mut r_labels = vec![-1];
        for i in 0..(r_ndim - 1) {
            r_labels.push((q_ndim + i) as i32);
        }

        let contracted = contract(&result.q, &q_labels, &result.r, &r_labels).unwrap();

        // Build the inverse permutation
        let perm: Vec<usize> = left_inds.iter().chain(right_inds.iter()).copied().collect();
        let mut inv_perm = vec![0; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            inv_perm[p] = i;
        }

        // Reshape to permuted shape
        let permuted_shape: Vec<usize> = left_inds
            .iter()
            .map(|&i| original_shape[i])
            .chain(right_inds.iter().map(|&i| original_shape[i]))
            .collect();

        let reshaped = DenseTensor::from_vec(contracted.data().to_vec(), &permuted_shape).unwrap();

        // Apply inverse permutation
        use crate::operations::permutedims;
        permutedims(&reshaped, &inv_perm).unwrap()
    }

    fn check_orthogonality<ElT: Scalar + Add<Output = ElT> + Mul<Output = ElT>>(
        q: &DenseTensor<ElT>,
        epsilon: f64,
    ) {
        // Q^H @ Q should be identity
        let q_ndim = q.ndim();
        let k = q.shape()[q_ndim - 1];

        // Flatten Q to 2D: (m, k)
        let m: usize = q.shape()[..q_ndim - 1].iter().product();

        let eps: <ElT as ComplexField>::Real = from_f64(epsilon);

        // Manual computation for simplicity
        for j in 0..k {
            for l in 0..k {
                let mut sum = ElT::zero();
                for i in 0..m {
                    let q_ij = *q.get_linear(j * m + i).unwrap();
                    let q_il = *q.get_linear(l * m + i).unwrap();
                    sum = sum + conj(&q_ij) * q_il;
                }
                let expected = if j == l { ElT::one() } else { ElT::zero() };
                let diff: <ElT as ComplexField>::Real = abs(&(sum + neg(&expected)));
                assert!(diff < eps, "Q^H @ Q should be identity at ({}, {})", j, l);
            }
        }
    }

    #[test]
    fn test_qr_2d() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let result = qr(&t, &[0], &[1]).unwrap();

        assert_eq!(result.q.shape(), &[2, 2]);
        assert_eq!(result.r.shape(), &[2, 3]);
        assert_eq!(result.rank, 2);

        // Reconstruct and compare
        let reconstructed = reconstruct(&result, &[2, 3], &[0], &[1]);
        assert_eq!(reconstructed.shape(), t.shape());

        for i in 0..t.len() {
            assert_relative_eq!(
                *reconstructed.get_linear(i).unwrap(),
                *t.get_linear(i).unwrap(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_qr_tall() {
        let t =
            DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]).unwrap();
        let result = qr(&t, &[0], &[1]).unwrap();

        assert_eq!(result.q.shape(), &[4, 2]); // m=4, k=min(4,2)=2
        assert_eq!(result.r.shape(), &[2, 2]); // k=2, n=2
        assert_eq!(result.rank, 2);

        // Check orthogonality
        check_orthogonality(&result.q, 1e-10);

        // Reconstruct and compare
        let reconstructed = reconstruct(&result, &[4, 2], &[0], &[1]);
        for i in 0..t.len() {
            assert_relative_eq!(
                *reconstructed.get_linear(i).unwrap(),
                *t.get_linear(i).unwrap(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_qr_wide() {
        let t =
            DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]).unwrap();
        let result = qr(&t, &[0], &[1]).unwrap();

        assert_eq!(result.q.shape(), &[2, 2]); // m=2, k=min(2,4)=2
        assert_eq!(result.r.shape(), &[2, 4]); // k=2, n=4
        assert_eq!(result.rank, 2);

        // Check orthogonality
        check_orthogonality(&result.q, 1e-10);

        // Reconstruct and compare
        let reconstructed = reconstruct(&result, &[2, 4], &[0], &[1]);
        for i in 0..t.len() {
            assert_relative_eq!(
                *reconstructed.get_linear(i).unwrap(),
                *t.get_linear(i).unwrap(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_qr_3d() {
        let t = DenseTensor::<f64>::ones(&[2, 3, 4]);
        let result = qr(&t, &[0, 1], &[2]).unwrap();

        // m = 2*3 = 6, n = 4, k = min(6, 4) = 4
        assert_eq!(result.q.shape(), &[2, 3, 4]);
        assert_eq!(result.r.shape(), &[4, 4]);
        assert_eq!(result.rank, 4);

        // Reconstruct and compare
        let reconstructed = reconstruct(&result, &[2, 3, 4], &[0, 1], &[2]);
        for i in 0..t.len() {
            assert_relative_eq!(
                *reconstructed.get_linear(i).unwrap(),
                *t.get_linear(i).unwrap(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_qr_orthogonality() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])
            .unwrap();
        let result = qr(&t, &[0], &[1]).unwrap();

        check_orthogonality(&result.q, 1e-10);
    }
}
