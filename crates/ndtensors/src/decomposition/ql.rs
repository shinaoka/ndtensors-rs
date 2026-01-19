//! QL Decomposition for tensors.
//!
//! This module provides QL decomposition for tensors by reshaping them
//! to matrices, computing the QL using a custom implementation based on
//! LAPACK's geqlf, and reshaping the results.

use crate::decomposition::util::{PermuteReshapeResult, permute_reshape};
use crate::error::TensorError;
use crate::scalar::Scalar;
use crate::tensor::DenseTensor;

/// Result of QL decomposition.
#[derive(Debug, Clone)]
pub struct QlResult<ElT: Scalar> {
    /// Orthogonal/unitary matrix Q reshaped to tensor form.
    /// Shape: [...left_dims..., k] where k = min(m, n)
    pub q: DenseTensor<ElT>,

    /// Lower triangular matrix L reshaped to tensor form.
    /// Shape: [k, ...right_dims...] where k = min(m, n)
    pub l: DenseTensor<ElT>,

    /// The rank k = min(m, n)
    pub rank: usize,
}

/// Compute the thin QL decomposition of a tensor.
///
/// The tensor is reshaped into a matrix by grouping the specified left indices
/// into rows and right indices into columns. The QL decomposition is then computed
/// and the results are reshaped back to tensor form.
///
/// For a matrix A of shape (m, n), the thin QL produces:
/// - Q: (m, k) where k = min(m, n), with orthonormal columns
/// - L: (k, n), lower triangular (zeros in upper right)
///
/// Such that A = Q @ L.
///
/// # Arguments
///
/// * `tensor` - The input tensor
/// * `left_inds` - Indices to place on the left (become Q's row dimensions)
/// * `right_inds` - Indices to place on the right (become L's column dimensions)
///
/// # Returns
///
/// `QlResult` containing:
/// - `q`: Orthogonal factor with shape [...left_dims..., k]
/// - `l`: Lower triangular factor with shape [k, ...right_dims...]
/// - `rank`: k = min(m, n)
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::decomposition::ql;
///
/// let t = Tensor::<f64>::ones(&[4, 3]);
/// let result = ql(&t, &[0], &[1]).unwrap();
///
/// // Q has shape [4, 3], L has shape [3, 3]
/// assert_eq!(result.q.shape(), &[4, 3]);
/// assert_eq!(result.l.shape(), &[3, 3]);
/// ```
pub fn ql<ElT: Scalar>(
    tensor: &DenseTensor<ElT>,
    left_inds: &[usize],
    right_inds: &[usize],
) -> Result<QlResult<ElT>, TensorError> {
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

    // Compute QL decomposition using custom implementation
    let (q_data, l_data) = ql_decomp(matrix.data(), m, n);

    // Reshape Q to [...left_dims..., k]
    let mut q_shape = left_dims;
    q_shape.push(k);
    let q = DenseTensor::from_vec(q_data, &q_shape)?;

    // Reshape L to [k, ...right_dims...]
    let mut l_shape = vec![k];
    l_shape.extend(right_dims);
    let l = DenseTensor::from_vec(l_data, &l_shape)?;

    Ok(QlResult { q, l, rank: k })
}

/// Compute QL decomposition of a matrix stored in column-major order.
/// Returns (Q_data, L_data) both in column-major order.
fn ql_decomp<ElT: Scalar>(data: &[ElT], m: usize, n: usize) -> (Vec<ElT>, Vec<ElT>) {
    use faer::MatRef;
    use faer::linalg::solvers::Qr;

    let k = m.min(n);

    // QL decomposition: A = Q * L where L is lower triangular (from the right)
    // We compute this by: reverse columns of A, do QR, reverse columns of Q and both dims of R
    //
    // Alternatively, use the relationship with QR:
    // A = Q * L  where L is lower triangular
    // A^T = L^T * Q^T where L^T is upper triangular
    // So we can: transpose A, do QR, then transpose results

    // For simplicity, we'll use the column-reversal approach:
    // 1. Reverse column order of A: A_rev[:, j] = A[:, n-1-j]
    // 2. Compute QR: A_rev = Q_rev * R_rev
    // 3. Q[:, j] = Q_rev[:, k-1-j]
    // 4. L[i, j] = R_rev[k-1-i, n-1-j]

    // Create reversed matrix (column-major)
    let mut a_rev = vec![ElT::zero(); m * n];
    for j in 0..n {
        for i in 0..m {
            // a_rev[i, j] = a[i, n-1-j]
            a_rev[i + j * m] = data[i + (n - 1 - j) * m];
        }
    }

    // Compute QR using faer
    let mat = MatRef::from_column_major_slice(&a_rev, m, n);
    let qr_result: Qr<ElT> = Qr::new(mat);

    let q_rev = qr_result.compute_thin_Q();
    let r_rev = qr_result.thin_R();

    // Extract Q with reversed columns
    let mut q_data = Vec::with_capacity(m * k);
    for j in 0..k {
        for i in 0..m {
            q_data.push(q_rev[(i, k - 1 - j)]);
        }
    }

    // Extract L with reversed rows and columns
    // L has shape (k, n), lower triangular from the right
    // L[i, j] = R_rev[k-1-i, n-1-j]
    let mut l_data = Vec::with_capacity(k * n);
    for j in 0..n {
        for i in 0..k {
            l_data.push(r_rev[(k - 1 - i, n - 1 - j)]);
        }
    }

    (q_data, l_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use faer_traits::ComplexField;
    use faer_traits::math_utils::{abs, conj, from_f64, neg};
    use std::ops::{Add, Mul};

    fn reconstruct<ElT: Scalar + Add<Output = ElT> + Mul<Output = ElT>>(
        result: &QlResult<ElT>,
        original_shape: &[usize],
        left_inds: &[usize],
        right_inds: &[usize],
    ) -> DenseTensor<ElT> {
        use crate::contract::contract;

        // Contract Q with L
        let q_ndim = result.q.ndim();
        let l_ndim = result.l.ndim();

        // Labels for Q: [1, 2, ..., q_ndim-1, -1]
        let mut q_labels: Vec<i32> = (1..q_ndim as i32).collect();
        q_labels.push(-1);

        // Labels for L: [-1, q_ndim, q_ndim+1, ...]
        let mut l_labels = vec![-1];
        for i in 0..(l_ndim - 1) {
            l_labels.push((q_ndim + i) as i32);
        }

        let contracted = contract(&result.q, &q_labels, &result.l, &l_labels).unwrap();

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

        // Manual computation
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

    fn check_lower_triangular<ElT: Scalar>(l: &DenseTensor<ElT>, epsilon: f64) {
        // L should be lower triangular (zeros in upper-right triangle)
        // For QL, the "diagonal" starts from the right side
        let k = l.shape()[0];
        let n: usize = l.shape()[1..].iter().product();

        let eps: <ElT as ComplexField>::Real = from_f64(epsilon);

        // For a k x n matrix L in QL:
        // If n >= k: upper triangle starts at column (n - k)
        // If n < k: L is dense (no upper triangle zeros required by definition)

        if n >= k {
            for i in 0..k {
                for j in (n - k + i + 1)..n {
                    let val = *l.get_linear(i + j * k).unwrap();
                    let val_abs: <ElT as ComplexField>::Real = abs(&val);
                    assert!(
                        val_abs < eps,
                        "L should be lower triangular, but L[{}, {}] = {:?}",
                        i,
                        j,
                        val
                    );
                }
            }
        }
    }

    #[test]
    fn test_ql_2d() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let result = ql(&t, &[0], &[1]).unwrap();

        assert_eq!(result.q.shape(), &[2, 2]);
        assert_eq!(result.l.shape(), &[2, 3]);
        assert_eq!(result.rank, 2);

        // Check orthogonality
        check_orthogonality(&result.q, 1e-10);

        // Check lower triangular
        check_lower_triangular(&result.l, 1e-10);

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
    fn test_ql_tall() {
        let t =
            DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2]).unwrap();
        let result = ql(&t, &[0], &[1]).unwrap();

        assert_eq!(result.q.shape(), &[4, 2]); // m=4, k=min(4,2)=2
        assert_eq!(result.l.shape(), &[2, 2]); // k=2, n=2
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
    fn test_ql_wide() {
        let t =
            DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]).unwrap();
        let result = ql(&t, &[0], &[1]).unwrap();

        assert_eq!(result.q.shape(), &[2, 2]); // m=2, k=min(2,4)=2
        assert_eq!(result.l.shape(), &[2, 4]); // k=2, n=4
        assert_eq!(result.rank, 2);

        // Check orthogonality
        check_orthogonality(&result.q, 1e-10);

        // Check lower triangular
        check_lower_triangular(&result.l, 1e-10);

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
    fn test_ql_3d() {
        let t = DenseTensor::<f64>::ones(&[2, 3, 4]);
        let result = ql(&t, &[0, 1], &[2]).unwrap();

        // m = 2*3 = 6, n = 4, k = min(6, 4) = 4
        assert_eq!(result.q.shape(), &[2, 3, 4]);
        assert_eq!(result.l.shape(), &[4, 4]);
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
}
