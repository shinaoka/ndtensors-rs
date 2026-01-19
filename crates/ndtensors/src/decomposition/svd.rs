//! Singular Value Decomposition (SVD) for tensors.
//!
//! This module provides SVD decomposition for tensors by reshaping them
//! to matrices, computing the SVD using faer, and reshaping the results.

use faer::linalg::solvers::{Svd, SvdError};
use faer_traits::math_utils::{conj, from_f64, real};

use crate::backend::AsFaerMat;
use crate::decomposition::util::{PermuteReshapeResult, permute_reshape};
use crate::error::TensorError;
use crate::scalar::Scalar;
use crate::tensor::DenseTensor;

/// Result of SVD decomposition.
#[derive(Debug, Clone)]
pub struct SvdResult<ElT: Scalar> {
    /// Left singular vectors reshaped to tensor form.
    /// Shape: [...left_dims..., rank]
    pub u: DenseTensor<ElT>,

    /// Singular values (1D tensor).
    /// Shape: [rank]
    pub s: DenseTensor<ElT>,

    /// Right singular vectors reshaped to tensor form.
    /// Shape: [rank, ...right_dims...]
    pub vt: DenseTensor<ElT>,

    /// Numerical rank (number of singular values above cutoff).
    pub rank: usize,
}

/// Compute the thin SVD of a tensor.
///
/// The tensor is reshaped into a matrix by grouping the specified left indices
/// into rows and right indices into columns. The SVD is then computed and the
/// results are reshaped back to tensor form.
///
/// # Arguments
///
/// * `tensor` - The input tensor
/// * `left_inds` - Indices to place on the left (become U's row dimensions)
/// * `right_inds` - Indices to place on the right (become V's column dimensions)
///
/// # Returns
///
/// `SvdResult` containing:
/// - `u`: Left singular vectors with shape [...left_dims..., rank]
/// - `s`: Singular values with shape [rank]
/// - `vt`: Right singular vectors with shape [rank, ...right_dims...]
/// - `rank`: Number of singular values
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::decomposition::svd;
///
/// let t = Tensor::<f64>::ones(&[2, 3, 4]);
/// let result = svd(&t, &[0, 1], &[2]).unwrap();
///
/// // U has shape [2, 3, rank], S has shape [rank], Vt has shape [rank, 4]
/// assert_eq!(result.u.ndim(), 3);
/// assert_eq!(result.s.ndim(), 1);
/// assert_eq!(result.vt.ndim(), 2);
/// ```
pub fn svd<ElT: Scalar>(
    tensor: &DenseTensor<ElT>,
    left_inds: &[usize],
    right_inds: &[usize],
) -> Result<SvdResult<ElT>, TensorError> {
    svd_truncated(tensor, left_inds, right_inds, None, None)
}

/// Compute truncated SVD of a tensor.
///
/// Like `svd`, but allows specifying a maximum rank and/or cutoff for singular values.
///
/// # Arguments
///
/// * `tensor` - The input tensor
/// * `left_inds` - Indices to place on the left
/// * `right_inds` - Indices to place on the right
/// * `max_rank` - Optional maximum number of singular values to keep
/// * `cutoff` - Optional singular value threshold (values below this are discarded)
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::decomposition::svd_truncated;
///
/// let t = Tensor::<f64>::ones(&[4, 4]);
/// let result = svd_truncated(&t, &[0], &[1], Some(2), Some(1e-10)).unwrap();
///
/// // At most 2 singular values kept
/// assert!(result.rank <= 2);
/// ```
pub fn svd_truncated<ElT: Scalar>(
    tensor: &DenseTensor<ElT>,
    left_inds: &[usize],
    right_inds: &[usize],
    max_rank: Option<usize>,
    cutoff: Option<f64>,
) -> Result<SvdResult<ElT>, TensorError> {
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
    let min_dim = m.min(n);

    // Convert to faer matrix
    let mat = matrix.as_faer_mat(m, n);

    // Compute thin SVD
    let svd_result: Svd<ElT> = Svd::new_thin(mat).map_err(|e: SvdError| TensorError::SvdError {
        message: format!("{:?}", e),
    })?;

    // Extract components
    let u_mat = svd_result.U();
    let s_diag = svd_result.S();
    let v_mat = svd_result.V();

    // Determine rank based on cutoff and max_rank
    let cutoff_elt: ElT = from_f64(cutoff.unwrap_or(0.0));
    let mut rank = min_dim;

    // Find number of singular values above cutoff
    // Singular values are real and non-negative, stored as ElT
    for k in 0..min_dim {
        // For cutoff comparison, we check if s_k < cutoff
        // Since singular values are real, we can compare directly
        if real(&s_diag[k]) < real(&cutoff_elt) {
            rank = k;
            break;
        }
    }

    // Apply max_rank constraint
    if let Some(max_r) = max_rank {
        rank = rank.min(max_r);
    }

    // Ensure at least rank 1
    rank = rank.max(1);

    // Extract U[:, :rank]
    let mut u_data = Vec::with_capacity(m * rank);
    for j in 0..rank {
        for i in 0..m {
            u_data.push(u_mat[(i, j)]);
        }
    }

    // Extract S[:rank] - singular values are real but stored as ElT
    let mut s_data = Vec::with_capacity(rank);
    for k in 0..rank {
        s_data.push(s_diag[k]);
    }

    // Extract V^H[:rank, :] (V is n x min_dim, V^H is min_dim x n)
    // In faer, V() returns V, so V^H[i,j] = conj(V[j,i])
    let mut vt_data = Vec::with_capacity(rank * n);
    for j in 0..n {
        for i in 0..rank {
            // V^H[i,j] = conj(V[j,i])
            vt_data.push(conj(&v_mat[(j, i)]));
        }
    }

    // Reshape U to [...left_dims..., rank]
    let mut u_shape = left_dims.clone();
    u_shape.push(rank);
    let u = DenseTensor::from_vec(u_data, &u_shape)?;

    // S is just a 1D tensor
    let s = DenseTensor::from_vec(s_data, &[rank])?;

    // Reshape V^H to [rank, ...right_dims...]
    let mut vt_shape = vec![rank];
    vt_shape.extend(right_dims);
    let vt = DenseTensor::from_vec(vt_data, &vt_shape)?;

    Ok(SvdResult { u, s, vt, rank })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::ops::{Add, Mul};

    fn reconstruct<ElT: Scalar + Add<Output = ElT> + Mul<Output = ElT>>(
        result: &SvdResult<ElT>,
        original_shape: &[usize],
        left_inds: &[usize],
        right_inds: &[usize],
    ) -> DenseTensor<ElT> {
        use crate::contract::contract;

        // Compute U @ diag(S) @ V^H
        // First, scale U by S: U_scaled[..., k] = U[..., k] * S[k]
        let u_shape = result.u.shape();
        let rank = result.rank;

        let mut u_scaled_data = result.u.data().to_vec();
        let u_rows: usize = u_shape[..u_shape.len() - 1].iter().product();

        for k in 0..rank {
            let s_k = *result.s.get_linear(k).unwrap();
            for i in 0..u_rows {
                let idx = k * u_rows + i;
                u_scaled_data[idx] = u_scaled_data[idx] * s_k;
            }
        }

        let u_scaled = DenseTensor::from_vec(u_scaled_data, u_shape).unwrap();

        // Contract U_scaled with V^H
        // U_scaled has shape [...left_dims..., rank]
        // V^H has shape [rank, ...right_dims...]
        // Result should have shape [...left_dims..., ...right_dims...]

        let u_ndim = u_scaled.ndim();
        let vt_ndim = result.vt.ndim();

        // Labels for U: [1, 2, ..., u_ndim-1, -1]
        let mut u_labels: Vec<i32> = (1..u_ndim as i32).collect();
        u_labels.push(-1);

        // Labels for V^H: [-1, u_ndim, u_ndim+1, ...]
        let mut vt_labels = vec![-1];
        for i in 0..(vt_ndim - 1) {
            vt_labels.push((u_ndim + i) as i32);
        }

        let contracted = contract(&u_scaled, &u_labels, &result.vt, &vt_labels).unwrap();

        // The result is in permuted order, need to permute back
        // Current order: [...left_dims..., ...right_dims...]
        // Need to map back to original_shape using inverse of permutation

        // Build the inverse permutation
        let perm: Vec<usize> = left_inds.iter().chain(right_inds.iter()).copied().collect();
        let mut inv_perm = vec![0; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            inv_perm[p] = i;
        }

        // Reshape to permuted shape first
        let mut permuted_shape: Vec<usize> = left_inds
            .iter()
            .map(|&i| original_shape[i])
            .chain(right_inds.iter().map(|&i| original_shape[i]))
            .collect();

        // Handle scalar case
        if permuted_shape.is_empty() {
            permuted_shape.push(1);
        }

        let reshaped = DenseTensor::from_vec(contracted.data().to_vec(), &permuted_shape).unwrap();

        // Apply inverse permutation
        use crate::operations::permutedims;
        permutedims(&reshaped, &inv_perm).unwrap()
    }

    #[test]
    fn test_svd_2d() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let result = svd(&t, &[0], &[1]).unwrap();

        assert_eq!(result.u.shape()[0], 2);
        assert_eq!(result.s.shape(), &[result.rank]);
        assert_eq!(result.vt.shape()[1], 3);

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
    fn test_svd_3d() {
        let t = DenseTensor::<f64>::ones(&[2, 3, 4]);
        let result = svd(&t, &[0, 1], &[2]).unwrap();

        assert_eq!(result.u.shape()[0], 2);
        assert_eq!(result.u.shape()[1], 3);
        assert_eq!(result.vt.shape()[1], 4);

        // Reconstruct and compare
        let reconstructed = reconstruct(&result, &[2, 3, 4], &[0, 1], &[2]);
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
    fn test_svd_truncated() {
        // Create a rank-2 matrix
        let t =
            DenseTensor::from_vec(vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0], &[4, 2]).unwrap();

        // Truncate to rank 1
        let result = svd_truncated(&t, &[0], &[1], Some(1), None).unwrap();

        assert_eq!(result.rank, 1);
        assert_eq!(result.u.shape(), &[4, 1]);
        assert_eq!(result.s.shape(), &[1]);
        assert_eq!(result.vt.shape(), &[1, 2]);
    }

    #[test]
    fn test_svd_singular_values_ordered() {
        let t = DenseTensor::from_vec(vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0], &[3, 3])
            .unwrap();
        let result = svd(&t, &[0], &[1]).unwrap();

        // Singular values should be in decreasing order
        for i in 0..result.rank - 1 {
            let s_i = *result.s.get_linear(i).unwrap();
            let s_i1 = *result.s.get_linear(i + 1).unwrap();
            assert!(s_i >= s_i1);
        }
    }
}
