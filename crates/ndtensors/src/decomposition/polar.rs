//! Polar decomposition for tensors.
//!
//! Decomposes a tensor A = U * P where:
//! - U is unitary (orthogonal for real)
//! - P is positive semi-definite Hermitian

use faer_traits::math_utils::conj;

use crate::backend::AsFaerMat;
use crate::decomposition::svd::svd;
use crate::decomposition::util::{PermuteReshapeResult, permute_reshape};
use crate::error::TensorError;
use crate::operations::diag_from_vec;
use crate::scalar::Scalar;
use crate::tensor::DenseTensor;

/// Result of polar decomposition.
#[derive(Debug, Clone)]
pub struct PolarResult<ElT: Scalar> {
    /// Unitary factor U.
    /// Shape: [...left_dims..., ...right_dims...]
    pub u: DenseTensor<ElT>,

    /// Positive semi-definite Hermitian factor P.
    /// Shape: [...right_dims..., ...right_dims...]
    pub p: DenseTensor<ElT>,
}

/// Compute the polar decomposition of a tensor.
///
/// The tensor is reshaped into a matrix by grouping the specified left indices
/// into rows and right indices into columns. The polar decomposition is computed
/// as A = U * P where U is unitary and P is positive semi-definite Hermitian.
///
/// # Algorithm
///
/// Uses SVD internally: A = U_svd * S * V^H
/// Then: U_polar = U_svd * V^H, P = V * S * V^H
///
/// # Arguments
///
/// * `tensor` - The input tensor
/// * `left_inds` - Indices to place on the left (become row dimensions)
/// * `right_inds` - Indices to place on the right (become column dimensions)
///
/// # Returns
///
/// `PolarResult` containing:
/// - `u`: Unitary factor with shape [...left_dims..., ...right_dims...]
/// - `p`: Positive semi-definite Hermitian factor with shape [...right_dims..., ...right_dims...]
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::decomposition::polar;
///
/// let t = Tensor::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
/// let result = polar(&t, &[0], &[1]).unwrap();
///
/// // U has shape [2, 3], P has shape [3, 3]
/// assert_eq!(result.u.shape(), &[2, 3]);
/// assert_eq!(result.p.shape(), &[3, 3]);
/// ```
pub fn polar<ElT: Scalar>(
    tensor: &DenseTensor<ElT>,
    left_inds: &[usize],
    right_inds: &[usize],
) -> Result<PolarResult<ElT>, TensorError> {
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

    // Compute SVD: A = U_svd * S * V^H
    // Our SVD returns u, s, vt where vt = V^H
    let svd_result = svd(&matrix, &[0], &[1])?;
    let u_svd = svd_result.u; // m x rank
    let s = svd_result.s; // rank
    let vt = svd_result.vt; // rank x n (this is V^H)
    let rank = svd_result.rank;

    // Compute U_polar = U_svd @ V^H
    // u_svd: m x rank, vt: rank x n -> U_polar: m x n
    let u_polar_2d = matrix_multiply(&u_svd, &vt, m, rank, n);

    // Compute P = V @ S @ V^H = (V^H)^H @ diag(S) @ V^H = vt^H @ diag(S) @ vt
    // First create diagonal matrix from S
    let s_diag = diag_from_vec(&s)?; // rank x rank

    // vt^H = conj(vt^T): n x rank
    let vt_h = hermitian_conjugate_2d(&vt, rank, n);

    // Compute vt^H @ diag(S) = n x rank @ rank x rank -> n x rank
    let vt_h_s = matrix_multiply(&vt_h, &s_diag, n, rank, rank);

    // Compute (vt^H @ S) @ vt = n x rank @ rank x n -> n x n
    let p_2d = matrix_multiply(&vt_h_s, &vt, n, rank, n);

    // Reshape U_polar to [...left_dims..., ...right_dims...]
    let mut u_shape = left_dims.clone();
    u_shape.extend(&right_dims);
    let u = DenseTensor::from_vec(u_polar_2d.data().to_vec(), &u_shape)?;

    // Reshape P to [...right_dims..., ...right_dims...]
    let mut p_shape = right_dims.clone();
    p_shape.extend(&right_dims);
    let p = DenseTensor::from_vec(p_2d.data().to_vec(), &p_shape)?;

    Ok(PolarResult { u, p })
}

/// Compute Hermitian conjugate (conjugate transpose) of a 2D tensor.
fn hermitian_conjugate_2d<ElT: Scalar>(
    tensor: &DenseTensor<ElT>,
    nrows: usize,
    ncols: usize,
) -> DenseTensor<ElT> {
    // Result has shape [ncols, nrows] (transposed)
    let mut data = Vec::with_capacity(nrows * ncols);

    // Column-major: output[j, i] = conj(input[i, j])
    // output is ncols x nrows in column-major
    for i in 0..nrows {
        for j in 0..ncols {
            // input[i, j] in column-major = data[i + j * nrows]
            let val = *tensor.get(&[i, j]).unwrap();
            data.push(conj(&val));
        }
    }

    DenseTensor::from_vec(data, &[ncols, nrows]).expect("hermitian_conjugate: valid shape")
}

/// Compute matrix multiplication C = A @ B using faer.
fn matrix_multiply<ElT: Scalar>(
    a: &DenseTensor<ElT>,
    b: &DenseTensor<ElT>,
    m: usize,
    k: usize,
    n: usize,
) -> DenseTensor<ElT> {
    use faer::mat::Mat;

    // Convert to faer matrices
    let a_mat = a.as_faer_mat(m, k);
    let b_mat = b.as_faer_mat(k, n);

    // Compute C = A @ B
    let c_mat: Mat<ElT> = a_mat * b_mat;

    // Extract data in column-major order
    let mut data = Vec::with_capacity(m * n);
    for j in 0..n {
        for i in 0..m {
            data.push(c_mat[(i, j)]);
        }
    }

    DenseTensor::from_vec(data, &[m, n]).expect("matrix_multiply: valid shape")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c64;
    use approx::assert_relative_eq;

    /// Check if U is unitary: U @ U^H = I (for f64)
    fn check_unitarity_f64(u: &DenseTensor<f64>, m: usize, n: usize, epsilon: f64) {
        let u_h = hermitian_conjugate_2d(u, m, n);
        let uuh = matrix_multiply(u, &u_h, m, n, m);

        for i in 0..m {
            for j in 0..m {
                let val = *uuh.get(&[i, j]).unwrap();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(val, expected, epsilon = epsilon);
            }
        }
    }

    /// Check if U is unitary: U @ U^H = I (for c64)
    fn check_unitarity_c64(u: &DenseTensor<c64>, m: usize, n: usize, epsilon: f64) {
        let u_h = hermitian_conjugate_2d(u, m, n);
        let uuh = matrix_multiply(u, &u_h, m, n, m);

        for i in 0..m {
            for j in 0..m {
                let val = *uuh.get(&[i, j]).unwrap();
                let expected_re = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(val.re, expected_re, epsilon = epsilon);
                assert_relative_eq!(val.im, 0.0, epsilon = epsilon);
            }
        }
    }

    /// Check if P is positive semi-definite for f64
    fn check_positive_semidefinite_f64(p: &DenseTensor<f64>, n: usize, epsilon: f64) {
        let p_mat = p.as_faer_mat(n, n);
        let evd = p_mat
            .self_adjoint_eigen(faer::Side::Lower)
            .expect("eigendecomposition failed");
        let eigenvalues = evd.S();

        for i in 0..n {
            let ev = eigenvalues[i];
            assert!(ev >= -epsilon, "Eigenvalue {} is negative: {}", i, ev);
        }
    }

    /// Check if P is positive semi-definite for c64
    fn check_positive_semidefinite_c64(p: &DenseTensor<c64>, n: usize, epsilon: f64) {
        let p_mat = p.as_faer_mat(n, n);
        let evd = p_mat
            .self_adjoint_eigen(faer::Side::Lower)
            .expect("eigendecomposition failed");
        let eigenvalues = evd.S();

        for i in 0..n {
            // Eigenvalues of Hermitian matrices are real, stored as c64 with zero imaginary part
            let ev = eigenvalues[i].re;
            assert!(ev >= -epsilon, "Eigenvalue {} is negative: {}", i, ev);
        }
    }

    #[test]
    fn test_polar_2d_square() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let result = polar(&t, &[0], &[1]).unwrap();

        assert_eq!(result.u.shape(), &[2, 2]);
        assert_eq!(result.p.shape(), &[2, 2]);

        check_unitarity_f64(&result.u, 2, 2, 1e-10);
        check_positive_semidefinite_f64(&result.p, 2, 1e-10);

        // Check reconstruction: A = U @ P
        let reconstructed = matrix_multiply(&result.u, &result.p, 2, 2, 2);
        for i in 0..t.len() {
            assert_relative_eq!(
                *reconstructed.get_linear(i).unwrap(),
                *t.get_linear(i).unwrap(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_polar_2d_rectangular() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let result = polar(&t, &[0], &[1]).unwrap();

        assert_eq!(result.u.shape(), &[2, 3]);
        assert_eq!(result.p.shape(), &[3, 3]);

        check_unitarity_f64(&result.u, 2, 3, 1e-10);
        check_positive_semidefinite_f64(&result.p, 3, 1e-10);

        // Check reconstruction
        let reconstructed = matrix_multiply(&result.u, &result.p, 2, 3, 3);
        for i in 0..t.len() {
            assert_relative_eq!(
                *reconstructed.get_linear(i).unwrap(),
                *t.get_linear(i).unwrap(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_polar_3d() {
        let t = DenseTensor::<f64>::ones(&[2, 3, 4]);
        let result = polar(&t, &[0, 1], &[2]).unwrap();

        assert_eq!(result.u.shape(), &[2, 3, 4]);
        assert_eq!(result.p.shape(), &[4, 4]);

        // Flatten to check reconstruction
        let u_2d = DenseTensor::from_vec(result.u.data().to_vec(), &[6, 4]).unwrap();
        let reconstructed = matrix_multiply(&u_2d, &result.p, 6, 4, 4);

        let t_2d = DenseTensor::from_vec(t.data().to_vec(), &[6, 4]).unwrap();
        for i in 0..t_2d.len() {
            assert_relative_eq!(
                *reconstructed.get_linear(i).unwrap(),
                *t_2d.get_linear(i).unwrap(),
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_polar_c64() {
        let t = DenseTensor::from_vec(
            vec![
                c64::new(1.0, 1.0),
                c64::new(2.0, -1.0),
                c64::new(3.0, 2.0),
                c64::new(4.0, 0.0),
            ],
            &[2, 2],
        )
        .unwrap();
        let result = polar(&t, &[0], &[1]).unwrap();

        assert_eq!(result.u.shape(), &[2, 2]);
        assert_eq!(result.p.shape(), &[2, 2]);

        check_unitarity_c64(&result.u, 2, 2, 1e-10);
        check_positive_semidefinite_c64(&result.p, 2, 1e-10);

        // Check reconstruction
        let reconstructed = matrix_multiply(&result.u, &result.p, 2, 2, 2);
        for i in 0..t.len() {
            let rec = *reconstructed.get_linear(i).unwrap();
            let orig = *t.get_linear(i).unwrap();
            assert_relative_eq!(rec.re, orig.re, epsilon = 1e-10);
            assert_relative_eq!(rec.im, orig.im, epsilon = 1e-10);
        }
    }
}
