//! Matrix exponential for tensors.
//!
//! Computes exp(A) where A is reshaped to a square matrix.

use std::ops::Mul;

use faer::linalg::solvers::Solve;
use faer::mat::Mat;
use faer_traits::math_utils::{conj, from_f64};

use crate::backend::AsFaerMat;
use crate::decomposition::util::{PermuteReshapeResult, permute_reshape};
use crate::error::TensorError;
use crate::scalar::{RealScalar, Scalar};
use crate::tensor::DenseTensor;

/// Compute the matrix exponential of a tensor.
///
/// The tensor is reshaped into a square matrix by grouping the specified left indices
/// into rows and right indices into columns. Then exp(A) is computed.
///
/// # Arguments
///
/// * `tensor` - The input tensor
/// * `left_inds` - Indices to place on the left (must produce same size as right)
/// * `right_inds` - Indices to place on the right
/// * `ishermitian` - If true, uses eigendecomposition (faster, more stable for Hermitian)
///
/// # Returns
///
/// A tensor with the same shape as input, containing exp(A).
///
/// # Errors
///
/// Returns error if the reshaped matrix is not square (left and right dimensions must match).
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::decomposition::matrix_exp;
///
/// // exp(zero matrix) = identity
/// let t = Tensor::<f64>::zeros(&[2, 2]);
/// let exp_t = matrix_exp(&t, &[0], &[1], false).unwrap();
/// assert!((exp_t.get(&[0, 0]).unwrap() - 1.0).abs() < 1e-10);
/// assert!((exp_t.get(&[1, 1]).unwrap() - 1.0).abs() < 1e-10);
/// ```
pub fn matrix_exp<ElT: Scalar + Mul<Output = ElT> + std::ops::Add<Output = ElT>>(
    tensor: &DenseTensor<ElT>,
    left_inds: &[usize],
    right_inds: &[usize],
    ishermitian: bool,
) -> Result<DenseTensor<ElT>, TensorError> {
    // Permute and reshape to matrix
    let PermuteReshapeResult {
        matrix,
        nrows,
        ncols,
        ..
    } = permute_reshape(tensor, left_inds, right_inds)?;

    // Matrix must be square
    if nrows != ncols {
        return Err(TensorError::NotSquareMatrix {
            rows: nrows,
            cols: ncols,
        });
    }

    let n = nrows;

    // Compute matrix exponential
    let exp_matrix = if ishermitian {
        matrix_exp_hermitian(&matrix, n)?
    } else {
        matrix_exp_pade(&matrix, n)?
    };

    // Get original permuted shape
    let mut perm_shape: Vec<usize> = left_inds.iter().map(|&i| tensor.shape()[i]).collect();
    perm_shape.extend(right_inds.iter().map(|&i| tensor.shape()[i]));

    // Build inverse permutation
    let perm: Vec<usize> = left_inds.iter().chain(right_inds.iter()).copied().collect();
    let mut inv_perm = vec![0; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv_perm[p] = i;
    }

    // Reshape to permuted shape
    let exp_reshaped = DenseTensor::from_vec(exp_matrix.data().to_vec(), &perm_shape)?;

    // Apply inverse permutation to get back to original shape
    crate::operations::permutedims(&exp_reshaped, &inv_perm)
}

/// Compute matrix exponential for Hermitian matrices using eigendecomposition.
///
/// For A = V @ diag(λ) @ V^H, we have:
/// exp(A) = V @ diag(exp(λ)) @ V^H
fn matrix_exp_hermitian<ElT: Scalar + Mul<Output = ElT>>(
    matrix: &DenseTensor<ElT>,
    n: usize,
) -> Result<DenseTensor<ElT>, TensorError> {
    let mat = matrix.as_faer_mat(n, n);

    // Compute eigendecomposition
    let evd =
        mat.self_adjoint_eigen(faer::Side::Lower)
            .map_err(|e| TensorError::MatrixExpError {
                message: format!("eigendecomposition failed: {:?}", e),
            })?;

    let v_mat = evd.U(); // Eigenvectors (columns)
    let eigenvalues = evd.S(); // Eigenvalues (real, diagonal)

    // Compute exp(λ) for each eigenvalue
    // For self-adjoint eigendecomposition, eigenvalues are real (stored as element type)
    let mut exp_lambda: Vec<ElT> = Vec::with_capacity(n);
    for i in 0..n {
        let lambda = eigenvalues[i];
        // For f64, lambda is f64. For c64, lambda is c64 but with zero imaginary part.
        // We extract the real part (which is always f64 per our Scalar impl).
        let lambda_real = lambda.real_part();
        // Compute exp - RealScalar::exp returns <ElT as Scalar>::Real
        let exp_val = RealScalar::exp(lambda_real);
        // Convert to f64 and then to element type
        exp_lambda.push(from_f64(exp_val.to_f64()));
    }

    // Compute V @ diag(exp(λ)) @ V^H
    // First: V_scaled = V @ diag(exp(λ)), i.e., scale each column of V by exp(λ_i)
    let mut v_scaled_data = Vec::with_capacity(n * n);
    for j in 0..n {
        let scale = exp_lambda[j];
        for i in 0..n {
            let v_ij = v_mat[(i, j)];
            v_scaled_data.push(v_ij * scale);
        }
    }
    let v_scaled = DenseTensor::from_vec(v_scaled_data, &[n, n])?;

    // Compute V^H
    let mut vh_data = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            // V^H[i,j] = conj(V[j,i])
            vh_data.push(conj(&v_mat[(j, i)]));
        }
    }
    let v_h = DenseTensor::from_vec(vh_data, &[n, n])?;

    // Compute V_scaled @ V^H
    matrix_multiply(&v_scaled, &v_h, n, n, n)
}

/// Compute matrix exponential using Padé approximation with scaling and squaring.
///
/// Algorithm from Higham (2005): "The Scaling and Squaring Method for the Matrix Exponential Revisited"
/// Uses [13/13] Padé approximation.
fn matrix_exp_pade<ElT: Scalar + Mul<Output = ElT> + std::ops::Add<Output = ElT>>(
    matrix: &DenseTensor<ElT>,
    n: usize,
) -> Result<DenseTensor<ElT>, TensorError> {
    // For very small matrices, use direct series expansion
    // For larger matrices, use Padé approximation

    // Estimate ||A||_1 (1-norm)
    let norm1 = matrix_1_norm(matrix, n);

    // Padé coefficients for [13/13] approximation
    // theta_13 ≈ 5.37 is the threshold for [13/13] Padé
    let theta_13 = 5.37;

    // Determine scaling factor
    let s = if norm1 > theta_13 {
        (norm1 / theta_13).ln().ceil() as i32 / std::f64::consts::LN_2.ln().ceil() as i32 + 1
    } else {
        0
    };
    let s = s.max(0) as usize;

    // Scale matrix: A_scaled = A / 2^s
    let scale_factor = 1.0 / (1u64 << s) as f64;
    let a_scaled = scale_matrix(matrix, n, scale_factor);

    // Compute Padé approximation
    let exp_scaled = pade_13(&a_scaled, n)?;

    // Square the result s times
    let mut result = exp_scaled;
    for _ in 0..s {
        result = matrix_multiply(&result, &result, n, n, n)?;
    }

    Ok(result)
}

/// Compute [13/13] Padé approximation of exp(A).
fn pade_13<ElT: Scalar + Mul<Output = ElT> + std::ops::Add<Output = ElT>>(
    a: &DenseTensor<ElT>,
    n: usize,
) -> Result<DenseTensor<ElT>, TensorError> {
    // Padé coefficients for [13/13]
    let b: [f64; 14] = [
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    ];

    // Compute powers of A: A^2, A^4, A^6
    let a2 = matrix_multiply(a, a, n, n, n)?;
    let a4 = matrix_multiply(&a2, &a2, n, n, n)?;
    let a6 = matrix_multiply(&a2, &a4, n, n, n)?;

    // Compute U and V polynomials
    // U = A @ (A^6 @ (b13*A^6 + b11*A^4 + b9*A^2) + b7*A^6 + b5*A^4 + b3*A^2 + b1*I)
    // V = A^6 @ (b12*A^6 + b10*A^4 + b8*A^2) + b6*A^6 + b4*A^4 + b2*A^2 + b0*I

    let identity = identity_matrix::<ElT>(n);

    // Inner part of U: A^6 @ (b13*A^6 + b11*A^4 + b9*A^2) + b7*A^6 + b5*A^4 + b3*A^2 + b1*I
    let u_inner = {
        // b13*A^6 + b11*A^4 + b9*A^2
        let term1 = matrix_add_scaled(
            &a6,
            b[13],
            &matrix_add_scaled(&a4, b[11], &scale_matrix(&a2, n, b[9]), n)?,
            n,
        )?;
        // A^6 @ term1
        let term2 = matrix_multiply(&a6, &term1, n, n, n)?;
        // + b7*A^6 + b5*A^4 + b3*A^2 + b1*I
        let term3 = matrix_add_scaled(&term2, 1.0, &scale_matrix(&a6, n, b[7]), n)?;
        let term4 = matrix_add_scaled(&term3, 1.0, &scale_matrix(&a4, n, b[5]), n)?;
        let term5 = matrix_add_scaled(&term4, 1.0, &scale_matrix(&a2, n, b[3]), n)?;
        matrix_add_scaled(&term5, 1.0, &scale_matrix(&identity, n, b[1]), n)?
    };

    // U = A @ u_inner
    let u = matrix_multiply(a, &u_inner, n, n, n)?;

    // V: A^6 @ (b12*A^6 + b10*A^4 + b8*A^2) + b6*A^6 + b4*A^4 + b2*A^2 + b0*I
    let v = {
        // b12*A^6 + b10*A^4 + b8*A^2
        let term1 = matrix_add_scaled(
            &a6,
            b[12],
            &matrix_add_scaled(&a4, b[10], &scale_matrix(&a2, n, b[8]), n)?,
            n,
        )?;
        // A^6 @ term1
        let term2 = matrix_multiply(&a6, &term1, n, n, n)?;
        // + b6*A^6 + b4*A^4 + b2*A^2 + b0*I
        let term3 = matrix_add_scaled(&term2, 1.0, &scale_matrix(&a6, n, b[6]), n)?;
        let term4 = matrix_add_scaled(&term3, 1.0, &scale_matrix(&a4, n, b[4]), n)?;
        let term5 = matrix_add_scaled(&term4, 1.0, &scale_matrix(&a2, n, b[2]), n)?;
        matrix_add_scaled(&term5, 1.0, &scale_matrix(&identity, n, b[0]), n)?
    };

    // exp(A) ≈ (V - U)^{-1} @ (V + U)
    let v_plus_u = matrix_add_scaled(&v, 1.0, &u, n)?;
    let v_minus_u = matrix_add_scaled(&v, -1.0, &u, n)?;

    // Solve (V - U) @ X = (V + U) for X
    solve_linear_system(&v_minus_u, &v_plus_u, n)
}

/// Compute matrix 1-norm (maximum absolute column sum).
fn matrix_1_norm<ElT: Scalar>(matrix: &DenseTensor<ElT>, n: usize) -> f64 {
    let mut max_sum = 0.0;
    for j in 0..n {
        let mut col_sum = 0.0;
        for i in 0..n {
            let val = *matrix.get(&[i, j]).unwrap();
            col_sum += RealScalar::sqrt(val.abs_sqr()).to_f64();
        }
        if col_sum > max_sum {
            max_sum = col_sum;
        }
    }
    max_sum
}

/// Create identity matrix.
fn identity_matrix<ElT: Scalar>(n: usize) -> DenseTensor<ElT> {
    let mut data = vec![ElT::zero(); n * n];
    for i in 0..n {
        data[i + i * n] = ElT::one();
    }
    DenseTensor::from_vec(data, &[n, n]).expect("identity_matrix: valid shape")
}

/// Scale a matrix by a scalar.
fn scale_matrix<ElT: Scalar + Mul<Output = ElT>>(
    matrix: &DenseTensor<ElT>,
    n: usize,
    scale: f64,
) -> DenseTensor<ElT> {
    let scale_elt: ElT = from_f64(scale);
    let data: Vec<ElT> = matrix.data().iter().map(|&x| x * scale_elt).collect();
    DenseTensor::from_vec(data, &[n, n]).expect("scale_matrix: valid shape")
}

/// Add two matrices: result = a + scale * b
fn matrix_add_scaled<ElT: Scalar + Mul<Output = ElT> + std::ops::Add<Output = ElT>>(
    a: &DenseTensor<ElT>,
    scale: f64,
    b: &DenseTensor<ElT>,
    n: usize,
) -> Result<DenseTensor<ElT>, TensorError> {
    let scale_elt: ElT = from_f64(scale);
    let data: Vec<ElT> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(&ai, &bi)| ai + bi * scale_elt)
        .collect();
    DenseTensor::from_vec(data, &[n, n])
}

/// Matrix multiplication C = A @ B.
fn matrix_multiply<ElT: Scalar>(
    a: &DenseTensor<ElT>,
    b: &DenseTensor<ElT>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<DenseTensor<ElT>, TensorError> {
    let a_mat = a.as_faer_mat(m, k);
    let b_mat = b.as_faer_mat(k, n);

    let c_mat: Mat<ElT> = a_mat * b_mat;

    let mut data = Vec::with_capacity(m * n);
    for j in 0..n {
        for i in 0..m {
            data.push(c_mat[(i, j)]);
        }
    }

    DenseTensor::from_vec(data, &[m, n])
}

/// Solve linear system A @ X = B for X using LU decomposition.
fn solve_linear_system<ElT: Scalar>(
    a: &DenseTensor<ElT>,
    b: &DenseTensor<ElT>,
    n: usize,
) -> Result<DenseTensor<ElT>, TensorError> {
    let a_mat = a.as_faer_mat(n, n);
    let b_mat = b.as_faer_mat(n, n);

    // LU decomposition with partial pivoting
    let lu = a_mat.partial_piv_lu();

    // Solve A @ X = B
    let mut x_mat = b_mat.to_owned();
    lu.solve_in_place(&mut x_mat);

    // Extract result
    let mut data = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            data.push(x_mat[(i, j)]);
        }
    }

    DenseTensor::from_vec(data, &[n, n])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c64;
    use approx::assert_relative_eq;

    #[test]
    fn test_exp_zero_is_identity() {
        // exp(0) = I
        let t = DenseTensor::<f64>::zeros(&[3, 3]);
        let exp_t = matrix_exp(&t, &[0], &[1], false).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(*exp_t.get(&[i, j]).unwrap(), expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_exp_hermitian_zero_is_identity() {
        // exp(0) = I (Hermitian path)
        let t = DenseTensor::<f64>::zeros(&[3, 3]);
        let exp_t = matrix_exp(&t, &[0], &[1], true).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(*exp_t.get(&[i, j]).unwrap(), expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_exp_diagonal() {
        // exp(diag(a, b)) = diag(e^a, e^b)
        let mut t = DenseTensor::<f64>::zeros(&[2, 2]);
        t.set(&[0, 0], 1.0).unwrap();
        t.set(&[1, 1], 2.0).unwrap();

        let exp_t = matrix_exp(&t, &[0], &[1], true).unwrap();

        assert_relative_eq!(*exp_t.get(&[0, 0]).unwrap(), 1.0_f64.exp(), epsilon = 1e-10);
        assert_relative_eq!(*exp_t.get(&[1, 1]).unwrap(), 2.0_f64.exp(), epsilon = 1e-10);
        assert_relative_eq!(*exp_t.get(&[0, 1]).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(*exp_t.get(&[1, 0]).unwrap(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_exp_inverse() {
        // exp(A) @ exp(-A) = I
        // Use a simple symmetric matrix
        let mut t = DenseTensor::<f64>::zeros(&[2, 2]);
        t.set(&[0, 0], 0.5).unwrap();
        t.set(&[0, 1], 0.3).unwrap();
        t.set(&[1, 0], 0.3).unwrap();
        t.set(&[1, 1], 0.4).unwrap();

        let exp_t = matrix_exp(&t, &[0], &[1], true).unwrap();

        // Create -t
        let neg_t = crate::operations::scale(&t, -1.0);
        let exp_neg_t = matrix_exp(&neg_t, &[0], &[1], true).unwrap();

        // Multiply exp(t) @ exp(-t)
        let result = super::matrix_multiply(&exp_t, &exp_neg_t, 2, 2, 2).unwrap();

        // Should be identity
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(*result.get(&[i, j]).unwrap(), expected, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_exp_general_small_matrix() {
        // Test with a known result: [[0, 1], [0, 0]]
        // exp([[0, 1], [0, 0]]) = [[1, 1], [0, 1]]
        let mut t = DenseTensor::<f64>::zeros(&[2, 2]);
        t.set(&[0, 1], 1.0).unwrap();

        let exp_t = matrix_exp(&t, &[0], &[1], false).unwrap();

        assert_relative_eq!(*exp_t.get(&[0, 0]).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(*exp_t.get(&[0, 1]).unwrap(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(*exp_t.get(&[1, 0]).unwrap(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(*exp_t.get(&[1, 1]).unwrap(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_exp_c64_hermitian() {
        // Hermitian matrix with complex off-diagonals
        // [[1, i], [-i, 2]]
        let mut t = DenseTensor::<c64>::zeros(&[2, 2]);
        t.set(&[0, 0], c64::new(1.0, 0.0)).unwrap();
        t.set(&[0, 1], c64::new(0.0, 1.0)).unwrap();
        t.set(&[1, 0], c64::new(0.0, -1.0)).unwrap();
        t.set(&[1, 1], c64::new(2.0, 0.0)).unwrap();

        let exp_t = matrix_exp(&t, &[0], &[1], true).unwrap();

        // Verify exp(A) @ exp(-A) = I
        let neg_t = crate::operations::scale(&t, c64::new(-1.0, 0.0));
        let exp_neg_t = matrix_exp(&neg_t, &[0], &[1], true).unwrap();

        let result = super::matrix_multiply(&exp_t, &exp_neg_t, 2, 2, 2).unwrap();

        for i in 0..2 {
            for j in 0..2 {
                let val = *result.get(&[i, j]).unwrap();
                let expected_re = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(val.re, expected_re, epsilon = 1e-9);
                assert_relative_eq!(val.im, 0.0, epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_exp_not_square() {
        let t = DenseTensor::<f64>::zeros(&[2, 3]);
        let result = matrix_exp(&t, &[0], &[1], false);
        assert!(result.is_err());
    }
}
