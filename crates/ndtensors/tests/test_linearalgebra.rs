//! Tests for linear algebra utilities.
//!
//! These tests mirror NDTensors.jl's test_linearalgebra.jl, covering:
//! - random_orthog: Random orthogonal matrix generation
//! - random_unitary: Random unitary matrix generation
//! - QR decomposition with positive factorization mode
//! - QL decomposition with positive factorization mode

use approx::assert_relative_eq;
use ndtensors::Tensor;
use ndtensors::c64;
use ndtensors::decomposition::{ql, ql_positive, qr, qr_positive};
use ndtensors::random::{random_orthog, random_unitary};

/// Test random_orthog generates matrices with orthonormal columns (tall) or rows (wide).
/// Mirrors: @testset "random_orthog" from test_linearalgebra.jl
#[test]
fn test_random_orthog() {
    let (n, m) = (10, 4);

    // Tall matrix: O^T * O should be I_m
    let o1 = random_orthog(n, m);
    assert_eq!(o1.shape(), &[n, m]);

    // Check O1^T * O1 ≈ I_m
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += o1.get(&[k, i]).unwrap() * o1.get(&[k, j]).unwrap();
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(sum, expected, epsilon = 1e-14);
        }
    }

    // Wide matrix: O * O^T should be I_m
    let o2 = random_orthog(m, n);
    assert_eq!(o2.shape(), &[m, n]);

    // Check O2 * O2^T ≈ I_m
    for i in 0..m {
        for j in 0..m {
            let mut sum = 0.0;
            for k in 0..n {
                sum += o2.get(&[i, k]).unwrap() * o2.get(&[j, k]).unwrap();
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(sum, expected, epsilon = 1e-14);
        }
    }
}

/// Test random_unitary generates matrices with orthonormal columns (tall) or rows (wide).
/// Mirrors: @testset "random_unitary" from test_linearalgebra.jl
#[test]
fn test_random_unitary() {
    let (n, m) = (10, 4);

    // Tall matrix: U^H * U should be I_m
    let u1 = random_unitary(n, m);
    assert_eq!(u1.shape(), &[n, m]);

    // Check U1^H * U1 ≈ I_m
    for i in 0..m {
        for j in 0..m {
            let mut sum = c64::new(0.0, 0.0);
            for k in 0..n {
                let u_ki = *u1.get(&[k, i]).unwrap();
                let u_kj = *u1.get(&[k, j]).unwrap();
                // U^H means conjugate transpose
                sum = c64::new(
                    sum.re + u_ki.re * u_kj.re + u_ki.im * u_kj.im,
                    sum.im + u_ki.re * u_kj.im - u_ki.im * u_kj.re,
                );
            }
            let expected_re = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(sum.re, expected_re, epsilon = 1e-14);
            assert_relative_eq!(sum.im, 0.0, epsilon = 1e-14);
        }
    }

    // Wide matrix: U * U^H should be I_m
    let u2 = random_unitary(m, n);
    assert_eq!(u2.shape(), &[m, n]);

    // Check U2 * U2^H ≈ I_m
    for i in 0..m {
        for j in 0..m {
            let mut sum = c64::new(0.0, 0.0);
            for k in 0..n {
                let u_ik = *u2.get(&[i, k]).unwrap();
                let u_jk = *u2.get(&[j, k]).unwrap();
                // U * U^H: sum over u_ik * conj(u_jk)
                sum = c64::new(
                    sum.re + u_ik.re * u_jk.re + u_ik.im * u_jk.im,
                    sum.im + u_ik.im * u_jk.re - u_ik.re * u_jk.im,
                );
            }
            let expected_re = if i == j { 1.0 } else { 0.0 };
            assert_relative_eq!(sum.re, expected_re, epsilon = 1e-14);
            assert_relative_eq!(sum.im, 0.0, epsilon = 1e-14);
        }
    }
}

/// Generic test for QX (QR or QL) decompositions.
/// Mirrors: @testset "QX testing" from test_linearalgebra.jl
mod qx_tests {
    use super::*;
    use ndtensors::contract::contract;
    use ndtensors::decomposition::{QlResult, QrResult};
    use ndtensors::scalar::{RealScalar, Scalar};

    /// Check Q^H * Q ≈ I
    fn check_orthogonality<ElT: Scalar>(q: &Tensor<ElT>, epsilon: f64)
    where
        <ElT as Scalar>::Real: RealScalar,
    {
        let q_shape = q.shape();
        let m: usize = q_shape[..q_shape.len() - 1].iter().product();
        let k = q_shape[q_shape.len() - 1];

        for i in 0..k {
            for j in 0..k {
                let mut sum_re = 0.0;
                let mut sum_im = 0.0;
                for r in 0..m {
                    let q_ri = *q.get_linear(r + i * m).unwrap();
                    let q_rj = *q.get_linear(r + j * m).unwrap();
                    // conj(q_ri) * q_rj
                    sum_re += q_ri.real_part().to_f64() * q_rj.real_part().to_f64()
                        + q_ri.imag_part().to_f64() * q_rj.imag_part().to_f64();
                    sum_im += q_ri.real_part().to_f64() * q_rj.imag_part().to_f64()
                        - q_ri.imag_part().to_f64() * q_rj.real_part().to_f64();
                }
                let expected_re = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum_re - expected_re).abs() < epsilon,
                    "Q^H*Q[{},{}] real part: {} != {}",
                    i,
                    j,
                    sum_re,
                    expected_re
                );
                assert!(
                    sum_im.abs() < epsilon,
                    "Q^H*Q[{},{}] imag part: {} != 0",
                    i,
                    j,
                    sum_im
                );
            }
        }
    }

    /// Check that Q * Q^H ≈ I (for square Q)
    fn check_orthogonality_full<ElT: Scalar>(q: &Tensor<ElT>, epsilon: f64)
    where
        <ElT as Scalar>::Real: RealScalar,
    {
        let q_shape = q.shape();
        let m: usize = q_shape[..q_shape.len() - 1].iter().product();
        let k = q_shape[q_shape.len() - 1];

        if m != k {
            return; // Only check for square Q
        }

        for i in 0..m {
            for j in 0..m {
                let mut sum_re = 0.0;
                let mut sum_im = 0.0;
                for c in 0..k {
                    let q_ic = *q.get_linear(i + c * m).unwrap();
                    let q_jc = *q.get_linear(j + c * m).unwrap();
                    // q_ic * conj(q_jc)
                    sum_re += q_ic.real_part().to_f64() * q_jc.real_part().to_f64()
                        + q_ic.imag_part().to_f64() * q_jc.imag_part().to_f64();
                    sum_im += q_ic.imag_part().to_f64() * q_jc.real_part().to_f64()
                        - q_ic.real_part().to_f64() * q_jc.imag_part().to_f64();
                }
                let expected_re = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum_re - expected_re).abs() < epsilon,
                    "Q*Q^H[{},{}] real part: {} != {}",
                    i,
                    j,
                    sum_re,
                    expected_re
                );
                assert!(
                    sum_im.abs() < epsilon,
                    "Q*Q^H[{},{}] imag part: {} != 0",
                    i,
                    j,
                    sum_im
                );
            }
        }
    }

    /// Check reconstruction A ≈ Q * X
    fn check_reconstruction_qr<
        ElT: Scalar + std::ops::Add<Output = ElT> + std::ops::Mul<Output = ElT>,
    >(
        result: &QrResult<ElT>,
        original: &Tensor<ElT>,
        epsilon: f64,
    ) where
        <ElT as Scalar>::Real: RealScalar,
    {
        let q_ndim = result.q.ndim();
        let r_ndim = result.r.ndim();

        let mut q_labels: Vec<i32> = (1..q_ndim as i32).collect();
        q_labels.push(-1);

        let mut r_labels = vec![-1];
        for i in 0..(r_ndim - 1) {
            r_labels.push((q_ndim + i) as i32);
        }

        let reconstructed = contract(&result.q, &q_labels, &result.r, &r_labels).unwrap();

        for i in 0..original.len() {
            let orig = *original.get_linear(i).unwrap();
            let recon = *reconstructed.get_linear(i).unwrap();
            let diff = (orig.real_part().to_f64() - recon.real_part().to_f64()).abs()
                + (orig.imag_part().to_f64() - recon.imag_part().to_f64()).abs();
            assert!(
                diff < epsilon,
                "Reconstruction QR failed at {}: orig={:?}, recon={:?}",
                i,
                orig,
                recon
            );
        }
    }

    fn check_reconstruction_ql<
        ElT: Scalar + std::ops::Add<Output = ElT> + std::ops::Mul<Output = ElT>,
    >(
        result: &QlResult<ElT>,
        original: &Tensor<ElT>,
        epsilon: f64,
    ) where
        <ElT as Scalar>::Real: RealScalar,
    {
        let q_ndim = result.q.ndim();
        let l_ndim = result.l.ndim();

        let mut q_labels: Vec<i32> = (1..q_ndim as i32).collect();
        q_labels.push(-1);

        let mut l_labels = vec![-1];
        for i in 0..(l_ndim - 1) {
            l_labels.push((q_ndim + i) as i32);
        }

        let reconstructed = contract(&result.q, &q_labels, &result.l, &l_labels).unwrap();

        for i in 0..original.len() {
            let orig = *original.get_linear(i).unwrap();
            let recon = *reconstructed.get_linear(i).unwrap();
            let diff = (orig.real_part().to_f64() - recon.real_part().to_f64()).abs()
                + (orig.imag_part().to_f64() - recon.imag_part().to_f64()).abs();
            assert!(
                diff < epsilon,
                "Reconstruction QL failed at {}: orig={:?}, recon={:?}",
                i,
                orig,
                recon
            );
        }
    }

    /// Check positive diagonal for QR
    fn check_positive_diagonal_qr<ElT: Scalar>(result: &QrResult<ElT>)
    where
        <ElT as Scalar>::Real: RealScalar,
    {
        let k = result.rank;
        for i in 0..k {
            let r_ii = *result.r.get(&[i, i]).unwrap();
            assert!(
                r_ii.real_part().to_f64() >= -1e-10,
                "R[{},{}].re = {} should be non-negative",
                i,
                i,
                r_ii.real_part().to_f64()
            );
            assert!(
                r_ii.imag_part().to_f64().abs() < 1e-10,
                "R[{},{}].im = {} should be zero",
                i,
                i,
                r_ii.imag_part().to_f64()
            );
        }
    }

    /// Check positive diagonal for QL
    fn check_positive_diagonal_ql<ElT: Scalar>(result: &QlResult<ElT>)
    where
        <ElT as Scalar>::Real: RealScalar,
    {
        let k = result.rank;
        let n = result.l.shape()[1];
        let diag_offset = n.saturating_sub(k);

        for i in 0..k.min(n) {
            let diag_col = diag_offset + i;
            if diag_col >= n {
                continue;
            }
            let l_ii = *result.l.get(&[i, diag_col]).unwrap();
            assert!(
                l_ii.real_part().to_f64() >= -1e-10,
                "L[{},{}].re = {} should be non-negative",
                i,
                diag_col,
                l_ii.real_part().to_f64()
            );
            assert!(
                l_ii.imag_part().to_f64().abs() < 1e-10,
                "L[{},{}].im = {} should be zero",
                i,
                diag_col,
                l_ii.imag_part().to_f64()
            );
        }
    }

    // ========== QR Tests ==========

    #[test]
    fn test_qr_f64_wide_positive() {
        let eps = f64::EPSILON * 100.0;
        let (n, m) = (4, 8);
        let a: Tensor<f64> = Tensor::randn(&[n, m]);

        let result = qr_positive(&a, &[0], &[1]).unwrap();
        check_reconstruction_qr(&result, &a, eps);
        check_orthogonality(&result.q, eps);
        check_orthogonality_full(&result.q, eps);
        check_positive_diagonal_qr(&result);
    }

    #[test]
    fn test_qr_f64_tall_positive() {
        let eps = f64::EPSILON * 100.0;
        let (n, m) = (8, 4);
        let a: Tensor<f64> = Tensor::randn(&[n, m]);

        let result = qr_positive(&a, &[0], &[1]).unwrap();
        check_reconstruction_qr(&result, &a, eps);
        check_orthogonality(&result.q, eps);
        check_positive_diagonal_qr(&result);
    }

    #[test]
    fn test_qr_c64_wide_positive() {
        let eps = f64::EPSILON * 100.0;
        let (n, m) = (4, 8);
        let a: Tensor<c64> = Tensor::randn(&[n, m]);

        let result = qr_positive(&a, &[0], &[1]).unwrap();
        check_reconstruction_qr(&result, &a, eps);
        check_orthogonality(&result.q, eps);
        check_orthogonality_full(&result.q, eps);
        check_positive_diagonal_qr(&result);
    }

    #[test]
    fn test_qr_c64_tall_positive() {
        let eps = f64::EPSILON * 100.0;
        let (n, m) = (8, 4);
        let a: Tensor<c64> = Tensor::randn(&[n, m]);

        let result = qr_positive(&a, &[0], &[1]).unwrap();
        check_reconstruction_qr(&result, &a, eps);
        check_orthogonality(&result.q, eps);
        check_positive_diagonal_qr(&result);
    }

    #[test]
    fn test_qr_f64_wide_no_positive() {
        let eps = f64::EPSILON * 100.0;
        let (n, m) = (4, 8);
        let a: Tensor<f64> = Tensor::randn(&[n, m]);

        let result = qr(&a, &[0], &[1]).unwrap();
        check_reconstruction_qr(&result, &a, eps);
        check_orthogonality(&result.q, eps);
        check_orthogonality_full(&result.q, eps);
    }

    #[test]
    fn test_qr_f64_tall_no_positive() {
        let eps = f64::EPSILON * 100.0;
        let (n, m) = (8, 4);
        let a: Tensor<f64> = Tensor::randn(&[n, m]);

        let result = qr(&a, &[0], &[1]).unwrap();
        check_reconstruction_qr(&result, &a, eps);
        check_orthogonality(&result.q, eps);
    }

    // ========== QL Tests ==========

    #[test]
    fn test_ql_f64_wide_positive() {
        let eps = f64::EPSILON * 100.0;
        let (n, m) = (4, 8);
        let a: Tensor<f64> = Tensor::randn(&[n, m]);

        let result = ql_positive(&a, &[0], &[1]).unwrap();
        check_reconstruction_ql(&result, &a, eps);
        check_orthogonality(&result.q, eps);
        check_orthogonality_full(&result.q, eps);
        check_positive_diagonal_ql(&result);
    }

    #[test]
    fn test_ql_f64_tall_positive() {
        let eps = f64::EPSILON * 100.0;
        let (n, m) = (8, 4);
        let a: Tensor<f64> = Tensor::randn(&[n, m]);

        let result = ql_positive(&a, &[0], &[1]).unwrap();
        check_reconstruction_ql(&result, &a, eps);
        check_orthogonality(&result.q, eps);
        check_positive_diagonal_ql(&result);
    }

    #[test]
    fn test_ql_c64_wide_positive() {
        let eps = f64::EPSILON * 100.0;
        let (n, m) = (4, 8);
        let a: Tensor<c64> = Tensor::randn(&[n, m]);

        let result = ql_positive(&a, &[0], &[1]).unwrap();
        check_reconstruction_ql(&result, &a, eps);
        check_orthogonality(&result.q, eps);
        check_orthogonality_full(&result.q, eps);
        check_positive_diagonal_ql(&result);
    }

    #[test]
    fn test_ql_c64_tall_positive() {
        let eps = f64::EPSILON * 100.0;
        let (n, m) = (8, 4);
        let a: Tensor<c64> = Tensor::randn(&[n, m]);

        let result = ql_positive(&a, &[0], &[1]).unwrap();
        check_reconstruction_ql(&result, &a, eps);
        check_orthogonality(&result.q, eps);
        check_positive_diagonal_ql(&result);
    }

    #[test]
    fn test_ql_f64_wide_no_positive() {
        let eps = f64::EPSILON * 100.0;
        let (n, m) = (4, 8);
        let a: Tensor<f64> = Tensor::randn(&[n, m]);

        let result = ql(&a, &[0], &[1]).unwrap();
        check_reconstruction_ql(&result, &a, eps);
        check_orthogonality(&result.q, eps);
        check_orthogonality_full(&result.q, eps);
    }

    #[test]
    fn test_ql_f64_tall_no_positive() {
        let eps = f64::EPSILON * 100.0;
        let (n, m) = (8, 4);
        let a: Tensor<f64> = Tensor::randn(&[n, m]);

        let result = ql(&a, &[0], &[1]).unwrap();
        check_reconstruction_ql(&result, &a, eps);
        check_orthogonality(&result.q, eps);
    }

    // ========== Singular Matrix Tests ==========
    // Test handling of matrices with rank deficiency

    #[test]
    fn test_qr_singular_f64() {
        let eps = f64::EPSILON * 1000.0;
        let (n, m) = (4, 8);

        // Create a singular matrix by making all rows equal
        let row: Tensor<f64> = Tensor::randn(&[1, m]);
        let mut data = Vec::with_capacity(n * m);
        for _ in 0..n {
            data.extend_from_slice(row.data());
        }
        let a = Tensor::from_vec(data, &[n, m]).unwrap();

        let result = qr_positive(&a, &[0], &[1]).unwrap();
        check_reconstruction_qr(&result, &a, eps);
        check_positive_diagonal_qr(&result);
    }

    #[test]
    fn test_ql_singular_f64() {
        let eps = f64::EPSILON * 1000.0;
        let (n, m) = (4, 8);

        // Create a singular matrix by making all rows equal
        let row: Tensor<f64> = Tensor::randn(&[1, m]);
        let mut data = Vec::with_capacity(n * m);
        for _ in 0..n {
            data.extend_from_slice(row.data());
        }
        let a = Tensor::from_vec(data, &[n, m]).unwrap();

        let result = ql_positive(&a, &[0], &[1]).unwrap();
        check_reconstruction_ql(&result, &a, eps);
        check_positive_diagonal_ql(&result);
    }

    #[test]
    fn test_qr_singular_c64() {
        let eps = f64::EPSILON * 1000.0;
        let (n, m) = (4, 8);

        // Create a singular matrix by making all rows equal
        let row: Tensor<c64> = Tensor::randn(&[1, m]);
        let mut data = Vec::with_capacity(n * m);
        for _ in 0..n {
            data.extend_from_slice(row.data());
        }
        let a = Tensor::from_vec(data, &[n, m]).unwrap();

        let result = qr_positive(&a, &[0], &[1]).unwrap();
        check_reconstruction_qr(&result, &a, eps);
        check_positive_diagonal_qr(&result);
    }

    #[test]
    fn test_ql_singular_c64() {
        let eps = f64::EPSILON * 1000.0;
        let (n, m) = (4, 8);

        // Create a singular matrix by making all rows equal
        let row: Tensor<c64> = Tensor::randn(&[1, m]);
        let mut data = Vec::with_capacity(n * m);
        for _ in 0..n {
            data.extend_from_slice(row.data());
        }
        let a = Tensor::from_vec(data, &[n, m]).unwrap();

        let result = ql_positive(&a, &[0], &[1]).unwrap();
        check_reconstruction_ql(&result, &a, eps);
        check_positive_diagonal_ql(&result);
    }
}
