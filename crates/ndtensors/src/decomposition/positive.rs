//! Positive QR and QL decompositions.
//!
//! This module provides QR and QL decompositions where the diagonal elements
//! of R or L are guaranteed to be non-negative and real. This makes the
//! decomposition unique.

use crate::decomposition::ql::{QlResult, ql};
use crate::decomposition::qr::{QrResult, qr};
use crate::error::TensorError;
use crate::scalar::{RealScalar, Scalar};
use crate::tensor::DenseTensor;

/// Compute the QR decomposition with non-negative diagonal of R.
///
/// This computes the thin QR decomposition and then adjusts the signs
/// of the columns of Q and rows of R so that the diagonal elements of R
/// are non-negative (and real for complex types).
///
/// This makes the QR decomposition unique.
///
/// # Arguments
///
/// * `tensor` - The input tensor
/// * `left_inds` - Indices to place on the left (become Q's row dimensions)
/// * `right_inds` - Indices to place on the right (become R's column dimensions)
///
/// # Returns
///
/// `QrResult` with non-negative diagonal in R.
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::decomposition::qr_positive;
///
/// let t = Tensor::<f64>::from_vec(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0], &[2, 3]).unwrap();
/// let result = qr_positive(&t, &[0], &[1]).unwrap();
///
/// // Check that diagonal of R is non-negative
/// let k = result.rank;
/// for i in 0..k {
///     let r_ii = *result.r.get(&[i, i]).unwrap();
///     assert!(r_ii >= 0.0, "R[{}, {}] = {} should be non-negative", i, i, r_ii);
/// }
/// ```
pub fn qr_positive<ElT: Scalar>(
    tensor: &DenseTensor<ElT>,
    left_inds: &[usize],
    right_inds: &[usize],
) -> Result<QrResult<ElT>, TensorError>
where
    <ElT as Scalar>::Real: RealScalar,
{
    let result = qr(tensor, left_inds, right_inds)?;
    Ok(make_qr_positive(result))
}

/// Make QR decomposition positive by adjusting signs.
fn make_qr_positive<ElT: Scalar>(mut result: QrResult<ElT>) -> QrResult<ElT>
where
    <ElT as Scalar>::Real: RealScalar,
{
    let k = result.rank;
    let q_shape = result.q.shape().to_vec();
    let r_shape = result.r.shape().to_vec();

    // Flatten dimensions for easier access
    let m: usize = q_shape[..q_shape.len() - 1].iter().product();
    let n: usize = r_shape[1..].iter().product();

    for j in 0..k {
        // Get diagonal element R[j, j]
        let r_jj = *result.r.get_linear(j + j * k).unwrap();

        // Compute sign (for complex: phase factor to make it real and non-negative)
        let (sign_q, sign_r) = compute_sign(r_jj);

        // If sign is already positive (or zero), no adjustment needed
        if is_positive_real(r_jj) {
            continue;
        }

        // Adjust column j of Q: Q[:, j] *= sign_q
        for i in 0..m {
            let idx = i + j * m;
            let q_ij = *result.q.get_linear(idx).unwrap();
            *result.q.get_linear_mut(idx).unwrap() = mul(q_ij, sign_q);
        }

        // Adjust row j of R: R[j, :] *= sign_r
        for l in 0..n {
            let idx = j + l * k;
            let r_jl = *result.r.get_linear(idx).unwrap();
            *result.r.get_linear_mut(idx).unwrap() = mul(r_jl, sign_r);
        }
    }

    result
}

/// Compute the QL decomposition with non-negative diagonal of L.
///
/// This computes the thin QL decomposition and then adjusts the signs
/// of the columns of Q and rows of L so that the diagonal elements of L
/// are non-negative (and real for complex types).
///
/// This makes the QL decomposition unique.
///
/// # Arguments
///
/// * `tensor` - The input tensor
/// * `left_inds` - Indices to place on the left (become Q's row dimensions)
/// * `right_inds` - Indices to place on the right (become L's column dimensions)
///
/// # Returns
///
/// `QlResult` with non-negative diagonal in L.
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::decomposition::ql_positive;
///
/// let t = Tensor::<f64>::from_vec(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0], &[2, 3]).unwrap();
/// let result = ql_positive(&t, &[0], &[1]).unwrap();
///
/// // Check that diagonal of L is non-negative
/// // For QL, diagonal is L[i, n - k + i] for i in 0..k
/// let k = result.rank;
/// let n = result.l.shape()[1];
/// for i in 0..k {
///     let l_ii = *result.l.get(&[i, n - k + i]).unwrap();
///     assert!(l_ii >= 0.0, "L[{}, {}] = {} should be non-negative", i, n - k + i, l_ii);
/// }
/// ```
pub fn ql_positive<ElT: Scalar>(
    tensor: &DenseTensor<ElT>,
    left_inds: &[usize],
    right_inds: &[usize],
) -> Result<QlResult<ElT>, TensorError>
where
    <ElT as Scalar>::Real: RealScalar,
{
    let result = ql(tensor, left_inds, right_inds)?;
    Ok(make_ql_positive(result))
}

/// Make QL decomposition positive by adjusting signs.
fn make_ql_positive<ElT: Scalar>(mut result: QlResult<ElT>) -> QlResult<ElT>
where
    <ElT as Scalar>::Real: RealScalar,
{
    let k = result.rank;
    let q_shape = result.q.shape().to_vec();
    let l_shape = result.l.shape().to_vec();

    // Flatten dimensions
    let m: usize = q_shape[..q_shape.len() - 1].iter().product();
    let n: usize = l_shape[1..].iter().product();

    // For QL, diagonal elements are at L[i, n - k + i] for i in 0..min(k, n)
    let diag_offset = n.saturating_sub(k);

    for i in 0..k.min(n) {
        let diag_col = diag_offset + i;
        if diag_col >= n {
            continue;
        }

        // Get diagonal element L[i, diag_col]
        let l_ii = *result.l.get_linear(i + diag_col * k).unwrap();

        // If sign is already positive (or zero), no adjustment needed
        if is_positive_real(l_ii) {
            continue;
        }

        // Compute sign
        let (sign_q, sign_l) = compute_sign(l_ii);

        // Adjust column i of Q: Q[:, i] *= sign_q
        for r in 0..m {
            let idx = r + i * m;
            let q_ri = *result.q.get_linear(idx).unwrap();
            *result.q.get_linear_mut(idx).unwrap() = mul(q_ri, sign_q);
        }

        // Adjust row i of L from column 0 to diag_col (inclusive): L[i, 0:diag_col+1] *= sign_l
        for c in 0..=diag_col {
            let idx = i + c * k;
            let l_ic = *result.l.get_linear(idx).unwrap();
            *result.l.get_linear_mut(idx).unwrap() = mul(l_ic, sign_l);
        }
    }

    result
}

/// Check if a scalar is positive real (non-negative real part, zero imaginary part).
fn is_positive_real<ElT: Scalar>(x: ElT) -> bool
where
    <ElT as Scalar>::Real: RealScalar,
{
    let re = x.real_part().to_f64();
    let im = x.imag_part().to_f64();

    // Check if imaginary part is essentially zero and real part is non-negative
    im.abs() < 1e-14 && re >= -1e-14
}

/// Compute the sign factors to make a diagonal element non-negative real.
/// Returns (sign_q, sign_r) such that:
/// - x * sign_r = |x| (real, non-negative)
/// - sign_q = conj(sign_r) to preserve Q*R product
fn compute_sign<ElT: Scalar>(x: ElT) -> (ElT, ElT)
where
    <ElT as Scalar>::Real: RealScalar,
{
    let re = x.real_part().to_f64();
    let im = x.imag_part().to_f64();
    let abs_x = (re * re + im * im).sqrt();

    if abs_x < 1e-14 {
        // Zero element, no adjustment needed
        return (ElT::one(), ElT::one());
    }

    use faer_traits::math_utils::{from_f64, mul as faer_mul};

    // unit_phase = x / |x|
    let inv_abs: ElT = from_f64(1.0 / abs_x);
    let unit_phase = faer_mul(&x, &inv_abs);

    // sign_r = conj(x) / |x| = conj(unit_phase)
    // so that x * sign_r = x * conj(x) / |x| = |x|² / |x| = |x|
    let sign_r = unit_phase.conjugate();

    // sign_q = conj(sign_r) = unit_phase
    // so that Q' * R' = (Q * sign_q) * (sign_r * R) = Q * R (signs cancel)
    let sign_q = unit_phase;

    (sign_q, sign_r)
}

/// Multiply two scalars.
fn mul<ElT: Scalar>(a: ElT, b: ElT) -> ElT {
    use faer_traits::math_utils::mul as faer_mul;
    faer_mul(&a, &b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::c64;

    fn check_reconstruction_qr<
        ElT: Scalar + std::ops::Add<Output = ElT> + std::ops::Mul<Output = ElT>,
    >(
        result: &QrResult<ElT>,
        original: &DenseTensor<ElT>,
        epsilon: f64,
    ) {
        use crate::contract::contract;

        // Contract Q with R
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
                "Reconstruction failed at index {}: orig={:?}, recon={:?}",
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
        original: &DenseTensor<ElT>,
        epsilon: f64,
    ) {
        use crate::contract::contract;

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
                "Reconstruction failed at index {}: orig={:?}, recon={:?}",
                i,
                orig,
                recon
            );
        }
    }

    #[test]
    fn test_qr_positive_f64() {
        let t = DenseTensor::from_vec(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0], &[2, 3]).unwrap();
        let result = qr_positive(&t, &[0], &[1]).unwrap();

        // Check diagonal of R is non-negative
        let k = result.rank;
        for i in 0..k {
            let r_ii = *result.r.get(&[i, i]).unwrap();
            assert!(
                r_ii >= -1e-14,
                "R[{}, {}] = {} should be non-negative",
                i,
                i,
                r_ii
            );
        }

        // Check reconstruction
        check_reconstruction_qr(&result, &t, 1e-10);
    }

    #[test]
    fn test_qr_positive_c64() {
        let t = DenseTensor::from_vec(
            vec![
                c64::new(1.0, -1.0),
                c64::new(-2.0, 2.0),
                c64::new(3.0, -3.0),
                c64::new(-4.0, 4.0),
                c64::new(5.0, -5.0),
                c64::new(-6.0, 6.0),
            ],
            &[2, 3],
        )
        .unwrap();
        let result = qr_positive(&t, &[0], &[1]).unwrap();

        // Check diagonal of R is non-negative real
        let k = result.rank;
        for i in 0..k {
            let r_ii = *result.r.get(&[i, i]).unwrap();
            assert!(
                r_ii.re >= -1e-14,
                "R[{}, {}].re = {} should be non-negative",
                i,
                i,
                r_ii.re
            );
            assert!(
                r_ii.im.abs() < 1e-10,
                "R[{}, {}].im = {} should be zero",
                i,
                i,
                r_ii.im
            );
        }

        // Check reconstruction
        check_reconstruction_qr(&result, &t, 1e-10);
    }

    #[test]
    fn test_ql_positive_f64() {
        let t = DenseTensor::from_vec(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0], &[2, 3]).unwrap();
        let result = ql_positive(&t, &[0], &[1]).unwrap();

        // Check diagonal of L is non-negative
        let k = result.rank;
        let n = result.l.shape()[1];
        for i in 0..k {
            let diag_col = n - k + i;
            let l_ii = *result.l.get(&[i, diag_col]).unwrap();
            assert!(
                l_ii >= -1e-14,
                "L[{}, {}] = {} should be non-negative",
                i,
                diag_col,
                l_ii
            );
        }

        // Check reconstruction
        check_reconstruction_ql(&result, &t, 1e-10);
    }

    #[test]
    fn test_ql_positive_c64() {
        let t = DenseTensor::from_vec(
            vec![
                c64::new(1.0, -1.0),
                c64::new(-2.0, 2.0),
                c64::new(3.0, -3.0),
                c64::new(-4.0, 4.0),
                c64::new(5.0, -5.0),
                c64::new(-6.0, 6.0),
            ],
            &[2, 3],
        )
        .unwrap();
        let result = ql_positive(&t, &[0], &[1]).unwrap();

        // Check diagonal of L is non-negative real
        let k = result.rank;
        let n = result.l.shape()[1];
        for i in 0..k {
            let diag_col = n - k + i;
            let l_ii = *result.l.get(&[i, diag_col]).unwrap();
            assert!(
                l_ii.re >= -1e-14,
                "L[{}, {}].re = {} should be non-negative",
                i,
                diag_col,
                l_ii.re
            );
            assert!(
                l_ii.im.abs() < 1e-10,
                "L[{}, {}].im = {} should be zero",
                i,
                diag_col,
                l_ii.im
            );
        }

        // Check reconstruction
        check_reconstruction_ql(&result, &t, 1e-10);
    }

    #[test]
    fn test_qr_positive_tall() {
        let t = DenseTensor::from_vec(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], &[4, 2])
            .unwrap();
        let result = qr_positive(&t, &[0], &[1]).unwrap();

        let k = result.rank;
        for i in 0..k {
            let r_ii = *result.r.get(&[i, i]).unwrap();
            assert!(r_ii >= -1e-14);
        }

        check_reconstruction_qr(&result, &t, 1e-10);
    }

    #[test]
    fn test_ql_positive_wide() {
        let t = DenseTensor::from_vec(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], &[2, 4])
            .unwrap();
        let result = ql_positive(&t, &[0], &[1]).unwrap();

        let k = result.rank;
        let n = result.l.shape()[1];
        for i in 0..k {
            let diag_col = n - k + i;
            let l_ii = *result.l.get(&[i, diag_col]).unwrap();
            assert!(l_ii >= -1e-14);
        }

        check_reconstruction_ql(&result, &t, 1e-10);
    }
}
