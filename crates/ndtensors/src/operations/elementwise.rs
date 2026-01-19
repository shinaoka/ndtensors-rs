//! Element-wise tensor operations.

use crate::error::TensorError;
use crate::scalar::Scalar;
use crate::tensor::DenseTensor;

/// Return a new tensor with element-wise complex conjugation.
///
/// For real tensors, this is effectively a copy (conjugate is identity).
/// For complex tensors, each element z becomes conj(z).
///
/// # Example
///
/// ```
/// use ndtensors::{Tensor, c64};
/// use ndtensors::operations::conj;
///
/// let t = Tensor::from_vec(vec![c64::new(1.0, 2.0), c64::new(3.0, -4.0)], &[2]).unwrap();
/// let tc = conj(&t);
/// assert_eq!(tc.get(&[0]).unwrap().im, -2.0);
/// assert_eq!(tc.get(&[1]).unwrap().im, 4.0);
/// ```
pub fn conj<ElT: Scalar>(tensor: &DenseTensor<ElT>) -> DenseTensor<ElT> {
    let data: Vec<ElT> = tensor.data().iter().map(|x| x.conjugate()).collect();
    DenseTensor::from_vec(data, tensor.shape()).expect("conj: shape unchanged")
}

/// Complex conjugate in-place.
///
/// # Example
///
/// ```
/// use ndtensors::{Tensor, c64};
/// use ndtensors::operations::conj_inplace;
///
/// let mut t = Tensor::from_vec(vec![c64::new(1.0, 2.0)], &[1]).unwrap();
/// conj_inplace(&mut t);
/// assert_eq!(t.get(&[0]).unwrap().im, -2.0);
/// ```
pub fn conj_inplace<ElT: Scalar>(tensor: &mut DenseTensor<ElT>) {
    for x in tensor.data_mut() {
        *x = x.conjugate();
    }
}

/// Extract the real part of each element.
///
/// Returns a tensor of the same shape with real-valued elements.
/// For real input, this is effectively a copy.
///
/// # Example
///
/// ```
/// use ndtensors::{Tensor, c64};
/// use ndtensors::operations::real;
///
/// let t = Tensor::from_vec(vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)], &[2]).unwrap();
/// let tr = real(&t);
/// assert_eq!(tr.shape(), &[2]);
/// assert_eq!(*tr.get(&[0]).unwrap(), 1.0);
/// assert_eq!(*tr.get(&[1]).unwrap(), 3.0);
/// ```
pub fn real<ElT: Scalar>(tensor: &DenseTensor<ElT>) -> DenseTensor<<ElT as Scalar>::Real> {
    let data: Vec<<ElT as Scalar>::Real> = tensor.data().iter().map(|x| x.real_part()).collect();
    DenseTensor::from_vec(data, tensor.shape()).expect("real: shape unchanged")
}

/// Extract the imaginary part of each element.
///
/// Returns a tensor of the same shape with real-valued elements.
/// For real input, returns a zero tensor.
///
/// # Example
///
/// ```
/// use ndtensors::{Tensor, c64};
/// use ndtensors::operations::imag;
///
/// let t = Tensor::from_vec(vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)], &[2]).unwrap();
/// let ti = imag(&t);
/// assert_eq!(ti.shape(), &[2]);
/// assert_eq!(*ti.get(&[0]).unwrap(), 2.0);
/// assert_eq!(*ti.get(&[1]).unwrap(), 4.0);
/// ```
pub fn imag<ElT: Scalar>(tensor: &DenseTensor<ElT>) -> DenseTensor<<ElT as Scalar>::Real> {
    let data: Vec<<ElT as Scalar>::Real> = tensor.data().iter().map(|x| x.imag_part()).collect();
    DenseTensor::from_vec(data, tensor.shape()).expect("imag: shape unchanged")
}

/// Multiply all elements by a scalar, returning a new tensor.
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::operations::scale;
///
/// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
/// let ts = scale(&t, 2.0);
/// assert_eq!(ts.data(), &[2.0, 4.0, 6.0]);
/// ```
pub fn scale<ElT: Scalar + std::ops::Mul<Output = ElT>>(
    tensor: &DenseTensor<ElT>,
    alpha: ElT,
) -> DenseTensor<ElT> {
    let data: Vec<ElT> = tensor.data().iter().map(|&x| x * alpha).collect();
    DenseTensor::from_vec(data, tensor.shape()).expect("scale: shape unchanged")
}

/// Scale tensor in-place.
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::operations::scale_inplace;
///
/// let mut t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
/// scale_inplace(&mut t, 2.0);
/// assert_eq!(t.data(), &[2.0, 4.0, 6.0]);
/// ```
pub fn scale_inplace<ElT: Scalar + std::ops::Mul<Output = ElT>>(
    tensor: &mut DenseTensor<ElT>,
    alpha: ElT,
) {
    for x in tensor.data_mut() {
        *x = *x * alpha;
    }
}

/// Apply a function to each element, returning a new tensor.
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::operations::apply;
///
/// let t = Tensor::from_vec(vec![1.0, 4.0, 9.0], &[3]).unwrap();
/// let ts = apply(&t, |x| x.sqrt());
/// assert!((ts.data()[0] - 1.0).abs() < 1e-10);
/// assert!((ts.data()[1] - 2.0).abs() < 1e-10);
/// assert!((ts.data()[2] - 3.0).abs() < 1e-10);
/// ```
pub fn apply<ElT: Scalar, F>(tensor: &DenseTensor<ElT>, f: F) -> DenseTensor<ElT>
where
    F: Fn(ElT) -> ElT,
{
    let data: Vec<ElT> = tensor.data().iter().map(|&x| f(x)).collect();
    DenseTensor::from_vec(data, tensor.shape()).expect("apply: shape unchanged")
}

/// Apply a function to each element in-place.
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::operations::apply_inplace;
///
/// let mut t = Tensor::from_vec(vec![1.0, 4.0, 9.0], &[3]).unwrap();
/// apply_inplace(&mut t, |x| x.sqrt());
/// assert!((t.data()[0] - 1.0).abs() < 1e-10);
/// ```
pub fn apply_inplace<ElT: Scalar, F>(tensor: &mut DenseTensor<ElT>, f: F)
where
    F: Fn(ElT) -> ElT,
{
    for x in tensor.data_mut() {
        *x = f(*x);
    }
}

/// Apply a binary function combining two tensors element-wise.
///
/// Both tensors must have the same shape.
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::operations::apply_binary;
///
/// let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
/// let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
/// let c = apply_binary(&a, &b, |x, y| x + y).unwrap();
/// assert_eq!(c.data(), &[5.0, 7.0, 9.0]);
/// ```
pub fn apply_binary<ElT: Scalar, F>(
    a: &DenseTensor<ElT>,
    b: &DenseTensor<ElT>,
    f: F,
) -> Result<DenseTensor<ElT>, TensorError>
where
    F: Fn(ElT, ElT) -> ElT,
{
    if a.shape() != b.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    let data: Vec<ElT> = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(&x, &y)| f(x, y))
        .collect();
    Ok(DenseTensor::from_vec(data, a.shape()).expect("apply_binary: shape unchanged"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c64;
    use approx::assert_relative_eq;

    #[test]
    fn test_conj_f64() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let tc = conj(&t);
        assert_eq!(tc.data(), t.data());
    }

    #[test]
    fn test_conj_c64() {
        let t = DenseTensor::from_vec(vec![c64::new(1.0, 2.0), c64::new(3.0, -4.0)], &[2]).unwrap();
        let tc = conj(&t);
        assert_eq!(tc.get(&[0]).unwrap().re, 1.0);
        assert_eq!(tc.get(&[0]).unwrap().im, -2.0);
        assert_eq!(tc.get(&[1]).unwrap().re, 3.0);
        assert_eq!(tc.get(&[1]).unwrap().im, 4.0);
    }

    #[test]
    fn test_conj_inplace_c64() {
        let mut t = DenseTensor::from_vec(vec![c64::new(1.0, 2.0)], &[1]).unwrap();
        conj_inplace(&mut t);
        assert_eq!(t.get(&[0]).unwrap().im, -2.0);
    }

    #[test]
    fn test_real_f64() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let tr = real(&t);
        assert_eq!(tr.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_real_c64() {
        let t = DenseTensor::from_vec(vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)], &[2]).unwrap();
        let tr = real(&t);
        assert_eq!(tr.data(), &[1.0, 3.0]);
    }

    #[test]
    fn test_imag_f64() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let ti = imag(&t);
        assert_eq!(ti.data(), &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_imag_c64() {
        let t = DenseTensor::from_vec(vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)], &[2]).unwrap();
        let ti = imag(&t);
        assert_eq!(ti.data(), &[2.0, 4.0]);
    }

    #[test]
    fn test_scale() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let ts = scale(&t, 2.0);
        assert_eq!(ts.data(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_scale_inplace() {
        let mut t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        scale_inplace(&mut t, 0.5);
        assert_eq!(t.data(), &[0.5, 1.0, 1.5]);
    }

    #[test]
    fn test_apply() {
        let t = DenseTensor::from_vec(vec![1.0, 4.0, 9.0], &[3]).unwrap();
        let ts = apply(&t, |x| x.sqrt());
        assert_relative_eq!(ts.data()[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(ts.data()[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(ts.data()[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_inplace() {
        let mut t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        apply_inplace(&mut t, |x| x * x);
        assert_eq!(t.data(), &[1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_apply_binary() {
        let a = DenseTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = DenseTensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        let c = apply_binary(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(c.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_apply_binary_shape_mismatch() {
        let a = DenseTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = DenseTensor::from_vec(vec![4.0, 5.0], &[2]).unwrap();
        let result = apply_binary(&a, &b, |x, y| x + y);
        assert!(result.is_err());
    }

    // =========================================================
    // Issue #36: Expanded complex tensor operation tests
    // =========================================================

    /// Test complex tensor operations match NDTensors.jl test_dense.jl patterns:
    /// - Create random complex tensor
    /// - Verify real(), imag(), conj() element-wise

    #[test]
    fn test_complex_real_imag_conj_3d() {
        // Match test_dense.jl pattern: 2x3x4 complex tensor
        let d1 = 2;
        let d2 = 3;
        let d3 = 4;
        let size = d1 * d2 * d3;

        // Create complex tensor with known values
        let data: Vec<c64> = (0..size)
            .map(|i| c64::new(i as f64, -(i as f64 * 0.5)))
            .collect();
        let t = DenseTensor::from_vec(data.clone(), &[d1, d2, d3]).unwrap();

        let rt = real(&t);
        let it = imag(&t);
        let ct = conj(&t);

        // Verify element-wise (mirrors Julia test pattern)
        for n1 in 0..d1 {
            for n2 in 0..d2 {
                for n3 in 0..d3 {
                    let idx = &[n1, n2, n3];
                    let orig = t.get(idx).unwrap();

                    assert_relative_eq!(*rt.get(idx).unwrap(), orig.re, epsilon = 1e-10);
                    assert_relative_eq!(*it.get(idx).unwrap(), orig.im, epsilon = 1e-10);
                    assert_relative_eq!(ct.get(idx).unwrap().re, orig.re, epsilon = 1e-10);
                    assert_relative_eq!(ct.get(idx).unwrap().im, -orig.im, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_complex_conj_preserves_type() {
        let t = DenseTensor::from_vec(vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)], &[2]).unwrap();
        let ct = conj(&t);

        // conj should preserve shape
        assert_eq!(ct.shape(), t.shape());
        assert_eq!(ct.len(), t.len());
    }

    #[test]
    fn test_real_imag_output_is_real() {
        let t = DenseTensor::from_vec(vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)], &[2]).unwrap();
        let rt = real(&t);
        let it = imag(&t);

        // Output type is f64 (the Real type of c64)
        // This is verified by the fact that we can compare with f64 values
        assert_eq!(*rt.get(&[0]).unwrap(), 1.0_f64);
        assert_eq!(*it.get(&[0]).unwrap(), 2.0_f64);
    }

    #[test]
    fn test_complex_conj_inplace_preserves_structure() {
        let mut t =
            DenseTensor::from_vec(vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)], &[2]).unwrap();
        let original_shape = t.shape().to_vec();
        conj_inplace(&mut t);
        assert_eq!(t.shape(), &original_shape);
    }

    #[test]
    fn test_double_conj_is_identity() {
        let t = DenseTensor::from_vec(vec![c64::new(1.0, 2.0), c64::new(3.0, -4.0)], &[2]).unwrap();
        let cc = conj(&conj(&t));

        for i in 0..t.len() {
            assert_relative_eq!(
                cc.get(&[i]).unwrap().re,
                t.get(&[i]).unwrap().re,
                epsilon = 1e-10
            );
            assert_relative_eq!(
                cc.get(&[i]).unwrap().im,
                t.get(&[i]).unwrap().im,
                epsilon = 1e-10
            );
        }
    }

    #[test]
    fn test_complex_scale() {
        let t = DenseTensor::from_vec(vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)], &[2]).unwrap();

        // Scale by real
        let scaled_real = scale(&t, c64::new(2.0, 0.0));
        assert_relative_eq!(scaled_real.get(&[0]).unwrap().re, 2.0, epsilon = 1e-10);
        assert_relative_eq!(scaled_real.get(&[0]).unwrap().im, 4.0, epsilon = 1e-10);

        // Scale by complex (multiply by i)
        let scaled_imag = scale(&t, c64::new(0.0, 1.0));
        // (1+2i) * i = i + 2i^2 = -2 + i
        assert_relative_eq!(scaled_imag.get(&[0]).unwrap().re, -2.0, epsilon = 1e-10);
        assert_relative_eq!(scaled_imag.get(&[0]).unwrap().im, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_complex_addition() {
        let a = DenseTensor::from_vec(vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)], &[2]).unwrap();
        let b = DenseTensor::from_vec(vec![c64::new(5.0, 6.0), c64::new(7.0, 8.0)], &[2]).unwrap();

        let c = apply_binary(&a, &b, |x, y| x + y).unwrap();

        assert_relative_eq!(c.get(&[0]).unwrap().re, 6.0, epsilon = 1e-10);
        assert_relative_eq!(c.get(&[0]).unwrap().im, 8.0, epsilon = 1e-10);
        assert_relative_eq!(c.get(&[1]).unwrap().re, 10.0, epsilon = 1e-10);
        assert_relative_eq!(c.get(&[1]).unwrap().im, 12.0, epsilon = 1e-10);
    }

    #[test]
    fn test_complex_subtraction() {
        let a = DenseTensor::from_vec(vec![c64::new(5.0, 6.0), c64::new(7.0, 8.0)], &[2]).unwrap();
        let b = DenseTensor::from_vec(vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)], &[2]).unwrap();

        let c = apply_binary(&a, &b, |x, y| x - y).unwrap();

        assert_relative_eq!(c.get(&[0]).unwrap().re, 4.0, epsilon = 1e-10);
        assert_relative_eq!(c.get(&[0]).unwrap().im, 4.0, epsilon = 1e-10);
        assert_relative_eq!(c.get(&[1]).unwrap().re, 4.0, epsilon = 1e-10);
        assert_relative_eq!(c.get(&[1]).unwrap().im, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_complex_multiplication() {
        let a = DenseTensor::from_vec(vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)], &[2]).unwrap();
        let b = DenseTensor::from_vec(vec![c64::new(5.0, 6.0), c64::new(7.0, 8.0)], &[2]).unwrap();

        let c = apply_binary(&a, &b, |x, y| x * y).unwrap();

        // (1+2i)*(5+6i) = 5 + 6i + 10i + 12i^2 = 5 + 16i - 12 = -7 + 16i
        assert_relative_eq!(c.get(&[0]).unwrap().re, -7.0, epsilon = 1e-10);
        assert_relative_eq!(c.get(&[0]).unwrap().im, 16.0, epsilon = 1e-10);

        // (3+4i)*(7+8i) = 21 + 24i + 28i + 32i^2 = 21 + 52i - 32 = -11 + 52i
        assert_relative_eq!(c.get(&[1]).unwrap().re, -11.0, epsilon = 1e-10);
        assert_relative_eq!(c.get(&[1]).unwrap().im, 52.0, epsilon = 1e-10);
    }

    #[test]
    fn test_apply_complex() {
        let t = DenseTensor::from_vec(vec![c64::new(4.0, 0.0), c64::new(9.0, 0.0)], &[2]).unwrap();

        let sqrt_t = apply(&t, |x| c64::new(x.re.sqrt(), x.im));

        assert_relative_eq!(sqrt_t.get(&[0]).unwrap().re, 2.0, epsilon = 1e-10);
        assert_relative_eq!(sqrt_t.get(&[1]).unwrap().re, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_norm_relation_real_imag() {
        use crate::operations::norm;

        // Test: norm(rT)^2 + norm(iT)^2 ≈ norm(T)^2
        // This is a property from NDTensors.jl test_blocksparse.jl
        let t = DenseTensor::from_vec(
            vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0), c64::new(5.0, 6.0)],
            &[3],
        )
        .unwrap();

        let rt = real(&t);
        let it = imag(&t);

        let norm_t = norm(&t);
        let norm_rt = norm(&rt);
        let norm_it = norm(&it);

        assert_relative_eq!(
            norm_rt * norm_rt + norm_it * norm_it,
            norm_t * norm_t,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_conj_real_tensor_is_identity() {
        // For real tensors, conj should be identity
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let ct = conj(&t);

        for i in 0..t.len() {
            assert_relative_eq!(t.data()[i], ct.data()[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_scale_commutativity() {
        // Test: data(A * 2.0) == data(2.0 * A)
        // This mirrors test_dense.jl line 105
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        let left = scale(&t, 2.0);
        // There's no right_scale, but mathematically a * scalar = scalar * a
        // so just verify scaling works correctly
        assert_eq!(left.data(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_scale_norm_relation() {
        // Test: norm(2 * T) / norm(T) ≈ 2
        // This mirrors test_dense.jl lines 206-210
        use crate::operations::norm;

        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let t2 = scale(&t, 2.0);

        let ratio = norm(&t2) / norm(&t);
        assert_relative_eq!(ratio, 2.0, epsilon = 1e-10);
    }
}
