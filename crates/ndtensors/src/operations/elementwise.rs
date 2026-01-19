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
}
