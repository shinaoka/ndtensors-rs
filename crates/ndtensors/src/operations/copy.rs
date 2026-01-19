//! Tensor copy operations.

use crate::error::TensorError;
use crate::scalar::Scalar;
use crate::tensor::DenseTensor;

/// Copy data from source tensor to destination tensor.
///
/// Both tensors must have the same shape.
///
/// # Errors
///
/// Returns error if shapes don't match.
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::operations::copy_into;
///
/// let src = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
/// let mut dst = Tensor::<f64>::zeros(&[3]);
/// copy_into(&mut dst, &src).unwrap();
/// assert_eq!(dst.data(), &[1.0, 2.0, 3.0]);
/// ```
pub fn copy_into<ElT: Scalar>(
    dest: &mut DenseTensor<ElT>,
    src: &DenseTensor<ElT>,
) -> Result<(), TensorError> {
    if dest.shape() != src.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: dest.len(),
            actual: src.len(),
        });
    }
    dest.data_mut().copy_from_slice(src.data());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c64;

    #[test]
    fn test_copy_into_f64() {
        let src = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let mut dst = DenseTensor::<f64>::zeros(&[2, 2]);
        copy_into(&mut dst, &src).unwrap();
        assert_eq!(dst.data(), src.data());
    }

    #[test]
    fn test_copy_into_c64() {
        let src =
            DenseTensor::from_vec(vec![c64::new(1.0, 2.0), c64::new(3.0, 4.0)], &[2]).unwrap();
        let mut dst = DenseTensor::<c64>::zeros(&[2]);
        copy_into(&mut dst, &src).unwrap();
        assert_eq!(dst.data(), src.data());
    }

    #[test]
    fn test_copy_into_shape_mismatch() {
        let src = DenseTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let mut dst = DenseTensor::<f64>::zeros(&[2]);
        let result = copy_into(&mut dst, &src);
        assert!(result.is_err());
    }
}
