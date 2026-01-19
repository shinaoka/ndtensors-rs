//! Diagonal tensor operations.

use crate::error::TensorError;
use crate::scalar::Scalar;
use crate::tensor::DenseTensor;

/// Extract the diagonal elements from a 2D tensor.
///
/// Returns a 1D tensor containing the diagonal elements (elements where row == col).
///
/// # Errors
///
/// Returns error if the tensor is not 2D.
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::operations::diag;
///
/// // Column-major: data = [1, 4, 7, 2, 5, 8, 3, 6, 9] for
/// // [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
/// let t = Tensor::from_vec(vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0], &[3, 3]).unwrap();
/// let d = diag(&t).unwrap();
/// assert_eq!(d.shape(), &[3]);
/// assert_eq!(d.data(), &[1.0, 5.0, 9.0]);
/// ```
pub fn diag<ElT: Scalar>(tensor: &DenseTensor<ElT>) -> Result<DenseTensor<ElT>, TensorError> {
    if tensor.ndim() != 2 {
        return Err(TensorError::RankMismatch {
            expected: 2,
            actual: tensor.ndim(),
        });
    }

    let shape = tensor.shape();
    let n = shape[0].min(shape[1]);
    let mut data = Vec::with_capacity(n);

    for i in 0..n {
        data.push(*tensor.get(&[i, i]).unwrap());
    }

    DenseTensor::from_vec(data, &[n])
}

/// Extract the diagonal elements from an N-dimensional tensor.
///
/// Returns a 1D tensor containing the elements where all indices are equal.
/// For a tensor with shape [d0, d1, ..., dn], returns elements at positions
/// [i, i, ..., i] for i in 0..min(d0, d1, ..., dn).
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::operations::diag_nd;
///
/// // 2x2x2 tensor
/// let mut t = Tensor::<f64>::zeros(&[2, 2, 2]);
/// t.set(&[0, 0, 0], 1.0).unwrap();
/// t.set(&[1, 1, 1], 8.0).unwrap();
/// t.set(&[0, 1, 0], 99.0).unwrap();  // off-diagonal
///
/// let d = diag_nd(&t);
/// assert_eq!(d.shape(), &[2]);
/// assert_eq!(d.data(), &[1.0, 8.0]);
/// ```
pub fn diag_nd<ElT: Scalar>(tensor: &DenseTensor<ElT>) -> DenseTensor<ElT> {
    let shape = tensor.shape();
    if shape.is_empty() {
        // Scalar tensor: return itself
        return DenseTensor::from_vec(tensor.data().to_vec(), &[1]).expect("diag_nd: scalar case");
    }

    let n = *shape.iter().min().unwrap();
    let ndim = tensor.ndim();
    let mut data = Vec::with_capacity(n);

    for i in 0..n {
        let indices: Vec<usize> = vec![i; ndim];
        data.push(*tensor.get(&indices).unwrap());
    }

    DenseTensor::from_vec(data, &[n]).expect("diag_nd: shape unchanged")
}

/// Create a diagonal 2D tensor from a 1D tensor.
///
/// Returns a square matrix with the input elements on the diagonal and zeros elsewhere.
///
/// # Errors
///
/// Returns error if the input is not 1D.
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::operations::diag_from_vec;
///
/// let d = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
/// let t = diag_from_vec(&d).unwrap();
/// assert_eq!(t.shape(), &[3, 3]);
/// assert_eq!(*t.get(&[0, 0]).unwrap(), 1.0);
/// assert_eq!(*t.get(&[1, 1]).unwrap(), 2.0);
/// assert_eq!(*t.get(&[2, 2]).unwrap(), 3.0);
/// assert_eq!(*t.get(&[0, 1]).unwrap(), 0.0);
/// ```
pub fn diag_from_vec<ElT: Scalar>(
    vector: &DenseTensor<ElT>,
) -> Result<DenseTensor<ElT>, TensorError> {
    if vector.ndim() != 1 {
        return Err(TensorError::RankMismatch {
            expected: 1,
            actual: vector.ndim(),
        });
    }

    let n = vector.shape()[0];
    let mut result = DenseTensor::<ElT>::zeros(&[n, n]);

    for i in 0..n {
        result.set(&[i, i], *vector.get(&[i]).unwrap())?;
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::c64;

    #[test]
    fn test_diag_square() {
        // Column-major 3x3 matrix
        let t = DenseTensor::from_vec(vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0], &[3, 3])
            .unwrap();
        let d = diag(&t).unwrap();
        assert_eq!(d.shape(), &[3]);
        assert_eq!(d.data(), &[1.0, 5.0, 9.0]);
    }

    #[test]
    fn test_diag_rectangular() {
        // 2x3 matrix (column-major)
        let t = DenseTensor::from_vec(vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], &[2, 3]).unwrap();
        let d = diag(&t).unwrap();
        assert_eq!(d.shape(), &[2]); // min(2, 3) = 2
        assert_eq!(d.data(), &[1.0, 5.0]);
    }

    #[test]
    fn test_diag_not_2d() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let result = diag(&t);
        assert!(result.is_err());
    }

    #[test]
    fn test_diag_from_vec() {
        let v = DenseTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let t = diag_from_vec(&v).unwrap();
        assert_eq!(t.shape(), &[3, 3]);
        assert_eq!(*t.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(*t.get(&[1, 1]).unwrap(), 2.0);
        assert_eq!(*t.get(&[2, 2]).unwrap(), 3.0);
        assert_eq!(*t.get(&[0, 1]).unwrap(), 0.0);
        assert_eq!(*t.get(&[1, 0]).unwrap(), 0.0);
    }

    #[test]
    fn test_diag_from_vec_not_1d() {
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let result = diag_from_vec(&t);
        assert!(result.is_err());
    }

    #[test]
    fn test_diag_c64() {
        let t = DenseTensor::from_vec(
            vec![
                c64::new(1.0, 1.0),
                c64::new(4.0, 4.0),
                c64::new(2.0, 2.0),
                c64::new(5.0, 5.0),
            ],
            &[2, 2],
        )
        .unwrap();
        let d = diag(&t).unwrap();
        assert_eq!(d.shape(), &[2]);
        assert_eq!(d.get(&[0]).unwrap().re, 1.0);
        assert_eq!(d.get(&[1]).unwrap().re, 5.0);
    }

    #[test]
    fn test_diag_nd_3d() {
        // 2x2x2 tensor
        let mut t = DenseTensor::<f64>::zeros(&[2, 2, 2]);
        t.set(&[0, 0, 0], 1.0).unwrap();
        t.set(&[1, 1, 1], 8.0).unwrap();
        t.set(&[0, 1, 0], 99.0).unwrap(); // off-diagonal

        let d = diag_nd(&t);
        assert_eq!(d.shape(), &[2]);
        assert_eq!(d.data(), &[1.0, 8.0]);
    }

    #[test]
    fn test_diag_nd_3d_non_cube() {
        // 2x3x4 tensor - min dim is 2
        let mut t = DenseTensor::<f64>::zeros(&[2, 3, 4]);
        t.set(&[0, 0, 0], 1.0).unwrap();
        t.set(&[1, 1, 1], 2.0).unwrap();

        let d = diag_nd(&t);
        assert_eq!(d.shape(), &[2]);
        assert_eq!(d.data(), &[1.0, 2.0]);
    }

    #[test]
    fn test_diag_nd_2d() {
        // 2D case should work the same as diag
        let t = DenseTensor::from_vec(vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0], &[3, 3])
            .unwrap();
        let d = diag_nd(&t);
        assert_eq!(d.shape(), &[3]);
        assert_eq!(d.data(), &[1.0, 5.0, 9.0]);
    }

    #[test]
    fn test_diag_nd_1d() {
        // 1D tensor: all indices are equal, so return all elements
        let t = DenseTensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let d = diag_nd(&t);
        assert_eq!(d.shape(), &[3]);
        assert_eq!(d.data(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_diag_nd_c64() {
        let mut t = DenseTensor::<c64>::zeros(&[2, 2, 2]);
        t.set(&[0, 0, 0], c64::new(1.0, 2.0)).unwrap();
        t.set(&[1, 1, 1], c64::new(3.0, 4.0)).unwrap();

        let d = diag_nd(&t);
        assert_eq!(d.shape(), &[2]);
        assert_eq!(d.get(&[0]).unwrap().re, 1.0);
        assert_eq!(d.get(&[0]).unwrap().im, 2.0);
        assert_eq!(d.get(&[1]).unwrap().re, 3.0);
        assert_eq!(d.get(&[1]).unwrap().im, 4.0);
    }
}
