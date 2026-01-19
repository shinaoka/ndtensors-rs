//! Zero-copy conversion between Tensor and faer matrices.
//!
//! This module provides traits and functions for converting between
//! ndtensors' `DenseTensor` and faer's matrix types (`MatRef`, `MatMut`, `Mat`).
//!
//! # Memory Layout
//!
//! Both ndtensors and faer use column-major (Fortran-style) storage order,
//! enabling zero-copy conversions for 2D tensors.
//!
//! # CPU-Only
//!
//! These conversions only work with CPU-backed tensors (`CpuBuffer`).
//! Future GPU backends will require separate implementations.

use faer::{Mat, MatMut, MatRef};

use crate::error::TensorError;
use crate::scalar::Scalar;
use crate::storage::{CpuBuffer, Dense};
use crate::tensor::Tensor;

/// Type alias for CPU-backed dense tensor.
pub type CpuDenseTensor<ElT> = Tensor<ElT, Dense<ElT, CpuBuffer<ElT>>>;

/// Extension trait for converting DenseTensor to faer matrix views.
///
/// This trait is implemented for CPU-backed dense tensors and provides
/// zero-copy conversion to faer's matrix reference types.
pub trait AsFaerMat<T: Scalar> {
    /// View tensor data as an immutable faer matrix (zero-copy).
    ///
    /// The tensor data is interpreted as a column-major matrix with the
    /// specified dimensions.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows in the matrix view
    /// * `cols` - Number of columns in the matrix view
    ///
    /// # Panics
    ///
    /// Panics if `rows * cols != tensor.len()`.
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::Tensor;
    /// use ndtensors::backend::AsFaerMat;
    ///
    /// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    /// let mat = t.as_faer_mat(2, 3);
    /// assert_eq!(mat.nrows(), 2);
    /// assert_eq!(mat.ncols(), 3);
    /// ```
    fn as_faer_mat(&self, rows: usize, cols: usize) -> MatRef<'_, T>;

    /// View tensor data as a mutable faer matrix (zero-copy).
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows in the matrix view
    /// * `cols` - Number of columns in the matrix view
    ///
    /// # Panics
    ///
    /// Panics if `rows * cols != tensor.len()`.
    fn as_faer_mat_mut(&mut self, rows: usize, cols: usize) -> MatMut<'_, T>;
}

impl<T: Scalar> AsFaerMat<T> for CpuDenseTensor<T> {
    fn as_faer_mat(&self, rows: usize, cols: usize) -> MatRef<'_, T> {
        assert_eq!(
            rows * cols,
            self.len(),
            "Matrix dimensions ({} x {} = {}) must match tensor size ({})",
            rows,
            cols,
            rows * cols,
            self.len()
        );
        // faer uses column-major order by default, same as ndtensors
        MatRef::from_column_major_slice(self.data(), rows, cols)
    }

    fn as_faer_mat_mut(&mut self, rows: usize, cols: usize) -> MatMut<'_, T> {
        assert_eq!(
            rows * cols,
            self.len(),
            "Matrix dimensions ({} x {} = {}) must match tensor size ({})",
            rows,
            cols,
            rows * cols,
            self.len()
        );
        MatMut::from_column_major_slice_mut(self.data_mut(), rows, cols)
    }
}

/// Create a DenseTensor from a faer matrix (copies data).
///
/// This function copies the matrix data into a new tensor with shape `[rows, cols]`.
///
/// # Example
///
/// ```
/// use faer::Mat;
/// use ndtensors::backend::tensor_from_faer_mat;
///
/// let mat = Mat::from_fn(2, 3, |i, j| (i * 3 + j) as f64);
/// let tensor = tensor_from_faer_mat(mat.as_ref());
/// assert_eq!(tensor.shape(), &[2, 3]);
/// ```
pub fn tensor_from_faer_mat<T: Scalar>(mat: MatRef<'_, T>) -> CpuDenseTensor<T> {
    let rows = mat.nrows();
    let cols = mat.ncols();

    // Copy data in column-major order
    let mut data = Vec::with_capacity(rows * cols);
    for j in 0..cols {
        for i in 0..rows {
            data.push(mat[(i, j)]);
        }
    }

    Tensor::from_vec(data, &[rows, cols]).expect("Shape should match data length")
}

/// Create an owned faer Mat from a DenseTensor (copies data).
///
/// # Example
///
/// ```
/// use ndtensors::Tensor;
/// use ndtensors::backend::faer_mat_from_tensor;
///
/// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
/// let mat = faer_mat_from_tensor(&t, 2, 3);
/// assert_eq!(mat.nrows(), 2);
/// assert_eq!(mat.ncols(), 3);
/// ```
pub fn faer_mat_from_tensor<T: Scalar>(
    tensor: &CpuDenseTensor<T>,
    rows: usize,
    cols: usize,
) -> Mat<T> {
    let mat_ref = tensor.as_faer_mat(rows, cols);
    mat_ref.to_owned()
}

/// Reshape a tensor to 2D and return as faer matrix view.
///
/// For a tensor with shape `[d0, d1, ..., dn]`, this creates a matrix view
/// where the first `split` dimensions are merged into rows and the remaining
/// dimensions are merged into columns.
///
/// # Arguments
///
/// * `tensor` - The input tensor
/// * `split` - The index at which to split dimensions (rows = dims[0..split], cols = dims[split..])
///
/// # Returns
///
/// A tuple of (MatRef, rows, cols) where rows and cols are the computed dimensions.
///
/// # Errors
///
/// Returns error if split is out of bounds.
pub fn reshape_to_matrix<T: Scalar>(
    tensor: &CpuDenseTensor<T>,
    split: usize,
) -> Result<(MatRef<'_, T>, usize, usize), TensorError> {
    let shape = tensor.shape();

    if split > shape.len() {
        return Err(TensorError::InvalidPermutation {
            perm: vec![split],
            ndim: shape.len(),
        });
    }

    let rows: usize = shape[..split].iter().product::<usize>().max(1);
    let cols: usize = shape[split..].iter().product::<usize>().max(1);

    Ok((tensor.as_faer_mat(rows, cols), rows, cols))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_as_faer_mat() {
        let t: CpuDenseTensor<f64> =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let mat = t.as_faer_mat(2, 3);
        assert_eq!(mat.nrows(), 2);
        assert_eq!(mat.ncols(), 3);

        // Check column-major order
        // t[0,0]=1, t[1,0]=2, t[0,1]=3, t[1,1]=4, t[0,2]=5, t[1,2]=6
        assert_relative_eq!(mat[(0, 0)], 1.0);
        assert_relative_eq!(mat[(1, 0)], 2.0);
        assert_relative_eq!(mat[(0, 1)], 3.0);
        assert_relative_eq!(mat[(1, 1)], 4.0);
        assert_relative_eq!(mat[(0, 2)], 5.0);
        assert_relative_eq!(mat[(1, 2)], 6.0);
    }

    #[test]
    fn test_as_faer_mat_mut() {
        let mut t: CpuDenseTensor<f64> = Tensor::zeros(&[2, 3]);

        {
            let mut mat = t.as_faer_mat_mut(2, 3);
            mat[(0, 0)] = 1.0;
            mat[(1, 1)] = 5.0;
        }

        assert_relative_eq!(*t.get(&[0, 0]).unwrap(), 1.0);
        assert_relative_eq!(*t.get(&[1, 1]).unwrap(), 5.0);
    }

    #[test]
    fn test_tensor_from_faer_mat() {
        let mat = Mat::from_fn(2, 3, |i, j| (i * 3 + j) as f64);
        let tensor = tensor_from_faer_mat(mat.as_ref());

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_relative_eq!(*tensor.get(&[0, 0]).unwrap(), 0.0);
        assert_relative_eq!(*tensor.get(&[1, 0]).unwrap(), 3.0);
        assert_relative_eq!(*tensor.get(&[0, 1]).unwrap(), 1.0);
        assert_relative_eq!(*tensor.get(&[1, 2]).unwrap(), 5.0);
    }

    #[test]
    fn test_faer_mat_from_tensor() {
        let t: CpuDenseTensor<f64> =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let mat = faer_mat_from_tensor(&t, 2, 3);
        assert_eq!(mat.nrows(), 2);
        assert_eq!(mat.ncols(), 3);
        assert_relative_eq!(mat[(0, 0)], 1.0);
        assert_relative_eq!(mat[(1, 2)], 6.0);
    }

    #[test]
    fn test_reshape_to_matrix() {
        // 2x3x4 tensor -> 6x4 matrix (split at 2)
        let t: CpuDenseTensor<f64> = Tensor::zeros(&[2, 3, 4]);
        let (mat, rows, cols) = reshape_to_matrix(&t, 2).unwrap();

        assert_eq!(rows, 6); // 2 * 3
        assert_eq!(cols, 4);
        assert_eq!(mat.nrows(), 6);
        assert_eq!(mat.ncols(), 4);
    }

    #[test]
    fn test_reshape_to_matrix_split_0() {
        // Split at 0: 1 row, all cols
        let t: CpuDenseTensor<f64> = Tensor::zeros(&[2, 3]);
        let (mat, rows, cols) = reshape_to_matrix(&t, 0).unwrap();

        assert_eq!(rows, 1);
        assert_eq!(cols, 6);
        assert_eq!(mat.nrows(), 1);
        assert_eq!(mat.ncols(), 6);
    }

    #[test]
    fn test_reshape_to_matrix_split_end() {
        // Split at end: all rows, 1 col
        let t: CpuDenseTensor<f64> = Tensor::zeros(&[2, 3]);
        let (mat, rows, cols) = reshape_to_matrix(&t, 2).unwrap();

        assert_eq!(rows, 6);
        assert_eq!(cols, 1);
        assert_eq!(mat.nrows(), 6);
        assert_eq!(mat.ncols(), 1);
    }

    #[test]
    #[should_panic(expected = "Matrix dimensions")]
    fn test_as_faer_mat_dimension_mismatch() {
        let t: CpuDenseTensor<f64> = Tensor::zeros(&[2, 3]);
        let _ = t.as_faer_mat(3, 3); // 9 != 6
    }

    #[test]
    fn test_zero_copy_verification() {
        let t: CpuDenseTensor<f64> =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let mat = t.as_faer_mat(2, 3);

        // Verify that the faer matrix uses the same memory
        // by checking that the data pointer matches
        let tensor_ptr = t.data().as_ptr();
        let mat_ptr = mat.as_ptr();

        assert_eq!(
            tensor_ptr, mat_ptr,
            "faer matrix should share memory with tensor"
        );
    }
}
