//! Dense n-dimensional tensor type.

use crate::error::TensorError;
use crate::scalar::Scalar;
use crate::storage::Dense;
use crate::strides::{cartesian_to_linear, compute_strides};

/// A dense n-dimensional tensor with column-major storage.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<T: Scalar> {
    storage: Dense<T>,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<T: Scalar> Tensor<T> {
    /// Create a new tensor with the given shape, zero-initialized.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndtensors::Tensor;
    ///
    /// let t: Tensor<f64> = Tensor::zeros(&[2, 3, 4]);
    /// assert_eq!(t.shape(), &[2, 3, 4]);
    /// assert_eq!(t.len(), 24);
    /// ```
    pub fn zeros(shape: &[usize]) -> Self {
        let strides = compute_strides(shape);
        let len: usize = shape.iter().product();
        Self {
            storage: Dense::zeros(len.max(1)), // At least 1 for scalar (empty shape)
            shape: shape.to_vec(),
            strides,
        }
    }

    /// Create tensor from data and shape.
    ///
    /// Data is expected to be in column-major order.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::ShapeMismatch` if data length doesn't match shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndtensors::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    /// assert_eq!(t.shape(), &[2, 3]);
    /// assert_eq!(t.get(&[0, 0]), Some(&1.0));
    /// assert_eq!(t.get(&[1, 0]), Some(&2.0)); // Column-major: [1,0] is second element
    /// assert_eq!(t.get(&[0, 1]), Some(&3.0)); // [0,1] is third element
    /// ```
    pub fn from_vec(data: Vec<T>, shape: &[usize]) -> Result<Self, TensorError> {
        let expected_len: usize = shape.iter().product::<usize>().max(1);
        if data.len() != expected_len {
            return Err(TensorError::ShapeMismatch {
                expected: expected_len,
                actual: data.len(),
            });
        }
        let strides = compute_strides(shape);
        Ok(Self {
            storage: Dense::from_vec(data),
            shape: shape.to_vec(),
            strides,
        })
    }

    /// Get the shape of the tensor.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the rank (number of dimensions).
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get total number of elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.storage.len()
    }

    /// Check if tensor is empty (has zero elements).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.storage.is_empty()
    }

    /// Get strides.
    #[inline]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get underlying data as slice.
    #[inline]
    pub fn data(&self) -> &[T] {
        self.storage.as_slice()
    }

    /// Get underlying data as mutable slice.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }

    /// Get element by linear index.
    #[inline]
    pub fn get_linear(&self, i: usize) -> Option<&T> {
        self.storage.as_slice().get(i)
    }

    /// Get mutable element by linear index.
    #[inline]
    pub fn get_linear_mut(&mut self, i: usize) -> Option<&mut T> {
        self.storage.as_mut_slice().get_mut(i)
    }

    /// Get element by cartesian indices.
    ///
    /// Returns `None` if indices are out of bounds or wrong number of indices.
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        if indices.len() != self.ndim() {
            return None;
        }
        for (&idx, &dim) in indices.iter().zip(self.shape.iter()) {
            if idx >= dim {
                return None;
            }
        }
        let linear = cartesian_to_linear(indices, &self.strides);
        self.get_linear(linear)
    }

    /// Get mutable element by cartesian indices.
    ///
    /// Returns `None` if indices are out of bounds or wrong number of indices.
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        if indices.len() != self.ndim() {
            return None;
        }
        for (&idx, &dim) in indices.iter().zip(self.shape.iter()) {
            if idx >= dim {
                return None;
            }
        }
        let linear = cartesian_to_linear(indices, &self.strides);
        self.get_linear_mut(linear)
    }

    /// Set element by cartesian indices.
    ///
    /// # Errors
    ///
    /// Returns error if indices are out of bounds or wrong number of indices.
    pub fn set(&mut self, indices: &[usize], value: T) -> Result<(), TensorError> {
        if indices.len() != self.ndim() {
            return Err(TensorError::WrongNumberOfIndices {
                expected: self.ndim(),
                actual: indices.len(),
            });
        }
        for (&idx, &dim) in indices.iter().zip(self.shape.iter()) {
            if idx >= dim {
                return Err(TensorError::IndexOutOfBounds {
                    index: idx,
                    dim_size: dim,
                });
            }
        }
        let linear = cartesian_to_linear(indices, &self.strides);
        self.storage[linear] = value;
        Ok(())
    }

    /// Fill all elements with a value.
    pub fn fill(&mut self, value: T) {
        for x in self.storage.as_mut_slice() {
            *x = value;
        }
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: &[usize]) -> Self {
        let mut t = Self::zeros(shape);
        t.fill(T::one());
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::c64;

    fn test_zeros_generic<T: Scalar>() {
        let t: Tensor<T> = Tensor::zeros(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.len(), 6);
        assert_eq!(t.strides(), &[1, 2]);
        for i in 0..6 {
            assert_eq!(*t.get_linear(i).unwrap(), T::zero());
        }
    }

    #[test]
    fn test_zeros_f64() {
        test_zeros_generic::<f64>();
    }

    #[test]
    fn test_zeros_c64() {
        test_zeros_generic::<c64>();
    }

    #[test]
    fn test_from_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_vec(data, &[2, 3]).unwrap();

        // Column-major order: data laid out as columns
        // [1, 2] [3, 4] [5, 6] -> stored as [1, 2, 3, 4, 5, 6]
        assert_eq!(t.get(&[0, 0]), Some(&1.0));
        assert_eq!(t.get(&[1, 0]), Some(&2.0));
        assert_eq!(t.get(&[0, 1]), Some(&3.0));
        assert_eq!(t.get(&[1, 1]), Some(&4.0));
        assert_eq!(t.get(&[0, 2]), Some(&5.0));
        assert_eq!(t.get(&[1, 2]), Some(&6.0));
    }

    #[test]
    fn test_from_vec_shape_mismatch() {
        let data = vec![1.0, 2.0, 3.0];
        let result = Tensor::from_vec(data, &[2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_out_of_bounds() {
        let t: Tensor<f64> = Tensor::zeros(&[2, 3]);
        assert_eq!(t.get(&[2, 0]), None); // Index 2 out of bounds for dim 2
        assert_eq!(t.get(&[0, 3]), None); // Index 3 out of bounds for dim 3
        assert_eq!(t.get(&[0]), None); // Wrong number of indices
        assert_eq!(t.get(&[0, 0, 0]), None); // Wrong number of indices
    }

    #[test]
    fn test_set() {
        let mut t: Tensor<f64> = Tensor::zeros(&[2, 3]);
        t.set(&[1, 2], 42.0).unwrap();
        assert_eq!(t.get(&[1, 2]), Some(&42.0));
    }

    #[test]
    fn test_fill() {
        let mut t: Tensor<f64> = Tensor::zeros(&[2, 3]);
        t.fill(5.0);
        for i in 0..6 {
            assert_eq!(*t.get_linear(i).unwrap(), 5.0);
        }
    }

    #[test]
    fn test_ones() {
        let t: Tensor<f64> = Tensor::ones(&[2, 3]);
        for i in 0..6 {
            assert_eq!(*t.get_linear(i).unwrap(), 1.0);
        }
    }

    #[test]
    fn test_scalar_tensor() {
        // 0-dimensional tensor (scalar)
        let t: Tensor<f64> = Tensor::zeros(&[]);
        assert_eq!(t.ndim(), 0);
        assert_eq!(t.len(), 1);
        assert_eq!(t.shape(), &[]);
    }

    #[test]
    fn test_1d_tensor() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        assert_eq!(t.get(&[0]), Some(&1.0));
        assert_eq!(t.get(&[1]), Some(&2.0));
        assert_eq!(t.get(&[2]), Some(&3.0));
    }

    #[test]
    fn test_3d_tensor() {
        let t: Tensor<f64> = Tensor::zeros(&[2, 3, 4]);
        assert_eq!(t.shape(), &[2, 3, 4]);
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.len(), 24);
        assert_eq!(t.strides(), &[1, 2, 6]);
    }
}
