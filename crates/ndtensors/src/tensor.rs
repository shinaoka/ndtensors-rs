//! N-dimensional tensor type with polymorphic storage.
//!
//! Following NDTensors.jl's design:
//! ```text
//! Tensor{ElT, N, StoreT<:TensorStorage, IndsT}
//! ├── DenseTensor   = Tensor where StoreT<:Dense
//! ├── DiagTensor    = Tensor where StoreT<:Diag (future)
//! └── BlockSparseTensor = Tensor where StoreT<:BlockSparse (future)
//! ```

use crate::error::TensorError;
use crate::scalar::Scalar;
use crate::storage::{Dense, TensorStorage};
use crate::strides::{cartesian_to_linear, compute_strides};
use std::marker::PhantomData;

/// A n-dimensional tensor with polymorphic storage.
///
/// This mirrors NDTensors.jl's `Tensor{ElT, N, StoreT, IndsT}`.
/// The storage type `StoreT` determines the storage layout (Dense, Diag, etc.).
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<ElT: Scalar, StoreT: TensorStorage<ElT> = Dense<ElT>> {
    storage: StoreT,
    shape: Vec<usize>,
    strides: Vec<usize>,
    _phantom: PhantomData<ElT>,
}

/// Type alias for dense tensors (most common case).
/// Matches NDTensors.jl's `DenseTensor = Tensor where StoreT<:Dense`.
pub type DenseTensor<ElT> = Tensor<ElT, Dense<ElT>>;

impl<ElT: Scalar, StoreT: TensorStorage<ElT>> Tensor<ElT, StoreT> {
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
            storage: StoreT::zeros(len.max(1)), // At least 1 for scalar (empty shape)
            shape: shape.to_vec(),
            strides,
            _phantom: PhantomData,
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
    /// use ndtensors::{DenseTensor, Tensor};
    ///
    /// let t: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    /// assert_eq!(t.shape(), &[2, 3]);
    /// assert_eq!(t.get(&[0, 0]), Some(&1.0));
    /// assert_eq!(t.get(&[1, 0]), Some(&2.0)); // Column-major: [1,0] is second element
    /// assert_eq!(t.get(&[0, 1]), Some(&3.0)); // [0,1] is third element
    /// ```
    pub fn from_vec(data: Vec<ElT>, shape: &[usize]) -> Result<Self, TensorError> {
        let expected_len: usize = shape.iter().product::<usize>().max(1);
        if data.len() != expected_len {
            return Err(TensorError::ShapeMismatch {
                expected: expected_len,
                actual: data.len(),
            });
        }
        let strides = compute_strides(shape);
        Ok(Self {
            storage: StoreT::from_vec(data),
            shape: shape.to_vec(),
            strides,
            _phantom: PhantomData,
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
    pub fn data(&self) -> &[ElT] {
        self.storage.as_slice()
    }

    /// Get underlying data as mutable slice.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [ElT] {
        self.storage.as_mut_slice()
    }

    /// Get element by linear index.
    #[inline]
    pub fn get_linear(&self, i: usize) -> Option<&ElT> {
        self.storage.as_slice().get(i)
    }

    /// Get mutable element by linear index.
    #[inline]
    pub fn get_linear_mut(&mut self, i: usize) -> Option<&mut ElT> {
        self.storage.as_mut_slice().get_mut(i)
    }

    /// Get element by cartesian indices.
    ///
    /// Returns `None` if indices are out of bounds or wrong number of indices.
    pub fn get(&self, indices: &[usize]) -> Option<&ElT> {
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
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut ElT> {
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
    pub fn set(&mut self, indices: &[usize], value: ElT) -> Result<(), TensorError> {
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
        self.storage.as_mut_slice()[linear] = value;
        Ok(())
    }

    /// Fill all elements with a value.
    pub fn fill(&mut self, value: ElT) {
        for x in self.storage.as_mut_slice() {
            *x = value;
        }
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: &[usize]) -> Self {
        let mut t = Self::zeros(shape);
        t.fill(ElT::one());
        t
    }
}

// DenseTensor-specific operations
// Following NDTensors.jl's dispatch pattern where operations are specialized by storage type
impl<ElT: Scalar> Tensor<ElT, Dense<ElT>> {
    /// Reshape the tensor to a new shape (zero-copy view).
    ///
    /// Creates a new tensor that shares the same underlying storage.
    /// The total number of elements must remain the same.
    ///
    /// # Arguments
    ///
    /// * `new_shape` - The new shape for the tensor
    ///
    /// # Returns
    ///
    /// A new tensor with the new shape, sharing the same underlying data.
    ///
    /// # Errors
    ///
    /// Returns an error if the total number of elements doesn't match.
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    ///
    /// // Reshape to 1D
    /// let t1d = t.reshape(&[6]).unwrap();
    /// assert_eq!(t1d.shape(), &[6]);
    ///
    /// // Reshape to 3x2
    /// let t3x2 = t.reshape(&[3, 2]).unwrap();
    /// assert_eq!(t3x2.shape(), &[3, 2]);
    ///
    /// // t, t1d, t3x2 share the same underlying data
    /// assert!(t.shares_storage_with(&t1d));
    /// assert!(t.shares_storage_with(&t3x2));
    /// ```
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, TensorError> {
        let current_len = self.len();
        let new_len: usize = new_shape.iter().product::<usize>().max(1);

        if current_len != new_len {
            return Err(TensorError::ShapeMismatch {
                expected: current_len,
                actual: new_len,
            });
        }

        let new_strides = compute_strides(new_shape);

        Ok(Self {
            storage: self.storage.view(),
            shape: new_shape.to_vec(),
            strides: new_strides,
            _phantom: PhantomData,
        })
    }

    /// Check if this tensor shares storage with another tensor.
    ///
    /// Returns `true` if both tensors point to the same underlying data.
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        self.storage
            .buffer()
            .shares_storage_with(other.storage.buffer())
    }

    /// Check if this tensor's storage is shared with other tensors.
    ///
    /// Returns `true` if there are other views of this tensor's data.
    pub fn is_view(&self) -> bool {
        self.storage.buffer().is_shared()
    }

    /// Permute the dimensions of the tensor.
    ///
    /// # Arguments
    ///
    /// * `perm` - Permutation of dimensions. `perm[i]` gives the source dimension
    ///   for the i-th dimension of the result.
    ///
    /// # Errors
    ///
    /// Returns error if `perm` is not a valid permutation of `0..ndim`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ndtensors::Tensor;
    ///
    /// // Create a 2x3 tensor
    /// let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    ///
    /// // Transpose (swap dimensions 0 and 1)
    /// let t2 = t.permutedims(&[1, 0]).unwrap();
    /// assert_eq!(t2.shape(), &[3, 2]);
    ///
    /// // t[i,j] == t2[j,i]
    /// assert_eq!(t.get(&[0, 0]), t2.get(&[0, 0]));
    /// assert_eq!(t.get(&[1, 0]), t2.get(&[0, 1]));
    /// assert_eq!(t.get(&[0, 2]), t2.get(&[2, 0]));
    /// ```
    pub fn permutedims(&self, perm: &[usize]) -> Result<Self, TensorError> {
        // Delegate to operations module (DenseTensor specialization)
        crate::operations::permutedims(self, perm)
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
        let t: DenseTensor<f64> = Tensor::from_vec(data, &[2, 3]).unwrap();

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
        let result = Tensor::<f64>::from_vec(data, &[2, 3]);
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
        let t: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
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

    #[test]
    fn test_permutedims_transpose() {
        // 2x3 matrix
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Transpose
        let t2 = t.permutedims(&[1, 0]).unwrap();
        assert_eq!(t2.shape(), &[3, 2]);

        // t[i,j] == t2[j,i]
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(t.get(&[i, j]), t2.get(&[j, i]));
            }
        }
    }

    #[test]
    fn test_permutedims_3d() {
        // 2x3x4 tensor
        let mut t: Tensor<f64> = Tensor::zeros(&[2, 3, 4]);
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    t.set(&[i, j, k], (i * 100 + j * 10 + k) as f64).unwrap();
                }
            }
        }

        // Permute [0,1,2] -> [2,0,1]: shape 2x3x4 -> 4x2x3
        let t2 = t.permutedims(&[2, 0, 1]).unwrap();
        assert_eq!(t2.shape(), &[4, 2, 3]);

        // t[i,j,k] == t2[k,i,j]
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_eq!(t.get(&[i, j, k]), t2.get(&[k, i, j]));
                }
            }
        }
    }

    #[test]
    fn test_permutedims_identity() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Identity permutation
        let t2 = t.permutedims(&[0, 1]).unwrap();
        assert_eq!(t2.shape(), &[2, 3]);
        assert_eq!(t.data(), t2.data());
    }

    #[test]
    fn test_permutedims_invalid() {
        let t: Tensor<f64> = Tensor::zeros(&[2, 3]);

        // Wrong number of dimensions
        assert!(t.permutedims(&[0]).is_err());
        assert!(t.permutedims(&[0, 1, 2]).is_err());

        // Invalid index
        assert!(t.permutedims(&[0, 2]).is_err());

        // Duplicate index
        assert!(t.permutedims(&[0, 0]).is_err());
    }

    #[test]
    fn test_dense_tensor_alias() {
        // Verify DenseTensor alias works
        let t: DenseTensor<f64> = DenseTensor::zeros(&[2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
    }

    #[test]
    fn test_reshape_basic() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Reshape to 1D
        let t1d = t.reshape(&[6]).unwrap();
        assert_eq!(t1d.shape(), &[6]);
        assert_eq!(t1d.len(), 6);

        // Reshape to 3x2
        let t3x2 = t.reshape(&[3, 2]).unwrap();
        assert_eq!(t3x2.shape(), &[3, 2]);

        // Data is the same
        for i in 0..6 {
            assert_eq!(t.get_linear(i), t1d.get_linear(i));
            assert_eq!(t.get_linear(i), t3x2.get_linear(i));
        }
    }

    #[test]
    fn test_reshape_shares_storage() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        let t1d = t.reshape(&[6]).unwrap();
        let t3x2 = t.reshape(&[3, 2]).unwrap();

        // All share the same storage
        assert!(t.shares_storage_with(&t1d));
        assert!(t.shares_storage_with(&t3x2));
        assert!(t1d.shares_storage_with(&t3x2));

        // All are views (shared)
        assert!(t.is_view());
        assert!(t1d.is_view());
        assert!(t3x2.is_view());
    }

    #[test]
    fn test_reshape_copy_on_write() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let mut t1d = t.reshape(&[6]).unwrap();

        // Initially shared
        assert!(t.shares_storage_with(&t1d));

        // Modify t1d
        t1d.set(&[0], 100.0).unwrap();

        // No longer shared after mutation
        assert!(!t.shares_storage_with(&t1d));

        // Original unchanged
        assert_eq!(*t.get_linear(0).unwrap(), 1.0);
        // Modified version changed
        assert_eq!(*t1d.get_linear(0).unwrap(), 100.0);
    }

    #[test]
    fn test_reshape_invalid_size() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();

        // Wrong total size
        assert!(t.reshape(&[5]).is_err());
        assert!(t.reshape(&[2, 2]).is_err());
        assert!(t.reshape(&[7]).is_err());
    }

    #[test]
    fn test_reshape_to_higher_dim() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]).unwrap();

        // Reshape 1D to 2D
        let t2d = t.reshape(&[2, 3]).unwrap();
        assert_eq!(t2d.shape(), &[2, 3]);

        // Reshape 1D to 3D
        let t3d = t.reshape(&[2, 1, 3]).unwrap();
        assert_eq!(t3d.shape(), &[2, 1, 3]);

        assert!(t.shares_storage_with(&t2d));
        assert!(t.shares_storage_with(&t3d));
    }

    #[test]
    fn test_not_view_single_ref() {
        let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        // Single reference is not a view
        assert!(!t.is_view());
    }
}
