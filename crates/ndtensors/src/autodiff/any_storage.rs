//! Type-erased tensor storage for backward pass saves.

use crate::scalar::Scalar;
use crate::tensor::DenseTensor;
use std::any::Any;
use std::fmt::Debug;

/// Type-erased tensor storage for backward pass.
///
/// This trait enables storing tensors of different storage types
/// (Dense, BlockSparse, etc.) in the computation graph without
/// carrying the storage type in the type signature.
///
/// The trait preserves the element type `T` while erasing the
/// storage type, allowing gradient operations to work correctly.
///
/// Note: This trait does not require `Send` because the computation
/// graph is thread-local and tensors are never shared across threads.
pub trait AnyStorage<T: Scalar>: Debug {
    /// Clone into a boxed trait object.
    fn clone_boxed(&self) -> Box<dyn AnyStorage<T>>;

    /// Downcast to concrete type.
    fn as_any(&self) -> &dyn Any;

    /// Number of elements.
    fn len(&self) -> usize;

    /// Check if empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Shape of the tensor.
    fn shape(&self) -> &[usize];

    /// Convert to dense CPU tensor (for gradient computation).
    fn to_dense_cpu(&self) -> DenseTensor<T>;
}

impl<T: Scalar> AnyStorage<T> for DenseTensor<T> {
    fn clone_boxed(&self) -> Box<dyn AnyStorage<T>> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn len(&self) -> usize {
        DenseTensor::len(self)
    }

    fn shape(&self) -> &[usize] {
        DenseTensor::shape(self)
    }

    fn to_dense_cpu(&self) -> DenseTensor<T> {
        self.clone()
    }
}

impl<T: Scalar> Clone for Box<dyn AnyStorage<T>> {
    fn clone(&self) -> Self {
        self.clone_boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_any_storage_clone_boxed() {
        let t: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let boxed: Box<dyn AnyStorage<f64>> = Box::new(t.clone());

        let cloned = boxed.clone_boxed();
        assert_eq!(cloned.len(), 3);
        assert_eq!(cloned.shape(), &[3]);
    }

    #[test]
    fn test_any_storage_to_dense_cpu() {
        let t: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let boxed: Box<dyn AnyStorage<f64>> = Box::new(t.clone());

        let dense = boxed.to_dense_cpu();
        assert_eq!(dense.data(), t.data());
    }

    #[test]
    fn test_any_storage_downcast() {
        let t: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let boxed: Box<dyn AnyStorage<f64>> = Box::new(t.clone());

        let downcasted = boxed.as_any().downcast_ref::<DenseTensor<f64>>();
        assert!(downcasted.is_some());
        assert_eq!(downcasted.unwrap().data(), t.data());
    }

    #[test]
    fn test_box_clone() {
        let t: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let boxed: Box<dyn AnyStorage<f64>> = Box::new(t);

        let cloned: Box<dyn AnyStorage<f64>> = boxed.clone();
        assert_eq!(cloned.len(), 3);
    }
}
