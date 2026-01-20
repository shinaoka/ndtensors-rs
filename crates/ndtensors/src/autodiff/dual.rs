//! DualTensor - Tensor with tangent vector for forward-mode automatic differentiation.
//!
//! Forward-mode AD propagates tangent vectors through computations using
//! Jacobian-vector products (JVP). Given a function f and input x with
//! tangent v, forward-mode computes:
//!   - primal: f(x)
//!   - tangent: J_f(x) * v (Jacobian-vector product)
//!
//! This is the dual to backward-mode AD (VJP), which computes v^T * J_f(x).
//!
//! # Example
//!
//! ```ignore
//! use ndtensors::autodiff::{DualTensor, dual_contract};
//! use ndtensors::Tensor;
//!
//! // Create dual tensors with tangent vectors
//! let a = DualTensor::with_tangent(
//!     Tensor::ones(&[2, 3]),
//!     Tensor::ones(&[2, 3]),  // tangent = dA
//! );
//! let b = DualTensor::new(Tensor::ones(&[3, 4])); // constant (dB = 0)
//!
//! // Forward pass propagates tangents using JVP
//! let c = dual_contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();
//!
//! // c.primal() = A @ B
//! // c.tangent() = dA @ B (product rule, since dB = 0)
//! ```

use crate::error::TensorError;
use crate::scalar::Scalar;
use crate::tensor::DenseTensor;

/// A tensor with an associated tangent vector for forward-mode AD.
///
/// `DualTensor` represents a value and its derivative in a given direction.
/// The tangent can be `None` to represent a zero tangent (constant value),
/// which allows efficient handling of inputs that don't require differentiation.
///
/// # Design
///
/// Uses concrete `DenseTensor<T>` rather than type-erased `Box<dyn AnyStorage<T>>`
/// for consistency with `TrackedTensor` and simplicity. Forward-mode AD doesn't
/// need graph storage (no tape), so type erasure provides no benefit.
#[derive(Debug, Clone)]
pub struct DualTensor<T: Scalar> {
    /// The primal (function value) tensor.
    primal: DenseTensor<T>,
    /// The tangent (derivative) tensor. None represents a zero tangent.
    tangent: Option<DenseTensor<T>>,
}

impl<T: Scalar> DualTensor<T> {
    /// Create a dual tensor with zero tangent (constant).
    ///
    /// Use this for inputs that are constants and should not be differentiated.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use ndtensors::autodiff::DualTensor;
    /// use ndtensors::Tensor;
    ///
    /// let constant = DualTensor::new(Tensor::ones(&[2, 3]));
    /// assert!(!constant.has_tangent());
    /// ```
    pub fn new(primal: DenseTensor<T>) -> Self {
        Self {
            primal,
            tangent: None,
        }
    }

    /// Create a dual tensor with an explicit tangent vector.
    ///
    /// # Arguments
    ///
    /// * `primal` - The function value
    /// * `tangent` - The derivative in some direction
    ///
    /// # Returns
    ///
    /// Returns an error if the tangent shape doesn't match the primal shape.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use ndtensors::autodiff::DualTensor;
    /// use ndtensors::Tensor;
    ///
    /// let primal = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
    /// let tangent = Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).unwrap();
    /// let dual = DualTensor::with_tangent(primal, tangent).unwrap();
    /// assert!(dual.has_tangent());
    /// ```
    pub fn with_tangent(
        primal: DenseTensor<T>,
        tangent: DenseTensor<T>,
    ) -> Result<Self, TensorError> {
        if primal.shape() != tangent.shape() {
            return Err(TensorError::InvalidOperation(format!(
                "tangent shape {:?} must match primal shape {:?}",
                tangent.shape(),
                primal.shape()
            )));
        }
        Ok(Self {
            primal,
            tangent: Some(tangent),
        })
    }

    /// Create from primal and optional tangent (internal use).
    pub(crate) fn from_primal_tangent(
        primal: DenseTensor<T>,
        tangent: Option<DenseTensor<T>>,
    ) -> Self {
        Self { primal, tangent }
    }

    /// Get the primal tensor.
    pub fn primal(&self) -> &DenseTensor<T> {
        &self.primal
    }

    /// Get the tangent tensor (None if zero).
    pub fn tangent(&self) -> Option<&DenseTensor<T>> {
        self.tangent.as_ref()
    }

    /// Consume and return primal and tangent.
    pub fn into_parts(self) -> (DenseTensor<T>, Option<DenseTensor<T>>) {
        (self.primal, self.tangent)
    }

    /// Check if this tensor has a non-zero tangent.
    pub fn has_tangent(&self) -> bool {
        self.tangent.is_some()
    }

    /// Get shape (same for primal and tangent).
    pub fn shape(&self) -> &[usize] {
        self.primal.shape()
    }

    /// Get number of dimensions.
    pub fn ndim(&self) -> usize {
        self.primal.ndim()
    }

    /// Get total number of elements.
    pub fn len(&self) -> usize {
        self.primal.len()
    }

    /// Check if tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.primal.is_empty()
    }

    /// Detach tangent, returning a new DualTensor with zero tangent.
    ///
    /// This is useful when you want to treat a computed value as a constant
    /// for subsequent operations.
    pub fn detach(&self) -> Self {
        Self {
            primal: self.primal.clone(),
            tangent: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_dual_tensor_new() {
        let primal: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let dual = DualTensor::new(primal.clone());

        assert_eq!(dual.shape(), &[3]);
        assert_eq!(dual.ndim(), 1);
        assert_eq!(dual.len(), 3);
        assert!(!dual.is_empty());
        assert!(!dual.has_tangent());
        assert!(dual.tangent().is_none());
        assert_eq!(dual.primal().data(), primal.data());
    }

    #[test]
    fn test_dual_tensor_with_tangent() {
        let primal: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let tangent: DenseTensor<f64> = Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).unwrap();
        let dual = DualTensor::with_tangent(primal.clone(), tangent.clone()).unwrap();

        assert!(dual.has_tangent());
        assert_eq!(dual.tangent().unwrap().data(), tangent.data());
        assert_eq!(dual.primal().data(), primal.data());
    }

    #[test]
    fn test_dual_tensor_shape_mismatch() {
        let primal: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let tangent: DenseTensor<f64> = Tensor::from_vec(vec![0.1, 0.2], &[2]).unwrap();

        let result = DualTensor::with_tangent(primal, tangent);
        assert!(result.is_err());
        match result {
            Err(TensorError::InvalidOperation(msg)) => {
                assert!(msg.contains("tangent shape"));
                assert!(msg.contains("[2]"));
                assert!(msg.contains("[3]"));
            }
            _ => panic!("Expected InvalidOperation error"),
        }
    }

    #[test]
    fn test_dual_tensor_2d() {
        let primal: DenseTensor<f64> = Tensor::ones(&[2, 3]);
        let tangent: DenseTensor<f64> = Tensor::zeros(&[2, 3]);
        let dual = DualTensor::with_tangent(primal, tangent).unwrap();

        assert_eq!(dual.shape(), &[2, 3]);
        assert_eq!(dual.ndim(), 2);
        assert_eq!(dual.len(), 6);
    }

    #[test]
    fn test_dual_tensor_detach() {
        let primal: DenseTensor<f64> = Tensor::ones(&[2, 3]);
        let tangent: DenseTensor<f64> = Tensor::ones(&[2, 3]);
        let dual = DualTensor::with_tangent(primal, tangent).unwrap();

        assert!(dual.has_tangent());

        let detached = dual.detach();
        assert!(!detached.has_tangent());
        assert_eq!(detached.primal().data(), dual.primal().data());
    }

    #[test]
    fn test_dual_tensor_into_parts() {
        let primal: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let tangent: DenseTensor<f64> = Tensor::from_vec(vec![0.1, 0.2, 0.3], &[3]).unwrap();
        let dual = DualTensor::with_tangent(primal.clone(), tangent.clone()).unwrap();

        let (p, t) = dual.into_parts();
        assert_eq!(p.data(), primal.data());
        assert_eq!(t.unwrap().data(), tangent.data());
    }

    #[test]
    fn test_dual_tensor_into_parts_no_tangent() {
        let primal: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let dual = DualTensor::new(primal.clone());

        let (p, t) = dual.into_parts();
        assert_eq!(p.data(), primal.data());
        assert!(t.is_none());
    }

    #[test]
    fn test_dual_tensor_clone() {
        let primal: DenseTensor<f64> = Tensor::ones(&[2, 3]);
        let tangent: DenseTensor<f64> = Tensor::ones(&[2, 3]);
        let dual = DualTensor::with_tangent(primal, tangent).unwrap();

        let cloned = dual.clone();
        assert_eq!(cloned.primal().data(), dual.primal().data());
        assert_eq!(
            cloned.tangent().unwrap().data(),
            dual.tangent().unwrap().data()
        );
    }
}
