//! TrackedTensor - Tensor with gradient tracking for automatic differentiation.

use super::graph::{NodeId, NodeRef, clear_graph_f64, with_graph_f64};
use crate::scalar::Scalar;
use crate::tensor::DenseTensor;

/// A tensor that tracks gradients for automatic differentiation.
///
/// This is the main user-facing type for AD operations. It wraps a
/// `DenseTensor` and optionally tracks it in the computation graph.
///
/// # Example
///
/// ```ignore
/// use ndtensors::autodiff::TrackedTensor;
/// use ndtensors::Tensor;
///
/// // Create a leaf tensor that requires gradient
/// let a = TrackedTensor::leaf(Tensor::ones(&[2, 3]));
/// assert!(a.requires_grad());
///
/// // Create a tensor that doesn't require gradient
/// let b = TrackedTensor::new(Tensor::ones(&[3, 4]));
/// assert!(!b.requires_grad());
/// ```
#[derive(Debug, Clone)]
pub struct TrackedTensor<T: Scalar> {
    /// The underlying tensor data.
    tensor: DenseTensor<T>,
    /// Node in computation graph (None if not tracking).
    node: Option<NodeRef<T>>,
    /// Whether this tensor requires gradient.
    requires_grad: bool,
}

impl<T: Scalar> TrackedTensor<T> {
    /// Create a tracked tensor that does not require gradient.
    pub fn new(tensor: DenseTensor<T>) -> Self {
        Self {
            tensor,
            node: None,
            requires_grad: false,
        }
    }

    /// Create from tensor with a node reference (used internally).
    pub fn from_tensor_with_grad(tensor: DenseTensor<T>, node: NodeRef<T>) -> Self {
        Self {
            tensor,
            node: Some(node),
            requires_grad: true,
        }
    }

    /// Get the underlying tensor.
    pub fn tensor(&self) -> &DenseTensor<T> {
        &self.tensor
    }

    /// Consume and return the underlying tensor.
    pub fn into_tensor(self) -> DenseTensor<T> {
        self.tensor
    }

    /// Get node reference if this tensor is in the computation graph.
    pub fn node(&self) -> Option<&NodeRef<T>> {
        self.node.as_ref()
    }

    /// Get node ID if tracked.
    pub fn node_id(&self) -> Option<NodeId> {
        self.node.as_ref().map(|n| n.id())
    }

    /// Check if this tensor requires gradient.
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Get shape.
    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    /// Get number of dimensions.
    pub fn ndim(&self) -> usize {
        self.tensor.ndim()
    }

    /// Get total number of elements.
    pub fn len(&self) -> usize {
        self.tensor.len()
    }

    /// Check if tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.tensor.is_empty()
    }

    /// Get data slice.
    pub fn data(&self) -> &[T] {
        self.tensor.data()
    }

    /// Detach from computation graph.
    ///
    /// Returns a new tensor that shares data but doesn't require grad.
    pub fn detach(&self) -> Self {
        Self {
            tensor: self.tensor.clone(),
            node: None,
            requires_grad: false,
        }
    }
}

// f64-specific methods that interact with the thread-local graph
impl TrackedTensor<f64> {
    /// Create a leaf tensor that requires gradient.
    ///
    /// Registers in the thread-local computation graph.
    pub fn leaf(tensor: DenseTensor<f64>) -> Self {
        let node = with_graph_f64(|g| g.create_leaf(true));
        Self {
            tensor,
            node: Some(node),
            requires_grad: true,
        }
    }

    /// Create with explicit requires_grad flag.
    pub fn with_requires_grad(tensor: DenseTensor<f64>, requires_grad: bool) -> Self {
        if requires_grad {
            Self::leaf(tensor)
        } else {
            Self::new(tensor)
        }
    }
}

/// Clear the computation graph (call after backward or between forward passes).
pub fn clear_graph() {
    clear_graph_f64();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_tracked_tensor_new() {
        let t: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let tracked = TrackedTensor::new(t.clone());

        assert!(!tracked.requires_grad());
        assert!(tracked.node().is_none());
        assert_eq!(tracked.data(), t.data());
    }

    #[test]
    fn test_tracked_tensor_leaf() {
        clear_graph();

        let t: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let tracked = TrackedTensor::leaf(t);

        assert!(tracked.requires_grad());
        assert!(tracked.node().is_some());
        assert_eq!(tracked.node_id().unwrap().index(), 0);
    }

    #[test]
    fn test_tracked_tensor_with_requires_grad() {
        clear_graph();

        let t: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        let with_grad = TrackedTensor::with_requires_grad(t.clone(), true);
        assert!(with_grad.requires_grad());

        let without_grad = TrackedTensor::with_requires_grad(t, false);
        assert!(!without_grad.requires_grad());
    }

    #[test]
    fn test_tracked_tensor_detach() {
        clear_graph();

        let t: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let tracked = TrackedTensor::leaf(t);

        let detached = tracked.detach();
        assert!(!detached.requires_grad());
        assert!(detached.node().is_none());
        assert_eq!(tracked.data(), detached.data());
    }

    #[test]
    fn test_tracked_tensor_shape() {
        let t: DenseTensor<f64> = Tensor::zeros(&[2, 3, 4]);
        let tracked = TrackedTensor::new(t);

        assert_eq!(tracked.shape(), &[2, 3, 4]);
        assert_eq!(tracked.ndim(), 3);
        assert_eq!(tracked.len(), 24);
    }

    #[test]
    fn test_multiple_leaves() {
        clear_graph();

        let t1: DenseTensor<f64> = Tensor::ones(&[2, 3]);
        let t2: DenseTensor<f64> = Tensor::ones(&[3, 4]);

        let leaf1 = TrackedTensor::leaf(t1);
        let leaf2 = TrackedTensor::leaf(t2);

        assert_eq!(leaf1.node_id().unwrap().index(), 0);
        assert_eq!(leaf2.node_id().unwrap().index(), 1);
    }
}
