//! Gradient storage container.

use super::graph::NodeId;
use crate::operations::apply_binary;
use crate::scalar::Scalar;
use crate::tensor::DenseTensor;
use std::collections::HashMap;
use std::ops::Add;

/// Container for accumulated gradients.
///
/// Stores gradients keyed by NodeId, with in-place accumulation
/// for nodes with multiple downstream paths.
#[derive(Debug)]
pub struct Gradients<T: Scalar> {
    grads: HashMap<NodeId, DenseTensor<T>>,
}

impl<T: Scalar + Add<Output = T>> Gradients<T> {
    /// Create empty gradient container.
    pub fn new() -> Self {
        Self {
            grads: HashMap::new(),
        }
    }

    /// Accumulate gradient for a node.
    ///
    /// If gradient already exists, adds to it (for multiple paths).
    pub fn accumulate(&mut self, id: NodeId, grad: DenseTensor<T>) {
        if let Some(existing) = self.grads.get_mut(&id) {
            // Add gradients element-wise
            *existing =
                apply_binary(existing, &grad, |a, b| a + b).expect("gradient shapes must match");
        } else {
            self.grads.insert(id, grad);
        }
    }

    /// Get gradient for a node.
    pub fn get(&self, id: NodeId) -> Option<&DenseTensor<T>> {
        self.grads.get(&id)
    }

    /// Remove and return gradient (for passing to backward functions).
    pub fn remove(&mut self, id: NodeId) -> Option<DenseTensor<T>> {
        self.grads.remove(&id)
    }

    /// Check if gradient exists for node.
    pub fn contains(&self, id: NodeId) -> bool {
        self.grads.contains_key(&id)
    }

    /// Number of stored gradients.
    pub fn len(&self) -> usize {
        self.grads.len()
    }

    /// Check if no gradients stored.
    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }

    /// Iterate over all gradients.
    pub fn iter(&self) -> impl Iterator<Item = (&NodeId, &DenseTensor<T>)> {
        self.grads.iter()
    }
}

impl<T: Scalar + Add<Output = T>> Default for Gradients<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_gradients_new() {
        let grads: Gradients<f64> = Gradients::new();
        assert!(grads.is_empty());
        assert_eq!(grads.len(), 0);
    }

    #[test]
    fn test_gradients_accumulate_single() {
        let mut grads: Gradients<f64> = Gradients::new();
        let id = NodeId::new_for_test(0);
        let grad: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        grads.accumulate(id, grad.clone());

        assert!(grads.contains(id));
        assert_eq!(grads.get(id).unwrap().data(), grad.data());
    }

    #[test]
    fn test_gradients_accumulate_multiple() {
        let mut grads: Gradients<f64> = Gradients::new();
        let id = NodeId::new_for_test(0);
        let grad1: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let grad2: DenseTensor<f64> = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();

        grads.accumulate(id, grad1);
        grads.accumulate(id, grad2);

        let accumulated = grads.get(id).unwrap();
        assert_eq!(accumulated.data(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_gradients_remove() {
        let mut grads: Gradients<f64> = Gradients::new();
        let id = NodeId::new_for_test(0);
        let grad: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();

        grads.accumulate(id, grad);
        assert!(grads.contains(id));

        let removed = grads.remove(id);
        assert!(removed.is_some());
        assert!(!grads.contains(id));
    }

    #[test]
    fn test_gradients_multiple_nodes() {
        let mut grads: Gradients<f64> = Gradients::new();
        let id1 = NodeId::new_for_test(0);
        let id2 = NodeId::new_for_test(1);
        let grad1: DenseTensor<f64> = Tensor::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        let grad2: DenseTensor<f64> = Tensor::from_vec(vec![3.0, 4.0, 5.0], &[3]).unwrap();

        grads.accumulate(id1, grad1);
        grads.accumulate(id2, grad2);

        assert_eq!(grads.len(), 2);
        assert_eq!(grads.get(id1).unwrap().shape(), &[2]);
        assert_eq!(grads.get(id2).unwrap().shape(), &[3]);
    }
}
