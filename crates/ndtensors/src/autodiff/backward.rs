//! Backward pass execution for reverse-mode automatic differentiation.

use super::gradients::Gradients;
use super::graph::{ComputationGraph, NodeId, with_graph_f64};
use super::tensor::TrackedTensor;
use crate::error::TensorError;
use crate::scalar::Scalar;
use crate::tensor::DenseTensor;
use std::collections::{HashSet, VecDeque};

/// Execute backward pass from a scalar loss.
///
/// Computes gradients for all leaf nodes that require_grad.
///
/// # Arguments
/// * `loss` - The scalar loss tensor (must have exactly 1 element)
///
/// # Returns
/// Gradients container with accumulated gradients for all nodes.
///
/// # Errors
/// Returns error if:
/// - Loss is not a scalar (len != 1)
/// - Loss is not in the computation graph
///
/// # Example
///
/// ```ignore
/// use ndtensors::autodiff::{TrackedTensor, tracked_contract, backward, clear_graph};
/// use ndtensors::Tensor;
///
/// clear_graph();
///
/// let a = TrackedTensor::leaf(Tensor::ones(&[2, 3]));
/// let b = TrackedTensor::leaf(Tensor::ones(&[3, 4]));
/// let c = tracked_contract(&a, &[1, -1], &b, &[-1, 2]).unwrap();
///
/// // Sum to scalar for backward
/// let ones = TrackedTensor::new(Tensor::ones(&[2, 4]));
/// let loss = tracked_contract(&c, &[-1, -2], &ones, &[-1, -2]).unwrap();
///
/// let grads = backward(&loss).unwrap();
/// let grad_a = grads.get(a.node_id().unwrap()).unwrap();
/// ```
pub fn backward(loss: &TrackedTensor<f64>) -> Result<Gradients<f64>, TensorError> {
    // Validate loss is scalar
    let loss_len = loss.tensor().len();
    if loss_len != 1 {
        return Err(TensorError::InvalidOperation(format!(
            "backward() requires scalar loss, got {} elements",
            loss_len
        )));
    }

    // Get the loss node
    let loss_node_id = loss.node_id().ok_or_else(|| {
        TensorError::InvalidOperation(
            "backward() called on tensor not in computation graph".to_string(),
        )
    })?;

    // Initialize gradients with loss gradient = 1.0
    let mut gradients = Gradients::new();
    let loss_grad = DenseTensor::from_vec(vec![1.0], &[1])?;
    gradients.accumulate(loss_node_id, loss_grad);

    // Topological sort via reverse BFS from loss
    let topo_order = with_graph_f64(|graph| topological_sort(graph, loss_node_id));

    // Backward pass in topological order (from loss to inputs)
    with_graph_f64(|graph| {
        for node_id in topo_order {
            // Get gradient for this node
            let grad_output = match gradients.remove(node_id) {
                Some(g) => g,
                None => continue, // No gradient flowing to this node
            };

            // Get node and its backward function
            let node = match graph.get_node(node_id) {
                Some(n) => n,
                None => continue,
            };

            if let Some(grad_fn) = node.grad_fn() {
                // Compute gradients for inputs
                let input_grads = grad_fn.backward(&grad_output);

                // Accumulate gradients
                for (input_id, input_grad) in input_grads {
                    gradients.accumulate(input_id, input_grad);
                }
            } else {
                // Leaf node - store gradient for user access
                gradients.accumulate(node_id, grad_output);
            }
        }
    });

    Ok(gradients)
}

/// Topological sort of computation graph nodes reachable from start.
///
/// Returns nodes in order such that each node comes before all nodes
/// that depend on it (correct order for backward pass).
fn topological_sort<T: Scalar>(graph: &ComputationGraph<T>, start: NodeId) -> Vec<NodeId> {
    let mut result = Vec::new();
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();

    queue.push_back(start);

    // BFS to find all reachable nodes
    while let Some(node_id) = queue.pop_front() {
        if visited.contains(&node_id) {
            continue;
        }
        visited.insert(node_id);
        result.push(node_id);

        // Add input nodes to queue
        if let Some(node) = graph.get_node(node_id) {
            if let Some(grad_fn) = node.grad_fn() {
                for input_id in grad_fn.inputs() {
                    if !visited.contains(&input_id) {
                        queue.push_back(input_id);
                    }
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use crate::autodiff::graph::{GradFn, clear_graph_f64};

    // Simple GradFn that passes gradient through unchanged
    #[derive(Debug)]
    struct IdentityBackward {
        input_id: NodeId,
    }

    impl GradFn<f64> for IdentityBackward {
        fn backward(&self, grad_output: &DenseTensor<f64>) -> Vec<(NodeId, DenseTensor<f64>)> {
            vec![(self.input_id, grad_output.clone())]
        }

        fn inputs(&self) -> Vec<NodeId> {
            vec![self.input_id]
        }
    }

    // GradFn that scales gradient
    #[derive(Debug)]
    struct ScaleBackward {
        input_id: NodeId,
        scale: f64,
    }

    impl GradFn<f64> for ScaleBackward {
        fn backward(&self, grad_output: &DenseTensor<f64>) -> Vec<(NodeId, DenseTensor<f64>)> {
            let scaled: Vec<f64> = grad_output.data().iter().map(|&x| x * self.scale).collect();
            let grad = DenseTensor::from_vec(scaled, grad_output.shape()).unwrap();
            vec![(self.input_id, grad)]
        }

        fn inputs(&self) -> Vec<NodeId> {
            vec![self.input_id]
        }
    }

    #[test]
    fn test_backward_single_leaf() {
        clear_graph_f64();

        // Create a leaf and a computed node
        let leaf = with_graph_f64(|g| g.create_leaf(true));
        let computed = with_graph_f64(|g| {
            g.create_node(
                Box::new(IdentityBackward {
                    input_id: leaf.id(),
                }),
                true,
            )
        });

        // Create loss tensor
        let loss = TrackedTensor::from_tensor_with_grad(
            Tensor::from_vec(vec![1.0], &[1]).unwrap(),
            computed,
        );

        let grads = backward(&loss).unwrap();

        // Leaf should have gradient = 1.0
        let grad = grads.get(leaf.id()).unwrap();
        assert_eq!(grad.data(), &[1.0]);
    }

    #[test]
    fn test_backward_chain() {
        clear_graph_f64();

        // Create chain: leaf -> scale(2) -> scale(3) -> loss
        let leaf = with_graph_f64(|g| g.create_leaf(true));
        let node1 = with_graph_f64(|g| {
            g.create_node(
                Box::new(ScaleBackward {
                    input_id: leaf.id(),
                    scale: 2.0,
                }),
                true,
            )
        });
        let node2 = with_graph_f64(|g| {
            g.create_node(
                Box::new(ScaleBackward {
                    input_id: node1.id(),
                    scale: 3.0,
                }),
                true,
            )
        });

        let loss =
            TrackedTensor::from_tensor_with_grad(Tensor::from_vec(vec![1.0], &[1]).unwrap(), node2);

        let grads = backward(&loss).unwrap();

        // Gradient should be 1.0 * 3.0 * 2.0 = 6.0
        let grad = grads.get(leaf.id()).unwrap();
        assert_eq!(grad.data(), &[6.0]);
    }

    #[test]
    fn test_backward_non_scalar_error() {
        clear_graph_f64();

        let leaf = with_graph_f64(|g| g.create_leaf(true));
        let computed = with_graph_f64(|g| {
            g.create_node(
                Box::new(IdentityBackward {
                    input_id: leaf.id(),
                }),
                true,
            )
        });

        // Non-scalar loss
        let loss = TrackedTensor::from_tensor_with_grad(
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(),
            computed,
        );

        let result = backward(&loss);
        assert!(result.is_err());
    }

    #[test]
    fn test_backward_not_in_graph_error() {
        clear_graph_f64();

        // Tensor not in graph
        let loss = TrackedTensor::new(Tensor::from_vec(vec![1.0], &[1]).unwrap());

        let result = backward(&loss);
        assert!(result.is_err());
    }
}
