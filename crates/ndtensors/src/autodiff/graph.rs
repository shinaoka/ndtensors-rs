//! Computation graph for reverse-mode automatic differentiation.

use crate::scalar::Scalar;
use crate::tensor::DenseTensor;
use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;

/// Unique identifier for a node in the computation graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

impl NodeId {
    /// Get the internal index.
    pub fn index(&self) -> usize {
        self.0
    }

    /// Create a NodeId for testing purposes.
    #[cfg(test)]
    pub(crate) fn new_for_test(index: usize) -> Self {
        Self(index)
    }
}

/// Reference to a computation graph node.
///
/// This is the user-facing handle to a node in the computation graph.
#[derive(Debug, Clone)]
pub struct NodeRef<T: Scalar> {
    id: NodeId,
    _phantom: PhantomData<T>,
}

impl<T: Scalar> NodeRef<T> {
    /// Get the node ID.
    pub fn id(&self) -> NodeId {
        self.id
    }
}

/// Backward function trait.
///
/// Computes gradients with respect to inputs given gradient of output.
/// Each operation (contract, add, etc.) implements this trait.
pub trait GradFn<T: Scalar>: Debug {
    /// Compute VJP: given grad_output, return gradients for each input.
    ///
    /// Returns Vec of (NodeId, gradient) pairs for inputs that require grad.
    fn backward(&self, grad_output: &DenseTensor<T>) -> Vec<(NodeId, DenseTensor<T>)>;

    /// Get input node IDs (for topological sort).
    fn inputs(&self) -> Vec<NodeId>;
}

/// A node in the computation graph.
#[derive(Debug)]
pub struct Node<T: Scalar> {
    /// Unique identifier.
    id: NodeId,
    /// Backward function (None for leaf nodes).
    grad_fn: Option<Box<dyn GradFn<T>>>,
    /// Whether this node requires gradient.
    requires_grad: bool,
}

impl<T: Scalar> Node<T> {
    /// Get node ID.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Get backward function reference.
    pub fn grad_fn(&self) -> Option<&dyn GradFn<T>> {
        self.grad_fn.as_deref()
    }

    /// Check if this node requires gradient.
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }
}

/// Thread-local computation graph.
///
/// Stores the DAG of tensor operations for reverse-mode AD.
/// Each thread has its own independent graph.
pub struct ComputationGraph<T: Scalar> {
    nodes: Vec<Node<T>>,
    next_id: usize,
}

impl<T: Scalar> ComputationGraph<T> {
    /// Create a new empty computation graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            next_id: 0,
        }
    }

    /// Create a leaf node (input tensor with requires_grad=true).
    pub fn create_leaf(&mut self, requires_grad: bool) -> NodeRef<T> {
        let id = NodeId(self.next_id);
        self.next_id += 1;

        self.nodes.push(Node {
            id,
            grad_fn: None,
            requires_grad,
        });

        NodeRef {
            id,
            _phantom: PhantomData,
        }
    }

    /// Create a computed node with backward function.
    pub fn create_node(&mut self, grad_fn: Box<dyn GradFn<T>>, requires_grad: bool) -> NodeRef<T> {
        let id = NodeId(self.next_id);
        self.next_id += 1;

        self.nodes.push(Node {
            id,
            grad_fn: Some(grad_fn),
            requires_grad,
        });

        NodeRef {
            id,
            _phantom: PhantomData,
        }
    }

    /// Get node by ID.
    pub fn get_node(&self, id: NodeId) -> Option<&Node<T>> {
        self.nodes.get(id.index())
    }

    /// Get all nodes.
    pub fn nodes(&self) -> &[Node<T>] {
        &self.nodes
    }

    /// Clear the graph (call after backward).
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.next_id = 0;
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

impl<T: Scalar> Default for ComputationGraph<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> Debug for ComputationGraph<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputationGraph")
            .field("num_nodes", &self.nodes.len())
            .field("next_id", &self.next_id)
            .finish()
    }
}

// Thread-local graph storage for f64
thread_local! {
    static GRAPH_F64: RefCell<ComputationGraph<f64>> = RefCell::new(ComputationGraph::new());
}

/// Access the thread-local computation graph for f64.
///
/// # Example
///
/// ```ignore
/// use ndtensors::autodiff::with_graph_f64;
///
/// with_graph_f64(|graph| {
///     let node = graph.create_leaf(true);
///     println!("Created node {:?}", node.id());
/// });
/// ```
pub fn with_graph_f64<R>(f: impl FnOnce(&mut ComputationGraph<f64>) -> R) -> R {
    GRAPH_F64.with(|g| f(&mut g.borrow_mut()))
}

/// Clear the thread-local computation graph for f64.
pub fn clear_graph_f64() {
    with_graph_f64(|g| g.clear());
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple GradFn implementation for testing
    #[derive(Debug)]
    struct TestGradFn {
        input_ids: Vec<NodeId>,
    }

    impl GradFn<f64> for TestGradFn {
        fn backward(&self, grad_output: &DenseTensor<f64>) -> Vec<(NodeId, DenseTensor<f64>)> {
            // Just pass through the gradient to all inputs
            self.input_ids
                .iter()
                .map(|&id| (id, grad_output.clone()))
                .collect()
        }

        fn inputs(&self) -> Vec<NodeId> {
            self.input_ids.clone()
        }
    }

    #[test]
    fn test_create_leaf() {
        let mut graph: ComputationGraph<f64> = ComputationGraph::new();

        let node1 = graph.create_leaf(true);
        let node2 = graph.create_leaf(false);

        assert_eq!(node1.id().index(), 0);
        assert_eq!(node2.id().index(), 1);
        assert_eq!(graph.len(), 2);

        let n1 = graph.get_node(node1.id()).unwrap();
        assert!(n1.requires_grad());
        assert!(n1.grad_fn().is_none());

        let n2 = graph.get_node(node2.id()).unwrap();
        assert!(!n2.requires_grad());
    }

    #[test]
    fn test_create_node_with_grad_fn() {
        let mut graph: ComputationGraph<f64> = ComputationGraph::new();

        let leaf1 = graph.create_leaf(true);
        let leaf2 = graph.create_leaf(true);

        let grad_fn = TestGradFn {
            input_ids: vec![leaf1.id(), leaf2.id()],
        };
        let computed = graph.create_node(Box::new(grad_fn), true);

        assert_eq!(computed.id().index(), 2);

        let node = graph.get_node(computed.id()).unwrap();
        assert!(node.requires_grad());
        assert!(node.grad_fn().is_some());

        let inputs = node.grad_fn().unwrap().inputs();
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].index(), 0);
        assert_eq!(inputs[1].index(), 1);
    }

    #[test]
    fn test_clear_graph() {
        let mut graph: ComputationGraph<f64> = ComputationGraph::new();

        graph.create_leaf(true);
        graph.create_leaf(true);
        assert_eq!(graph.len(), 2);

        graph.clear();
        assert_eq!(graph.len(), 0);
        assert!(graph.is_empty());
    }

    #[test]
    fn test_thread_local_graph() {
        clear_graph_f64();

        with_graph_f64(|g| {
            g.create_leaf(true);
            g.create_leaf(true);
        });

        let count = with_graph_f64(|g| g.len());
        assert_eq!(count, 2);

        clear_graph_f64();

        let count = with_graph_f64(|g| g.len());
        assert_eq!(count, 0);
    }
}
