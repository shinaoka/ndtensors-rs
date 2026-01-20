//! Hessian-vector product via forward-on-reverse AD
//!
//! HVP: H @ v where H = ∇²f (Hessian matrix)
//!
//! Method (forward-on-reverse):
//!   1. g(x) = ∇f(x)           # gradient function (reverse-mode)
//!   2. H @ v = Jg(x) @ v      # JVP of gradient (forward-mode)
//!
//! # Example
//!
//! ```ignore
//! use ndtensors::autodiff::hvp_scalar;
//!
//! // f(x) = x^2, H = 2
//! let result = hvp_scalar(|x| x * x, 3.0, 1.0);
//! assert!((result - 2.0).abs() < 1e-10);
//! ```

use petgraph::algo::toposort;
use petgraph::graph::{DiGraph, NodeIndex};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Node ID for the computation graph
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScalarNodeId(usize);

/// Combined dual + tracked scalar for forward-on-reverse AD
///
/// This type tracks both:
/// - Reverse-mode graph (for computing gradients via backward pass)
/// - Forward-mode tangent (for computing JVP of the gradient)
///
/// Uses petgraph for proper topological sort to handle DAGs correctly.
#[derive(Clone)]
pub struct DualTrackedScalar {
    inner: Rc<RefCell<DualTrackedInner>>,
    /// Unique ID for this node in the computation graph
    id: ScalarNodeId,
}

// Thread-local counter for unique node IDs
thread_local! {
    static NEXT_ID: RefCell<usize> = const { RefCell::new(0) };
}

fn next_node_id() -> ScalarNodeId {
    NEXT_ID.with(|id| {
        let current = *id.borrow();
        *id.borrow_mut() = current + 1;
        ScalarNodeId(current)
    })
}

fn reset_node_ids() {
    NEXT_ID.with(|id| {
        *id.borrow_mut() = 0;
    });
}

struct DualTrackedInner {
    /// Primal value
    primal: f64,
    /// Forward-mode tangent (direction for HVP)
    tangent: f64,
    /// Accumulated gradient (from backward pass)
    grad: f64,
    /// Accumulated gradient's tangent (this is the HVP result!)
    grad_tangent: f64,
    /// Gradient function for backward pass
    grad_fn: Option<Box<dyn DualGradFn>>,
}

/// Gradient function that propagates both gradients and their tangents
trait DualGradFn {
    /// Compute gradients AND their tangents for each input
    /// Returns Vec<(grad, grad_tangent)>
    fn backward(&self, grad_output: f64, grad_tangent_output: f64) -> Vec<(f64, f64)>;

    /// Get node IDs of inputs (for building the graph)
    fn input_ids(&self) -> Vec<ScalarNodeId>;

    /// Get references to input scalars for gradient accumulation
    fn inputs(&self) -> Vec<Rc<RefCell<DualTrackedInner>>>;
}

impl DualTrackedScalar {
    /// Create a new dual tracked scalar (leaf node)
    pub fn new(primal: f64, tangent: f64) -> Self {
        Self {
            inner: Rc::new(RefCell::new(DualTrackedInner {
                primal,
                tangent,
                grad: 0.0,
                grad_tangent: 0.0,
                grad_fn: None,
            })),
            id: next_node_id(),
        }
    }

    /// Get the node ID
    pub fn id(&self) -> ScalarNodeId {
        self.id
    }

    /// Get primal value
    pub fn primal(&self) -> f64 {
        self.inner.borrow().primal
    }

    /// Get tangent
    pub fn tangent(&self) -> f64 {
        self.inner.borrow().tangent
    }

    /// Get gradient (after backward)
    pub fn grad(&self) -> f64 {
        self.inner.borrow().grad
    }

    /// Get gradient's tangent (this is the HVP result!)
    pub fn grad_tangent(&self) -> f64 {
        self.inner.borrow().grad_tangent
    }

    fn inner_ref(&self) -> Rc<RefCell<DualTrackedInner>> {
        Rc::clone(&self.inner)
    }

    fn with_grad_fn(primal: f64, tangent: f64, grad_fn: Box<dyn DualGradFn>) -> Self {
        Self {
            inner: Rc::new(RefCell::new(DualTrackedInner {
                primal,
                tangent,
                grad: 0.0,
                grad_tangent: 0.0,
                grad_fn: Some(grad_fn),
            })),
            id: next_node_id(),
        }
    }

    /// Backward pass with tangent propagation using topological sort
    fn backward(&self, all_nodes: &HashMap<ScalarNodeId, Rc<RefCell<DualTrackedInner>>>) {
        // Build the computation graph
        let mut graph: DiGraph<ScalarNodeId, ()> = DiGraph::new();
        let mut node_to_idx: HashMap<ScalarNodeId, NodeIndex> = HashMap::new();

        // Add all nodes to the graph
        for &node_id in all_nodes.keys() {
            let idx = graph.add_node(node_id);
            node_to_idx.insert(node_id, idx);
        }

        // Add edges (from inputs to outputs)
        for (&node_id, inner) in all_nodes.iter() {
            let inner_borrowed = inner.borrow();
            if let Some(ref grad_fn) = inner_borrowed.grad_fn {
                let output_idx = node_to_idx[&node_id];
                for input_id in grad_fn.input_ids() {
                    if let Some(&input_idx) = node_to_idx.get(&input_id) {
                        // Edge from input to output (forward direction)
                        graph.add_edge(input_idx, output_idx, ());
                    }
                }
            }
        }

        // Get topological order (forward order)
        let topo_order = toposort(&graph, None).expect("Graph should be acyclic");

        // Initialize the output node's gradient
        {
            let mut inner = self.inner.borrow_mut();
            inner.grad = 1.0;
            inner.grad_tangent = 0.0;
        }

        // Process in reverse topological order (backward pass)
        for node_idx in topo_order.into_iter().rev() {
            let node_id = graph[node_idx];
            let inner = &all_nodes[&node_id];

            let grad_fn_data = {
                let inner_borrowed = inner.borrow();
                inner_borrowed.grad_fn.as_ref().map(|gf| {
                    let grads = gf.backward(inner_borrowed.grad, inner_borrowed.grad_tangent);
                    let inputs = gf.inputs();
                    (grads, inputs)
                })
            };

            if let Some((input_grads, inputs)) = grad_fn_data {
                for (input_inner, (grad, grad_tangent)) in inputs.iter().zip(input_grads.iter()) {
                    let mut inp = input_inner.borrow_mut();
                    inp.grad += grad;
                    inp.grad_tangent += grad_tangent;
                }
            }
        }
    }
}

// --- Multiplication ---

struct MulBackward {
    x: Rc<RefCell<DualTrackedInner>>,
    y: Rc<RefCell<DualTrackedInner>>,
    x_id: ScalarNodeId,
    y_id: ScalarNodeId,
    x_primal: f64,
    y_primal: f64,
    x_tangent: f64,
    y_tangent: f64,
}

impl DualGradFn for MulBackward {
    fn backward(&self, grad_output: f64, grad_tangent_output: f64) -> Vec<(f64, f64)> {
        // z = x * y
        // dz/dx = y,  dz/dy = x
        //
        // grad_x = grad_z * y
        // grad_y = grad_z * x
        //
        // For tangent propagation (chain rule on gradient computation):
        // d(grad_x)/dt = d(grad_z * y)/dt = grad_z' * y + grad_z * y'
        // d(grad_y)/dt = d(grad_z * x)/dt = grad_z' * x + grad_z * x'

        let grad_x = grad_output * self.y_primal;
        let grad_y = grad_output * self.x_primal;

        let grad_tangent_x = grad_tangent_output * self.y_primal + grad_output * self.y_tangent;
        let grad_tangent_y = grad_tangent_output * self.x_primal + grad_output * self.x_tangent;

        vec![(grad_x, grad_tangent_x), (grad_y, grad_tangent_y)]
    }

    fn input_ids(&self) -> Vec<ScalarNodeId> {
        vec![self.x_id, self.y_id]
    }

    fn inputs(&self) -> Vec<Rc<RefCell<DualTrackedInner>>> {
        vec![Rc::clone(&self.x), Rc::clone(&self.y)]
    }
}

impl std::ops::Mul for &DualTrackedScalar {
    type Output = DualTrackedScalar;

    fn mul(self, rhs: &DualTrackedScalar) -> DualTrackedScalar {
        let x_primal = self.primal();
        let y_primal = rhs.primal();
        let x_tangent = self.tangent();
        let y_tangent = rhs.tangent();

        // Forward: primal and tangent
        let primal = x_primal * y_primal;
        let tangent = x_tangent * y_primal + x_primal * y_tangent;

        let grad_fn = MulBackward {
            x: self.inner_ref(),
            y: rhs.inner_ref(),
            x_id: self.id,
            y_id: rhs.id,
            x_primal,
            y_primal,
            x_tangent,
            y_tangent,
        };

        DualTrackedScalar::with_grad_fn(primal, tangent, Box::new(grad_fn))
    }
}

// --- Addition ---

struct AddBackward {
    x: Rc<RefCell<DualTrackedInner>>,
    y: Rc<RefCell<DualTrackedInner>>,
    x_id: ScalarNodeId,
    y_id: ScalarNodeId,
}

impl DualGradFn for AddBackward {
    fn backward(&self, grad_output: f64, grad_tangent_output: f64) -> Vec<(f64, f64)> {
        // z = x + y
        // dz/dx = 1, dz/dy = 1
        // grad_x = grad_z, grad_y = grad_z
        // Tangent passes through unchanged
        vec![
            (grad_output, grad_tangent_output),
            (grad_output, grad_tangent_output),
        ]
    }

    fn input_ids(&self) -> Vec<ScalarNodeId> {
        vec![self.x_id, self.y_id]
    }

    fn inputs(&self) -> Vec<Rc<RefCell<DualTrackedInner>>> {
        vec![Rc::clone(&self.x), Rc::clone(&self.y)]
    }
}

impl std::ops::Add for &DualTrackedScalar {
    type Output = DualTrackedScalar;

    fn add(self, rhs: &DualTrackedScalar) -> DualTrackedScalar {
        let primal = self.primal() + rhs.primal();
        let tangent = self.tangent() + rhs.tangent();

        let grad_fn = AddBackward {
            x: self.inner_ref(),
            y: rhs.inner_ref(),
            x_id: self.id,
            y_id: rhs.id,
        };

        DualTrackedScalar::with_grad_fn(primal, tangent, Box::new(grad_fn))
    }
}

// --- HVP API ---

/// Compute Hessian-vector product for a scalar function: H @ v
///
/// Uses forward-on-reverse AD: computes JVP of the gradient function.
/// Handles DAG structures correctly using topological sort.
///
/// # Arguments
/// * `f` - Scalar function f: DualTrackedScalar → DualTrackedScalar
/// * `x` - Point at which to evaluate
/// * `v` - Direction vector for HVP
///
/// # Returns
/// * `H @ v` - Hessian-vector product (scalar)
///
/// # Example
///
/// ```ignore
/// use ndtensors::autodiff::hvp_scalar;
///
/// // f(x) = x^2, f''(x) = 2
/// let hvp = hvp_scalar(|x| x * x, 3.0, 1.0);
/// assert!((hvp - 2.0).abs() < 1e-10);
/// ```
pub fn hvp_scalar<F>(f: F, x: f64, v: f64) -> f64
where
    F: Fn(&DualTrackedScalar) -> DualTrackedScalar,
{
    // Reset node IDs for a fresh computation
    reset_node_ids();

    // Create input with tangent direction v
    let x_dual_tracked = DualTrackedScalar::new(x, v);

    // Forward pass: computes f(x) and propagates tangents
    let y = f(&x_dual_tracked);

    // Collect all nodes in the computation graph
    let all_nodes = collect_all_nodes(&y);

    // Backward pass: computes gradients AND their tangents
    y.backward(&all_nodes);

    // The tangent of the gradient is the HVP result
    x_dual_tracked.grad_tangent()
}

/// Collect all nodes reachable from the output by traversing grad_fn inputs
fn collect_all_nodes(
    output: &DualTrackedScalar,
) -> HashMap<ScalarNodeId, Rc<RefCell<DualTrackedInner>>> {
    let mut nodes = HashMap::new();
    let mut stack = vec![(output.id, output.inner_ref())];

    while let Some((id, inner)) = stack.pop() {
        if nodes.contains_key(&id) {
            continue;
        }
        nodes.insert(id, Rc::clone(&inner));

        let inner_borrowed = inner.borrow();
        if let Some(ref grad_fn) = inner_borrowed.grad_fn {
            let input_ids = grad_fn.input_ids();
            let inputs = grad_fn.inputs();
            for (input_id, input_inner) in input_ids.into_iter().zip(inputs.into_iter()) {
                if !nodes.contains_key(&input_id) {
                    stack.push((input_id, input_inner));
                }
            }
        }
    }

    nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hvp_quadratic() {
        // f(x) = x^2
        // f'(x) = 2x
        // f''(x) = 2 (constant Hessian)
        // HVP = f''(x) * v = 2 * v

        let hvp = hvp_scalar(|x| x * x, 3.0, 1.0);

        // H @ v = 2 * 1 = 2
        assert!((hvp - 2.0).abs() < 1e-10, "Expected 2.0, got {}", hvp);
    }

    #[test]
    fn test_hvp_cubic() {
        // f(x) = x^3 = x * x * x
        // f'(x) = 3x^2
        // f''(x) = 6x
        // HVP at x=2 with v=1: f''(2) * 1 = 12

        let hvp = hvp_scalar(|x| &(x * x) * x, 2.0, 1.0);

        // H @ v = 6 * 2 * 1 = 12
        assert!((hvp - 12.0).abs() < 1e-10, "Expected 12.0, got {}", hvp);
    }

    #[test]
    fn test_hvp_with_different_v() {
        // f(x) = x^2
        // f''(x) = 2
        // HVP = 2 * v

        let hvp = hvp_scalar(|x| x * x, 5.0, 3.0);

        // H @ v = 2 * 3 = 6
        assert!((hvp - 6.0).abs() < 1e-10, "Expected 6.0, got {}", hvp);
    }

    #[test]
    fn test_hvp_quartic_dag() {
        // f(x) = x^4 = (x^2)^2 (DAG structure - x^2 is reused)
        // f'(x) = 4x^3
        // f''(x) = 12x^2
        // HVP at x=2 with v=1: f''(2) * 1 = 12 * 4 = 48
        //
        // This tests DAG support with topological sort

        let hvp = hvp_scalar(
            |x| {
                let x2 = x * x;
                &x2 * &x2
            },
            2.0,
            1.0,
        );

        // H @ v = 12 * 4 * 1 = 48
        assert!((hvp - 48.0).abs() < 1e-10, "Expected 48.0, got {}", hvp);
    }

    #[test]
    fn test_hvp_quartic_tree() {
        // f(x) = x^4 = x * x * x * x (tree structure)
        // Same result as DAG version

        let hvp = hvp_scalar(|x| &(&(x * x) * x) * x, 2.0, 1.0);

        // H @ v = 12 * 4 * 1 = 48
        assert!((hvp - 48.0).abs() < 1e-10, "Expected 48.0, got {}", hvp);
    }

    #[test]
    fn test_hvp_sum_of_squares() {
        // f(x) = x^2 + x^2 = 2x^2
        // f'(x) = 4x
        // f''(x) = 4
        // HVP = 4 * v

        let hvp = hvp_scalar(|x| &(x * x) + &(x * x), 3.0, 2.0);

        // H @ v = 4 * 2 = 8
        assert!((hvp - 8.0).abs() < 1e-10, "Expected 8.0, got {}", hvp);
    }
}
