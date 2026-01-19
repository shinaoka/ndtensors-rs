//! Contraction properties for GEMM optimization.
//!
//! This module provides the `ContractionProperties` struct that analyzes
//! tensor contraction patterns and determines the optimal GEMM-based approach.
//!
//! # Design
//!
//! Following NDTensors.jl's `ContractionProperties` from `contraction_logic.jl`,
//! this struct computes:
//! - Which indices are contracted vs. uncontracted
//! - GEMM dimensions (dleft, dmid, dright)
//! - Required permutations for A, B, and C

/// Properties computed for efficient tensor contraction via GEMM.
///
/// For a contraction `C = A * B` where some indices are summed over:
/// - `dleft` = product of A's uncontracted dimensions
/// - `dmid` = product of contracted dimensions
/// - `dright` = product of B's uncontracted dimensions
///
/// The contraction becomes: `C(dleft, dright) = A(dleft, dmid) * B(dmid, dright)`
#[derive(Debug, Clone)]
pub struct ContractionProperties {
    /// Contracted index pairs: (index in A, index in B)
    pub contracted_pairs: Vec<(usize, usize)>,

    /// Indices in A that are not contracted (appear in output)
    pub uncontracted_a: Vec<usize>,

    /// Indices in B that are not contracted (appear in output)
    pub uncontracted_b: Vec<usize>,

    /// Whether A needs permutation before GEMM
    pub permute_a: bool,

    /// Whether B needs permutation before GEMM
    pub permute_b: bool,

    /// Whether C needs permutation after GEMM
    pub permute_c: bool,

    /// GEMM dimension: product of uncontracted A dimensions
    pub dleft: usize,

    /// GEMM dimension: product of contracted dimensions
    pub dmid: usize,

    /// GEMM dimension: product of uncontracted B dimensions
    pub dright: usize,

    /// Permutation for A: reorder to [uncontracted..., contracted...]
    pub perm_a: Vec<usize>,

    /// Permutation for B: reorder to [contracted..., uncontracted...]
    pub perm_b: Vec<usize>,

    /// Permutation for C: reorder output if needed
    pub perm_c: Vec<usize>,

    /// Output labels in the order they appear after GEMM (before perm_c)
    pub output_labels: Vec<i32>,
}

impl ContractionProperties {
    /// Compute contraction properties from labels and shapes.
    ///
    /// Labels convention (following NDTensors.jl):
    /// - Negative labels: contracted indices (summed over)
    /// - Positive labels: uncontracted indices (appear in output)
    ///
    /// # Arguments
    ///
    /// * `labels_a` - Labels for tensor A
    /// * `shape_a` - Shape of tensor A
    /// * `labels_b` - Labels for tensor B
    /// * `shape_b` - Shape of tensor B
    ///
    /// # Example
    ///
    /// ```
    /// use ndtensors::contract::ContractionProperties;
    ///
    /// // Matrix multiplication: C[i,k] = A[i,j] * B[j,k]
    /// // A[1,-1], B[-1,2] -> C[1,2]
    /// let props = ContractionProperties::compute(
    ///     &[1, -1], &[2, 3],  // A is 2x3
    ///     &[-1, 2], &[3, 4],  // B is 3x4
    /// );
    ///
    /// assert_eq!(props.dleft, 2);   // A's uncontracted dim
    /// assert_eq!(props.dmid, 3);    // contracted dim
    /// assert_eq!(props.dright, 4);  // B's uncontracted dim
    /// ```
    pub fn compute(
        labels_a: &[i32],
        shape_a: &[usize],
        labels_b: &[i32],
        shape_b: &[usize],
    ) -> Self {
        // Find contracted pairs (negative labels that appear in both)
        let mut contracted_pairs = Vec::new();
        for (i, &la) in labels_a.iter().enumerate() {
            if la < 0 {
                for (j, &lb) in labels_b.iter().enumerate() {
                    if la == lb {
                        contracted_pairs.push((i, j));
                        break;
                    }
                }
            }
        }

        // Find uncontracted indices
        let contracted_a: Vec<usize> = contracted_pairs.iter().map(|&(i, _)| i).collect();
        let contracted_b: Vec<usize> = contracted_pairs.iter().map(|&(_, j)| j).collect();

        let uncontracted_a: Vec<usize> = (0..labels_a.len())
            .filter(|i| !contracted_a.contains(i))
            .collect();
        let uncontracted_b: Vec<usize> = (0..labels_b.len())
            .filter(|j| !contracted_b.contains(j))
            .collect();

        // Compute GEMM dimensions
        let dleft: usize = uncontracted_a
            .iter()
            .map(|&i| shape_a[i])
            .product::<usize>()
            .max(1);
        let dmid: usize = contracted_pairs
            .iter()
            .map(|&(i, _)| shape_a[i])
            .product::<usize>()
            .max(1);
        let dright: usize = uncontracted_b
            .iter()
            .map(|&j| shape_b[j])
            .product::<usize>()
            .max(1);

        // Compute permutations
        // For A: need [uncontracted..., contracted...] order
        let ideal_a: Vec<usize> = uncontracted_a
            .iter()
            .chain(contracted_a.iter())
            .copied()
            .collect();
        let perm_a = ideal_a.clone();
        let permute_a = !is_identity_perm(&perm_a);

        // For B: need [contracted..., uncontracted...] order
        let ideal_b: Vec<usize> = contracted_b
            .iter()
            .chain(uncontracted_b.iter())
            .copied()
            .collect();
        let perm_b = ideal_b.clone();
        let permute_b = !is_identity_perm(&perm_b);

        // Output labels: [A's positive labels in order, B's positive labels in order]
        let mut output_labels = Vec::new();
        for &i in &uncontracted_a {
            output_labels.push(labels_a[i]);
        }
        for &j in &uncontracted_b {
            output_labels.push(labels_b[j]);
        }

        // Check if output needs permutation (should be sorted by label value)
        let mut sorted_output = output_labels.clone();
        sorted_output.sort();
        let permute_c = output_labels != sorted_output;

        // Compute perm_c to sort output labels
        let perm_c = if permute_c {
            let mut indices: Vec<usize> = (0..output_labels.len()).collect();
            indices.sort_by_key(|&i| output_labels[i]);
            // perm_c[i] = where to find the i-th sorted element in the unsorted array
            let mut perm = vec![0; output_labels.len()];
            for (new_pos, &old_pos) in indices.iter().enumerate() {
                perm[new_pos] = old_pos;
            }
            perm
        } else {
            (0..output_labels.len()).collect()
        };

        Self {
            contracted_pairs,
            uncontracted_a,
            uncontracted_b,
            permute_a,
            permute_b,
            permute_c,
            dleft,
            dmid,
            dright,
            perm_a,
            perm_b,
            perm_c,
            output_labels,
        }
    }

    /// Check if this is a simple matrix multiplication (no permutation needed).
    pub fn is_simple_matmul(&self) -> bool {
        !self.permute_a && !self.permute_b && !self.permute_c
    }

    /// Check if this is an outer product (no contracted indices).
    pub fn is_outer_product(&self) -> bool {
        self.contracted_pairs.is_empty()
    }

    /// Check if this is a full contraction (scalar result).
    pub fn is_full_contraction(&self) -> bool {
        self.uncontracted_a.is_empty() && self.uncontracted_b.is_empty()
    }
}

/// Check if a permutation is the identity permutation.
fn is_identity_perm(perm: &[usize]) -> bool {
    perm.iter().enumerate().all(|(i, &p)| i == p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_multiply() {
        // C[i,k] = A[i,j] * B[j,k]
        // A[1,-1], B[-1,2]
        let props = ContractionProperties::compute(&[1, -1], &[2, 3], &[-1, 2], &[3, 4]);

        assert_eq!(props.contracted_pairs, vec![(1, 0)]);
        assert_eq!(props.uncontracted_a, vec![0]);
        assert_eq!(props.uncontracted_b, vec![1]);
        assert_eq!(props.dleft, 2);
        assert_eq!(props.dmid, 3);
        assert_eq!(props.dright, 4);
        assert!(!props.permute_a); // Already [uncontracted, contracted]
        assert!(!props.permute_b); // Already [contracted, uncontracted]
        assert!(!props.permute_c); // Output [1,2] is sorted
        assert!(props.is_simple_matmul());
    }

    #[test]
    fn test_inner_product() {
        // scalar = A[i] * B[i]
        // A[-1], B[-1]
        let props = ContractionProperties::compute(&[-1], &[5], &[-1], &[5]);

        assert_eq!(props.contracted_pairs, vec![(0, 0)]);
        assert!(props.uncontracted_a.is_empty());
        assert!(props.uncontracted_b.is_empty());
        assert_eq!(props.dleft, 1);
        assert_eq!(props.dmid, 5);
        assert_eq!(props.dright, 1);
        assert!(props.is_full_contraction());
    }

    #[test]
    fn test_outer_product() {
        // C[i,j] = A[i] * B[j]
        // A[1], B[2]
        let props = ContractionProperties::compute(&[1], &[3], &[2], &[4]);

        assert!(props.contracted_pairs.is_empty());
        assert_eq!(props.uncontracted_a, vec![0]);
        assert_eq!(props.uncontracted_b, vec![0]);
        assert_eq!(props.dleft, 3);
        assert_eq!(props.dmid, 1);
        assert_eq!(props.dright, 4);
        assert!(props.is_outer_product());
    }

    #[test]
    fn test_tensor_contraction_with_permutation() {
        // C[i,l] = A[i,j,k] * B[k,j,l]
        // A[1,-1,-2], B[-2,-1,2]
        // Contracted: j(-1), k(-2)
        let props =
            ContractionProperties::compute(&[1, -1, -2], &[2, 3, 4], &[-2, -1, 2], &[4, 3, 5]);

        assert_eq!(props.contracted_pairs.len(), 2);
        assert_eq!(props.uncontracted_a, vec![0]); // i
        assert_eq!(props.uncontracted_b, vec![2]); // l
        assert_eq!(props.dleft, 2);
        assert_eq!(props.dmid, 12); // 3 * 4
        assert_eq!(props.dright, 5);

        // A needs permutation: [0,1,2] -> [0,1,2] (uncontracted=0, contracted=1,2)
        // Actually [i,j,k] with contracted j,k should stay [0,1,2]
        assert!(!props.permute_a);

        // B needs permutation: contracted_pairs are ordered by A's labels
        // contracted_pairs = [(1,1), (2,0)] -> contracted_b = [1, 0]
        // ideal_b = [contracted..., uncontracted...] = [1, 0, 2]
        // This requires permutation from [0, 1, 2] to [1, 0, 2]
        assert!(props.permute_b);
        assert_eq!(props.perm_b, vec![1, 0, 2]);
    }

    #[test]
    fn test_permutation_needed() {
        // C[j,i] = A[i,k] * B[k,j]
        // A[2,-1], B[-1,1]
        // This should produce C in order [2,1] but we want [1,2]
        let props = ContractionProperties::compute(&[2, -1], &[3, 4], &[-1, 1], &[4, 5]);

        assert_eq!(props.dleft, 3);
        assert_eq!(props.dmid, 4);
        assert_eq!(props.dright, 5);

        // Output is [2, 1] but should be sorted to [1, 2]
        assert_eq!(props.output_labels, vec![2, 1]);
        assert!(props.permute_c);
    }

    #[test]
    fn test_3d_tensor_contraction() {
        // C[i,j,l] = A[i,j,k] * B[k,l]
        // A[1,2,-1], B[-1,3]
        let props = ContractionProperties::compute(&[1, 2, -1], &[2, 3, 4], &[-1, 3], &[4, 5]);

        assert_eq!(props.contracted_pairs, vec![(2, 0)]);
        assert_eq!(props.uncontracted_a, vec![0, 1]);
        assert_eq!(props.uncontracted_b, vec![1]);
        assert_eq!(props.dleft, 6); // 2 * 3
        assert_eq!(props.dmid, 4);
        assert_eq!(props.dright, 5);
    }
}
