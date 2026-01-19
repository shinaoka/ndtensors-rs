//! Block type for BlockSparse storage.
//!
//! A Block represents coordinates identifying a specific block in a block-sparse tensor.
//! It mirrors NDTensors.jl's `Block{N}` type.

use smallvec::SmallVec;
use std::hash::{Hash, Hasher};

/// A block coordinate with precomputed hash.
///
/// Uses `SmallVec<[usize; 8]>` for efficient storage:
/// - Stack allocation for common cases (N <= 8)
/// - Heap fallback for larger dimensions
///
/// # Example
/// ```
/// use ndtensors::storage::blocksparse::Block;
///
/// let block = Block::new(&[1, 2, 3]);
/// assert_eq!(block.ndims(), 3);
/// assert_eq!(block[0], 1);
/// assert_eq!(block[1], 2);
/// assert_eq!(block[2], 3);
/// ```
#[derive(Clone, Debug)]
pub struct Block {
    coords: SmallVec<[usize; 8]>,
    hash: u64,
}

impl Block {
    /// Create a new Block from coordinates.
    pub fn new(coords: &[usize]) -> Self {
        let coords: SmallVec<[usize; 8]> = coords.iter().copied().collect();
        let hash = compute_hash(&coords);
        Self { coords, hash }
    }

    /// Create a new Block by collecting coordinates from an iterator.
    pub fn collect_from<I: IntoIterator<Item = usize>>(iter: I) -> Self {
        let coords: SmallVec<[usize; 8]> = iter.into_iter().collect();
        let hash = compute_hash(&coords);
        Self { coords, hash }
    }

    /// Get the number of dimensions.
    #[inline]
    pub fn ndims(&self) -> usize {
        self.coords.len()
    }

    /// Get the coordinates as a slice.
    #[inline]
    pub fn coords(&self) -> &[usize] {
        &self.coords
    }

    /// Get the precomputed hash value.
    #[inline]
    pub fn precomputed_hash(&self) -> u64 {
        self.hash
    }

    /// Create a permuted block with reordered coordinates.
    pub fn permute(&self, perm: &[usize]) -> Self {
        assert_eq!(
            perm.len(),
            self.ndims(),
            "permutation length must match block dimensions"
        );
        let new_coords: SmallVec<[usize; 8]> = perm.iter().map(|&i| self.coords[i]).collect();
        let hash = compute_hash(&new_coords);
        Self {
            coords: new_coords,
            hash,
        }
    }

    /// Create a new block with one coordinate changed.
    pub fn set(&self, index: usize, value: usize) -> Self {
        let mut new_coords = self.coords.clone();
        new_coords[index] = value;
        let hash = compute_hash(&new_coords);
        Self {
            coords: new_coords,
            hash,
        }
    }
}

impl std::ops::Index<usize> for Block {
    type Output = usize;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.coords[index]
    }
}

impl PartialEq for Block {
    fn eq(&self, other: &Self) -> bool {
        // Fast path: check hash first
        if self.hash != other.hash {
            return false;
        }
        self.coords == other.coords
    }
}

impl Eq for Block {}

impl Hash for Block {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Use precomputed hash
        state.write_u64(self.hash);
    }
}

impl PartialOrd for Block {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Block {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Lexicographic ordering (like CartesianIndex comparison in Julia)
        self.coords.cmp(&other.coords)
    }
}

impl std::fmt::Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Block(")?;
        for (i, &c) in self.coords.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", c)?;
        }
        write!(f, ")")
    }
}

/// Compute hash for block coordinates.
///
/// Uses FNV-1a hash algorithm for fast hashing of integer sequences.
fn compute_hash(coords: &[usize]) -> u64 {
    // FNV-1a hash
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;
    for &coord in coords {
        hash ^= coord as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

impl<const N: usize> From<[usize; N]> for Block {
    fn from(coords: [usize; N]) -> Self {
        Self::new(&coords)
    }
}

impl From<&[usize]> for Block {
    fn from(coords: &[usize]) -> Self {
        Self::new(coords)
    }
}

impl From<Vec<usize>> for Block {
    fn from(coords: Vec<usize>) -> Self {
        Self::new(&coords)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_block_creation() {
        let block = Block::new(&[1, 2, 3]);
        assert_eq!(block.ndims(), 3);
        assert_eq!(block[0], 1);
        assert_eq!(block[1], 2);
        assert_eq!(block[2], 3);
    }

    #[test]
    fn test_block_from_array() {
        let block: Block = [1, 2, 3].into();
        assert_eq!(block.ndims(), 3);
        assert_eq!(block.coords(), &[1, 2, 3]);
    }

    #[test]
    fn test_block_equality() {
        let b1 = Block::new(&[1, 2, 3]);
        let b2 = Block::new(&[1, 2, 3]);
        let b3 = Block::new(&[1, 2, 4]);

        assert_eq!(b1, b2);
        assert_ne!(b1, b3);
    }

    #[test]
    fn test_block_hash_consistency() {
        let b1 = Block::new(&[1, 2, 3]);
        let b2 = Block::new(&[1, 2, 3]);

        assert_eq!(b1.precomputed_hash(), b2.precomputed_hash());

        // Can be used as HashMap key
        let mut map = HashMap::new();
        map.insert(b1.clone(), 42);
        assert_eq!(map.get(&b2), Some(&42));
    }

    #[test]
    fn test_block_ordering() {
        let b1 = Block::new(&[1, 2]);
        let b2 = Block::new(&[1, 3]);
        let b3 = Block::new(&[2, 1]);

        assert!(b1 < b2);
        assert!(b2 < b3);
        assert!(b1 < b3);
    }

    #[test]
    fn test_block_permute() {
        let block = Block::new(&[10, 20, 30]);
        let permuted = block.permute(&[2, 0, 1]);
        assert_eq!(permuted.coords(), &[30, 10, 20]);
    }

    #[test]
    fn test_block_set() {
        let block = Block::new(&[1, 2, 3]);
        let modified = block.set(1, 5);
        assert_eq!(modified.coords(), &[1, 5, 3]);
        // Original unchanged
        assert_eq!(block.coords(), &[1, 2, 3]);
    }

    #[test]
    fn test_block_display() {
        let block = Block::new(&[1, 2, 3]);
        assert_eq!(format!("{}", block), "Block(1, 2, 3)");
    }

    #[test]
    fn test_empty_block() {
        let block = Block::new(&[]);
        assert_eq!(block.ndims(), 0);
        assert_eq!(format!("{}", block), "Block()");
    }

    #[test]
    fn test_large_block() {
        // Test with more than 8 dimensions (heap allocation)
        let coords: Vec<usize> = (0..12).collect();
        let block = Block::new(&coords);
        assert_eq!(block.ndims(), 12);
        for i in 0..12 {
            assert_eq!(block[i], i);
        }
    }
}
