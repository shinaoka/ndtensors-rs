# Plan: C API Tensor Creation Functions (ones, rand, randn)

## Goal
Issue #48の一部として、C APIに `ones`, `rand`, `randn` 関数を追加する。

## Step 0: Create GitHub Issue

```bash
gh issue create --title "feat(capi): add ones, rand, randn tensor creation functions" \
  --body "## Summary
Add C API functions for tensor creation:
- \`ndt_tensor_f64_ones\` - create tensor filled with 1.0
- \`ndt_tensor_f64_rand\` - create tensor with uniform random values [0, 1)
- \`ndt_tensor_f64_randn\` - create tensor with standard normal random values

Part of #48 (NDTensors.jl compatible C API)

## Implementation
- Follow existing \`ndt_tensor_f64_zeros\` pattern
- Rust backend already has \`Tensor::ones()\`, \`Tensor::random()\`, \`Tensor::randn()\`
- Add corresponding tests"
```

## Files to Modify
- `crates/ndtensors-capi/src/lib.rs` - 3つの関数とテストを追加

## Implementation

### 1. Add `ndt_tensor_f64_ones` (after line 122)

```rust
/// Create a new tensor filled with ones.
///
/// # Arguments
/// * `shape` - Pointer to array of dimensions
/// * `ndim` - Number of dimensions
/// * `status` - Pointer to receive status code
///
/// # Returns
/// Pointer to new tensor, or null on error
///
/// # Ownership
/// On success, the caller is responsible for releasing the tensor
/// by calling `ndt_tensor_f64_release`.
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_ones(
    shape: *const size_t,
    ndim: size_t,
    status: *mut StatusCode,
) -> *mut ndt_tensor_f64
```

Implementation: Same pattern as `zeros`, but call `Tensor::<f64>::ones(shape_slice)`

### 2. Add `ndt_tensor_f64_rand`

```rust
/// Create a new tensor with uniform random values in [0, 1).
///
/// # Arguments
/// * `shape` - Pointer to array of dimensions
/// * `ndim` - Number of dimensions
/// * `status` - Pointer to receive status code
///
/// # Returns
/// Pointer to new tensor, or null on error
///
/// # Ownership
/// On success, the caller is responsible for releasing the tensor
/// by calling `ndt_tensor_f64_release`.
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_rand(
    shape: *const size_t,
    ndim: size_t,
    status: *mut StatusCode,
) -> *mut ndt_tensor_f64
```

Implementation: Same pattern as `zeros`, but call `Tensor::<f64>::random(shape_slice)`

### 3. Add `ndt_tensor_f64_randn`

```rust
/// Create a new tensor with standard normal random values (mean=0, std=1).
///
/// # Arguments
/// * `shape` - Pointer to array of dimensions
/// * `ndim` - Number of dimensions
/// * `status` - Pointer to receive status code
///
/// # Returns
/// Pointer to new tensor, or null on error
///
/// # Ownership
/// On success, the caller is responsible for releasing the tensor
/// by calling `ndt_tensor_f64_release`.
#[unsafe(no_mangle)]
pub extern "C" fn ndt_tensor_f64_randn(
    shape: *const size_t,
    ndim: size_t,
    status: *mut StatusCode,
) -> *mut ndt_tensor_f64
```

Implementation: Same pattern as `zeros`, but call `Tensor::<f64>::randn(shape_slice)`

### 4. Add Tests (in `#[cfg(test)] mod tests`)

| Test | Description |
|------|-------------|
| `test_tensor_ones` | Create 2x3 tensor, verify all values are 1.0 |
| `test_tensor_ones_scalar` | Create scalar (ndim=0), verify value is 1.0 |
| `test_tensor_rand` | Create 2x3 tensor, verify values in [0, 1) |
| `test_tensor_rand_null_status` | Verify null status returns null |
| `test_tensor_randn` | Create tensor, verify values are roughly normal |
| `test_tensor_randn_null_shape_with_ndim` | Verify null shape with ndim>0 returns error |

## Verification

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test -p ndtensors-capi
cargo llvm-cov --all-features --workspace --fail-under-lines 80
```

## Notes
- Rust backend already has `Tensor::ones()`, `Tensor::random()`, `Tensor::randn()` implemented
- Follow exact pattern from `ndt_tensor_f64_zeros` (lines 81-122)
- Random functions use thread-local RNG (thread-safe)
