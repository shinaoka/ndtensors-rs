# Agent Guidelines for ndtensors-rs

Read before starting work:
- `README.md` - Project overview
- `docs/design.md` - Design decisions, NDTensors.jl architecture reference, and migration checklist

## Core Principle: NDTensors.jl Compatibility

**This project is a technical feasibility study for replacing NDTensors.jl's backend with Rust.**

For this to be meaningful:
- **Data structures must mirror NDTensors.jl** (Storage trait hierarchy, Tensor wrapper, etc.)
- **Dispatch hierarchy must match** (High-level API → Storage-specific → Backend)
- **Module organization should parallel Julia's** for easy comparison

See "NDTensors.jl Architecture Reference" section in `docs/design.md` for detailed mapping.

**Rationale**: Without structural compatibility, migration from Julia to Rust backend becomes impractical, defeating the project's purpose.

### Generic Design Requirements

**Keep code generic over Storage types and Backends.** This mirrors NDTensors.jl's architecture:

| NDTensors.jl | ndtensors-rs | Description |
|--------------|--------------|-------------|
| `Dense{ElT, DataT}` | `Dense<T, D: DataBuffer<T>>` | Storage with backend-agnostic data |
| `BlockSparse{...}` | `BlockSparse<T, D>` (future) | Sparse block storage |
| `Vector{T}` | `CpuBuffer<T>` | CPU data backend |
| `CuVector{T}` | `CudaBuffer<T>` (future) | CUDA GPU backend |
| `AbstractArray` dispatch | `DataBuffer` trait | Backend abstraction |

**Guidelines:**
- Operations should be generic over `DataBuffer` trait, not hardcoded to `Vec<T>`
- Backend-specific optimizations (e.g., faer for CPU) go in specialized impls
- Use cargo features for optional backends: `default = ["cpu"]`, `cuda`, `metal`
- CPU-only code paths must not break when GPU backends are added later

## Development Stage

**Early development** - no backward compatibility required. Remove deprecated code immediately.

## General Guidelines

- Use same language as past conversations (Japanese if previous was Japanese)
- Source code and docs in English

### Reference Sources

**Prefer local `extern/` directory over web sources.** This repository includes local copies of reference projects:

```
extern/
├── ITensors.jl/     # NDTensors.jl source (in NDTensors/)
├── faer/            # faer linear algebra library
└── ...
```

When investigating NDTensors.jl or faer implementation details:
1. **First**: Check `extern/` for local source files
2. **Only if not available locally**: Use web search or fetch

This ensures consistent references and works offline.

## Code Style

`cargo fmt` for formatting, `cargo clippy --workspace --all-targets -- -D warnings` for linting. Avoid `unwrap()`/`expect()` in library code.

**Always run `cargo fmt --all` before committing changes.**

## Error Handling

- `anyhow` for internal error handling and context
- `thiserror` for public API error types

## Testing

```bash
cargo test                    # Full suite
cargo test --test test_name   # Specific test
```

- Private functions: `#[cfg(test)]` module in source file
- Integration tests: `tests/` directory
- **Test tolerance changes**: When relaxing test tolerances, always seek explicit user approval.

## API Design

Only make functions `pub` when truly public API.

### C API Documentation

For C API functions that return pointers to newly allocated objects:
- **Document ownership in docstrings.** Clearly state that the caller is responsible for releasing the returned objects.
- Example:
  ```rust
  /// # Ownership
  /// On success, `result_out` will point to a newly allocated tensor.
  /// The caller is responsible for releasing this tensor by calling
  /// `ndt_tensor_f64_release` when it is no longer needed.
  ```

### Layering and Maintainability

**Respect crate boundaries and abstraction layers.**

- **Never access low-level APIs or internal data structures from downstream crates.** Use high-level public methods instead.
- **Use high-level APIs.** If downstream code needs low-level access, create appropriate high-level APIs rather than exposing internal details.

**This applies to both library code and test code.**

### Code Deduplication

- **Avoid duplicate test code.** Use macros, functions, or generic functions to share test logic.
- **Example pattern for testing f64/Complex64:**

```rust
fn test_op_generic<T: Scalar + From<f64>>() { /* test */ }

#[test]
fn test_op_f64() { test_op_generic::<f64>(); }
#[test]
fn test_op_c64() { test_op_generic::<Complex64>(); }
```

## Git Workflow

**Never push/create PR without user approval.**

### Pre-PR Checks

Before creating a PR, always run lint checks locally:

```bash
cargo fmt --all          # Format all code
cargo clippy --workspace --all-targets -- -D warnings             # Check for common issues
cargo test               # Run all tests
```

| Change Type | Workflow |
|-------------|----------|
| Minor fixes | Branch + PR with auto-merge |
| Large features | Worktree + PR with auto-merge |

```bash
# Minor: branch workflow
git checkout -b fix-name && git add -A && git commit -m "msg"
cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings  # Lint before push
git push -u origin fix-name
gh pr create --base main --title "Title" --body "Desc"
gh pr merge --auto --squash --delete-branch

# Check PR before update
gh pr view <NUM> --json state  # Never push to merged PR

# Monitor CI
gh pr checks <NUM>
gh run view <RUN_ID> --log-failed
```

**Before creating PR**: Verify README.md is accurate (project structure, examples).
