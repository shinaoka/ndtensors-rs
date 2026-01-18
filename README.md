# ndtensors-rs

**Unofficial experimental Rust port of [NDTensors.jl](https://github.com/ITensor/ITensors.jl/tree/main/NDTensors)**

This is a technical feasibility study exploring whether NDTensors.jl can be ported to Rust while maintaining API compatibility, enabling ITensors.jl to use a Rust backend.

## Goal

```
Current:
  ITensors.jl → NDTensors.jl (Pure Julia)

Target:
  ITensors.jl → NDTensors.jl (Julia wrapper) → ndtensors-rs (Rust via C API)
```

## Why Rust?

- **Faster precompilation**: Eliminate JIT overhead, enable rapid trial-and-error (vibe coding) for ITensors.jl-dependent libraries
- **Maintainability**: Rust's type system catches bugs at compile time, making refactoring safer

## Key Challenges

- Zero-copy data sharing between Julia and Rust
- Automatic differentiation integration (ChainRules.jl)
- Performance parity with pure Julia implementation

## Design Document

See [docs/design.md](docs/design.md) for technical details.

## Citation

This is a port of **NDTensors.jl** from **ITensors.jl**. Please cite:

> M. Fishman, S. R. White, E. M. Stoudenmire, "The ITensor Software Library for Tensor Network Calculations", SciPost Phys. Codebases 4 (2022), arXiv:2007.14822

## License

MIT License (see [LICENSE-MIT](../../LICENSE-MIT))
