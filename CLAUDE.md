# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an unofficial experimental Rust port of NDTensors.jl, exploring technical feasibility of replacing ITensors.jl's backend with Rust.

Before working, read:
- `README.md` - Project overview and goals
- `docs/design.md` - Technical design decisions
- `AGENTS.md` - Development guidelines

## Critical Rules

- **Never push directly to main** - all changes via pull requests
- **Always run `cargo fmt --all` before committing**
- **Review Cargo.toml changes** - use `git diff Cargo.toml` to catch accidental dependency changes
- **Citation required** - This project requires citing ITensors.jl paper (see README.md)

@AGENTS.md
