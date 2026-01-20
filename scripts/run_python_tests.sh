#!/bin/bash
set -euo pipefail

# Run Python tests for ndtensors-rs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_PKG_DIR="$PROJECT_ROOT/python/ndtensors_rs"

echo "=== Setting up Python environment ==="
cd "$PYTHON_PKG_DIR"

# Sync dependencies with uv (creates venv and installs deps)
uv sync --extra dev

echo "=== Running Python tests ==="
# conftest.py automatically builds Rust library if needed
uv run pytest -v

echo "=== All Python tests passed ==="
