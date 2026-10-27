#!/bin/bash
# Run the benchmarks on the pandas versions the README quotes. Needs uv.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python benchmarks/run.py "$@"
uv run --with pyarrow python benchmarks/run.py "$@"
uv run --isolated --python 3.12 --with "pandas==2.2.3" --with-editable . python benchmarks/run.py "$@"
