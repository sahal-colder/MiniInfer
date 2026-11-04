#!/usr/bin/env bash
set -euo pipefail
BENCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$BENCH_ROOT/run_suite.py" "$@"
