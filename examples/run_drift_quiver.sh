#!/usr/bin/env bash
set -euo pipefail
uv run python examples/visualize_drift.py "$@"
