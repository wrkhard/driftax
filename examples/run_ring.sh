#!/usr/bin/env bash
set -euo pipefail
uv run python examples/ring_train.py "$@"
