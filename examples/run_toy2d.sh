#!/usr/bin/env bash
set -euo pipefail
uv run python examples/toy2d_train_mlp.py "$@"
