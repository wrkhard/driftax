#!/usr/bin/env bash
set -euo pipefail
uv run python examples/mnist_train.py "$@"
