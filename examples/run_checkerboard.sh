#!/usr/bin/env bash
set -euo pipefail
driftjax-train-toy --dataset checkerboard --steps 2000 --temp 0.05 --plot_every 500
