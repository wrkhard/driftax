#!/usr/bin/env bash
set -euo pipefail
driftjax-train-toy --dataset swiss_roll --steps 2000 --temp 0.07 --plot_every 500
