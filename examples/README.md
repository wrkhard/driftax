# driftax examples

Run with `uv` from the repo root:

```bash
uv sync --extra dev
uv run python -m pip install -e .
uv run python examples/mnist_train.py --samples_per_class 3
uv run python examples/ring_train.py --y0 0.0
uv run python examples/toy2d_train_mlp.py --dataset checkerboard
uv run python examples/visualize_drift.py --dataset checkerboard --temp 0.2
```

Or use the provided wrappers:

```bash
bash examples/run_mnist.sh
bash examples/run_ring.sh --y0 0.0
bash examples/run_toy2d.sh --dataset swiss
bash examples/run_drift_quiver.sh --dataset checkerboard
```


Ring: try stronger conditioning with `--temp_y 0.03` (smaller => stronger).
