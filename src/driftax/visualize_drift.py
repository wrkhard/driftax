from __future__ import annotations

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .datasets import get_sampler
from .drift import compute_drift


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="checkerboard")
    p.add_argument("--temp", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise", type=float, default=0.05)
    args = p.parse_args()

    sampler = get_sampler(args.dataset)

    key = jax.random.PRNGKey(args.seed)
    key, k_gen, k_pos = jax.random.split(key, 3)

    gen = 0.5 * jax.random.normal(k_gen, (100, 2), dtype=jnp.float32)
    pos = sampler(k_pos, 500, noise=args.noise)

    drift = compute_drift(gen, pos, temp=args.temp)

    gen_np = np.array(gen)
    pos_np = np.array(pos)
    drift_np = np.array(drift)

    plt.figure(figsize=(8, 8))
    plt.scatter(pos_np[:, 0], pos_np[:, 1], s=5, alpha=0.3, c="blue", label="Data (p)")
    plt.scatter(gen_np[:, 0], gen_np[:, 1], s=30, c="orange", label="Generated (q)")
    plt.quiver(gen_np[:, 0], gen_np[:, 1], drift_np[:, 0], drift_np[:, 1],
               scale=3, color="black", alpha=0.7, width=0.004)
    plt.legend()
    plt.title("Drift Vectors: Generated points drift toward data")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
