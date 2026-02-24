from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

from driftax.datasets import sample_checkerboard, sample_swiss_roll
from driftax.drift import compute_V


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="checkerboard")
    p.add_argument("--temp", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    sampler = sample_checkerboard if args.dataset.lower().startswith("check") else sample_swiss_roll

    key = jax.random.PRNGKey(args.seed)
    key, k_gen, k_pos = jax.random.split(key, 3)

    gen = 0.5 * jax.random.normal(k_gen, (100, 2), dtype=jnp.float32)
    pos = sampler(k_pos, 500, noise=0.05)
    V = compute_V(gen, pos, gen, temp=args.temp)

    plt.figure(figsize=(8, 8))
    plt.scatter(np.array(pos)[:, 0], np.array(pos)[:, 1], s=5, alpha=0.3, c="blue", label="Data (p)")
    plt.scatter(np.array(gen)[:, 0], np.array(gen)[:, 1], s=30, c="orange", label="Generated (q)")
    plt.quiver(np.array(gen)[:, 0], np.array(gen)[:, 1], np.array(V)[:, 0], np.array(V)[:, 1],
               scale=3, color="black", alpha=0.7, width=0.004)
    plt.legend()
    plt.title("Drift vectors: generated points drift toward data")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
