from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn

from driftax.datasets import sample_checkerboard, sample_swiss_roll
from driftax.drift import drifting_loss_features


class MLP(nn.Module):
    in_dim: int = 32
    hidden: int = 256
    out_dim: int = 2

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden)(z); x = nn.silu(x)
        x = nn.Dense(self.hidden)(x); x = nn.silu(x)
        x = nn.Dense(self.hidden)(x); x = nn.silu(x)
        x = nn.Dense(self.out_dim)(x)
        return x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="checkerboard")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--plot_every", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temp", type=float, default=0.05)
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--noise", type=float, default=0.05)
    args = p.parse_args()

    sampler = sample_checkerboard if args.dataset.lower().startswith("check") else sample_swiss_roll
    print("JAX devices:", jax.devices())

    key = jax.random.PRNGKey(42)
    key, k_init = jax.random.split(key)

    model = MLP(in_dim=args.z_dim, hidden=256, out_dim=2)
    params = model.init(k_init, jnp.zeros((1, args.z_dim), dtype=jnp.float32))
    opt = optax.adam(args.lr)
    opt_state = opt.init(params)

    @jax.jit
    def step_fn(params, opt_state, key):
        key, k_pos, k_z = jax.random.split(key, 3)
        pos = sampler(k_pos, 2048, noise=args.noise)
        z = jax.random.normal(k_z, (2048, args.z_dim), dtype=jnp.float32)
        gen = model.apply(params, z)

        loss = drifting_loss_features(
            x_feat=gen, pos_feat=pos,
            temps=(args.temp,), neg_feat=gen,
            feature_normalize=False, drift_normalize=False,
        )
        grads = jax.grad(lambda p: drifting_loss_features(
            x_feat=model.apply(p, z), pos_feat=pos,
            temps=(args.temp,), neg_feat=model.apply(p, z),
            feature_normalize=False, drift_normalize=False,
        ))(params)

        updates, opt_state2 = opt.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, key, loss

    loss_hist = []
    for s in trange(1, args.steps + 1, desc=f"train[toy2d-{args.dataset}]"):
        params, opt_state, key, loss = step_fn(params, opt_state, key)
        loss_hist.append(float(loss))

        if s == 1 or (args.plot_every and s % args.plot_every == 0):
            key, k_gt, k_vis = jax.random.split(key, 3)
            gt = np.array(sampler(k_gt, 5000, noise=args.noise))
            z = jax.random.normal(k_vis, (5000, args.z_dim), dtype=jnp.float32)
            vis = np.array(model.apply(params, z))

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
            ax1.scatter(gt[:, 0], gt[:, 1], s=2, alpha=0.3, c="black")
            ax1.set_title("Target"); ax1.set_aspect("equal"); ax1.axis("off")
            ax2.scatter(vis[:, 0], vis[:, 1], s=2, alpha=0.3, c="tab:orange")
            ax2.set_title(f"Generated (step {s})"); ax2.set_aspect("equal"); ax2.axis("off")
            plt.tight_layout()
            plt.show()

    loss_hist = np.asarray(loss_hist, dtype=np.float32)
    plt.figure(figsize=(6, 3))
    plt.plot(loss_hist, alpha=0.8)
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.title(f"Toy drifting loss ({args.dataset})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
