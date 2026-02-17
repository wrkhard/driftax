from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt
from tqdm import trange

from .datasets import get_sampler
from .models import MLPGenerator
from .drift import drifting_loss
from .ckpt import save_checkpoint


@dataclass
class TrainConfig:
    dataset: str = "checkerboard"
    steps: int = 2000
    data_batch_size: int = 2048
    gen_batch_size: int = 2048
    lr: float = 1e-3
    temp: float = 0.05
    z_dim: int = 32
    hidden: int = 256
    plot_every: int = 500
    seed: int = 42
    noise: float = 0.05
    out_dir: str = "runs"


def make_run_dir(cfg: TrainConfig) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.out_dir, f"{ts}_{cfg.dataset}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "ckpt"), exist_ok=True)
    return run_dir


def plot_samples(run_dir: str, step: int, gt: np.ndarray, gen: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    ax1.scatter(gt[:, 0], gt[:, 1], s=2, alpha=0.3, c="black")
    ax1.set_title("Target"); ax1.set_aspect("equal"); ax1.axis("off")
    ax2.scatter(gen[:, 0], gen[:, 1], s=2, alpha=0.3, c="tab:orange")
    ax2.set_title(f"Generated (step {step})"); ax2.set_aspect("equal"); ax2.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(run_dir, f"samples_step{step:05d}.png"), dpi=160)
    plt.close(fig)


def plot_loss(run_dir: str, loss_hist: np.ndarray) -> None:
    fig = plt.figure(figsize=(6, 3))
    plt.plot(loss_hist, alpha=0.8)
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.yscale("log"); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="checkerboard")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--data_batch_size", type=int, default=2048)
    p.add_argument("--gen_batch_size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temp", type=float, default=0.05)
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--plot_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise", type=float, default=0.05)
    p.add_argument("--out_dir", type=str, default="runs")
    args = p.parse_args()

    cfg = TrainConfig(
        dataset=args.dataset,
        steps=args.steps,
        data_batch_size=args.data_batch_size,
        gen_batch_size=args.gen_batch_size,
        lr=args.lr,
        temp=args.temp,
        z_dim=args.z_dim,
        hidden=args.hidden,
        plot_every=args.plot_every,
        seed=args.seed,
        noise=args.noise,
        out_dir=args.out_dir,
    )

    sampler = get_sampler(cfg.dataset)
    run_dir = make_run_dir(cfg)

    # init model + optimizer
    key = jax.random.PRNGKey(cfg.seed)
    key, key_init = jax.random.split(key)
    model = MLPGenerator(in_dim=cfg.z_dim, hidden=cfg.hidden, out_dim=2)
    params = model.init(key_init, jnp.zeros((1, cfg.z_dim), dtype=jnp.float32))

    opt = optax.adam(cfg.lr)
    opt_state = opt.init(params)

    @jax.jit
    def train_step(params, opt_state, key):
        key, key_pos, key_z = jax.random.split(key, 3)
        pos = sampler(key_pos, cfg.data_batch_size, noise=cfg.noise)
        z = jax.random.normal(key_z, (cfg.gen_batch_size, cfg.z_dim), dtype=jnp.float32)
        gen = model.apply(params, z)
        loss = drifting_loss(gen, pos, temp=cfg.temp)

        grads = jax.grad(lambda p: drifting_loss(model.apply(p, z), pos, temp=cfg.temp))(params)
        updates, opt_state2 = opt.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, key, loss

    loss_hist = []
    ema = None

    for step in trange(1, cfg.steps + 1, desc=f"train[{cfg.dataset}]"):
        params, opt_state, key, loss = train_step(params, opt_state, key)
        loss_val = float(loss)
        loss_hist.append(loss_val)
        ema = loss_val if ema is None else 0.96 * ema + 0.04 * loss_val

        if step % 50 == 0:
            trange(0).set_description(f"train[{cfg.dataset}] loss={ema:.2e}")

        if step == 1 or (cfg.plot_every > 0 and step % cfg.plot_every == 0):
            # sample for visualization
            key, key_gt, key_vis = jax.random.split(key, 3)
            gt = np.array(sampler(key_gt, 5000, noise=cfg.noise))
            z = jax.random.normal(key_vis, (5000, cfg.z_dim), dtype=jnp.float32)
            vis = np.array(model.apply(params, z))
            plot_samples(run_dir, step, gt, vis)

            save_checkpoint(
                os.path.join(run_dir, "ckpt", f"step{step:05d}.pkl"),
                {"params": params, "opt_state": opt_state, "step": step, "cfg": cfg.__dict__},
            )

    loss_arr = np.asarray(loss_hist, dtype=np.float32)
    np.save(os.path.join(run_dir, "loss.npy"), loss_arr)
    plot_loss(run_dir, loss_arr)
    print(f"Saved run to: {run_dir}")


if __name__ == "__main__":
    main()
