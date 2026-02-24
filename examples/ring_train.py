from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

import jax
import jax.numpy as jnp
import optax

from driftax.datasets import inverse_ring_toy
from driftax.models import DiT1D, DiT1DConfig, CondMLP
from driftax.drift import drifting_loss_conditional_features


def plot_full_gt_slices_and_gen(
    xt: np.ndarray,
    yt: np.ndarray,
    xg_list: list[np.ndarray],
    y0_list: list[float],
    *,
    slice_tol: float = 0.05,
):
    """Row of 3 panels: full GT ring (faint) + GT slice (black) + generated (orange)."""
    fig, axes = plt.subplots(1, len(y0_list), figsize=(4 * len(y0_list), 4), sharex=False, sharey=False)
    if len(y0_list) == 1:
        axes = [axes]

    for i, (ax, y0, xg) in enumerate(zip(axes, y0_list, xg_list)):
        ax.scatter(xt[:, 0], xt[:, 1], s=2, alpha=0.08, c="black", label="GT ring p(x)" if i == 0 else None)

        m = np.abs(yt - y0) < slice_tol
        ax.scatter(xt[m, 0], xt[m, 1], s=2, alpha=0.25, c="black", label="GT slice ~p(x|y)" if i == 0 else None)

        ax.scatter(xg[:, 0], xg[:, 1], s=2, alpha=0.25, c="tab:orange", label="gen q(x|y)" if i == 0 else None)

        ax.set_title(f"y={y0:+.2f}")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)

    axes[0].legend(loc="lower left", markerscale=3)
    fig.tight_layout()
    plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--plot_every", type=int, default=500)
    p.add_argument("--lr", type=float, default=2e-4)

    # continuous conditioning strength
    p.add_argument("--temp_y", type=float, default=0.05, help="Smaller => stronger conditioning in y.")
    p.add_argument("--y0_list", type=str, default="-0.6,0.0,0.6")
    p.add_argument("--slice_tol", type=float, default=0.05, help="For GT slice visualization only.")
    p.add_argument("--n_gt", type=int, default=80000)
    p.add_argument("--n_gen", type=int, default=2000)
    args = p.parse_args()

    print("JAX devices:", jax.devices())

    y0_list = [float(v) for v in args.y0_list.split(",")]

    cfg = DiT1DConfig(length=2, patch=1, dim=256, depth=6, heads=4, cond_dim=256, drop=0.0)
    model = DiT1D(cfg)
    cond_net = CondMLP(in_dim=1, out_dim=cfg.cond_dim, hidden=256)

    key = jax.random.PRNGKey(1)
    key, k1, k2 = jax.random.split(key, 3)

    dummy_noise = jnp.zeros((1, 2), dtype=jnp.float32)
    dummy_y = jnp.zeros((1, 1), dtype=jnp.float32)
    cond_params = cond_net.init(k1, dummy_y)
    cond0 = cond_net.apply(cond_params, dummy_y)
    model_params = model.init(k2, dummy_noise, cond0, train=True)

    params = {"cond": cond_params, "model": model_params}
    opt = optax.adamw(args.lr)
    opt_state = opt.init(params)

    temps_x = (0.02, 0.05, 0.2)

    def loss_fn(params, key):
        key, kdata, kz = jax.random.split(key, 3)

        # paired real samples (pos_x, pos_y)
        pos_x, pos_y = inverse_ring_toy(kdata, args.batch)  # [B,2], [B,1]

        # generate conditioned on the *real* pos_y (continuous conditioning)
        y_in = pos_y
        c = cond_net.apply(params["cond"], y_in)
        z = jax.random.normal(kz, (args.batch, 2), dtype=jnp.float32)
        x_gen = model.apply(params["model"], z, c, train=True)

        # drift in x-space, but with y-aware kernel so cross-y mixing is suppressed
        return drifting_loss_conditional_features(
            x_feat=x_gen,
            y=y_in,
            pos_feat=pos_x,
            pos_y=pos_y,
            temps_x=temps_x,
            temp_y=float(args.temp_y),
            neg_feat=x_gen,
            neg_y=y_in,
            feature_normalize=True,
            drift_normalize=True,
        )

    @jax.jit
    def step_fn(params, opt_state, key):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, key))(params)
        updates, opt_state2 = opt.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, key, loss

    # fixed GT snapshot for visualization
    kgt = jax.random.PRNGKey(123)
    x_true, y_true = inverse_ring_toy(kgt, args.n_gt)
    xt = np.array(x_true)
    yt = np.array(y_true).squeeze(-1)

    def sample_generated(params, key, y0: float):
        y_in = jnp.full((args.n_gen, 1), y0, dtype=jnp.float32)
        c = cond_net.apply(params["cond"], y_in)
        z = jax.random.normal(key, (args.n_gen, 2), dtype=jnp.float32)
        xg = model.apply(params["model"], z, c, train=False)
        return np.array(xg)

    loss_hist = []
    for s in trange(1, args.steps + 1, desc="train[ring-continuous]"):
        params, opt_state, key, loss = step_fn(params, opt_state, key)
        loss_hist.append(float(loss))

        if s == 1 or (args.plot_every and s % args.plot_every == 0) or s == args.steps:
            print(f"step {s} loss {float(loss):.3e}  (temp_y={args.temp_y})")
            xg_list = []
            for i, y0 in enumerate(y0_list):
                key, kvis = jax.random.split(key)
                xg_list.append(sample_generated(params, kvis, y0))
            plot_full_gt_slices_and_gen(xt, yt, xg_list, y0_list, slice_tol=args.slice_tol)

    loss_hist = np.asarray(loss_hist, dtype=np.float32)
    plt.figure(figsize=(6, 3))
    plt.plot(loss_hist, alpha=0.8)
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.title("Ring drifting loss (continuous y-aware)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
