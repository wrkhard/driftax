#!/usr/bin/env python
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

from driftax.datasets import load_mnist_npz, preprocess_mnist
from driftax.drift import drifting_loss_features
from driftax.models import DriftDiT2D, DiTLatent2DConfig



def show_grid_save(imgs: np.ndarray, out_png: Path, nrow: int = 10, title: str = ""):
    imgs01 = (np.clip(imgs, -1, 1) + 1.0) * 0.5
    H, W = imgs01.shape[1], imgs01.shape[2]
    grid = np.zeros((nrow * H, nrow * W), dtype=np.float32)
    for i in range(min(nrow * nrow, imgs01.shape[0])):
        r = i // nrow
        c = i % nrow
        grid[r * H : (r + 1) * H, c * W : (c + 1) * W] = imgs01[i, :, :, 0]
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def pad32(x: np.ndarray) -> np.ndarray:
    pad_val = -1.0 if float(x.min()) < -0.5 else 0.0
    pad = ((0, 0), (2, 2), (2, 2), (0, 0))
    return np.pad(x, pad, mode="constant", constant_values=pad_val).astype(np.float32)


def make_class_index(labels_np: np.ndarray, num_classes: int = 10):
    idxs = [np.where(labels_np == c)[0] for c in range(num_classes)]
    counts = np.array([len(v) for v in idxs], dtype=np.int32)
    M = int(max(counts))
    table = np.zeros((num_classes, M), dtype=np.int32)
    for c in range(num_classes):
        v = idxs[c]
        table[c, : len(v)] = v
        table[c, len(v) :] = v[0] if len(v) else 0
    return table, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="outputs/mnist_pixel_drift")
    ap.add_argument("--seed", type=int, default=0)

    # training schedule
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_steps", type=int, default=1000)
    ap.add_argument("--grad_clip", type=float, default=2.0)
    ap.add_argument("--ema_decay", type=float, default=0.999)

    # batch structure (per class)
    ap.add_argument("--batch_nc", type=int, default=10)
    ap.add_argument("--batch_n_pos", type=int, default=32)
    ap.add_argument("--batch_n_neg", type=int, default=32)

    # drifting temps
    ap.add_argument("--temps", type=float, nargs="+", default=[0.02, 0.05, 0.2])

    # CFG training/inference
    ap.add_argument("--label_dropout", type=float, default=0.1)
    ap.add_argument("--alpha_min", type=float, default=1.0)
    ap.add_argument("--alpha_max", type=float, default=3.0)
    ap.add_argument("--cfg_alpha", type=float, default=1.5)

    # plotting/logging
    ap.add_argument("--plot_every", type=int, default=500)
    ap.add_argument("--print_every", type=int, default=50)

    # model size
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--drop", type=float, default=0.0)

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[mnist_train] JAX devices:", jax.devices())

  
    x_train_u8, y_train_u8, *_ = load_mnist_npz(args.data_dir)
    x_train = preprocess_mnist(x_train_u8).astype(np.float32)
    y_train = y_train_u8.astype(np.int32)

    x_train32 = pad32(x_train)
    show_grid_save(
        x_train32[np.random.default_rng(0).choice(len(x_train32), 100, replace=False)],
        out_dir / "real_grid.png",
        title="Real MNIST (padded 32x32)",
    )

    x_train_d = jnp.asarray(x_train32)
    y_train_d = jnp.asarray(y_train)
    N = x_train_d.shape[0]

    class_table, class_counts = make_class_index(y_train, 10)
    class_table_d = jnp.asarray(class_table)
    class_counts_d = jnp.asarray(class_counts).astype(jnp.float32)

    def sample_real_by_class(key, cls_vec, n_pos: int):
        u = jax.random.uniform(key, (cls_vec.shape[0], n_pos), dtype=jnp.float32)
        cnt = jnp.take(class_counts_d, cls_vec)[:, None]
        r = jnp.minimum((u * cnt).astype(jnp.int32), (cnt.astype(jnp.int32) - 1))
        idx = class_table_d[cls_vec[:, None], r]
        return x_train_d[idx]  

  
    dit_cfg = DiTLatent2DConfig(
        h=32, w=32, ch= 1,
        patch= int(args.patch),
        dim=int(args.dim),
        depth=int(args.depth),
        heads=int(args.heads),
        cond_dim=256,
        num_context_tokens=0,
        drop=float(args.drop),
    )
    model = DriftDiT2D(dit_cfg=dit_cfg, num_classes=10, label_dropout=float(args.label_dropout))

    key = jax.random.PRNGKey(args.seed)
    key, kinit = jax.random.split(key, 2)

    B_init = int(args.batch_nc) * int(args.batch_n_neg)
    dummy_z = jnp.zeros((B_init, 32, 32, 1), dtype=jnp.float32)
    dummy_lab = jnp.zeros((B_init,), dtype=jnp.int32)
    dummy_alpha = jnp.ones((B_init,), dtype=jnp.float32)
    params = model.init({"params": kinit, "drop": kinit}, dummy_z, dummy_lab, dummy_alpha, train=True)

    # opt with warmup
    lr_sched = optax.join_schedules(
        [
            optax.linear_schedule(0.0, float(args.lr), int(args.warmup_steps)),
            optax.constant_schedule(float(args.lr)),
        ],
        boundaries=[int(args.warmup_steps)],
    )

    opt = optax.chain(
        optax.clip_by_global_norm(float(args.grad_clip)),
        optax.adamw(lr_sched, weight_decay=float(args.weight_decay)),
    )
    opt_state = opt.init(params)

    ema_params = params

    temps = tuple(float(t) for t in args.temps)
    nc = int(args.batch_nc)
    npos = int(args.batch_n_pos)
    nneg = int(args.batch_n_neg)

    def loss_fn(p, key_step):
        key_step, kc, kp, kz, ka, kd = jax.random.split(key_step, 6)

        cls_all = jnp.arange(10, dtype=jnp.int32)
        cls_vec = jax.random.choice(kc, cls_all, shape=(nc,), replace=False)

        x_pos = sample_real_by_class(kp, cls_vec, npos)

        z = jax.random.normal(kz, (nc, nneg, 32, 32, 1), dtype=jnp.float32)
        z_flat = z.reshape(nc * nneg, 32, 32, 1)

        labels = jnp.repeat(cls_vec, nneg)

        alpha = jax.random.uniform(
            ka,
            (nc * nneg,),
            minval=float(args.alpha_min),
            maxval=float(args.alpha_max),
            dtype=jnp.float32,
        )

        x_gen = model.apply(p, z_flat, labels, alpha, train=True, rngs={"drop": kd})
        x_gen = x_gen.reshape(nc, nneg, 32, 32, 1)

        def one_class(xg_c, xp_c):
            xg = xg_c.reshape(nneg, -1)
            xp = xp_c.reshape(npos, -1)
            return drifting_loss_features(
                x_feat=xg,
                pos_feat=xp,
                temps=temps,
                neg_feat=xg,
                feature_normalize=True,
                drift_normalize=True,
                mask_self_in_neg=True,
            )

        losses = jax.vmap(one_class)(x_gen, x_pos)
        return jnp.mean(losses)

    @jax.jit
    def step_fn(p, opt_state, ema_p, key_step):
        loss, grads = jax.value_and_grad(loss_fn)(p, key_step)
        updates, opt_state2 = opt.update(grads, opt_state, p)
        p2 = optax.apply_updates(p, updates)
        ema2 = jax.tree_util.tree_map(
            lambda e, x: float(args.ema_decay) * e + (1.0 - float(args.ema_decay)) * x,
            ema_p, p2
        )
        return p2, opt_state2, ema2, loss

    def sample_grid(ema_p, key_samp, alpha: float):
        cls = jnp.repeat(jnp.arange(10, dtype=jnp.int32), 10)
        z = jax.random.normal(key_samp, (cls.shape[0], 32, 32, 1), dtype=jnp.float32)
        x = model.apply(ema_p, z, cls, float(alpha), method=DriftDiT2D.forward_with_cfg)
        return np.array(jnp.clip(x, -1, 1))

    losses = []
    ema_loss = None

    for s in range(1, int(args.steps) + 1):
        key, kstep = jax.random.split(key)
        params, opt_state, ema_params, loss = step_fn(params, opt_state, ema_params, kstep)

        lv = float(loss)
        losses.append(lv)
        ema_loss = lv if ema_loss is None else 0.98 * ema_loss + 0.02 * lv

        if s == 1 or (s % int(args.print_every) == 0) or (s == int(args.steps)):
            print(f"[train] step {s:5d} loss={lv:.3e} (ema {ema_loss:.3e})")

        if s == 1 or (int(args.plot_every) and s % int(args.plot_every) == 0) or (s == int(args.steps)):
            key, ks = jax.random.split(key)
            imgs = sample_grid(ema_params, ks, alpha=float(args.cfg_alpha))
            show_grid_save(imgs, out_dir / f"samples_step{s:05d}.png", title=f"EMA samples (step {s})")

    plt.figure(figsize=(6, 3))
    plt.plot(np.asarray(losses, dtype=np.float32), alpha=0.9)
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("MNIST drifting loss (pixel space, class-conditional)")
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=180)
    plt.close()

    metrics = {
        "loss_last": float(losses[-1]) if losses else None,
        "steps":int(args.steps),
        "temps":[float(t) for t in temps],
        "batch_nc": nc,
        "batch_n_pos": npos,
        "batch_n_neg": nneg,
        "ema_decay": float(args.ema_decay),
        "cfg_alpha": float(args.cfg_alpha),
        "args": vars(args),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("[mnist_train] done. outputs in", out_dir)


if __name__ == "__main__":
    main()