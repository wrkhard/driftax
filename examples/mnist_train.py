from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

import jax
import jax.numpy as jnp
import optax

from driftax.datasets import load_mnist_npz, preprocess_mnist
from driftax.models import DiTLatent2D, DiTLatent2DConfig, ClassEmbed, TinyConvEncoder
from driftax.drift import drifting_loss_features


def show_grid(imgs, nrow=10, title=""):
    imgs01 = (np.clip(imgs, -1, 1) + 1.0) * 0.5
    N, H, W, C = imgs01.shape
    ncol = int(np.ceil(N / nrow))
    grid = np.ones((ncol * H, nrow * W), dtype=np.float32)
    for i in range(N):
        r = i // nrow
        c = i % nrow
        grid[r*H:(r+1)*H, c*W:(c+1)*W] = imgs01[i,:,:,0]
    plt.figure(figsize=(nrow, ncol))
    plt.imshow(grid, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def make_class_index(labels_np, num_classes=10):
    idxs = [np.where(labels_np == c)[0] for c in range(num_classes)]
    counts = np.array([len(v) for v in idxs], dtype=np.int32)
    M = int(max(counts))
    table = np.zeros((num_classes, M), dtype=np.int32)
    for c in range(num_classes):
        v = idxs[c]
        table[c, :len(v)] = v
        table[c, len(v):] = v[0] if len(v) else 0
    return table, counts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--plot_every", type=int, default=200)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--samples_per_class", type=int, default=3)
    args = p.parse_args()

    print("JAX devices:", jax.devices())

    x_train_u8, y_train_u8, *_ = load_mnist_npz("data")
    x_train = preprocess_mnist(x_train_u8)
    y_train = y_train_u8.astype(np.int32)

    x_train_d = jnp.asarray(x_train)

    class_table, class_counts = make_class_index(y_train, 10)
    class_table_d = jnp.asarray(class_table)
    class_counts_d = jnp.asarray(class_counts).astype(jnp.float32)

    cfg = DiTLatent2DConfig(h=28, w=28, ch=1, patch=2, dim=256, depth=6, heads=8, cond_dim=256)
    gen = DiTLatent2D(cfg)
    class_emb = ClassEmbed(num_classes=10, out_dim=cfg.cond_dim)
    phi = TinyConvEncoder(base=32)

    key = jax.random.PRNGKey(0)
    key, k0, k1, k2 = jax.random.split(key, 4)
    dummy_z = jnp.zeros((1, cfg.h, cfg.w, cfg.ch), dtype=jnp.float32)
    dummy_cls = jnp.zeros((1,), dtype=jnp.int32)

    params = {}
    params["class_emb"] = class_emb.init(k0, dummy_cls)
    cond0 = class_emb.apply(params["class_emb"], dummy_cls)
    params["gen"] = gen.init(k1, dummy_z, cond0, train=True)
    params["phi"] = phi.init(k2, jnp.zeros((1, 28, 28, 1), dtype=jnp.float32), train=False)

    opt = optax.adamw(args.lr)
    opt_state = opt.init(params)
    temps = (0.02, 0.05, 0.2)

    def sample_pos_images(key, cls):
        u = jax.random.uniform(key, cls.shape, dtype=jnp.float32)
        cnt = jnp.take(class_counts_d, cls)
        r = jnp.minimum((u * cnt).astype(jnp.int32), (cnt.astype(jnp.int32) - 1))
        idx = class_table_d[cls, r]
        return x_train_d[idx]

    def loss_fn(params, key):
        key, kz, kc, kp = jax.random.split(key, 4)
        cls = jax.random.randint(kc, (args.batch,), 0, 10)
        cond = class_emb.apply(params["class_emb"], cls)
        z = jax.random.normal(kz, (args.batch, cfg.h, cfg.w, cfg.ch), dtype=jnp.float32)
        x_gen = gen.apply(params["gen"], z, cond, train=True)
        x_pos = sample_pos_images(kp, cls)

        fx_list = phi.apply(params["phi"], x_gen, train=False)
        fp_list = phi.apply(params["phi"], x_pos, train=False)

        loss = 0.0
        for fx, fp in zip(fx_list, fp_list):
            fxv = jnp.mean(fx, axis=(1, 2))
            fpv = jnp.mean(fp, axis=(1, 2))
            loss = loss + drifting_loss_features(
                x_feat=fxv, pos_feat=fpv,
                temps=temps, neg_feat=fxv,
                feature_normalize=True, drift_normalize=True,
            )
        return loss / float(len(fx_list))

    @jax.jit
    def step_fn(params, opt_state, key):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, key))(params)
        updates, opt_state2 = opt.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, key, loss

    def sample_grid(params, key):
        cls = jnp.repeat(jnp.arange(10, dtype=jnp.int32), args.samples_per_class)
        cond = class_emb.apply(params["class_emb"], cls)
        z = jax.random.normal(key, (cls.shape[0], cfg.h, cfg.w, cfg.ch), dtype=jnp.float32)
        x = gen.apply(params["gen"], z, cond, train=False)
        return np.array(x)

    loss_hist = []
    for s in trange(1, args.steps + 1, desc="train[mnist]"):
        params, opt_state, key, loss = step_fn(params, opt_state, key)
        loss_hist.append(float(loss))

        if s == 1 or (args.plot_every and s % args.plot_every == 0) or s == args.steps:
            key, kvis = jax.random.split(key)
            imgs = sample_grid(params, kvis)
            show_grid(imgs, nrow=10, title=f"Generated digits (step {s})")

    loss_hist = np.asarray(loss_hist, dtype=np.float32)
    plt.figure(figsize=(6, 3))
    plt.plot(loss_hist, alpha=0.8)
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.title("MNIST drifting loss")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
