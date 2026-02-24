#!/usr/bin/env python
from __future__ import annotations

import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax
import flax

from driftax.datasets import load_mnist_npz, preprocess_mnist
from driftax.drift import drifting_loss_features
from driftax.models import (
    DiTLatent2D, DiTLatent2DConfig, ClassEmbed,
    SDVAEConfig, SDVAETokenizer,
    ResNetGNConfig, ResNetGNClassifier, ResNetGNEncoder,
)


def show_grid_save(imgs: np.ndarray, out_png: Path, nrow: int = 10, title: str = ""):
    # imgs: [-1,1], [N,H,W,1]
    imgs01 = (np.clip(imgs, -1, 1) + 1.0) * 0.5
    H, W = imgs01.shape[1], imgs01.shape[2]
    grid = np.zeros((nrow * H, nrow * W), dtype=np.float32)
    for i in range(min(nrow*nrow, imgs01.shape[0])):
        r = i // nrow
        c = i % nrow
        grid[r*H:(r+1)*H, c*W:(c+1)*W] = imgs01[i, :, :, 0]
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.title(title)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def pad32(x: np.ndarray) -> np.ndarray:
    # x: [N,28,28,1], in [-1,1] typically
    pad_val = -1.0 if float(x.min()) < -0.5 else 0.0
    pad = ((0, 0), (2, 2), (2, 2), (0, 0))
    return np.pad(x, pad, mode="constant", constant_values=pad_val).astype(np.float32)


def feats_to_vecs(feats, grid: int = 4):
    # feats: list of [B,H,W,C] -> list of [B, grid*grid*C]
    out = []
    for f in feats:
        B, H, W, C = f.shape
        ys = jnp.linspace(0, H - 1, grid).round().astype(jnp.int32)
        xs = jnp.linspace(0, W - 1, grid).round().astype(jnp.int32)
        yy, xx = jnp.meshgrid(ys, xs, indexing="ij")
        pts = f[:, yy.reshape(-1), xx.reshape(-1), :]  # [B, g*g, C]
        out.append(pts.reshape(B, -1))
    return out


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
    ap.add_argument("--data_dir", type=str, default="data", help="directory containing mnist.npz or where to download")
    ap.add_argument("--out_dir", type=str, default="outputs/mnist_tok", help="output directory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tok_steps", type=int, default=3000)
    ap.add_argument("--phi_steps", type=int, default=1500)
    ap.add_argument("--gen_steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--tok_batch", type=int, default=256)
    ap.add_argument("--phi_batch", type=int, default=256)
    ap.add_argument("--plot_every", type=int, default=500)
    ap.add_argument("--print_every", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Devices
    print("[mnist_train] JAX devices:", jax.devices())

    # Load MNIST
    x_train_u8, y_train_u8, x_test_u8, y_test_u8 = load_mnist_npz(args.data_dir)
    x_train = preprocess_mnist(x_train_u8).astype(np.float32)  # expected [-1,1], [N,28,28,1]
    y_train = y_train_u8.astype(np.int32)

    x_train32 = pad32(x_train)
    np.random.default_rng(0)
    show_grid_save(x_train32[np.random.default_rng(0).choice(len(x_train32), 100, replace=False)],
                   out_dir / "real_grid.png", title="Real MNIST (padded 32x32)")

    x_train_d = jnp.asarray(x_train32)
    y_train_d = jnp.asarray(y_train)
    N = x_train_d.shape[0]

    key = jax.random.PRNGKey(args.seed)
    key, kinit = jax.random.split(key)

    
    tok_cfg = SDVAEConfig(in_ch=1, z_ch=4, base_ch=64, ch_mult=(1, 2, 4), num_res_blocks=2, dropout=0.0)
    tok = SDVAETokenizer(tok_cfg)
    tok_params = tok.init(kinit, jnp.zeros((1, 32, 32, 1), jnp.float32), key, train=True)

    lr_tok = 2e-4
    opt_tok = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr_tok, weight_decay=1e-4))
    tok_state = opt_tok.init(tok_params)

    def kl_normal(mean, logvar):
        return 0.5 * jnp.mean(jnp.sum(jnp.exp(logvar) + mean**2 - 1.0 - logvar, axis=-1))

    @jax.jit
    def tok_step(params, opt_state, key_step, xb):
        def loss_fn(p):
            z, mean, logvar, xhat = tok.apply(p, xb, key_step, train=True)
            recon = jnp.mean((xhat - xb) ** 2)
            kl = kl_normal(mean, logvar)
            return recon + 1e-3 * kl, (recon, kl)
        (loss, (recon, kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state2 = opt_tok.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, loss, recon, kl

    rng = np.random.default_rng(args.seed)
    tok_losses = []
    for s in range(1, args.tok_steps + 1):
        idx = rng.integers(0, N, size=(args.tok_batch,))
        xb = x_train_d[idx]
        key, kstep = jax.random.split(key)
        tok_params, tok_state, loss, recon, kl = tok_step(tok_params, tok_state, kstep, xb)
        tok_losses.append(float(loss))
        if s == 1 or (s % (args.print_every * 3) == 0) or s == args.tok_steps:
            print(f"[tok] step {s:4d} loss={float(loss):.4e} recon={float(recon):.4e} kl={float(kl):.4e}")

    # recon viz
    key, kvis = jax.random.split(key)
    idx = rng.choice(N, size=100, replace=False)
    xb = x_train_d[idx]
    z, mean, logvar, xhat = tok.apply(tok_params, xb, kvis, train=False)
    show_grid_save(np.array(xhat), out_dir / "tok_recon.png", title="Tokenizer reconstructions")

  
    key, kinit = jax.random.split(key)
    phi_cfg = ResNetGNConfig(in_ch=1, base_ch=64, num_blocks=(2, 2, 2, 2), num_classes=10)
    phi_cls = ResNetGNClassifier(phi_cfg)
    phi_enc = ResNetGNEncoder(phi_cfg)
    phi_params = phi_cls.init(kinit, jnp.zeros((1, 32, 32, 1), jnp.float32))

    lr_phi = 2e-4
    opt_phi = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr_phi, weight_decay=1e-4))
    phi_state = opt_phi.init(phi_params)

    @jax.jit
    def phi_step(params, opt_state, xb, yb):
        def loss_fn(p):
            logits = phi_cls.apply(p, xb)
            loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, yb))
            acc = jnp.mean(jnp.argmax(logits, axis=-1) == yb)
            return loss, acc
        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state2 = opt_phi.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, loss, acc

    phi_losses, phi_accs = [], []
    for s in range(1, args.phi_steps + 1):
        idx = rng.integers(0, N, size=(args.phi_batch,))
        xb = x_train_d[idx]
        yb = y_train_d[idx]
        phi_params, phi_state, loss, acc = phi_step(phi_params, phi_state, xb, yb)
        phi_losses.append(float(loss)); phi_accs.append(float(acc))
        if s == 1 or (s % (args.print_every * 3) == 0) or s == args.phi_steps:
            print(f"[phi] step {s:4d} ce={float(loss):.4f} acc={float(acc):.3f}")

 
    cfg = DiTLatent2DConfig(h=8, w=8, ch=4, patch=2, dim=256, depth=6, heads=8, cond_dim=256, num_context_tokens=0, drop=0.0)
    gen = DiTLatent2D(cfg)
    class_emb = ClassEmbed(num_classes=10, out_dim=cfg.cond_dim)

    key, k0, k1 = jax.random.split(key, 3)
    dummy_z = jnp.zeros((1, cfg.h, cfg.w, cfg.ch), dtype=jnp.float32)
    dummy_cls = jnp.zeros((1,), dtype=jnp.int32)
    gen_params = {}
    gen_params["class_emb"] = class_emb.init(k0, dummy_cls)
    cond0 = class_emb.apply(gen_params["class_emb"], dummy_cls)
    gen_params["gen"] = gen.init(k1, dummy_z, cond0, train=True)

    lr_gen = 1e-4
    opt_gen = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr_gen, weight_decay=1e-4))
    gen_state = opt_gen.init(gen_params)

    class_table, class_counts = make_class_index(y_train, 10)
    class_table_d = jnp.asarray(class_table)
    class_counts_d = jnp.asarray(class_counts).astype(jnp.float32)

    def sample_pos_images(keyp, cls):
        u = jax.random.uniform(keyp, cls.shape, dtype=jnp.float32)
        cnt = jnp.take(class_counts_d, cls)
        r = jnp.minimum((u * cnt).astype(jnp.int32), (cnt.astype(jnp.int32) - 1))
        idx = class_table_d[cls, r]
        return x_train_d[idx]

    temps = (0.05,)

    def loss_fn(params, key_step):
        key_step, kz, kc, kp = jax.random.split(key_step, 4)
        cls = jax.random.randint(kc, (args.batch,), 0, 10)
        cond = class_emb.apply(params["class_emb"], cls)

        z = jax.random.normal(kz, (args.batch, cfg.h, cfg.w, cfg.ch), dtype=jnp.float32)
        lat_gen = gen.apply(params["gen"], z, cond, train=True)

        x_gen = tok.apply(tok_params, lat_gen, train=False, method=SDVAETokenizer.decode)
        x_pos = sample_pos_images(kp, cls)

        fx = phi_enc.apply(phi_params, x_gen)
        fp = phi_enc.apply(phi_params, x_pos)
        fxv_list = feats_to_vecs(fx, grid=4)
        fpv_list = feats_to_vecs(fp, grid=4)

        loss = 0.0
        for fxv, fpv in zip(fxv_list, fpv_list):
            loss = loss + drifting_loss_features(
                x_feat=fxv,
                pos_feat=fpv,
                temps=temps,
                neg_feat=fxv,
                feature_normalize=True,
                drift_normalize=False,
            )
        return loss / float(len(fxv_list))

    @jax.jit
    def gen_step(params, opt_state, key_step):
        loss, grads = jax.value_and_grad(lambda p: loss_fn(p, key_step))(params)
        updates, opt_state2 = opt_gen.update(grads, opt_state, params)
        params2 = optax.apply_updates(params, updates)
        return params2, opt_state2, loss

    def sample_grid(params, key_samp):
        cls = jnp.repeat(jnp.arange(10, dtype=jnp.int32), 10)
        cond = class_emb.apply(params["class_emb"], cls)
        z = jax.random.normal(key_samp, (cls.shape[0], cfg.h, cfg.w, cfg.ch), dtype=jnp.float32)
        lat = gen.apply(params["gen"], z, cond, train=False)
        x = tok.apply(tok_params, lat, train=False, method=SDVAETokenizer.decode)
        return np.array(x)

    losses = []
    for s in range(1, args.gen_steps + 1):
        key, kstep = jax.random.split(key)
        gen_params, gen_state, loss = gen_step(gen_params, gen_state, kstep)
        lv = float(loss)
        losses.append(lv)

        if s == 1 or (s % args.print_every == 0) or s == args.gen_steps:
            print(f"[gen] step {s:4d} loss {lv:.3e}")

        if s == 1 or (args.plot_every and s % args.plot_every == 0) or s == args.gen_steps:
            key, kvis = jax.random.split(key)
            imgs = sample_grid(gen_params, kvis)
            show_grid_save(imgs, out_dir / f"samples_step{s:05d}.png", title=f"Generated digits (step {s})")

    # loss plot
    plt.figure(figsize=(6, 3))
    plt.plot(np.asarray(losses, dtype=np.float32), alpha=0.9)
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.title("MNIST drifting loss (SDVAE + ResNetGN φ)")
    plt.tight_layout()
    plt.savefig(out_dir / "loss.png", dpi=180)
    plt.close()

    # Save a light metadata bundle
    metrics = {
        "tok_loss_last": float(tok_losses[-1]) if tok_losses else None,
        "phi_ce_last": float(phi_losses[-1]) if phi_losses else None,
        "phi_acc_last": float(phi_accs[-1]) if phi_accs else None,
        "gen_loss_last": float(losses[-1]) if losses else None,
        "args": vars(args),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save checkpoints as flax state dict (optional)
    import pickle
    ckpt = {
        "tok": flax.serialization.to_state_dict(tok_params),
        "phi": flax.serialization.to_state_dict(phi_params),
        "gen": flax.serialization.to_state_dict(gen_params),
    }
    with open(out_dir / "checkpoint_state_dict.pkl", "wb") as f:
        pickle.dump(ckpt, f)

    print("[mnist_train] done. outputs in", out_dir)


if __name__ == "__main__":
    main()
