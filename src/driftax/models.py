from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, List

import flax.linen as nn
import jax
import jax.numpy as jnp


# ============================================================
# Core Transformer building blocks (DiT-like)
# ============================================================

class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (self.dim,))
        norm = jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x / norm * scale


def apply_rope(x: jnp.ndarray, *, base: float = 10000.0) -> jnp.ndarray:
    """1D RoPE for [B,T,H,Dh] or [B,T,D]."""
    if x.ndim == 4:
        _, t, _, dh = x.shape
        x_ = x
    elif x.ndim == 3:
        _, t, d = x.shape
        dh = d
        x_ = x[:, :, None, :]
    else:
        raise ValueError(f"RoPE expects 3D/4D input, got {x.shape}")

    half = dh // 2
    freq_seq = jnp.arange(half, dtype=jnp.float32)
    inv_freq = 1.0 / (base ** (freq_seq / float(half)))
    pos = jnp.arange(t, dtype=jnp.float32)
    angles = pos[:, None] * inv_freq[None, :]  # [T, half]
    sin = jnp.sin(angles)[None, :, None, :]
    cos = jnp.cos(angles)[None, :, None, :]

    x1 = x_[..., :half]
    x2 = x_[..., half:2 * half]
    rot1 = x1 * cos - x2 * sin
    rot2 = x1 * sin + x2 * cos
    out = jnp.concatenate([rot1, rot2, x_[..., 2 * half :]], axis=-1)

    if x.ndim == 3:
        out = out[:, :, 0, :]
    return out


class MultiHeadSelfAttention(nn.Module):
    dim: int
    num_heads: int
    qk_norm: bool = True
    rope: bool = True
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        b, t, d = x.shape
        head_dim = d // self.num_heads
        assert head_dim * self.num_heads == d

        qkv = nn.Dense(3 * d, use_bias=False)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape(b, t, self.num_heads, head_dim)
        k = k.reshape(b, t, self.num_heads, head_dim)
        v = v.reshape(b, t, self.num_heads, head_dim)

        if self.rope:
            q = apply_rope(q)
            k = apply_rope(k)

        if self.qk_norm:
            q = q / (jnp.sqrt(jnp.sum(q * q, axis=-1, keepdims=True) + 1e-6))
            k = k / (jnp.sqrt(jnp.sum(k * k, axis=-1, keepdims=True) + 1e-6))

        attn = jnp.einsum("bthd,bshd->bhts", q, k) / math.sqrt(head_dim)
        attn = jax.nn.softmax(attn, axis=-1)
        if self.attn_drop > 0:
            attn = nn.Dropout(self.attn_drop)(attn, deterministic=not train)

        out = jnp.einsum("bhts,bshd->bthd", attn, v).reshape(b, t, d)
        out = nn.Dense(d)(out)
        if self.proj_drop > 0:
            out = nn.Dropout(self.proj_drop)(out, deterministic=not train)
        return out


class SwiGLU(nn.Module):
    dim: int
    mult: float = 4.0
    drop: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        hidden = int(self.dim * self.mult)
        x1 = nn.Dense(hidden)(x)
        x2 = nn.Dense(hidden)(x)
        x = nn.silu(x2) * x1
        if self.drop > 0:
            x = nn.Dropout(self.drop)(x, deterministic=not train)
        x = nn.Dense(self.dim)(x)
        return x


class AdaLNZero(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, cond: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        h = nn.Dense(
            3 * self.dim,
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(cond)
        shift, scale, gate = jnp.split(h, 3, axis=-1)
        return shift, scale, gate


class DiTBlock(nn.Module):
    dim: int
    num_heads: int
    mlp_mult: float = 4.0
    attn_drop: float = 0.0
    drop: float = 0.0
    rope: bool = True
    qk_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        norm1 = RMSNorm(self.dim)(x)
        s1, sc1, g1 = AdaLNZero(self.dim)(cond)
        norm1 = norm1 * (1.0 + sc1[:, None, :]) + s1[:, None, :]
        attn_out = MultiHeadSelfAttention(
            dim=self.dim,
            num_heads=self.num_heads,
            qk_norm=self.qk_norm,
            rope=self.rope,
            attn_drop=self.attn_drop,
            proj_drop=self.drop,
        )(norm1, train=train)
        x = x + (jax.nn.tanh(g1)[:, None, :] * attn_out)

        norm2 = RMSNorm(self.dim)(x)
        s2, sc2, g2 = AdaLNZero(self.dim)(cond)
        norm2 = norm2 * (1.0 + sc2[:, None, :]) + s2[:, None, :]
        mlp_out = SwiGLU(self.dim, mult=self.mlp_mult, drop=self.drop)(norm2, train=train)
        x = x + (jax.nn.tanh(g2)[:, None, :] * mlp_out)
        return x


class CondMLP(nn.Module):
    in_dim: int
    out_dim: int
    hidden: int = 512

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden)(x); x = nn.silu(x)
        x = nn.Dense(self.hidden)(x); x = nn.silu(x)
        x = nn.Dense(self.out_dim)(x)
        return x


class ClassEmbed(nn.Module):
    num_classes: int
    out_dim: int

    @nn.compact
    def __call__(self, cls: jnp.ndarray) -> jnp.ndarray:
        return nn.Embed(self.num_classes, self.out_dim)(cls)


@dataclass
class DiT1DConfig:
    length: int = 128
    in_ch: int = 1
    patch: int = 4
    dim: int = 384
    depth: int = 8
    heads: int = 6
    mlp_mult: float = 4.0
    cond_dim: int = 256
    num_context_tokens: int = 0
    drop: float = 0.0


class DiT1D(nn.Module):
    cfg: DiT1DConfig

    @nn.compact
    def __call__(self, noise: jnp.ndarray, cond: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        cfg = self.cfg
        if noise.ndim == 2:
            x = noise[:, :, None]
        else:
            x = noise
        b, L, C = x.shape
        assert L == cfg.length and C == cfg.in_ch and (L % cfg.patch == 0)

        T = L // cfg.patch
        x = x.reshape(b, T, cfg.patch * C)
        x = nn.Dense(cfg.dim)(x)

        if cfg.num_context_tokens > 0:
            ctx = self.param("ctx_tokens", nn.initializers.normal(0.02), (cfg.num_context_tokens, cfg.dim))
            ctx = jnp.broadcast_to(ctx[None, :, :], (b, cfg.num_context_tokens, cfg.dim))
            x = jnp.concatenate([ctx, x], axis=1)

        pos = self.param("pos_emb", nn.initializers.normal(0.02), (x.shape[1], cfg.dim))
        x = x + pos[None, :, :]

        for _ in range(cfg.depth):
            x = DiTBlock(
                dim=cfg.dim,
                num_heads=cfg.heads,
                mlp_mult=cfg.mlp_mult,
                drop=cfg.drop,
                attn_drop=cfg.drop,
                rope=True,
                qk_norm=True,
            )(x, cond, train=train)

        x = RMSNorm(cfg.dim)(x)
        x = nn.Dense(cfg.patch * C)(x)

        if cfg.num_context_tokens > 0:
            x = x[:, cfg.num_context_tokens :, :]

        x = x.reshape(b, L, C)
        return x[:, :, 0] if cfg.in_ch == 1 else x


@dataclass
class DiTLatent2DConfig:
    h: int = 32
    w: int = 32
    ch: int = 4
    patch: int = 2
    dim: int = 256
    depth: int = 6
    heads: int = 8
    mlp_mult: float = 4.0
    cond_dim: int = 256
    num_context_tokens: int = 0
    drop: float = 0.0


class DiTLatent2D(nn.Module):
    cfg: DiTLatent2DConfig

    @nn.compact
    def __call__(self, noise: jnp.ndarray, cond: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        cfg = self.cfg
        b, h, w, c = noise.shape
        assert (h, w, c) == (cfg.h, cfg.w, cfg.ch)
        assert h % cfg.patch == 0 and w % cfg.patch == 0

        ph = pw = cfg.patch
        th = h // ph
        tw = w // pw
        t = th * tw

        x = noise.reshape(b, th, ph, tw, pw, c)
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5)).reshape(b, t, ph * pw * c)
        x = nn.Dense(cfg.dim)(x)

        if cfg.num_context_tokens > 0:
            ctx = self.param("ctx_tokens", nn.initializers.normal(0.02), (cfg.num_context_tokens, cfg.dim))
            ctx = jnp.broadcast_to(ctx[None, :, :], (b, cfg.num_context_tokens, cfg.dim))
            x = jnp.concatenate([ctx, x], axis=1)

        pos = self.param("pos_emb", nn.initializers.normal(0.02), (x.shape[1], cfg.dim))
        x = x + pos[None, :, :]

        for _ in range(cfg.depth):
            x = DiTBlock(
                dim=cfg.dim,
                num_heads=cfg.heads,
                mlp_mult=cfg.mlp_mult,
                drop=cfg.drop,
                attn_drop=cfg.drop,
                rope=True,
                qk_norm=True,
            )(x, cond, train=train)

        x = RMSNorm(cfg.dim)(x)
        x = nn.Dense(ph * pw * c)(x)

        if cfg.num_context_tokens > 0:
            x = x[:, cfg.num_context_tokens :, :]

        x = x.reshape(b, th, tw, ph, pw, c)
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5)).reshape(b, h, w, c)
        return x


class TinyConvEncoder(nn.Module):
    base: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> List[jnp.ndarray]:
        feats: List[jnp.ndarray] = []
        h = nn.Conv(self.base, (3, 3), padding="SAME")(x); h = nn.silu(h)
        feats.append(h)
        h = nn.Conv(self.base * 2, (3, 3), strides=(2, 2), padding="SAME")(h); h = nn.silu(h)
        feats.append(h)
        h = nn.Conv(self.base * 4, (3, 3), strides=(2, 2), padding="SAME")(h); h = nn.silu(h)
        feats.append(h)
        h = nn.Conv(self.base * 8, (3, 3), strides=(2, 2), padding="SAME")(h); h = nn.silu(h)
        feats.append(h)
        return feats


# ============================================================
# SD-VAE tokenizer (Stable Diffusion style) + ResNet-GN encoder
# ============================================================

@dataclass
class SDVAEConfig:
    """Stable Diffusion style tokenizer (scaled down for MNIST)."""
    in_ch: int = 1
    z_ch: int = 4
    base_ch: int = 64
    ch_mult: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    dropout: float = 0.0


class _GN(nn.Module):
    num_groups: int = 32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Ensure num_groups divides channels
        c = x.shape[-1]
        g = min(self.num_groups, c)
        while g > 1 and (c % g) != 0:
            g -= 1
        return nn.GroupNorm(num_groups=g)(x)


class _ResBlock(nn.Module):
    ch: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        h = _GN()(x)
        h = nn.silu(h)
        h = nn.Conv(self.ch, (3, 3), padding="SAME")(h)

        h = _GN()(h)
        h = nn.silu(h)
        if self.dropout > 0:
            h = nn.Dropout(self.dropout)(h, deterministic=not train)
        h = nn.Conv(self.ch, (3, 3), padding="SAME")(h)

        if x.shape[-1] != self.ch:
            x = nn.Conv(self.ch, (1, 1), padding="SAME")(x)
        return x + h


class _Downsample(nn.Module):
    ch: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.Conv(self.ch, (3, 3), strides=(2, 2), padding="SAME")(x)


class _Upsample(nn.Module):
    ch: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)
        return nn.Conv(self.ch, (3, 3), padding="SAME")(x)


class SDVAEEncoder(nn.Module):
    """Encoder part (separate scope to avoid Linen param name collisions)."""
    cfg: SDVAEConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
        cfg = self.cfg
        h = nn.Conv(cfg.base_ch, (3, 3), padding="SAME")(x)
        ch = cfg.base_ch
        for i, mult in enumerate(cfg.ch_mult):
            ch = cfg.base_ch * mult
            for _ in range(cfg.num_res_blocks):
                h = _ResBlock(ch, dropout=cfg.dropout)(h, train=train)
            if i != len(cfg.ch_mult) - 1:
                h = _Downsample(ch)(h)

        h = _GN()(h); h = nn.silu(h)
        h = nn.Conv(ch, (3, 3), padding="SAME")(h)

        mean = nn.Conv(cfg.z_ch, (1, 1), padding="SAME")(h)
        logvar = nn.Conv(cfg.z_ch, (1, 1), padding="SAME")(h)
        logvar = jnp.clip(logvar, -30.0, 20.0)
        return mean, logvar


class SDVAEDecoder(nn.Module):
    """Decoder part (separate scope)."""
    cfg: SDVAEConfig

    @nn.compact
    def __call__(self, z: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        cfg = self.cfg
        ch = cfg.base_ch * cfg.ch_mult[-1]
        h = nn.Conv(ch, (3, 3), padding="SAME")(z)
        for i, mult in enumerate(reversed(cfg.ch_mult)):
            ch = cfg.base_ch * mult
            for _ in range(cfg.num_res_blocks):
                h = _ResBlock(ch, dropout=cfg.dropout)(h, train=train)
            if i != len(cfg.ch_mult) - 1:
                h = _Upsample(ch)(h)

        h = _GN()(h); h = nn.silu(h)
        x = nn.Conv(cfg.in_ch, (3, 3), padding="SAME")(h)
        x = jnp.tanh(x)  # SD-style [-1,1]
        return x


class SDVAETokenizer(nn.Module):
    """Small KL-VAE tokenizer in the style of SD-VAE (Stable Diffusion)."""
    cfg: SDVAEConfig

    def setup(self):
        self.enc = SDVAEEncoder(self.cfg)
        self.dec = SDVAEDecoder(self.cfg)

    def encode(self, x: jnp.ndarray, *, train: bool) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return self.enc(x, train=train)

    def decode(self, z: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        return self.dec(z, train=train)

    def reparam(self, key: jax.Array, mean: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
        eps = jax.random.normal(key, mean.shape, dtype=mean.dtype)
        return mean + jnp.exp(0.5 * logvar) * eps

    def __call__(
        self, x: jnp.ndarray, key: jax.Array, *, train: bool
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        mean, logvar = self.encode(x, train=train)
        z = self.reparam(key, mean, logvar)
        xhat = self.decode(z, train=train)
        return z, mean, logvar, xhat


@dataclass
class ResNetGNConfig:
    in_ch: int = 1
    base_ch: int = 64
    num_blocks: Tuple[int, int, int, int] = (2, 2, 2, 2)  # like ResNet-18 stages
    num_classes: int = 10


class _BasicBlockGN(nn.Module):
    ch: int
    stride: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.Conv(self.ch, (3, 3), strides=(self.stride, self.stride), padding="SAME")(x)
        h = _GN()(h); h = nn.silu(h)
        h = nn.Conv(self.ch, (3, 3), padding="SAME")(h)
        h = _GN()(h)

        if self.stride != 1 or x.shape[-1] != self.ch:
            x = nn.Conv(self.ch, (1, 1), strides=(self.stride, self.stride), padding="SAME")(x)
            x = _GN()(x)

        return nn.silu(x + h)


class ResNetGNEncoder(nn.Module):
    cfg: ResNetGNConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> List[jnp.ndarray]:
        cfg = self.cfg
        feats: List[jnp.ndarray] = []

        h = nn.Conv(cfg.base_ch, (3, 3), padding="SAME")(x)
        h = _GN()(h); h = nn.silu(h)
        feats.append(h)

        ch = cfg.base_ch
        for si, nb in enumerate(cfg.num_blocks):
            for bi in range(nb):
                stride = 2 if (bi == 0 and si > 0) else 1
                h = _BasicBlockGN(ch, stride=stride)(h)
            feats.append(h)
            ch = min(ch * 2, cfg.base_ch * 8)

        return feats


class ResNetGNClassifier(nn.Module):
    cfg: ResNetGNConfig

    def setup(self):
        self.enc = ResNetGNEncoder(self.cfg)
        self.head = nn.Dense(self.cfg.num_classes)

    def encode(self, x: jnp.ndarray) -> List[jnp.ndarray]:
        return self.enc(x)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        feats = self.encode(x)
        h = feats[-1]
        h = jnp.mean(h, axis=(1, 2))
        logits = self.head(h)
        return logits



# ============================================================
# Baselines: MDN and Conditional Flow Matching (CFM)
# ============================================================

@dataclass
class MDNConfig:
    num_mixtures: int = 8
    hidden: int = 128
    depth: int = 3
    min_scale: float = 1e-3


class MDN1D(nn.Module):
    """Conditional Mixture Density Network for 1D target: p(x | y)."""
    cfg: MDNConfig

    @nn.compact
    def __call__(self, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = y
        for _ in range(self.cfg.depth):
            x = nn.Dense(self.cfg.hidden)(x)
            x = nn.gelu(x)
        out = nn.Dense(3 * self.cfg.num_mixtures)(x)
        logits, means, scales_raw = jnp.split(out, 3, axis=-1)
        scales = nn.softplus(scales_raw) + self.cfg.min_scale
        return logits, means, scales


def mdn_log_prob_1d(
    logits: jnp.ndarray, means: jnp.ndarray, scales: jnp.ndarray, x: jnp.ndarray
) -> jnp.ndarray:
    if x.ndim == 2:
        x = x[:, 0]
    log_pi = jax.nn.log_softmax(logits, axis=-1)
    z = (x[:, None] - means) / scales
    log_norm = -0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(scales)
    log_exp = -0.5 * (z * z)
    log_comp = log_norm + log_exp
    return jax.scipy.special.logsumexp(log_pi + log_comp, axis=-1)


def mdn_nll_1d(params: dict, model: MDN1D, y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    logits, means, scales = model.apply(params, y)
    lp = mdn_log_prob_1d(logits, means, scales, x)
    return -jnp.mean(lp)


def mdn_mixture_pdf_1d(
    logits: jnp.ndarray, means: jnp.ndarray, scales: jnp.ndarray, grid: jnp.ndarray
) -> jnp.ndarray:
    pi = jax.nn.softmax(logits, axis=-1)
    z = (grid[:, None] - means[None, :]) / scales[None, :]
    norm = (1.0 / (jnp.sqrt(2.0 * jnp.pi) * scales[None, :])) * jnp.exp(-0.5 * z * z)
    return jnp.sum(pi[None, :] * norm, axis=-1)


@dataclass
class CFMConfig:
    hidden: int = 256
    depth: int = 4
    sigma: float = 0.0
    steps: int = 50


class CondVelocityMLP(nn.Module):
    """Conditional velocity field v_theta(y, x_t, t) -> dx/dt."""
    cfg: CFMConfig

    @nn.compact
    def __call__(self, y: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        if y.ndim == 1:
            y = y[:, None]
        if t.ndim == 0:
            t = jnp.full((x_t.shape[0], 1), t, dtype=x_t.dtype)
        elif t.ndim == 1:
            t = t[:, None]
        inp = jnp.concatenate([y, x_t, t], axis=-1)
        h = inp
        for _ in range(self.cfg.depth):
            h = nn.Dense(self.cfg.hidden)(h)
            h = nn.gelu(h)
        v = nn.Dense(2)(h)
        return v


def cfm_batch(
    key: jax.Array,
    x1: jnp.ndarray,
    y: jnp.ndarray,
    sigma: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    k0, kt, ke = jax.random.split(key, 3)
    B = x1.shape[0]
    x0 = jax.random.normal(k0, (B, 2), dtype=x1.dtype)
    t = jax.random.uniform(kt, (B, 1), minval=0.0, maxval=1.0, dtype=x1.dtype)
    eps = jax.random.normal(ke, (B, 2), dtype=x1.dtype)
    x_t = (1.0 - t) * x0 + t * x1 + float(sigma) * eps
    u = x1 - x0
    return x_t, t, u, y


def cfm_loss(
    params: dict,
    model: CondVelocityMLP,
    key: jax.Array,
    x1: jnp.ndarray,
    y: jnp.ndarray,
    sigma: float = 0.0,
) -> jnp.ndarray:
    x_t, t, u, y_in = cfm_batch(key, x1, y, sigma=sigma)
    v = model.apply(params, y_in, x_t, t)
    return jnp.mean(jnp.sum((v - u) ** 2, axis=-1))


def cfm_sample(
    params: dict,
    model: CondVelocityMLP,
    key: jax.Array,
    y: jnp.ndarray,
    *,
    steps: int = 50,
) -> jnp.ndarray:
    if y.ndim == 1:
        y = y[:, None]
    B = y.shape[0]
    x = jax.random.normal(key, (B, 2), dtype=jnp.float32)
    dt = 1.0 / float(steps)

    def body(i, x):
        t = jnp.full((B, 1), i * dt, dtype=jnp.float32)
        v = model.apply(params, y, x, t)
        return x + dt * v

    x = jax.lax.fori_loop(0, steps, body, x)
    return x
