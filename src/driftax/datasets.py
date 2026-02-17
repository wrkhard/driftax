from __future__ import annotations

import math
import jax
import jax.numpy as jnp


def sample_checkerboard(key: jax.Array, n: int, noise: float = 0.05) -> jnp.ndarray:
    """
    Checkerboard sampler:
    - pick b in {0,1}
    - i,j in {0,2} offset by b
    - uniform in each square, then scale/shift to roughly [-1,1]
    """
    key_b, key_i, key_j, key_u, key_v, key_eps = jax.random.split(key, 6)
    b = jax.random.randint(key_b, (n,), 0, 2)
    i = jax.random.randint(key_i, (n,), 0, 2) * 2 + b
    j = jax.random.randint(key_j, (n,), 0, 2) * 2 + b
    u = jax.random.uniform(key_u, (n,))
    v = jax.random.uniform(key_v, (n,))
    pts = jnp.stack([i + u, j + v], axis=1) - 2.0
    pts = pts / 2.0
    if noise > 0:
        pts = pts + noise * jax.random.normal(key_eps, pts.shape)
    return pts.astype(jnp.float32)


def sample_swiss_roll(key: jax.Array, n: int, noise: float = 0.03) -> jnp.ndarray:
    """2D swiss roll, normalized to ~[-1,1]."""
    key_u, key_eps = jax.random.split(key, 2)
    u = jax.random.uniform(key_u, (n,))
    t = 0.5 * math.pi + 4.0 * math.pi * u
    pts = jnp.stack([t * jnp.cos(t), t * jnp.sin(t)], axis=1)
    pts = pts / (jnp.max(jnp.abs(pts)) + 1e-8)
    if noise > 0:
        pts = pts + noise * jax.random.normal(key_eps, pts.shape)
    return pts.astype(jnp.float32)


def get_sampler(name: str):
    name = name.lower()
    if name in ("checkerboard", "check", "cb"):
        return sample_checkerboard
    if name in ("swiss_roll", "swiss", "sr"):
        return sample_swiss_roll
    raise ValueError(f"Dataset needs to be 'checkerboard' or 'swiss_roll': {name}")
