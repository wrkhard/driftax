from __future__ import annotations

import math
from typing import Tuple

import jax
import jax.numpy as jnp


def sample_checkerboard(key: jax.Array, n: int, noise: float = 0.05) -> jnp.ndarray:
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
    key_u, key_eps = jax.random.split(key, 2)
    u = jax.random.uniform(key_u, (n,))
    t = 0.5 * math.pi + 4.0 * math.pi * u
    pts = jnp.stack([t * jnp.cos(t), t * jnp.sin(t)], axis=1)
    pts = pts / (jnp.max(jnp.abs(pts)) + 1e-8)
    if noise > 0:
        pts = pts + noise * jax.random.normal(key_eps, pts.shape)
    return pts.astype(jnp.float32)


def inverse_linear_toy(
    key: jax.Array,
    batch: int,
    dim_x: int = 128,
    dim_y: int = 64,
    noise_std: float = 0.05,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    kA, kx, ke = jax.random.split(key, 3)
    A = jax.random.normal(kA, (dim_y, dim_x), dtype=jnp.float32) / jnp.sqrt(dim_x)
    x = jax.random.normal(kx, (batch, dim_x), dtype=jnp.float32)
    eps = noise_std * jax.random.normal(ke, (batch, dim_y), dtype=jnp.float32)
    y = x @ A.T + eps
    return x, y, A


def inverse_ring_toy(
    key: jax.Array,
    batch: int,
    radius: float = 1.0,
    x_noise: float = 0.02,
    y_noise_base: float = 0.03,
    y_noise_slope: float = 0.07,
):
    """Toy conditional inverse problem on a ring (bimodal posterior).

    Sample x on a noisy ring in R^2:
        x = (r cos θ, r sin θ) + εx
    Observe a 1D measurement:
        y = x0 + εy
    where εy is heteroscedastic: std = y_noise_base + y_noise_slope * |x1|.

    Conditioning on y yields a bimodal posterior over x1 (upper/lower arc).
    Returns:
        x: [B,2]
        y: [B,1]
    """
    k_theta, kx, ky = jax.random.split(key, 3)
    theta = 2.0 * jnp.pi * jax.random.uniform(k_theta, (batch,), dtype=jnp.float32)
    x0 = radius * jnp.cos(theta)
    x1 = radius * jnp.sin(theta)
    x = jnp.stack([x0, x1], axis=1)

    if x_noise > 0:
        x = x + x_noise * jax.random.normal(kx, x.shape, dtype=jnp.float32)

    y_std = y_noise_base + y_noise_slope * jnp.abs(x[:, 1])
    y = x[:, 0:1] + (y_std[:, None] * jax.random.normal(ky, (batch, 1), dtype=jnp.float32))
    return x.astype(jnp.float32), y.astype(jnp.float32)




import os
import urllib.request
from typing import Tuple

import numpy as np


MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def _download(url: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)


def load_mnist_npz(cache_dir: str = "data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST as numpy arrays. Downloads mnist.npz if missing.

    Returns:
        x_train: [60000, 28, 28] uint8
        y_train: [60000] uint8
        x_test:  [10000, 28, 28] uint8
        y_test:  [10000] uint8
    """
    path = os.path.join(cache_dir, "mnist.npz")
    if not os.path.exists(path):
        _download(MNIST_URL, path)

    with np.load(path) as data:
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]
    return x_train, y_train, x_test, y_test


def preprocess_mnist(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32) / 127.5 - 1.0
    return x[..., None]  # [N,28,28,1]
