from __future__ import annotations

import jax
import jax.numpy as jnp


def pairwise_cdist(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Euclidean cdist between a:[N,D] and b:[M,D] -> [N,M]."""
    diff = a[:, None, :] - b[None, :, :]
    return jnp.sqrt(jnp.sum(diff * diff, axis=-1) + eps)


def compute_drift(gen: jnp.ndarray, pos: jnp.ndarray, temp: float = 0.05) -> jnp.ndarray:
    """
    Compute drift field V with attention-based kernel.

    Args:
        gen: Generated samples [G, D]
        pos: Data samples [P, D]
        temp: Temperature for softmax-like kernel (smaller => sharper)

    Returns:
        V: Drift vectors [G, D]
    """
    targets = jnp.concatenate([gen, pos], axis=0)  # [G+P, D]
    G = gen.shape[0]

    dist = pairwise_cdist(gen, targets)  # [G, G+P]

    # mask self distances for gen vs gen diagonal (first G columns)
    diag = jnp.eye(G, dtype=bool)
    dist = dist.at[:, :G].set(jnp.where(diag, 1e6, dist[:, :G]))

    kernel = jnp.exp(-dist / temp)  # [G, G+P]

    row_sums = jnp.sum(kernel, axis=1, keepdims=True)      # [G, 1]
    col_sums = jnp.sum(kernel, axis=0, keepdims=True)      # [1, G+P]
    normalizer = jnp.sqrt(jnp.clip(row_sums * col_sums, a_min=1e-12))  # [G, G+P]
    normalized_kernel = kernel / normalizer

    # positive drift toward data
    sum_gen = jnp.sum(normalized_kernel[:, :G], axis=-1, keepdims=True)  # [G,1]
    pos_coeff = normalized_kernel[:, G:] * sum_gen                       # [G,P]
    pos_V = pos_coeff @ targets[G:]                                      # [G,D]

    # negative drift away from current generated mass
    sum_pos = jnp.sum(normalized_kernel[:, G:], axis=-1, keepdims=True)  # [G,1]
    neg_coeff = normalized_kernel[:, :G] * sum_pos                       # [G,G]
    neg_V = neg_coeff @ targets[:G]                                      # [G,D]

    return pos_V - neg_V


def drifting_loss(gen: jnp.ndarray, pos: jnp.ndarray, *, temp: float = 0.05) -> jnp.ndarray:
    """Drifting loss: MSE(gen, stopgrad(gen + V))."""
    V = compute_drift(jax.lax.stop_gradient(gen), jax.lax.stop_gradient(pos), temp=temp)
    target = jax.lax.stop_gradient(gen + V)
    return jnp.mean(jnp.sum((gen - target) ** 2, axis=-1))
