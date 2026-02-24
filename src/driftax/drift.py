from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp


def _cdist_l2(a: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Euclidean distance matrix between a:[N,D] and b:[M,D] -> [N,M]."""
    a2 = jnp.sum(a * a, axis=-1, keepdims=True)        # [N,1]
    b2 = jnp.sum(b * b, axis=-1, keepdims=True).T      # [1,M]
    dist2 = jnp.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0)
    return jnp.sqrt(dist2 + eps)


def _softmax2d_rowcol(logit: jnp.ndarray) -> jnp.ndarray:
    """ A = sqrt(softmax_row * softmax_col)"""
    a_row = jax.nn.softmax(logit, axis=1)  # over columns
    a_col = jax.nn.softmax(logit, axis=0)  # over rows
    return jnp.sqrt(a_row * a_col)


def compute_V(
    x: jnp.ndarray,
    y_pos: jnp.ndarray,
    y_neg: jnp.ndarray,
    temp: float,
    *,
    neg_log_weights: Optional[jnp.ndarray] = None,
    mask_self_in_neg: bool = True,
) -> jnp.ndarray:
    """Compute drifting field V (Algorithm 2) with optional weighted negatives (CFG-style)."""
    n = x.shape[0]
    n_pos = y_pos.shape[0]
    n_neg = y_neg.shape[0]

    dist_pos = _cdist_l2(x, y_pos)  # [N, N_pos]
    dist_neg = _cdist_l2(x, y_neg)  # [N, N_neg]

    if mask_self_in_neg and n_neg >= n:
        dist_neg = dist_neg.at[:, :n].add(jnp.eye(n, dtype=dist_neg.dtype) * 1e6)

    logit_pos = -dist_pos / temp
    logit_neg = -dist_neg / temp
    if neg_log_weights is not None:
        logit_neg = logit_neg + neg_log_weights[None, :]

    logit = jnp.concatenate([logit_pos, logit_neg], axis=1)  # [N, N_pos+N_neg]
    A = _softmax2d_rowcol(logit)

    A_pos = A[:, :n_pos]
    A_neg = A[:, n_pos:]

    W_pos = A_pos * jnp.sum(A_neg, axis=1, keepdims=True)
    W_neg = A_neg * jnp.sum(A_pos, axis=1, keepdims=True)

    drift_pos = W_pos @ y_pos
    drift_neg = W_neg @ y_neg
    return drift_pos - drift_neg


def _mean_pairwise_dist(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(_cdist_l2(x, y))


def normalize_features(
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    target_mean_dist: float = 1.0,
    stopgrad_scale: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scale features so mean pairwise distance is ~ target_mean_dist (Appendix A.6)."""
    mean_dist = _mean_pairwise_dist(x, y)
    feat_scale = mean_dist / target_mean_dist
    if stopgrad_scale:
        feat_scale = jax.lax.stop_gradient(feat_scale)
    x_n = x / (feat_scale + 1e-12)
    y_n = y / (feat_scale + 1e-12)
    return x_n, y_n, feat_scale


def normalize_drift(
    V: jnp.ndarray,
    *,
    target_mean_norm: float = 1.0,
    stopgrad_scale: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Scale drift magnitudes (Appendix A.6)."""
    mean_norm = jnp.mean(jnp.sqrt(jnp.sum(V * V, axis=-1) + 1e-12))
    drift_scale = mean_norm / target_mean_norm
    if stopgrad_scale:
        drift_scale = jax.lax.stop_gradient(drift_scale)
    return V / (drift_scale + 1e-12), drift_scale


def drifting_loss_features(
    x_feat: jnp.ndarray,
    pos_feat: jnp.ndarray,
    *,
    temps: Sequence[float] = (0.02, 0.05, 0.2),
    neg_feat: Optional[jnp.ndarray] = None,
    neg_log_weights: Optional[jnp.ndarray] = None,
    feature_normalize: bool = True,
    drift_normalize: bool = True,
) -> jnp.ndarray:
    """Drifting loss in feature space with multi-T aggregation (Appendix A.6)."""
    if neg_feat is None:
        neg_feat = x_feat

    if feature_normalize:
        targets = jnp.concatenate([pos_feat, neg_feat], axis=0)
        x_n, targets_n, _ = normalize_features(x_feat, targets, stopgrad_scale=True)
        pos_n = targets_n[: pos_feat.shape[0]]
        neg_n = targets_n[pos_feat.shape[0] :]
    else:
        x_n, pos_n, neg_n = x_feat, pos_feat, neg_feat

    Vs = []
    for T in temps:
        Vs.append(compute_V(x_n, pos_n, neg_n, float(T), neg_log_weights=neg_log_weights, mask_self_in_neg=True))
    V = sum(Vs) / float(len(Vs))

    if drift_normalize:
        V, _ = normalize_drift(V, stopgrad_scale=False)

    target = jax.lax.stop_gradient(x_n + V)
    return jnp.mean(jnp.sum((x_n - target) ** 2, axis=-1))




def compute_V_conditional(
    x: jnp.ndarray,            # [N, Dx]
    y: jnp.ndarray,            # [N, Dy]
    pos_x: jnp.ndarray,        # [P, Dx]
    pos_y: jnp.ndarray,        # [P, Dy]
    neg_x: jnp.ndarray,        # [M, Dx]
    neg_y: jnp.ndarray,        # [M, Dy]
    temp_x: float,
    temp_y: float,
    *,
    neg_log_weights: jnp.ndarray | None = None,
    mask_self_in_neg: bool = True,
) -> jnp.ndarray:
    """Compute drift in x-space with a y-aware kernel.

    Similarity logits are:
        - (||x - x'|| / temp_x + ||y - y'|| / temp_y)

    This prevents positives from other conditions (far in y) from pulling x.
    """
    # distances
    dist_pos_x = _cdist(x, pos_x)  # [N,P]
    dist_neg_x = _cdist(x, neg_x)  # [N,M]
    dist_pos_y = _cdist(y, pos_y)  # [N,P]
    dist_neg_y = _cdist(y, neg_y)  # [N,M]

    n = x.shape[0]
    if mask_self_in_neg and (neg_x.shape[0] >= n):
        big = jnp.eye(n, dtype=dist_neg_x.dtype) * 1e6
        dist_neg_x = dist_neg_x.at[:, :n].add(big)
        dist_neg_y = dist_neg_y.at[:, :n].add(big)

    logit_pos = -(dist_pos_x / float(temp_x) + dist_pos_y / float(temp_y))
    logit_neg = -(dist_neg_x / float(temp_x) + dist_neg_y / float(temp_y))
    if neg_log_weights is not None:
        logit_neg = logit_neg + neg_log_weights[None, :]

    logit = jnp.concatenate([logit_pos, logit_neg], axis=1)
    A = _softmax2d_rowcol(logit)

    P = pos_x.shape[0]
    A_pos = A[:, :P]
    A_neg = A[:, P:]

    W_pos = A_pos * jnp.sum(A_neg, axis=1, keepdims=True)
    W_neg = A_neg * jnp.sum(A_pos, axis=1, keepdims=True)

    return (W_pos @ pos_x) - (W_neg @ neg_x)


def drifting_loss_conditional_features(
    x_feat: jnp.ndarray,
    y: jnp.ndarray,
    pos_feat: jnp.ndarray,
    pos_y: jnp.ndarray,
    *,
    temps_x: tuple[float, ...] = (0.02, 0.05, 0.2),
    temp_y: float = 0.05,
    neg_feat: jnp.ndarray | None = None,
    neg_y: jnp.ndarray | None = None,
    neg_log_weights: jnp.ndarray | None = None,
    feature_normalize: bool = True,
    drift_normalize: bool = True,
) -> jnp.ndarray:
    """Conditional drifting loss in a feature space (Sec. 3.4-style), continuous y.

    Implements:
        E || phi(x) - stopgrad(phi(x) + V_phi(x)) ||^2
    but computes V using y-aware logits so conditioning is respected.

    - x_feat: [N,D] features of generated samples phi(x)
    - y:      [N,Dy] conditions for generated samples
    - pos_feat,pos_y: positives from data distribution with their conditions
    """
    if neg_feat is None:
        neg_feat = x_feat
    if neg_y is None:
        neg_y = y

    # optional feature normalization (like existing drifting_loss_features)
    if feature_normalize:
        targets = jnp.concatenate([pos_feat, neg_feat], axis=0)
        x_n, targets_n, _ = normalize_features(x_feat, targets, stopgrad_scale=True)
        pos_n = targets_n[: pos_feat.shape[0]]
        neg_n = targets_n[pos_feat.shape[0] :]
    else:
        x_n, pos_n, neg_n = x_feat, pos_feat, neg_feat

    Vs = []
    for Tx in temps_x:
        Vs.append(
            compute_V_conditional(
                x_n, y,
                pos_n, pos_y,
                neg_n, neg_y,
                temp_x=float(Tx),
                temp_y=float(temp_y),
                neg_log_weights=neg_log_weights,
                mask_self_in_neg=True,
            )
        )
    V = sum(Vs) / float(len(Vs))

    if drift_normalize:
        V, _ = normalize_drift(V, stopgrad_scale=False)

    target = jax.lax.stop_gradient(x_n + V)
    return jnp.mean(jnp.sum((x_n - target) ** 2, axis=-1))





def drifting_loss(
    x: jnp.ndarray,
    pos: jnp.ndarray,
    *,
    temp: float = 0.05,
    feature_normalize: bool = False,
    drift_normalize: bool = False,
) -> jnp.ndarray:
    """Drifting loss: MSE(x, stopgrad(x + V)) where V is computed against `pos`.

    This mirrors the original toy loss used in the colab / toy examples.
    """
    V = compute_V(x, pos, x, temp=float(temp), mask_self_in_neg=True)
    if feature_normalize:
        # normalize x and V in the same way drifting_loss_features does (simple option)
        x_n, _, scale = normalize_features(x, x, stopgrad_scale=True)
        V = V / (scale + 1e-12)
        x_use = x_n
    else:
        x_use = x

    if drift_normalize:
        V, _ = normalize_drift(V, stopgrad_scale=False)

    target = jax.lax.stop_gradient(x_use + V)
    return jnp.mean(jnp.sum((x_use - target) ** 2, axis=-1))
