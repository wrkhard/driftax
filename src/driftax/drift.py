from __future__ import annotations

from typing import List, Optional, Sequence

import jax
import jax.numpy as jnp


def _cdist_l2(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    a2 = jnp.sum(a * a, axis=1, keepdims=True)
    b2 = jnp.sum(b * b, axis=1, keepdims=True).T
    return jnp.sqrt(jnp.maximum(a2 + b2 - 2.0 * (a @ b.T), 0.0) + 1e-12)


def compute_V(
    x: jnp.ndarray,
    y_pos: jnp.ndarray,
    y_neg: jnp.ndarray,
    temperature: float,
    *,
    mask_self: bool = True,
) -> jnp.ndarray:
    N = x.shape[0]
    dist_pos = _cdist_l2(x, y_pos)
    dist_neg = _cdist_l2(x, y_neg)

    if mask_self and (y_neg.shape[0] == N):
        dist_neg = dist_neg + jnp.eye(N, dtype=dist_neg.dtype) * 1e6

    logit_pos = -dist_pos / float(temperature)
    logit_neg = -dist_neg / float(temperature)
    logit = jnp.concatenate([logit_pos, logit_neg], axis=1)

    A_row = jax.nn.softmax(logit, axis=1)
    A_col = jax.nn.softmax(logit, axis=0)
    A = jnp.sqrt(A_row * A_col)

    P = y_pos.shape[0]
    A_pos = A[:, :P]
    A_neg = A[:, P:]

    W_pos = A_pos * jnp.sum(A_neg, axis=1, keepdims=True)
    W_neg = A_neg * jnp.sum(A_pos, axis=1, keepdims=True)

    return (W_pos @ y_pos) - (W_neg @ y_neg)


def compute_V_multi_temperature(
    x: jnp.ndarray,
    y_pos: jnp.ndarray,
    y_neg: jnp.ndarray,
    temperatures: Sequence[float] = (0.02, 0.05, 0.2),
    *,
    mask_self: bool = True,
    normalize_each: bool = True,
) -> jnp.ndarray:
    V_total = jnp.zeros_like(x)
    for tau in temperatures:
        V_tau = compute_V(x, y_pos, y_neg, float(tau), mask_self=mask_self)
        if normalize_each:
            vnorm = jnp.sqrt(jnp.mean(V_tau * V_tau) + 1e-8)
            V_tau = V_tau / (vnorm + 1e-8)
        V_total = V_total + V_tau
    return V_total


def l2_normalize(x: jnp.ndarray, axis: int = -1, eps: float = 1e-12) -> jnp.ndarray:
    return x / jnp.sqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)


def global_avg_pool(feat_map: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(feat_map, axis=(1, 2))


def drifting_loss_features(
    x_feat: jnp.ndarray,
    pos_feat: jnp.ndarray,
    *,
    temps: Sequence[float] = (0.02, 0.05, 0.2),
    neg_feat: Optional[jnp.ndarray] = None,
    feature_normalize: bool = True,
    drift_normalize: bool = True,
    normalize_each_temp: Optional[bool] = None,
    mask_self_in_neg: bool = True,
) -> jnp.ndarray:
    if neg_feat is None:
        neg_feat = x_feat

    x = x_feat
    p = pos_feat
    n = neg_feat

    if feature_normalize:
        x = l2_normalize(x, axis=-1)
        p = l2_normalize(p, axis=-1)
        n = l2_normalize(n, axis=-1)

    if normalize_each_temp is None:
        normalize_each_temp = bool(drift_normalize)

    V = compute_V_multi_temperature(
        x, p, n,
        temperatures=temps,
        mask_self=mask_self_in_neg and (n.shape[0] == x.shape[0]),
        normalize_each=bool(normalize_each_temp),
    )

    target = jax.lax.stop_gradient(x + V)
    return jnp.mean(jnp.sum((x - target) ** 2, axis=-1))


def drifting_loss_multiscale_pooled(
    x_maps: List[jnp.ndarray],
    pos_maps: List[jnp.ndarray],
    *,
    temps: Sequence[float] = (0.02, 0.05, 0.2),
    feature_normalize: bool = True,
    drift_normalize: bool = True,
) -> jnp.ndarray:
    assert len(x_maps) == len(pos_maps)
    loss = 0.0
    for xm, pm in zip(x_maps, pos_maps):
        xv = global_avg_pool(xm)
        pv = global_avg_pool(pm)
        loss = loss + drifting_loss_features(
            x_feat=xv,
            pos_feat=pv,
            temps=temps,
            neg_feat=xv,
            feature_normalize=feature_normalize,
            drift_normalize=drift_normalize,
        )
    return loss / float(len(x_maps))


def compute_V_conditional(
    x: jnp.ndarray,
    y: jnp.ndarray,
    pos_x: jnp.ndarray,
    pos_y: jnp.ndarray,
    neg_x: jnp.ndarray,
    neg_y: jnp.ndarray,
    *,
    temp_x: float,
    temp_y: float,
    beta_y: float = 1.0,
    mask_self_in_neg: bool = True,
) -> jnp.ndarray:
    dx_pos = _cdist_l2(x, pos_x)
    dx_neg = _cdist_l2(x, neg_x)
    dy_pos = _cdist_l2(y, pos_y)
    dy_neg = _cdist_l2(y, neg_y)

    n = x.shape[0]
    if mask_self_in_neg and (neg_x.shape[0] >= n):
        big = jnp.eye(n, dtype=dx_neg.dtype) * 1e6
        dx_neg = dx_neg.at[:, :n].add(big)
        dy_neg = dy_neg.at[:, :n].add(big)

    logit_pos = -(dx_pos / float(temp_x) + float(beta_y) * (dy_pos / float(temp_y)))
    logit_neg = -(dx_neg / float(temp_x) + float(beta_y) * (dy_neg / float(temp_y)))
    logit = jnp.concatenate([logit_pos, logit_neg], axis=1)

    A_row = jax.nn.softmax(logit, axis=1)
    A_col = jax.nn.softmax(logit, axis=0)
    A = jnp.sqrt(A_row * A_col)

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
    temps_x: Sequence[float] = (0.05,),
    temp_y: float = 0.05,
    beta_y: float = 3.0,
    feature_normalize: bool = True,
    drift_normalize: bool = True,
) -> jnp.ndarray:
    neg_feat = x_feat
    neg_y = y

    x = x_feat
    px = pos_feat
    nx = neg_feat

    if feature_normalize:
        x = l2_normalize(x, axis=-1)
        px = l2_normalize(px, axis=-1)
        nx = l2_normalize(nx, axis=-1)

    V = jnp.zeros_like(x)
    for Tx in temps_x:
        Vt = compute_V_conditional(
            x, y,
            px, pos_y,
            nx, neg_y,
            temp_x=float(Tx),
            temp_y=float(temp_y),
            beta_y=float(beta_y),
            mask_self_in_neg=True,
        )
        if drift_normalize:
            vnorm = jnp.sqrt(jnp.mean(Vt * Vt) + 1e-8)
            Vt = Vt / (vnorm + 1e-8)
        V = V + Vt

    target = jax.lax.stop_gradient(x + V)
    return jnp.mean(jnp.sum((x - target) ** 2, axis=-1))


def drifting_loss(gen=None, pos=None, compute_drift=None, **kwargs):
    if callable(compute_drift):
        V = compute_drift(gen, pos)
        target = jax.lax.stop_gradient(gen + V)
        return jnp.mean(jnp.sum((gen - target) ** 2, axis=-1))
    return drifting_loss_features(**kwargs)
