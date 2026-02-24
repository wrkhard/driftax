import jax
import jax.numpy as jnp

from driftax.drift import compute_V
from driftax.models import (
    DiT1D, DiT1DConfig,
    DiTLatent2D, DiTLatent2DConfig,
    CondMLP,
    MDN1D, MDNConfig, mdn_nll_1d,
    CondVelocityMLP, CFMConfig, cfm_loss, cfm_sample,
)


def test_compute_v():
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    x = jax.random.normal(k1, (8, 4))
    y = jax.random.normal(k2, (16, 4))
    V = compute_V(x, y, x, temp=0.1)
    assert V.shape == x.shape


def test_dit1d_forward():
    cfg = DiT1DConfig(length=16, patch=4, dim=64, depth=2, heads=4, cond_dim=32)
    model = DiT1D(cfg)
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    z = jax.random.normal(k1, (2, 16))
    c = jax.random.normal(k2, (2, 32))
    params = model.init(k1, z, c, train=False)
    out = model.apply(params, z, c, train=False)
    assert out.shape == (2, 16)


def test_dit2d_forward():
    cfg = DiTLatent2DConfig(h=8, w=8, ch=1, patch=2, dim=64, depth=2, heads=4, cond_dim=32)
    model = DiTLatent2D(cfg)
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    z = jax.random.normal(k1, (2, 8, 8, 1))
    c = jax.random.normal(k2, (2, 32))
    params = model.init(k1, z, c, train=False)
    out = model.apply(params, z, c, train=False)
    assert out.shape == (2, 8, 8, 1)


def test_condmlp():
    mlp = CondMLP(in_dim=1, out_dim=8, hidden=16)
    key = jax.random.PRNGKey(0)
    params = mlp.init(key, jnp.zeros((1, 1), dtype=jnp.float32))
    y = mlp.apply(params, jnp.ones((4, 1), dtype=jnp.float32))
    assert y.shape == (4, 8)


def test_mdn_forward():
    cfg = MDNConfig(num_mixtures=4, hidden=16, depth=2)
    m = MDN1D(cfg)
    key = jax.random.PRNGKey(0)
    params = m.init(key, jnp.zeros((1,1), dtype=jnp.float32))
    y = jnp.zeros((8,1), dtype=jnp.float32)
    x = jnp.zeros((8,1), dtype=jnp.float32)
    nll = mdn_nll_1d(params, m, y, x)
    assert nll.shape == ()


def test_cfm_forward_and_sample():
    cfg = CFMConfig(hidden=32, depth=2, sigma=0.0, steps=10)
    m = CondVelocityMLP(cfg)
    key = jax.random.PRNGKey(0)
    params = m.init(key, jnp.zeros((1,1), dtype=jnp.float32), jnp.zeros((1,2), dtype=jnp.float32), jnp.zeros((1,1), dtype=jnp.float32))
    x1 = jax.random.normal(key, (16,2))
    y = jnp.zeros((16,1), dtype=jnp.float32)
    loss = cfm_loss(params, m, key, x1, y, sigma=0.0)
    assert loss.shape == ()
    xs = cfm_sample(params, m, key, y, steps=10)
    assert xs.shape == (16,2)
