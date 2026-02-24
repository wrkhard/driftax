import jax
import jax.numpy as jnp
from driftax.drift import compute_V, drifting_loss_conditional_features
from driftax.models import DiT1D, DiT1DConfig, DiTLatent2D, DiTLatent2DConfig, CondMLP

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


def test_conditional_loss_runs():
    import jax
    import jax.numpy as jnp
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (32, 2))
    y = jax.random.normal(k2, (32, 1))
    pos_x = jax.random.normal(k3, (64, 2))
    pos_y = jax.random.normal(k2, (64, 1))
    loss = drifting_loss_conditional_features(x_feat=x, y=y, pos_feat=pos_x, pos_y=pos_y, temp_y=0.1)
    assert loss.shape == ()
