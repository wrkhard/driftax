import jax
import jax.numpy as jnp
from driftax.drift import compute_drift, drifting_loss

def test_shapes():
    key = jax.random.PRNGKey(0)
    key, k1, k2 = jax.random.split(key, 3)
    gen = jax.random.normal(k1, (16, 2))
    pos = jax.random.normal(k2, (32, 2))
    V = compute_drift(gen, pos, temp=0.1)
    assert V.shape == gen.shape
    loss = drifting_loss(gen, pos, temp=0.1)
    assert loss.shape == ()
