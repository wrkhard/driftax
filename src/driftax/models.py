from __future__ import annotations

from dataclasses import dataclass
import flax.linen as nn
import jax.numpy as jnp


class MLPGenerator(nn.Module):
    """MLP: z -> x (toy 2D). 3 hidden layers with SiLU."""
    in_dim: int = 32
    hidden: int = 256
    out_dim: int = 2

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden)(z)
        x = nn.silu(x)
        x = nn.Dense(self.hidden)(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden)(x)
        x = nn.silu(x)
        x = nn.Dense(self.out_dim)(x)
        return x
