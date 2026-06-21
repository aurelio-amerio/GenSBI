"""Elementwise transformers parameterised per-dimension by a conditioner.

Pure functions of (value, params) — no learnable state of their own.
"""

import jax
import jax.numpy as jnp
from jax import Array


def _clamp(a: Array, lo: float, hi: float) -> Array:
    """Clamp with a straight-through gradient (NumPyro IAF trick)."""
    return a + jax.lax.stop_gradient(jnp.clip(a, lo, hi) - a)


class Affine:
    """Elementwise affine transform with log-scale clamping.

    params layout per dim: ``[shift mu, log-scale a]`` (``num_params == 2``).
    forward (noise->data): ``x = u * exp(a) + mu``, logdet ``= +sum(a)``.
    inverse (data->noise): ``u = (x - mu) * exp(-a)``, logdet ``= -sum(a)``.
    """

    num_params = 2

    def __init__(self, clamp_min: float = -5.0, clamp_max: float = 3.0):
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def _split(self, params: Array) -> tuple[Array, Array]:
        mu = params[..., 0]
        a = _clamp(params[..., 1], self.clamp_min, self.clamp_max)
        return mu, a

    def forward(self, u: Array, params: Array) -> tuple[Array, Array]:
        mu, a = self._split(params)
        x = u * jnp.exp(a) + mu
        return x, jnp.sum(a)

    def inverse(self, x: Array, params: Array) -> tuple[Array, Array]:
        mu, a = self._split(params)
        u = (x - mu) * jnp.exp(-a)
        return u, -jnp.sum(a)

    def forward_dim(self, u_i: Array, params_i: Array) -> Array:
        """Scalar forward for one dim (used by the sequential sampling scan)."""
        mu = params_i[0]
        a = _clamp(params_i[1], self.clamp_min, self.clamp_max)
        return u_i * jnp.exp(a) + mu
