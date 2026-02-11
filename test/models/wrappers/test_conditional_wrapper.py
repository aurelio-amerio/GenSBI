import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx
from jax import Array

from gensbi.models.wrappers.conditional import ConditionalWrapper

from gensbi.utils.math import _expand_dims, _expand_time


class DummyModel(nnx.Module):
    def __call__(
        self,
        obs: Array,
        obs_ids: Array,
        cond: Array,
        cond_ids: Array,
        t: Array,
        *args,
        **kwargs,
    ):
        # Ensure x and t are arrays and compatible for broadcasting
        x = _expand_dims(obs)
        t = _expand_time(t)

        t = t[..., None]

        res = x + t
        return res


def test_conditional_wrapper():
    model = DummyModel()
    wrapper = ConditionalWrapper(model)

    t = jnp.zeros((2, 1))
    obs = jnp.zeros((2, 3, 4))
    obs_ids = jnp.zeros((2, 3, 1))
    cond = jnp.zeros((2, 4, 4))
    cond_ids = jnp.zeros((2, 4, 1))

    out = wrapper(
        t,
        obs,
        obs_ids,
        cond,
        cond_ids,
    )
    assert out.shape == obs.shape
