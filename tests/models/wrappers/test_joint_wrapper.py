import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx
from jax import Array

from gensbi.models.wrappers.joint import JointWrapper

from gensbi.utils.math import _expand_dims, _expand_time


class DummyModel(nnx.Module):
    def __call__(
        self,
        obs: Array,
        t: Array,
        node_ids: Array,
        condition_mask: Array,
        *args,
        **kwargs,
    ):
        # Ensure x and t are arrays and compatible for broadcasting
        x = _expand_dims(obs)
        t = _expand_time(t)

        t = t[..., None]

        res = x + t
        return res


def test_joint_wrapper():
    model = DummyModel()
    wrapper = JointWrapper(model)

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
        conditioned=True,
    )
    assert out.shape == obs.shape

    out = wrapper(
        t,
        obs,
        obs_ids,
        cond,
        cond_ids,
        conditioned=False,
    )
    assert out.shape == obs.shape
