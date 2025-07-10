import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx
from jax import Array

from gensbi.utils.model_wrapping import ModelWrapper, GuidedModelWrapper


class DummyModel(nnx.Module):
    def __call__(self, x: Array, t: Array, args=None, conditioned=True, **kwargs):
        # Ensure x and t are arrays and compatible for broadcasting
        x = jnp.array(x)
        t = jnp.array(t)
        # Broadcast t to x's shape if needed
        if t.shape != x.shape:
            t = jnp.broadcast_to(t, x.shape)
        return x + t if conditioned else x - t


def test_model_wrapper_call_and_vector_field():
    model = DummyModel()
    wrapper = ModelWrapper(model)
    x = jnp.ones((2, 3))
    t = jnp.ones((2, 1))
    out = wrapper(x, t)
    assert out.shape == (2, 3)
    vf = wrapper.get_vector_field()
    vf_out = vf(t, x, None)
    assert vf_out.shape == (2, 3)


def test_model_wrapper_divergence():
    model = DummyModel()
    wrapper = ModelWrapper(model)
    x = jnp.ones((2, 2))
    t = jnp.ones((2, 1))
    div_fn = wrapper.get_divergence()
    div = div_fn(t, x, None)
    assert div.shape == (2,)


def test_guided_model_wrapper_call_and_vector_field():
    model = DummyModel()
    wrapper = GuidedModelWrapper(model, cfg_scale=0.5)
    x = jnp.ones((2, 3))
    t = jnp.ones((2, 1))
    out = wrapper(x, t)
    assert out.shape == (2, 3)
    vf = wrapper.get_vector_field()
    vf_out = vf(t, x, None)
    assert vf_out.shape == (2, 3)
