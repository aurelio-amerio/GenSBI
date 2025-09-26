import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx
from jax import Array

from gensbi.utils.model_wrapping import ModelWrapper, GuidedModelWrapper


class DummyModel(nnx.Module):
    def __call__(self, x: Array, t: Array, *args, conditioned=True, **kwargs):
        # Ensure x and t are arrays and compatible for broadcasting
        x = jnp.asarray(x)
        t = jnp.asarray(t)

        # we expect x of shape (batch_size, features, C) and t of shape (batch_size, 1), we need to expand t to (batch_size, 1, 1)
        if t.ndim == 1:
            t = t[..., None, None]
        elif t.ndim == 2:
            t = t[..., None]
            
        res = x + t if conditioned else x - t
        return res


def test_model_wrapper_call_and_vector_field():
    model = DummyModel()
    wrapper = ModelWrapper(model)
    x = jnp.ones((2, 3, 1))
    t = jnp.ones((2, 1))
    out = wrapper(t,x)
    assert out.shape == (2, 3, 1)
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
    assert div.shape == (2,), f"Expected divergence shape (2,), got {div.shape}"


def test_guided_model_wrapper_call_and_vector_field():
    model = DummyModel()
    wrapper = GuidedModelWrapper(model, cfg_scale=0.5)
    x = jnp.ones((2, 3))
    t = jnp.ones((2, 1))
    out = wrapper(t,x)
    assert out.shape == (2, 3, 1), f"Expected output shape (2, 3, 1), got {out.shape}"
    vf = wrapper.get_vector_field()
    vf_out = vf(t, x, None)
    assert vf_out.shape == (2, 3), f"Expected vector field shape (2, 3), got {vf_out.shape}"