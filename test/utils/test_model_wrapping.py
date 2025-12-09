import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx
from jax import Array

from gensbi.utils.model_wrapping import ModelWrapper # , GuidedModelWrapper

from gensbi.utils.math import _expand_dims, _expand_time


class DummyModel(nnx.Module):
    def __call__(self, x: Array, t: Array, *args, conditioned=True, **kwargs):
        # Ensure x and t are arrays and compatible for broadcasting
        x = _expand_dims(x)
        t = _expand_time(t)
 
        t = t[..., None]
            
        res = x + t if conditioned else x - t
        return res

    
def test_util_functions():
    x = jnp.ones((2, 3))
    t = jnp.ones((2,))
    expanded_x = _expand_dims(x)
    expanded_t = _expand_time(t)
    assert expanded_x.shape == (2, 3, 1), f"Expected shape (2, 3, 1), got {expanded_x.shape}"
    assert expanded_t.shape == (2, 1), f"Expected shape (2, 1), got {expanded_t.shape}"


def test_model_wrapper_call_and_vector_field():
    model = DummyModel()
    wrapper = ModelWrapper(model)
    x = jnp.ones((2, 3, 1))
    t = jnp.ones((2, 1))
    out = wrapper(t,x)
    assert out.shape == (2, 3, 1)
    vf = wrapper.get_vector_field()
    vf_out = vf(t, x, None)
    assert vf_out.shape == (2, 3,1)


def test_model_wrapper_divergence():
    model = DummyModel()
    wrapper = ModelWrapper(model)
    div_fn = wrapper.get_divergence()

    x = jnp.ones((3, 4, 2))
    t = jnp.ones((3,))
    div = div_fn(t, x, None)
    assert div.shape == (3,), f"Expected divergence shape (3,), got {div.shape}"

    x = jnp.ones((1, 2, 4))
    t = jnp.ones((1,))
    div = div_fn(t, x, None)
    assert div.shape == (), f"Expected divergence shape (), got {div.shape}"


# def test_guided_model_wrapper_call_and_vector_field():
#     model = DummyModel()
#     wrapper = GuidedModelWrapper(model, cfg_scale=0.5)
#     x = jnp.ones((2, 3, 1))
#     t = jnp.ones((2, 1, 1))
#     out = wrapper(t,x)
#     assert out.shape == (2, 3, 1), f"Expected output shape (2, 3, 1), got {out.shape}"
#     vf = wrapper.get_vector_field()
#     vf_out = vf(t, x, None)
#     assert vf_out.shape == (2, 3, 1), f"Expected vector field shape (2, 3, 1), got {vf_out.shape}"