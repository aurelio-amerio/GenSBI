#%%
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

from flax import nnx

import pytest

from gensbi.diagnostics import PosteriorWrapper

#%%
class MockPipeline:
    def __init__(
        self,
    ):

        self.dim_obs = 3
        self.ch_obs = 2

        self.dim_cond = 4
        self.ch_cond = 3

    def sample(self, key, cond, nsamples, *args, **kwargs):
        shape = (nsamples, self.dim_obs, self.ch_obs)
        return jnp.zeros(shape)

    def sample_batched(
        self,
        key,
        cond,
        nsamples,
        *args,
        **kwargs,
    ):
        shape = (nsamples, cond.shape[0], self.dim_obs, self.ch_obs)
        return jnp.zeros(shape)

#%%   
def test_distribution_wrapper_sampling():
    pipeline = MockPipeline()
    wrapper = PosteriorWrapper(pipeline, rngs=nnx.Rngs(0))

    # Test single sampling
    samples = wrapper.sample(sample_shape=(4,), x = jnp.zeros((1, 4, 3)))

    assert samples.shape == (4, pipeline.dim_obs*pipeline.ch_obs), f"Unexpected shape: {samples.shape}"

    # Test batched sampling
    samples_batched = wrapper.sample_batched(
        sample_shape=(5,),
        x = jnp.zeros((2, 4, 3))
    )
    assert samples_batched.shape == (5, 2, pipeline.dim_obs*pipeline.ch_obs), f"Unexpected shape: {samples_batched.shape}"

