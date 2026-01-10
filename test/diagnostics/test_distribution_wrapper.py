# %%
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

from flax import nnx

import pytest

from gensbi.diagnostics import PosteriorWrapper


# %%
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


# %%


def test_posterior_wrapper_processing():
    pipeline = MockPipeline()
    wrapper = PosteriorWrapper(pipeline, rngs=nnx.Rngs(0))

    xs = jnp.ones((10, pipeline.dim_cond, pipeline.ch_cond))
    xs_raveled = wrapper._ravel(jnp.array(xs))

    assert xs_raveled.shape == (10, pipeline.dim_cond * pipeline.ch_cond)

    xs_unraveled = wrapper._unravel_xs(xs_raveled)
    assert xs_unraveled.shape == (10, pipeline.dim_cond, pipeline.ch_cond)

    theta = jnp.ones((15, pipeline.dim_obs, pipeline.ch_obs))
    theta_raveled = wrapper._ravel(jnp.array(theta))
    assert theta_raveled.shape == (15, pipeline.dim_obs * pipeline.ch_obs)

    theta_unraveled = wrapper._unravel_theta(theta_raveled)
    assert theta_unraveled.shape == (15, pipeline.dim_obs, pipeline.ch_obs)

    xs = jnp.ones((20, pipeline.dim_cond * pipeline.ch_cond))
    xs_processed = wrapper._process_x(jnp.array(xs))
    assert xs_processed.shape == (20, pipeline.dim_cond * pipeline.ch_cond)
    xs = jnp.ones((25, pipeline.dim_cond, pipeline.ch_cond))
    xs_processed_2 = wrapper._process_x(jnp.array(xs))
    assert xs_processed_2.shape == (25, pipeline.dim_cond * pipeline.ch_cond)

    return


def test_distribution_wrapper_sampling():
    pipeline = MockPipeline()
    wrapper = PosteriorWrapper(pipeline, rngs=nnx.Rngs(0))

    # Test single sampling
    samples = wrapper.sample(sample_shape=(4,), x=jnp.zeros((1, 4, 3)))

    assert samples.shape == (
        4,
        pipeline.dim_obs * pipeline.ch_obs,
    ), f"Unexpected shape: {samples.shape}"

    # Test batched sampling
    samples_batched = wrapper.sample_batched(sample_shape=(5,), x=jnp.zeros((2, 4, 3)))
    assert samples_batched.shape == (
        5,
        2,
        pipeline.dim_obs * pipeline.ch_obs,
    ), f"Unexpected shape: {samples_batched.shape}"
