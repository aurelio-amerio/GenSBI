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


def test_posterior_wrapper_init():
    pipeline = MockPipeline()
    rngs = nnx.Rngs(0)
    
    # Test initialization with explicit shapes
    wrapper = PosteriorWrapper(
        pipeline, 
        rngs=rngs, 
        theta_shape=(3, 2), 
        x_shape=(4, 3)
    )
    assert wrapper.dim_theta == 3
    assert wrapper.ch_theta == 2
    assert wrapper.dim_x == 4
    assert wrapper.ch_x == 3

    # Test initialization with derived shapes (conditional)
    wrapper = PosteriorWrapper(pipeline, rngs=rngs)
    assert wrapper.dim_theta == pipeline.dim_obs
    assert wrapper.ch_theta == pipeline.ch_obs
    assert wrapper.dim_x == pipeline.dim_cond
    assert wrapper.ch_x == pipeline.ch_cond

    # Test initialization with derived shapes (unconditional / no ch_cond)
    pipeline.ch_cond = None
    wrapper = PosteriorWrapper(pipeline, rngs=rngs)
    assert wrapper.ch_x == pipeline.ch_obs # Defaults to ch_theta if cond is None


def test_posterior_wrapper_defaults():
    pipeline = MockPipeline()
    wrapper = PosteriorWrapper(pipeline, rngs=nnx.Rngs(0))
    
    # Test set_default_x
    default_x = jnp.ones((1, 4, 3))
    wrapper.set_default_x(default_x)
    assert wrapper.default_x is not None
    assert wrapper.default_x.shape == (1, 4*3) # Processed shape

    # Test sapmle with default x
    samples = wrapper.sample(sample_shape=(4,))
    assert samples.shape == (4, pipeline.dim_obs * pipeline.ch_obs)

    # Test sample_batched with default x
    samples_batched = wrapper.sample_batched(sample_shape=(5,))
    # Note: MockPipeline returns (nsamples, cond.shape[0], ...)
    # default_x has batch size 1, so result should handle that
    assert samples_batched.shape == (5, 1, pipeline.dim_obs * pipeline.ch_obs)


def test_process_x_validation():
    pipeline = MockPipeline()
    wrapper = PosteriorWrapper(pipeline, rngs=nnx.Rngs(0))

    # Test 3D input with wrong channels
    with pytest.raises(AssertionError, match="Wrong number of channels"):
        wrapper._process_x(jnp.zeros((1, 4, 99))) # Wrong channels
    
    # Test invalid dimensions
    with pytest.raises(AssertionError, match="x must be of shape"):
        wrapper._process_x(jnp.zeros((1, 4, 3, 5))) # 4D


def test_batched_sample_kwargs():
    pipeline = MockPipeline()
    wrapper = PosteriorWrapper(pipeline, rngs=nnx.Rngs(0), chunk_size=10)
    
    # Verify chunk_size is passed/overridden
    # We can't easily check internal call args with just MockPipeline unless we modify it to store args
    # But we can verify it runs without error
    wrapper.sample_batched(sample_shape=(5,), x=jnp.zeros((2, 4, 3)), chunk_size=20)
    # Coverage will confirm lines are hit

