import os
os.environ['JAX_PLATFORMS']="cpu"

import jax.numpy as jnp
import jax

import numpy as np
import torch

from flax import nnx

from gensbi.diagnostics import PosteriorWrapper
from gensbi.recipes.pipeline import _get_batch_sampler

class DummyPipeline():
    
    def __init__(self):

        self.dim_obs = 4
        self.ch_obs = 2
        self.dim_cond = 6
        self.ch_cond = 3
    
    def get_sampler(
        self,
        x_o,
        **kwargs,
    ):

        def sampler(key, nsamples):
            samples = jnp.ones((nsamples, self.dim_obs, self.ch_obs))
            return samples
        
        return sampler

    def sample(
        self,
        key,
        x_o,
        nsamples=10_000,
        **kwargs,
    ):

        sampler_ = self.get_sampler(
            x_o,
        )
        
        samples = sampler_(key, nsamples)

        return samples
    
    def sample_batched(
        self,
        key,
        x_o,
        nsamples,
        *args,
        chunk_size = 50,
        show_progress_bars=True,
        **kwargs,
    ):

        cond = x_o

        sampler = self.get_sampler(cond, *args, **kwargs)
        batched_sampler = _get_batch_sampler(
            sampler,
            ncond=cond.shape[0],
            chunk_size=chunk_size,
            show_progress_bars=show_progress_bars,
        )

        keys = jax.random.split(key, nsamples)

        res = batched_sampler(keys)

        return res
    
    
def test_posterior_wrapper_processing():
    pipeline = DummyPipeline()
    wrapper = PosteriorWrapper(pipeline, rngs=nnx.Rngs(0))
    
    xs = torch.Tensor(np.ones((10, pipeline.dim_cond, pipeline.ch_cond)))
    xs_raveled = wrapper._ravel(jnp.array(xs))
    
    assert xs_raveled.shape == (10, pipeline.dim_cond * pipeline.ch_cond)
    
    xs_unraveled = wrapper._unravel_xs(xs_raveled)
    assert xs_unraveled.shape == (10, pipeline.dim_cond, pipeline.ch_cond)
    
    theta = torch.Tensor(np.ones((15, pipeline.dim_obs,  pipeline.ch_obs)))
    theta_raveled = wrapper._ravel(jnp.array(theta))
    assert theta_raveled.shape == (15, pipeline.dim_obs * pipeline.ch_obs)
    
    theta_unraveled = wrapper._unravel_theta(theta_raveled)
    assert theta_unraveled.shape == (15, pipeline.dim_obs, pipeline.ch_obs)
    
    xs = torch.Tensor(np.ones((20, pipeline.dim_cond * pipeline.ch_cond)))
    xs_processed = wrapper._process_x(jnp.array(xs))
    assert xs_processed.shape == (20, pipeline.dim_cond * pipeline.ch_cond)
    xs = torch.Tensor(np.ones((25, pipeline.dim_cond, pipeline.ch_cond)))
    xs_processed_2 = wrapper._process_x(jnp.array(xs))
    assert xs_processed_2.shape == (25, pipeline.dim_cond * pipeline.ch_cond)
    
    return
    
    
def test_posterior_wrapper_sampling():

    pipeline = DummyPipeline()
    wrapper = PosteriorWrapper(pipeline, rngs=nnx.Rngs(0))

    # set default x
    x_o = torch.Tensor(np.ones((1, pipeline.dim_cond, pipeline.ch_cond)))
    wrapper.set_default_x(x_o)

    # sample from the posterior
    samples = wrapper.sample(sample_shape=(5,))

    assert samples.shape == (5, pipeline.dim_obs * pipeline.ch_obs)

    # test with different x shape
    x_o_2 = torch.Tensor(np.ones((1, pipeline.dim_cond * pipeline.ch_cond)))

    samples_2 = wrapper.sample(sample_shape=(3,), x=x_o_2)

    assert samples_2.shape == (3, pipeline.dim_obs * pipeline.ch_obs)
    
    # test batch sampling
    x_o_batch = torch.Tensor(np.ones((200, pipeline.dim_cond, pipeline.ch_cond)))

    samples_batch = wrapper.sample_batched(sample_shape=(500,), x=x_o_batch)

    assert samples_batch.shape == (500, 200, pipeline.dim_obs * pipeline.ch_obs)