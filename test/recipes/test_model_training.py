import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx


# def test_flux_cfmloss_runs():
#     path = AffineProbPath(scheduler=CondOTScheduler())
#     loss = FluxCFMLoss(path)
#     def vf(x, obs_ids, cond, cond_ids, t, conditioned=True):
#         return x + 1
#     x0 = jnp.ones((2, 2))
#     x1 = jnp.ones((2, 2))
#     t = jnp.ones((2,))
#     cond = jnp.ones((2, 2))
#     obs_ids = jnp.array([0, 1])
#     cond_ids = jnp.array([2, 3])
#     batch = (x0, x1, t)
#     result = loss(vf, batch, cond, obs_ids, cond_ids)
#     assert result is not None


from gensbi.models import SimformerParams, FluxParams
from gensbi.recipes import SimformerFlowPipeline, SimformerDiffusionPipeline, FluxFlowPipeline, FluxDiffusionPipeline

import itertools


nsamples = 1000
rng = jax.random.PRNGKey(0)

dim_theta = 2
dim_data = 7
dim_joint = dim_theta + dim_data


theta = jax.random.normal(rng, (nsamples, dim_theta, 1))
x = jax.random.normal(rng, (nsamples, dim_data, 1))

data = jnp.concatenate([theta, x], axis=1)

train_data = data[:800].reshape(10, -1, dim_joint, 1)
val_data = data[800:].reshape(10, -1, dim_joint, 1)

train_dataset = itertools.cycle(train_data)
val_dataset = itertools.cycle(val_data)

params_simf = SimformerParams(
    rngs = nnx.Rngs(0),
    dim_value = 2,
    dim_id = 2, 
    dim_condition = 2, 
    dim_joint= dim_joint,
    fourier_features = 32,
    num_heads = 2,
    num_layers = 1,
    widening_factor = 3,
    qkv_features = 10, 
    num_hidden_layers = 1)

pipeline_smf_flow = SimformerFlowPipeline(
    train_dataset, val_dataset, dim_theta, dim_data, params_simf
)

pipeline_smf_diff = SimformerDiffusionPipeline(
    train_dataset, val_dataset, dim_theta, dim_data, params_simf
)

params_flux = FluxParams(
            in_channels=1,
            vec_in_dim=None,
            context_in_dim=1,
            mlp_ratio=1,
            qkv_multiplier=1,
            num_heads=2,
            depth=2,
            depth_single_blocks=2,
            axes_dim=[2,],
            use_rope = False,
            qkv_bias=True,
            obs_dim = dim_theta,
            cond_dim = dim_data,
            theta=20,
            rngs=nnx.Rngs(default=42),
            param_dtype=jnp.float32,
        )

pipeline_flux_flow = FluxFlowPipeline(
    train_dataset, val_dataset, dim_theta, dim_data, params_flux
)

pipeline_flux_diff = FluxDiffusionPipeline(
    train_dataset, val_dataset, dim_theta, dim_data, params_flux
)

def test_simformer_flow_training_step():
    pipeline_smf_flow.train(nnx.Rngs(0), nsteps=2, save_model=False)
    sample = pipeline_smf_flow.sample(jax.random.PRNGKey(1), jnp.arange(dim_data)[None,...], nsamples=32)
    assert sample.shape == (32, dim_theta), f"Expected shape (32, {dim_theta}), got {sample.shape}"

def test_simformer_diff_training_step():
    pipeline_smf_diff.train(nnx.Rngs(0), nsteps=2, save_model=False)
    sample = pipeline_smf_diff.sample(jax.random.PRNGKey(1), jnp.arange(dim_data)[None,...], nsamples=32)
    assert sample.shape == (32, dim_theta), f"Expected shape (32, {dim_theta}), got {sample.shape}"

def test_flux_flow_training_step():
    pipeline_flux_flow.train(nnx.Rngs(0), nsteps=2, save_model=False)
    sample = pipeline_flux_flow.sample(jax.random.PRNGKey(1), jnp.arange(dim_data)[None,...], nsamples=32)
    assert sample.shape == (32, dim_theta), f"Expected shape (32, {dim_theta}), got {sample.shape}"

def test_flux_diff_training_step():
    pipeline_flux_diff.train(nnx.Rngs(0), nsteps=2, save_model=False)
    sample = pipeline_flux_diff.sample(jax.random.PRNGKey(1), jnp.arange(dim_data)[None,...], nsamples=32)
    assert sample.shape == (32, dim_theta), f"Expected shape (32, {dim_theta}), got {sample.shape}"

