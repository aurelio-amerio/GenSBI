"""
Pipeline for training and using a Flux1 model for simulation-based inference.

Example:
    .. code-block:: python
        import itertools
        import jax
        from jax import numpy as jnp
        from gensbi.recipes import SimformerPipeline

        # Define your training and validation datasets.
        train_data = jax.random.rand((1024, 4)) # your training dataset
        val_data = jax.random.rand((128, 4)) # your validation dataset

        batch_size = 32

        train_batch = train_data.reshape(-1, batch_size, train_data.shape[-1])
        val_batch = val_data.reshape(-1, batch_size, val_data.shape[-1])

        # Create datasets iterators (in this case with itertools, although a grain dataset is recommended)
        train_dataset = itertools.cycle(train_batch)
        val_dataset = itertools.cycle(val_batch)

        # Define the model
        dim_theta = 2  # Dimension of the parameter space
        dim_x = 2      # Dimension of the observation space
        pipeline = SimformerPipeline(train_dataset, val_dataset, dim_theta, dim_x)

        # Train the model
        rngs = jax.random.PRNGKey(0)
        pipeline.train(rngs)

        # Sample from the posterior
        x_o = jnp.array([0.5, -0.2])  # Example
        samples = pipeline.sample(rngs, x_o, nsamples=10000, step_size=0.01)
    
"""

import jax
import jax.numpy as jnp
from flax import nnx
import optax
from optax.contrib import reduce_on_plateau
from numpyro import distributions as dist
from tqdm.auto import tqdm
from functools import partial
import orbax.checkpoint as ocp

from gensbi.flow_matching.path import AffineProbPath
from gensbi.flow_matching.path.scheduler import CondOTScheduler
from gensbi.flow_matching.solver import ODESolver

from gensbi.diffusion.path import EDMPath
from gensbi.diffusion.path.scheduler import EDMScheduler, VEScheduler
from gensbi.diffusion.solver import SDESolver

from einops import repeat

from gensbi.models import (
    Simformer,
    SimformerParams,
    SimformerCFMLoss,
    SimformerWrapper,
    SimformerDiffLoss,
)

from gensbi.utils.model_wrapping import _expand_dims

import os

from gensbi.recipes.pipeline import AbstractPipeline, ModelEMA


def sample_strutured_conditional_mask(
    key,
    num_samples,
    theta_dim,
    x_dim,
    p_joint=0.2,
    p_posterior=0.2,
    p_likelihood=0.2,
    p_rnd1=0.2,
    p_rnd2=0.2,
    rnd1_prob=0.3,
    rnd2_prob=0.7,
):
    """
    Sample structured conditional masks for the Simformer model.

    Parameters
    ----------
    key : jax.random.PRNGKey
        Random key for sampling.
    num_samples : int
        Number of samples to generate.
    theta_dim : int
        Dimension of the parameter space.
    x_dim : int
        Dimension of the observation space.
    p_joint : float
        Probability of selecting the joint mask.
    p_posterior : float
        Probability of selecting the posterior mask.
    p_likelihood : float
        Probability of selecting the likelihood mask.
    p_rnd1 : float
        Probability of selecting the first random mask.
    p_rnd2 : float
        Probability of selecting the second random mask.
    rnd1_prob : float
        Probability of a True value in the first random mask.
    rnd2_prob : float
        Probability of a True value in the second random mask.

    Returns
    -------
    condition_mask : jnp.ndarray
        Array of shape (num_samples, theta_dim + x_dim) with boolean masks.

    """
    # Joint, posterior, likelihood, random1_mask, random2_mask
    key1, key2, key3 = jax.random.split(key, 3)
    joint_mask = jnp.array([False] * (theta_dim + x_dim), dtype=jnp.bool_)
    posterior_mask = jnp.array([False] * theta_dim + [True] * x_dim, dtype=jnp.bool_)
    likelihood_mask = jnp.array([True] * theta_dim + [False] * x_dim, dtype=jnp.bool_)
    random1_mask = jax.random.bernoulli(
        key2, rnd1_prob, shape=(theta_dim + x_dim,)
    ).astype(jnp.bool_)
    random2_mask = jax.random.bernoulli(
        key3, rnd2_prob, shape=(theta_dim + x_dim,)
    ).astype(jnp.bool_)
    mask_options = jnp.stack(
        [joint_mask, posterior_mask, likelihood_mask, random1_mask, random2_mask],
        axis=0,
    )  # (5, theta_dim + x_dim)
    idx = jax.random.choice(
        key1,
        5,
        shape=(num_samples,),
        p=jnp.array([p_joint, p_posterior, p_likelihood, p_rnd1, p_rnd2]),
    )
    condition_mask = mask_options[idx]
    all_ones_mask = jnp.all(condition_mask, axis=-1)
    # If all are ones, then set to false
    condition_mask = jnp.where(all_ones_mask[..., None], False, condition_mask)
    return condition_mask[...,None]


class SimformerFlowPipeline(AbstractPipeline):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        dim_theta: int,
        dim_x: int,
        params=None,
        training_config=None,
    ):
        """
        Flow pipeline for training and using a Simformer model for simulation-based inference.

        Parameters
        ----------
        train_dataset : grain dataset or iterator over batches
            Training dataset.
        val_dataset : grain dataset or iterator over batches
            Validation dataset.
        dim_theta : int
            Dimension of the parameter space.
        dim_x : int
            Dimension of the observation space.
        params : SimformerParams, optional
            Parameters for the Simformer model. If None, default parameters are used.
        training_config : dict, optional
            Configuration for training. If None, default configuration is used.

        """
        super().__init__(
            train_dataset, val_dataset, dim_theta, dim_x, params, training_config
        )

        self.path = AffineProbPath(scheduler=CondOTScheduler())

        self.loss_fn = SimformerCFMLoss(self.path)

        self.undirected_edge_mask = jnp.ones(
            (self.dim_joint, self.dim_joint), dtype=jnp.bool_
        )

        self.p0_dist_model = dist.Independent(
            dist.Normal(
                loc=jnp.zeros((self.dim_joint, 1)), scale=jnp.ones((self.dim_joint, 1))
            ),
            reinterpreted_batch_ndims=1,
        )

    def _make_model(self):
        """
        Create and return the Simformer model to be trained.
        """
        model = Simformer(self.params)
        return model

    def _get_default_params(self):
        """
        Return default parameters for the Simformer model.
        """
        params = SimformerParams(
            rngs=nnx.Rngs(0),
            dim_value=40,
            dim_id=40,
            dim_condition=10,
            dim_joint=self.dim_joint,
            fourier_features=128,
            num_heads=4,
            num_layers=8,
            widening_factor=3,
            qkv_features=40,
            num_hidden_layers=1,
        )
        return params

    def get_loss_fn(
        self,
    ):
        def loss_fn(
            model,
            x_1,
            key: jax.random.PRNGKey,
        ):
            batch_size = x_1.shape[0]
            rng_x0, rng_t, rng_condition = jax.random.split(key, 3)
            x_0 = self.p0_dist_model.sample(rng_x0, (batch_size,))
            t = jax.random.uniform(rng_t, x_1.shape[0])
            batch = (x_0, x_1, t)

            condition_mask = sample_strutured_conditional_mask(
                rng_condition,
                batch_size,
                self.dim_theta,
                self.dim_x,
            )

            edge_masks = self.undirected_edge_mask

            loss = self.loss_fn(
                model,
                batch,
                node_ids=self.node_ids,
                edge_mask=edge_masks,
                condition_mask=condition_mask,
            )
            return loss

        return loss_fn

    def _wrap_model(self):
        self.model_wrapped = SimformerWrapper(self.model)
        self.ema_model_wrapped = SimformerWrapper(self.ema_model)
        return

    def sample(self, key, x_o, nsamples=10_000, step_size=0.01, use_ema=True):
        if use_ema:
            model = self.ema_model_wrapped
        else:
            model = self.model_wrapped

        x_init = jax.random.normal(key, (nsamples, self.dim_theta))
        # cond = jnp.broadcast_to(x_o[..., None], (1, self.dim_x, 1))
        cond = _expand_dims(x_o)

        solver = ODESolver(velocity_model=model)
        model_extras = {
            "cond": cond,
            "obs_ids": self.obs_ids,
            "cond_ids": self.cond_ids,
            "edge_mask": self.undirected_edge_mask,
        }

        sampler_ = solver.get_sampler(
            method="Dopri5",
            step_size=step_size,
            return_intermediates=False,
            model_extras=model_extras,
        )
        samples = sampler_(x_init)
        return samples


class SimformerDiffusionPipeline(AbstractPipeline):
    def __init__(
        self,
        train_dataset,
        val_dataset,
        dim_theta: int,
        dim_x: int,
        params=None,
        training_config=None,
    ):
        """
        Diffusion pipeline for training and using a Simformer model for simulation-based inference.

        Parameters
        ----------
        train_dataset : grain dataset or iterator over batches
            Training dataset.
        val_dataset : grain dataset or iterator over batches
            Validation dataset.
        dim_theta : int
            Dimension of the parameter space.
        dim_x : int
            Dimension of the observation space.
        params : SimformerParams, optional
            Parameters for the Simformer model. If None, default parameters are used.
        training_config : dict, optional
            Configuration for training. If None, default configuration is used.

        """
        super().__init__(
            train_dataset, val_dataset, dim_theta, dim_x, params, training_config
        )

        self.path = EDMPath(
            scheduler=EDMScheduler(
                sigma_min=self.training_config["sigma_min"],
                sigma_max=self.training_config["sigma_max"],
            )
        )

        self.loss_fn = SimformerDiffLoss(self.path)

        self.undirected_edge_mask = jnp.ones(
            (self.dim_joint, self.dim_joint), dtype=jnp.bool_
        )

    def _make_model(self):
        """
        Create and return the Simformer model to be trained.
        """
        model = Simformer(self.params)
        return model

    def _get_default_params(self):
        """
        Return default parameters for the Simformer model.
        """
        params = SimformerParams(
            dim_value=40,
            dim_id=40,
            dim_condition=10,
            dim_joint=self.dim_joint,
            fourier_features=128,
            num_heads=4,
            num_layers=8,
            widening_factor=3,
            qkv_features=40,
            rngs=nnx.Rngs(0),
            num_hidden_layers=1,
        )
        return params

    @classmethod
    def _get_default_training_config(cls):
        config = super()._get_default_training_config()
        config.update(
            {
                "sigma_min": 0.002,  # from edm paper
                "sigma_max": 80.0,
            }
        )
        return config

    def get_loss_fn(
        self,
    ):
        def loss_fn(
            model,
            x_1,
            key: jax.random.PRNGKey,
        ):
            batch_size = x_1.shape[0]

            rng_x0, rng_sigma, rng_condition = jax.random.split(key, 3)

            sigma = self.path.sample_sigma(rng_sigma, x_1.shape[0])
            sigma = repeat(sigma, f"b -> b {'1 ' * (x_1.ndim - 1)}")

            batch = (x_1, sigma)

            condition_mask = sample_strutured_conditional_mask(
                rng_condition,
                batch_size,
                self.dim_theta,
                self.dim_x,
            )

            edge_masks = self.undirected_edge_mask

            loss = self.loss_fn(
                rng_x0,
                model,
                batch,
                condition_mask=condition_mask,
                node_ids=self.node_ids,
                edge_mask=edge_masks,
            )
            return loss

        return loss_fn

    def _wrap_model(self):
        self.model_wrapped = SimformerWrapper(self.model)
        self.ema_model_wrapped = SimformerWrapper(self.ema_model)
        return

    def sample(self, key, x_o, nsamples=10_000, nsteps=18, use_ema=True):
        if use_ema:
            model = self.ema_model_wrapped
        else:
            model = self.model_wrapped

        key1, key2 = jax.random.split(key, 2)

        cond = _expand_dims(x_o)

        solver = SDESolver(score_model=model, path=self.path)

        model_extras = {
            "cond": cond,
            "obs_ids": self.obs_ids,
            "cond_ids": self.cond_ids,
            "edge_mask": self.undirected_edge_mask,
        }

        x_init = self.path.sample_prior(key1, (nsamples, self.dim_theta, 1)) 

        samples = solver.sample(key2, x_init, nsteps=nsteps, model_extras=model_extras)

        return jnp.squeeze(samples, axis=-1)
