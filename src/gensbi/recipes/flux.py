"""
Pipeline for training and using a Flux1 model for simulation-based inference.

Example:
    .. code-block:: python
        import itertools
        import jax
        from jax import numpy as jnp
        from gensbi.recipes import FluxPipeline

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
        pipeline = FluxPipeline(train_dataset, val_dataset, dim_theta, dim_x)

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

from gensbi.models import Flux, FluxParams, FluxCFMLoss, FluxWrapper, FluxDiffLoss

from einops import repeat

from gensbi.utils.model_wrapping import _expand_dims

import os

from gensbi.recipes.pipeline import AbstractPipeline


class FluxFlowPipeline(AbstractPipeline):
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
        Flow pipeline for training and using a Flux1 model for simulation-based inference.

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
        params : FluxParams, optional
            Parameters for the Flux model. If None, default parameters are used.
        training_config : dict, optional
            Configuration for training. If None, default configuration is used.

        """
        super().__init__(
            train_dataset, val_dataset, dim_theta, dim_x, params, training_config
        )

        # self.cond_ids = self.cond_ids.reshape(1, -1, 1)
        # self.obs_ids = self.obs_ids.reshape(1, -1, 1)
        self.cond_ids = _expand_dims(self.cond_ids)
        self.obs_ids = _expand_dims(self.obs_ids)

        self.path = AffineProbPath(scheduler=CondOTScheduler())

        self.loss_fn = FluxCFMLoss(self.path)

        self.p0_dist_model = dist.Independent(
            dist.Normal(
                loc=jnp.zeros((self.dim_theta, 1)), scale=jnp.ones((self.dim_theta, 1))
            ),
            reinterpreted_batch_ndims=1,
        )

    def _make_model(self):
        """
        Create and return the Flux model to be trained.
        """
        model = Flux(self.params)
        return model

    def _get_default_params(self):
        """
        Return default parameters for the Flux model.
        """
        params = FluxParams(
            in_channels=1,
            vec_in_dim=None,
            context_in_dim=1,
            mlp_ratio=4,
            qkv_multiplier=1,
            num_heads=6,
            depth=8,
            depth_single_blocks=16,
            axes_dim=[6],
            qkv_bias=True,
            obs_dim=self.dim_theta,
            cond_dim=self.dim_x,
            theta=20,
            rngs=nnx.Rngs(default=42),
            param_dtype=jnp.float32,
        )
        return params

    def get_loss_fn(
        self,
    ):
        def loss_fn(model, batch, key: jax.random.PRNGKey):
            obs = batch[:, : self.dim_theta, ...]
            cond = batch[:, self.dim_theta :, ...]

            batch_size = batch.shape[0]

            rng_x0, rng_t = jax.random.split(key, 2)

            x_1 = obs
            x_0 = self.p0_dist_model.sample(rng_x0, (batch_size,))
            t = jax.random.uniform(rng_t, x_1.shape[0])

            batch = (x_0, x_1, t)

            loss = self.loss_fn(model, batch, cond, self.obs_ids, self.cond_ids)
            return loss

        return loss_fn

    def _wrap_model(self):
        self.model_wrapped = FluxWrapper(self.model)
        self.ema_model_wrapped = FluxWrapper(self.ema_model)
        return

    def sample(self, rng, x_o, nsamples=10_000, step_size=0.01, use_ema=True):
        if use_ema:
            vf_wrapped = self.ema_model_wrapped
        else:
            vf_wrapped = self.model_wrapped

        x_init = jax.random.normal(rng, (nsamples, self.dim_theta))
        # cond = jnp.broadcast_to(x_o[..., None], (1, self.dim_x, 1))
        cond = _expand_dims(x_o)

        solver = ODESolver(velocity_model=vf_wrapped)
        model_extras = {
            "cond": cond,
            "obs_ids": self.obs_ids,
            "cond_ids": self.cond_ids,
        }

        sampler_ = solver.get_sampler(
            method="Dopri5",
            step_size=step_size,
            return_intermediates=False,
            model_extras=model_extras,
        )
        samples = sampler_(x_init)
        return samples


class FluxDiffusionPipeline(AbstractPipeline):
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
        Diffusion pipeline for training and using a Flux1 model for simulation-based inference.

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
        params : FluxParams, optional
            Parameters for the Flux model. If None, default parameters are used.
        training_config : dict, optional
            Configuration for training. If None, default configuration is used.

        """
        super().__init__(
            train_dataset, val_dataset, dim_theta, dim_x, params, training_config
        )

        self.cond_ids = _expand_dims(self.cond_ids)
        self.obs_ids = _expand_dims(self.obs_ids)

        self.path = EDMPath(
            scheduler=EDMScheduler(
                sigma_min=self.training_config["sigma_min"],
                sigma_max=self.training_config["sigma_max"],
            )
        )

        self.loss_fn = FluxDiffLoss(self.path)

    def _make_model(self):
        """
        Create and return the Flux model to be trained.
        """
        model = Flux(self.params)
        return model

    def _get_default_params(self):
        """
        Return default parameters for the Flux model.
        """
        params = FluxParams(
            in_channels=1,
            vec_in_dim=None,
            context_in_dim=1,
            mlp_ratio=4,
            qkv_multiplier=1,
            num_heads=6,
            depth=8,
            depth_single_blocks=16,
            axes_dim=[6],
            qkv_bias=True,
            obs_dim=self.dim_theta,
            cond_dim=self.dim_x,
            theta=20,
            rngs=nnx.Rngs(default=42),
            param_dtype=jnp.float32,
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
        def loss_fn(model, batch, key: jax.random.PRNGKey):
            # jax debug print(batch.shape)
            # (batch_size, dim_theta + dim_x)

            obs = jnp.take_along_axis(batch, self.obs_ids, axis=1)
            cond = jnp.take_along_axis(batch, self.cond_ids, axis=1)
            # obs = batch[:, : self.dim_theta, ...]
            # cond = batch[:, self.dim_theta :, ...]

            rng_x0, rng_sigma = jax.random.split(key, 2)

            x_1 = obs
            sigma = self.path.sample_sigma(rng_sigma, x_1.shape[0])
            sigma = repeat(sigma, f"b -> b {'1 ' * (x_1.ndim - 1)}")  # TODO fixme

            batch = (x_1, sigma)
            loss = self.loss_fn(rng_x0, model, batch, cond, self.obs_ids, self.cond_ids)
            return loss

        return loss_fn

    def _wrap_model(self):
        self.model_wrapped = FluxWrapper(self.model)
        self.ema_model_wrapped = FluxWrapper(self.ema_model)
        return

    def sample(self, rng, x_o, nsamples=10_000, nsteps=18, use_ema=True):
        if use_ema:
            model = self.ema_model_wrapped
        else:
            model = self.model_wrapped

        key1, key2 = jax.random.split(rng, 2)

        # cond = jnp.broadcast_to(x_o[..., None], (1, self.dim_x, 1))
        cond = _expand_dims(x_o)

        solver = SDESolver(score_model=model, path=self.path)

        model_extras = {
            "cond": cond,
            "obs_ids": self.obs_ids,
            "cond_ids": self.cond_ids,
        }

        x_init = self.path.sample_prior(key1, (nsamples, self.dim_theta, 1))

        samples = solver.sample(key2, x_init, nsteps=nsteps, model_extras=model_extras)

        return jnp.squeeze(samples, axis=-1)
