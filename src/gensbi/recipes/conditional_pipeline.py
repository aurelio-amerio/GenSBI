"""
Pipeline for training and using a Conditional model for simulation-based inference.

Examples:
    .. code-block:: python
    
        import grain
        import numpy as np
        import jax
        from jax import numpy as jnp
        from gensbi.recipes import ConditionalPipeline

        # Define your training and validation datasets.
        train_data = jax.random.rand((1024, 4)) # your training dataset
        val_data = jax.random.rand((128, 4)) # your validation dataset

        batch_size = 32

        train_dataset_grain = (
            grain.MapDataset.source(np.array(train_data)[...,None])
            .shuffle(42)
            .repeat()
            .to_iter_dataset()
            .batch(batch_size)
            # .mp_prefetch() # Uncomment if you want to use multiprocessing prefetching
        )

        val_dataset_grain = (
            grain.MapDataset.source(np.array(val_data)[...,None])
            .shuffle(42)
            .repeat()
            .to_iter_dataset()
            .batch(batch_size) 
            # .mp_prefetch() # Uncomment if you want to use multiprocessing prefetching
        )
        
        # Define your model 
        model = ...  # your nnx.Module model here, e.g., a simple MLP, or the Flux1 model
        # if you define a custom model, it should take as input the following arguments:
        #    t: Array,
        #    obs: Array,
        #    obs_ids: Array,
        #    cond: Array,
        #    cond_ids: Array, 
        
        # the obs should have shape (batch_size, dim_theta, c), 
        # the cond should have shape (batch_size, dim_x, c),
        # obs_ids and cond_ids should match obs and cond respectively,
        # and the output will be of the same shape as obs

        dim_theta = 2  # Dimension of the parameter space
        dim_x = 2      # Dimension of the observation space
        pipeline = ConditionalPipeline(model, train_dataset_grain, val_dataset_grain, dim_theta, dim_x)

        # Train the model
        rngs = jax.random.PRNGKey(0)
        pipeline.train(rngs)

        # Sample from the posterior
        x_o = jnp.array([0.5, -0.2])  # Example
        samples = pipeline.sample(rngs, x_o, nsamples=10000, step_size=0.01)
    
    .. note::
    
        If you plan on using multiprocessing prefetching, ensure that your script is wrapped in a `if __name__ == "__main__":` guard. See https://docs.python.org/3/library/multiprocessing.html
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

from gensbi.models import  ConditionalCFMLoss, ConditionalWrapper, ConditionalDiffLoss

from einops import repeat

from gensbi.utils.model_wrapping import _expand_dims

import os

import yaml

from gensbi.recipes.pipeline import AbstractPipeline


class ConditionalFlowPipeline(AbstractPipeline):
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        dim_theta: int,
        dim_x: int,
        params=None,
        training_config=None,
    ):
        """
        Flow pipeline for training and using a Conditional model for simulation-based inference.

        Parameters
        ----------
        model: nnx.Module
            The model to be trained.
        train_dataset : grain dataset or iterator over batches
            Training dataset.
        val_dataset : grain dataset or iterator over batches
            Validation dataset.
        dim_theta : int
            Dimension of the parameter space.
        dim_x : int
            Dimension of the observation space.
        params : ConditionalParams, optional
            Parameters for the Conditional model. If None, default parameters are used.
        training_config : dict, optional
            Configuration for training. If None, default configuration is used.

        """
        super().__init__(
            train_dataset, val_dataset, dim_theta, dim_x, model, params, training_config
        )

        # self.cond_ids = self.cond_ids.reshape(1, -1, 1)
        # self.obs_ids = self.obs_ids.reshape(1, -1, 1)
        self.cond_ids = _expand_dims(self.cond_ids)
        self.obs_ids = _expand_dims(self.obs_ids)

        self.path = AffineProbPath(scheduler=CondOTScheduler())

        self.loss_fn = ConditionalCFMLoss(self.path)

        self.p0_dist_model = dist.Independent(
            dist.Normal(
                loc=jnp.zeros((self.dim_theta, 1)), scale=jnp.ones((self.dim_theta, 1))
            ),
            reinterpreted_batch_ndims=1,
        )

    @classmethod
    def init_pipeline_from_config(
        cls,
    ):
        raise NotImplementedError(
            "Initialization from config not implemented for ConditionalFlowPipeline."
        )

    def _make_model(self):
        raise NotImplementedError(
            "Model creation not implemented for ConditionalFlowPipeline."
        )

    def _get_default_params(self):
        raise NotImplementedError(
            "Default parameters not implemented for ConditionalFlowPipeline."
        )

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
        self.model_wrapped = ConditionalWrapper(self.model)
        self.ema_model_wrapped = ConditionalWrapper(self.ema_model)
        return

    def sample(
        self, rng, x_o, nsamples=10_000, step_size=0.01, use_ema=True, time_grid=None, **model_extras
    ):
        if use_ema:
            vf_wrapped = self.ema_model_wrapped
        else:
            vf_wrapped = self.model_wrapped

        if time_grid is None:
            time_grid = jnp.array([0.0, 1.0])
            return_intermediates = False
        else:
            assert jnp.all(time_grid[:-1] <= time_grid[1:])
            return_intermediates = True

        x_init = jax.random.normal(rng, (nsamples, self.dim_theta))
        # cond = jnp.broadcast_to(x_o[..., None], (1, self.dim_x, 1))
        cond = _expand_dims(x_o)

        solver = ODESolver(velocity_model=vf_wrapped)
        model_extras = {
            "cond": cond,
            "obs_ids": self.obs_ids,
            "cond_ids": self.cond_ids,
            **model_extras
        }

        sampler_ = solver.get_sampler(
            method="Dopri5",
            step_size=step_size,
            return_intermediates=return_intermediates,
            model_extras=model_extras,
            time_grid=time_grid,
        )
        samples = sampler_(x_init)
        return samples
    
    def compute_unnorm_logprob(
        self, x_1, x_o, step_size=0.01, use_ema=True, time_grid=None, **model_extras
    ):
        if use_ema:
            model = self.ema_model_wrapped
        else:
            model = self.model_wrapped

        if time_grid is None:
            time_grid = jnp.array([1.0, 0.0])
            return_intermediates = False
        else:
            # assert time grid is decreasing
            assert jnp.all(time_grid[:-1] >= time_grid[1:])
            return_intermediates = True

        solver = ODESolver(velocity_model=model)

        # x_1 = _expand_dims(x_1)
        assert (
            x_1.ndim == 2
        ), "x_1 must be of shape (num_samples, dim_obs), currently sampling for multiple channels is not supported."
        cond = _expand_dims(x_o)

        p0_cond = dist.Independent(
            dist.Normal(
                loc=jnp.zeros((x_1.shape[1],)), scale=jnp.ones((x_1.shape[1],))
            ),
            reinterpreted_batch_ndims=1,
        )

        model_extras = {
            "cond": cond,
            "obs_ids": self.obs_ids,
            "cond_ids": self.cond_ids,
            **model_extras
        }

        logp_sampler = solver.get_unnormalized_logprob(
            time_grid=time_grid,
            method="Dopri5",
            step_size=step_size,
            log_p0=p0_cond.log_prob,
            model_extras=model_extras,
            return_intermediates=return_intermediates,
        )

        if len(x_1)>4:
            # we trigger precompilation first
            _ = logp_sampler(x_1[:4])

        exact_log_p = logp_sampler(x_1)
        return exact_log_p


class ConditionalDiffusionPipeline(AbstractPipeline):
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        dim_theta: int,
        dim_x: int,
        params=None,
        training_config=None,
    ):
        """
        Diffusion pipeline for training and using a Conditional model for simulation-based inference.

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
        params : ConditionalParams, optional
            Parameters for the Conditional model. If None, default parameters are used.
        training_config : dict, optional
            Configuration for training. If None, default configuration is used.

        """
        super().__init__(
            train_dataset, val_dataset, dim_theta, dim_x, model, params, training_config
        )

        self.cond_ids = _expand_dims(self.cond_ids)
        self.obs_ids = _expand_dims(self.obs_ids)

        self.path = EDMPath(
            scheduler=EDMScheduler(
                sigma_min=self.training_config["sigma_min"],
                sigma_max=self.training_config["sigma_max"],
            )
        )

        self.loss_fn = ConditionalDiffLoss(self.path)

    @classmethod
    def init_pipeline_from_config(
        cls,
    ):
        raise NotImplementedError(
            "Initialization from config not implemented for ConditionalDiffusionPipeline."
        )

    def _make_model(self):
        raise NotImplementedError(
            "Model creation not implemented for ConditionalDiffusionPipeline."
        )

    def _get_default_params(self):
        raise NotImplementedError(
            "Default parameters not implemented for ConditionalDiffusionPipeline."
        )

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
        self.model_wrapped = ConditionalWrapper(self.model)
        self.ema_model_wrapped = ConditionalWrapper(self.ema_model)
        return

    def sample(
        self,
        rng,
        x_o,
        nsamples=10_000,
        nsteps=18,
        use_ema=True,
        return_intermediates=False,
        **model_extras
    ):
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
            **model_extras
        }

        x_init = self.path.sample_prior(key1, (nsamples, self.dim_theta, 1))

        samples = solver.sample(
            key2,
            x_init,
            nsteps=nsteps,
            model_extras=model_extras,
            return_intermediates=return_intermediates,
        )

        return jnp.squeeze(samples, axis=-1)
