"""
Pipeline for training and using a Flux1 model for simulation-based inference.

Examples:
    .. code-block:: python
    
        import grain
        import numpy as np
        import jax
        from jax import numpy as jnp
        from gensbi.recipes import JointPipeline

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
        model = ...  # your nnx.Module model here, e.g., a simple MLP, or the Simformer or Flux1Joint model
        # if you define a custom model, it should take as input the following arguments:
        #    t: Array,
        #    obs: Array,
        #    node_ids: Array,
        #    condition_mask: Array,
        #    *args,
        #    **kwargs,
        
        # the obs should have shape (batch_size, dim_joint, c), 
        # node_ids and condition_mask should match obs shape,
        # and the output will be of the same shape as obs
        
        dim_theta = 2  # Dimension of the parameter space
        dim_x = 2      # Dimension of the observation space
        pipeline = JointPipeline(model, train_dataset_grain, val_dataset_grain, dim_theta, dim_x)

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

from einops import repeat

from gensbi.models import (
    JointCFMLoss,
    JointWrapper,
    JointDiffLoss,
)

import numpyro.distributions as dist

from gensbi.utils.model_wrapping import _expand_dims

import os
import yaml

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
    Sample structured conditional masks for the Joint model.

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
    return condition_mask[..., None]


class JointFlowPipeline(AbstractPipeline):
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
        Flow pipeline for training and using a Joint model for simulation-based inference.

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
        params : JointParams, optional
            Parameters for the Joint model. If None, default parameters are used.
        training_config : dict, optional
            Configuration for training. If None, default configuration is used.

        """
        super().__init__(
            train_dataset, val_dataset, dim_theta, dim_x, model, params, training_config
        )

        self.cond_ids = _expand_dims(self.cond_ids)
        self.obs_ids = _expand_dims(self.obs_ids)
        self.node_ids = _expand_dims(self.node_ids)

        self.path = AffineProbPath(scheduler=CondOTScheduler())

        self.loss_fn = JointCFMLoss(self.path)

        self.p0_dist_model = dist.Independent(
            dist.Normal(
                loc=jnp.zeros((self.dim_joint, 1)), scale=jnp.ones((self.dim_joint, 1))
            ),
            reinterpreted_batch_ndims=1,
        )
        
        if self.dim_x == 0:
            raise ValueError("JointFlowPipeline initialized as unconditional since dim_x=0. Please use `UnconditionalFlowPipeline` instead.")


    @classmethod
    def init_pipeline_from_config(cls):
        raise NotImplementedError(
            "init_pipeline_from_config is not implemented for JointFlowPipeline."
        )

    def _make_model(self):
        raise NotImplementedError(
            "_make_model is not implemented for JointFlowPipeline."
        )   

    def _get_default_params(self):
        raise NotImplementedError(
            "_get_default_params is not implemented for JointFlowPipeline."
        )

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

            loss = self.loss_fn(
                model,
                batch,
                node_ids=self.node_ids,
                condition_mask=condition_mask,
            )
            return loss

        return loss_fn

    def _wrap_model(self):
        self.model_wrapped = JointWrapper(self.model)
        self.ema_model_wrapped = JointWrapper(self.ema_model)
        return

    def sample(
        self, key, x_o, nsamples=10_000, step_size=0.01, use_ema=True, time_grid=None, **model_extras
    ):
            
        if use_ema:
            model = self.ema_model_wrapped
        else:
            model = self.model_wrapped

        if time_grid is None:
            time_grid = jnp.array([0.0, 1.0])
            return_intermediates = False
        else:
            assert jnp.all(time_grid[:-1] <= time_grid[1:])
            return_intermediates = True

        x_init = jax.random.normal(key, (nsamples, self.dim_theta))
        # cond = jnp.broadcast_to(x_o[..., None], (1, self.dim_x, 1))
        cond = _expand_dims(x_o)

        solver = ODESolver(velocity_model=model)

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

        exact_log_p = logp_sampler(x_1)
        return exact_log_p


class JointDiffusionPipeline(AbstractPipeline):
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
        Diffusion pipeline for training and using a Joint model for simulation-based inference.

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
        params : optional
            Parameters for the Joint model. If None, default parameters are used.
        training_config : dict, optional
            Configuration for training. If None, default configuration is used.

        """
        super().__init__(
            train_dataset, val_dataset, dim_theta, dim_x, model, params, training_config
        )

        self.cond_ids = _expand_dims(self.cond_ids)
        self.obs_ids = _expand_dims(self.obs_ids)
        self.node_ids = _expand_dims(self.node_ids)
        

        self.path = EDMPath(
            scheduler=EDMScheduler(
                sigma_min=self.training_config["sigma_min"],
                sigma_max=self.training_config["sigma_max"],
            )
        )

        self.loss_fn = JointDiffLoss(self.path)
        
        if self.dim_x == 0:
            raise ValueError("JointFlowPipeline initialized as unconditional since dim_x=0. Please use `UnconditionalFlowPipeline` instead.")

    @classmethod
    def init_pipeline_from_config(
        cls,
    ):
        raise NotImplementedError(
            "init_pipeline_from_config is not implemented for JointDiffusionPipeline."
        )

    def _make_model(self):
        raise NotImplementedError(
            "_make_model is not implemented for JointDiffusionPipeline."
        )

    def _get_default_params(self):
        raise NotImplementedError(
            "_get_default_params is not implemented for JointDiffusionPipeline."
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

            loss = self.loss_fn(
                rng_x0,
                model,
                batch,
                condition_mask=condition_mask,
                node_ids=self.node_ids,
            )
            return loss

        return loss_fn

    def _wrap_model(self):
        self.model_wrapped = JointWrapper(self.model)
        self.ema_model_wrapped = JointWrapper(self.ema_model)
        return

    def sample(
        self,
        key,
        x_o,
        nsamples=10_000,
        nsteps=18,
        use_ema=True,
        return_intermediates=False,
        **model_extras,
    ):
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
            **model_extras,
            
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
