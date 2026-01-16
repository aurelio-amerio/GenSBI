"""
Pipeline for training and using a Conditional model for simulation-based inference.
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

from typing import Union, Tuple

from gensbi.flow_matching.path import AffineProbPath
from gensbi.flow_matching.path.scheduler import CondOTScheduler
from gensbi.flow_matching.solver import ODESolver

from gensbi.diffusion.path import EDMPath
from gensbi.diffusion.path.scheduler import EDMScheduler, VEScheduler
from gensbi.diffusion.solver import SDESolver

from gensbi.models import ConditionalCFMLoss, ConditionalWrapper, ConditionalDiffLoss

from einops import repeat

from gensbi.models.flux1 import model
from gensbi.utils.model_wrapping import _expand_dims

import os

import yaml

from gensbi.recipes.pipeline import AbstractPipeline

from gensbi.recipes.utils import init_ids_1d, init_ids_2d


class ConditionalFlowPipeline(AbstractPipeline):
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
    dim_obs : int or tuple of int
        Dimension of the parameter space (number of tokens).
        Can represent unstructured data, time-series, or patchified 2D images. For images, provide a tuple (height, width).
    dim_cond : int or tuple of int
        Dimension of the observation space (number of tokens).
        Can represent unstructured data, time-series, or patchified 2D images. For images, provide a tuple (height, width).
    ch_obs : int, optional
        Number of channels per token in the observation data. Default is 1.
    ch_cond : int, optional
        Number of channels per token in the conditional data. Default is 1.
    params : ConditionalParams, optional
        Parameters for the Conditional model. If None, default parameters are used.
    training_config : dict, optional
        Configuration for training. If None, default configuration is used.

    Examples
    --------
    Minimal example on how to instantiate and use the ConditionalFlowPipeline:

    .. literalinclude:: /examples/conditional_flow_pipeline.py
        :language: python
        :linenos:

    .. image:: /examples/conditional_flow_pipeline_marginals.png
        :width: 600

    .. note::
        If you plan on using multiprocessing prefetching, ensure that your script is wrapped
        in a ``if __name__ == "__main__":`` guard.
        See https://docs.python.org/3/library/multiprocessing.html

    .. note::
        Sampling in the latent space (latent diffusion/flow) is not currently supported.

    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        dim_obs: Union[int, Tuple[int, int]],
        dim_cond: Union[int, Tuple[int, int]],
        ch_obs=1,
        ch_cond=1,
        id_embedding_strategy=("absolute", "absolute"),
        params=None,
        training_config=None,
    ):

        # if latent diffusion is enabled, make sure to adjust the dimensionality accordingly of the transformer model

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            ch_obs=ch_obs,
            ch_cond=ch_cond,
            params=params,
            training_config=training_config,
        )

        embeddings_1d = ["absolute", "pos1d", "rope1d"]
        embeddings_2d = ["pos2d", "rope2d"]

        if id_embedding_strategy[0] in embeddings_1d:
            obs_ids = init_ids_1d(dim_obs, semantic_id=0)
        elif id_embedding_strategy[0] in embeddings_2d:
            obs_ids = init_ids_2d(dim_obs, semantic_id=0)
        else:
            raise ValueError(
                f"Unknown id embedding strategy: {id_embedding_strategy[0]}"
            )

        if id_embedding_strategy[1] in embeddings_1d:
            cond_ids = init_ids_1d(dim_cond, semantic_id=1)
        elif id_embedding_strategy[1] in embeddings_2d:
            cond_ids = init_ids_2d(dim_cond, semantic_id=1)
        else:
            raise ValueError(
                f"Unknown id embedding strategy: {id_embedding_strategy[1]}"
            )

        self.obs_ids = obs_ids
        self.cond_ids = cond_ids

        self.path = AffineProbPath(scheduler=CondOTScheduler())

        self.loss_fn = ConditionalCFMLoss(self.path)

        self.p0_obs = dist.Independent(
            dist.Normal(
                loc=jnp.zeros((self.dim_obs, self.ch_obs)),
                scale=jnp.ones((self.dim_obs, self.ch_obs)),
            ),
            reinterpreted_batch_ndims=2,
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
            # obs = batch[:, : self.dim_obs, ...]
            # cond = batch[:, self.dim_obs :, ...]
            obs, cond = batch
            rng_x0, rng_t = jax.random.split(key, 2)

            batch_size = obs.shape[0]

            x_1 = obs
            # x_0 = self.p0_obs.sample(rng_x0, (batch_size,))
            x_0 = jax.random.normal(rng_x0, (batch_size, self.dim_obs, self.ch_obs))
            t = jax.random.uniform(rng_t, x_1.shape[0])

            obs_batch = (x_0, x_1, t)

            loss = self.loss_fn(model, obs_batch, cond, self.obs_ids, self.cond_ids)
            return loss

        return loss_fn

    # need to change wrt
    # def _get_optimizer(self):
    #     """
    #     Construct the optimizer for training, including learning rate scheduling and gradient clipping.

    #     Returns
    #     -------
    #     optimizer : nnx.Optimizer
    #         The optimizer instance for the model.
    #     """

    #     # sbi_model_params = nnx.All(nnx.Param, nnx.PathContains('sbi_model'))
    #     # sbi_model_params = nnx.All(nnx.Param, nnx.PathContains("model"))

    #     opt = optax.chain(
    #         optax.adaptive_grad_clip(10.0),
    #         optax.adamw(self.training_config["max_lr"]),
    #         reduce_on_plateau(
    #             patience=self.training_config["patience"],
    #             cooldown=self.training_config["cooldown"],
    #             factor=self.training_config["factor"],
    #             rtol=self.training_config["rtol"],
    #             accumulation_size=self.training_config["accumulation_size"],
    #             min_scale=self.training_config["min_scale"],
    #         ),
    #     )
    #     if self.training_config["multistep"] > 1:
    #         opt = optax.MultiSteps(opt, self.training_config["multistep"])

    #     # optimizer = nnx.Optimizer(self.model, opt, wrt=sbi_model_params)
    #     optimizer = nnx.Optimizer(self.model, opt, wrt=nnx.Param)
    #     return optimizer

    # need to select the right weights to apply the updates
    def get_train_step_fn(self, loss_fn):
        """
        Return the training step function, which performs a single optimization step.

        Returns
        -------
        train_step : Callable
            JIT-compiled training step function.
        """
        # sbi_model_params = nnx.All(nnx.Param, nnx.PathContains('sbi_model'))
        # sbi_model_params = nnx.All(nnx.Param, nnx.PathContains("model"))

        @nnx.jit  # something bad happens here
        def train_step(model, optimizer, batch, key: jax.random.PRNGKey):
            # diff_state = nnx.DiffState(
            #     0, sbi_model_params
            # )  # filter head params of the first argument
            # loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
            #     model, batch, key
            # )
            loss, grads = nnx.value_and_grad(loss_fn)(model, batch, key)
            optimizer.update(model, grads, value=loss)
            return loss

        return train_step

    def _wrap_model(self):
        self.model_wrapped = ConditionalWrapper(self.model)
        self.ema_model_wrapped = ConditionalWrapper(self.ema_model)
        return

    def get_sampler(
        self,
        x_o,
        step_size=0.01,
        use_ema=True,
        time_grid=None,
        **model_extras,
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

        cond = _expand_dims(x_o)

        solver = ODESolver(velocity_model=vf_wrapped)

        model_extras = {
            "cond": cond,
            "obs_ids": self.obs_ids,
            "cond_ids": self.cond_ids,
            **model_extras,
        }

        sampler_ = solver.get_sampler(
            method="Dopri5",
            step_size=step_size,
            return_intermediates=return_intermediates,
            model_extras=model_extras,
            time_grid=time_grid,
        )

        def sampler(key, nsamples):
            x_init = jax.random.normal(key, (nsamples, self.dim_obs, self.ch_obs))

            samples = sampler_(x_init)

            return samples

        return sampler

    def sample(
        self,
        key,
        x_o,
        nsamples=10_000,
        step_size=0.01,
        use_ema=True,
        time_grid=None,
        **model_extras,
    ):

        sampler_ = self.get_sampler(
            x_o,
            step_size=step_size,
            use_ema=use_ema,
            time_grid=time_grid,
            **model_extras,
        )

        samples = sampler_(key, nsamples)

        return samples

    # def compute_unnorm_logprob(
    #     self, x_1, x_o, step_size=0.01, use_ema=True, time_grid=None, **model_extras
    # ):
    #     if use_ema:
    #         model = self.ema_model_wrapped
    #     else:
    #         model = self.model_wrapped

    #     if time_grid is None:
    #         time_grid = jnp.array([1.0, 0.0])
    #         return_intermediates = False
    #     else:
    #         # assert time grid is decreasing
    #         assert jnp.all(time_grid[:-1] >= time_grid[1:])
    #         return_intermediates = True

    #     solver = ODESolver(velocity_model=model)

    #     # x_1 = _expand_dims(x_1)
    #     assert (
    #         x_1.ndim == 2
    #     ), "x_1 must be of shape (num_samples, dim_obs), currently sampling for multiple channels is not supported."
    #     cond = _expand_dims(x_o)

    #     model_extras = {
    #         "cond": cond,
    #         "obs_ids": self.obs_ids,
    #         "cond_ids": self.cond_ids,
    #         **model_extras,
    #     }

    #     logp_sampler = solver.get_unnormalized_logprob(
    #         time_grid=time_grid,
    #         method="Dopri5",
    #         step_size=step_size,
    #         log_p0=self.p0_obs.log_prob,
    #         model_extras=model_extras,
    #         return_intermediates=return_intermediates,
    #     )

    #     if len(x_1) > 4:
    #         # we trigger precompilation first
    #         _ = logp_sampler(x_1[:4])

    #     exact_log_p = logp_sampler(x_1)
    #     return exact_log_p


class ConditionalDiffusionPipeline(AbstractPipeline):
    """
    Diffusion pipeline for training and using a Conditional model for simulation-based inference.

    Parameters
    ----------
    train_dataset : grain dataset or iterator over batches
        Training dataset.
    val_dataset : grain dataset or iterator over batches
        Validation dataset.
    dim_obs : int or tuple of int
        Dimension of the parameter space (number of tokens).
        Can represent unstructured data, time-series, or patchified 2D images. For images, provide a tuple (height, width).
    dim_cond : int or tuple of int
        Dimension of the observation space (number of tokens).
        Can represent unstructured data, time-series, or patchified 2D images. For images, provide a tuple (height, width).
    params : ConditionalParams, optional
        Parameters for the Conditional model. If None, default parameters are used.
    training_config : dict, optional
        Configuration for training. If None, default configuration is used.

    Examples
    --------
    Minimal example on how to instantiate and use the ConditionalDiffusionPipeline:

    .. literalinclude:: /examples/conditional_diffusion_pipeline.py
        :language: python
        :linenos:

    .. image:: /examples/conditional_diffusion_pipeline_marginals.png
        :width: 600

    .. note::
        If you plan on using multiprocessing prefetching, ensure that your script is wrapped
        in a ``if __name__ == "__main__":`` guard.
        See https://docs.python.org/3/library/multiprocessing.html

    .. note::
        Sampling in the latent space (latent diffusion/flow) is not currently supported.

    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        dim_obs: Union[int, Tuple[int, int]],
        dim_cond: Union[int, Tuple[int, int]],
        ch_obs=1,
        ch_cond=1,
        id_embedding_strategy=("absolute", "absolute"),
        params=None,
        training_config=None,
    ):

        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=dim_cond,
            ch_obs=ch_obs,
            ch_cond=ch_cond,
            params=params,
            training_config=training_config,
        )

        # # Flux1 uses different ids for obs and cond
        # obs_ids = jnp.zeros((1, dim_obs, 2), dtype=jnp.int32)
        # obs_ids = obs_ids.at[..., 0].set(jnp.arange(dim_obs))

        # cond_ids = jnp.zeros((1, dim_cond, 2), dtype=jnp.int32)
        # cond_ids = cond_ids.at[..., 0].set(jnp.arange(dim_cond))
        # cond_ids = cond_ids.at[..., 1].set(
        #     1
        # )  # set second channel to 1 for conditioning tokens

        embeddings_1d = ["absolute", "pos1d", "rope1d"]
        embeddings_2d = ["pos2d", "rope2d"]

        if id_embedding_strategy[0] in embeddings_1d:
            obs_ids = init_ids_1d(dim_obs, semantic_id=0)
        elif id_embedding_strategy[0] in embeddings_2d:
            obs_ids = init_ids_2d(dim_obs, semantic_id=0)
        else:
            raise ValueError(
                f"Unknown id embedding strategy: {id_embedding_strategy[0]}"
            )

        if id_embedding_strategy[1] in embeddings_1d:
            cond_ids = init_ids_1d(dim_cond, semantic_id=1)
        elif id_embedding_strategy[1] in embeddings_2d:
            cond_ids = init_ids_2d(dim_cond, semantic_id=1)
        else:
            raise ValueError(
                f"Unknown id embedding strategy: {id_embedding_strategy[1]}"
            )

        self.obs_ids = obs_ids
        self.cond_ids = cond_ids

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
    def get_default_training_config(cls):
        config = super().get_default_training_config()
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
            # (batch_size, dim_obs + dim_cond)

            # obs = jnp.take_along_axis(batch, self.obs_ids, axis=1)
            # cond = jnp.take_along_axis(batch, self.cond_ids, axis=1)
            # obs = batch[:, : self.dim_obs, ...]
            # cond = batch[:, self.dim_obs :, ...]

            obs, cond = batch

            rng_x0, rng_sigma = jax.random.split(key, 2)

            x_1 = obs
            # sigma = self.path.sample_sigma(rng_sigma, (x_1.shape[0],))
            sigma = self.path.sample_sigma(rng_sigma, (x_1.shape[0], 1, 1))
            # sigma = repeat(sigma, f"b -> b {'1 ' * (x_1.ndim - 1)}")  # TODO fixme

            obs_batch = (x_1, sigma)
            loss = self.loss_fn(
                rng_x0, model, obs_batch, cond, self.obs_ids, self.cond_ids
            )
            return loss

        return loss_fn

    # def _get_optimizer(self):
    #     """
    #     Construct the optimizer for training, including learning rate scheduling and gradient clipping.

    #     Returns
    #     -------
    #     optimizer : nnx.Optimizer
    #         The optimizer instance for the model.
    #     """
    #     # sbi_model_params = nnx.All(nnx.Param, nnx.PathContains("sbi_model"))
    #     # sbi_model_params = nnx.All(nnx.Param, nnx.PathContains("model"))

    #     opt = optax.chain(
    #         optax.adaptive_grad_clip(10.0),
    #         optax.adamw(self.training_config["max_lr"]),
    #         reduce_on_plateau(
    #             patience=self.training_config["patience"],
    #             cooldown=self.training_config["cooldown"],
    #             factor=self.training_config["factor"],
    #             rtol=self.training_config["rtol"],
    #             accumulation_size=self.training_config["accumulation_size"],
    #             min_scale=self.training_config["min_scale"],
    #         ),
    #     )
    #     if self.training_config["multistep"] > 1:
    #         opt = optax.MultiSteps(opt, self.training_config["multistep"])

    #     # optimizer = nnx.Optimizer(self.model, opt, wrt=sbi_model_params)
    #     optimizer = nnx.Optimizer(self.model, opt, wrt=nnx.Param)
    #     return optimizer

    def get_train_step_fn(self, loss_fn):
        """
        Return the training step function, which performs a single optimization step.

        Returns
        -------
        train_step : Callable
            JIT-compiled training step function.
        """
        # sbi_model_params = nnx.All(nnx.Param, nnx.PathContains("sbi_model"))
        # sbi_model_params = nnx.All(nnx.Param, nnx.PathContains("model"))

        @nnx.jit
        def train_step(model, optimizer, batch, key: jax.random.PRNGKey):
            # diff_state = nnx.DiffState(0, sbi_model_params)
            # loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
            #     model, batch, key
            # )
            loss, grads = nnx.value_and_grad(loss_fn)(model, batch, key)
            optimizer.update(model, grads, value=loss)
            return loss

        return train_step

    def _wrap_model(self):
        self.model_wrapped = ConditionalWrapper(self.model)
        self.ema_model_wrapped = ConditionalWrapper(self.ema_model)
        return

    def get_sampler(
        self,
        x_o,
        nsteps=18,
        use_ema=True,
        return_intermediates=False,
        **model_extras,
    ):
        if use_ema:
            model = self.ema_model_wrapped
        else:
            model = self.model_wrapped

        cond = _expand_dims(x_o)

        solver = SDESolver(score_model=model, path=self.path)

        model_extras = {
            "cond": cond,
            "obs_ids": self.obs_ids,
            "cond_ids": self.cond_ids,
            **model_extras,
        }

        sampler_ = solver.get_sampler(
            nsteps=nsteps,
            return_intermediates=return_intermediates,
            model_extras=model_extras,
        )

        def sampler(key, nsamples):
            key1, key2 = jax.random.split(key, 2)
            x_init = self.path.sample_prior(key1, (nsamples, self.dim_obs, self.ch_obs))
            samples = sampler_(key2, x_init)

            return samples

        return sampler

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

        sampler = self.get_sampler(
            x_o,
            nsteps=nsteps,
            use_ema=use_ema,
            return_intermediates=return_intermediates,
            **model_extras,
        )
        return sampler(key, nsamples)
