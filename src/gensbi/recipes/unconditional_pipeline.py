"""
Pipeline for training and using a Unconditional model for simulation-based inference.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from numpyro import distributions as dist


from gensbi.flow_matching.path import AffineProbPath
from gensbi.flow_matching.path.scheduler import CondOTScheduler
from gensbi.flow_matching.solver import ODESolver

from gensbi.diffusion.path import EDMPath
from gensbi.diffusion.path.scheduler import EDMScheduler, VEScheduler, VPScheduler
from gensbi.diffusion.solver import SDESolver

from gensbi.models import (
    UnconditionalCFMLoss,
    UnconditionalWrapper,
    UnconditionalDiffLoss,
)

from gensbi.recipes.utils import init_ids_1d

from einops import repeat

from gensbi.utils.model_wrapping import _expand_dims

from gensbi.recipes.pipeline import AbstractPipeline


class UnconditionalFlowPipeline(AbstractPipeline):
    """
    Flow pipeline for training and using an Unconditional model for simulation-based inference.

    Parameters
    ----------
    model : nnx.Module
        The model to be trained.
    train_dataset : grain dataset or iterator over batches
        Training dataset.
    val_dataset : grain dataset or iterator over batches
        Validation dataset.
    dim_obs : int
        Dimension of the parameter space.
    ch_obs : int
        Number of channels in the observation space.
    params : optional
        Parameters for the model. Serves no use if a custom model is provided.
    training_config : dict, optional
        Configuration for training. If None, default configuration is used.

    Examples
    --------
    Minimal example on how to instantiate and use the UnconditionalFlowPipeline:

    .. literalinclude:: /examples/unconditional_flow_pipeline.py
        :language: python
        :linenos:

    .. image:: /examples/unconditional_flow_samples.png
        :width: 600

    .. note::
        If you plan on using multiprocessing prefetching, ensure that your script is wrapped
        in a ``if __name__ == "__main__":`` guard.
        See https://docs.python.org/3/library/multiprocessing.html
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        dim_obs: int,
        ch_obs: int = 1,
        params=None,
        training_config=None,
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=0,
            ch_obs=ch_obs,
            params=params,
            training_config=training_config,
        )

        self.obs_ids = init_ids_1d(self.dim_obs)

        self.path = AffineProbPath(scheduler=CondOTScheduler())

        self.loss_fn = UnconditionalCFMLoss(self.path)

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
            "Initialization from config not implemented for UnconditionalFlowPipeline."
        )

    def _make_model(self):
        raise NotImplementedError(
            "Model creation not implemented for UnconditionalFlowPipeline."
        )

    def _get_default_params(self):
        raise NotImplementedError(
            "Default parameters not implemented for UnconditionalFlowPipeline."
        )

    def get_loss_fn(
        self,
    ):
        def loss_fn(model, batch, key: jax.random.PRNGKey):
            obs = batch

            batch_size = batch.shape[0]

            rng_x0, rng_t = jax.random.split(key, 2)

            x_1 = obs
            # x_0 = self.p0_obs.sample(rng_x0, (batch_size,))
            x_0 = jax.random.normal(rng_x0, (batch_size, self.dim_obs, self.ch_obs))
            t = jax.random.uniform(rng_t, x_1.shape[0])

            batch = (x_0, x_1, t)
            condition_mask = jnp.zeros((*x_1.shape[:-1], 1), dtype=jnp.bool_)

            loss = self.loss_fn(
                model, batch, node_ids=self.obs_ids, condition_mask=condition_mask
            )
            return loss

        return loss_fn

    def _wrap_model(self):
        self.model_wrapped = UnconditionalWrapper(self.model)
        self.ema_model_wrapped = UnconditionalWrapper(self.ema_model)
        return

    def get_sampler(
        self,
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

        solver = ODESolver(velocity_model=vf_wrapped)
        model_extras = {"obs_ids": self.obs_ids, **model_extras}

        sampler_ = solver.get_sampler(
            method="Dopri5",
            step_size=step_size,
            return_intermediates=return_intermediates,
            model_extras=model_extras,
            time_grid=time_grid,
        )

        def sampler(key, nsamples):
            x_init = jax.random.normal(key, (nsamples, self.dim_obs, self.ch_obs))
            return sampler_(x_init)

        return sampler

    def sample(
        self,
        key,
        nsamples=10_000,
        step_size=0.01,
        use_ema=True,
        time_grid=None,
        **model_extras,
    ):
        sampler = self.get_sampler(
            step_size=step_size,
            use_ema=use_ema,
            time_grid=time_grid,
            **model_extras,
        )
        samples = sampler(key, nsamples)
        return samples

    def sample_batched(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError(
            "Batched sampling not implemented for UnconditionalFlowPipeline."
        )

    # def compute_unnorm_logprob(
    #     self, x_1, step_size=0.01, use_ema=True, time_grid=None, **model_extras
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

    #     # todo need to check the model extras, is that node_ids instead?
    #     model_extras = {"obs_ids": self.obs_ids, **model_extras}

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


class UnconditionalDiffusionPipeline(AbstractPipeline):
    """
    Diffusion pipeline for training and using an Unconditional model for simulation-based inference.

    Parameters
    ----------
    model : nnx.Module
        The model to be trained.
    train_dataset : grain dataset or iterator over batches
        Training dataset.
    val_dataset : grain dataset or iterator over batches
        Validation dataset.
    dim_obs : int
        Dimension of the parameter space.
    ch_obs : int
        Number of channels in the observation space.
    params : optional
        Parameters for the model. Serves no use if a custom model is provided.
    training_config : dict, optional
        Configuration for training. If None, default configuration is used.

    Examples
    --------
    Minimal example on how to instantiate and use the UnconditionalDiffusionPipeline:

    .. literalinclude:: /examples/unconditional_diffusion_pipeline.py
        :language: python
        :linenos:

    .. image:: /examples/unconditional_diffusion_pipeline_samples.png
        :width: 600

    .. note::
        If you plan on using multiprocessing prefetching, ensure that your script is wrapped
        in a ``if __name__ == "__main__":`` guard.
        See https://docs.python.org/3/library/multiprocessing.html

    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        dim_obs: int,
        ch_obs: int = 1,
        params=None,
        # sde="EDM",
        training_config=None,
    ):
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            dim_obs=dim_obs,
            dim_cond=0,
            ch_obs=ch_obs,
            params=params,
            training_config=training_config,
        )

        self.obs_ids = init_ids_1d(self.dim_obs)

        sigma_min = self.training_config.get("sigma_min", 0.002)
        sigma_max = self.training_config.get("sigma_max", 80.0)
        self.path = EDMPath(
            scheduler=EDMScheduler(
                sigma_min=sigma_min,
                sigma_max=sigma_max,
            )
        )

        # self.sde = sde
        # if sde == "EDM":
        #     sigma_min = self.training_config.get("sigma_min", 0.002)
        #     sigma_max = self.training_config.get("sigma_max", 80.0)
        #     self.path = EDMPath(
        #         scheduler=EDMScheduler(
        #             sigma_min=sigma_min,
        #             sigma_max=sigma_max,
        #         )
        #     )
        # elif sde == "VE":
        #     sigma_min = self.training_config.get("sigma_min", 0.001)
        #     sigma_max = self.training_config.get("sigma_max", 15.0)
        #     self.path = EDMPath(scheduler=VEScheduler(
        #         sigma_min=sigma_min,
        #         sigma_max=sigma_max,
        #     ))
        # elif sde == "VP":
        #     beta_min = self.training_config.get("beta_min", 0.1)
        #     beta_max = self.training_config.get("beta_max", 20.0)
        #     self.path = EDMPath(scheduler=VPScheduler(
        #         beta_min = beta_min,
        #         beta_max = beta_max,
        #     ))
        # else:
        #     raise ValueError(f"Unknown sde type: {sde}")

        self.loss_fn = UnconditionalDiffLoss(self.path)

    @classmethod
    def init_pipeline_from_config(
        cls,
    ):
        raise NotImplementedError(
            "Initialization from config not implemented for UnconditionalDiffusionPipeline."
        )

    def _make_model(self):
        raise NotImplementedError(
            "Model creation not implemented for UnconditionalDiffusionPipeline."
        )

    def _get_default_params(self):
        raise NotImplementedError(
            "Default parameters not implemented for UnconditionalDiffusionPipeline."
        )

    @classmethod
    def get_default_training_config(cls, sde="EDM"):
        config = super().get_default_training_config()
        config.update(
            {
                "sigma_min": 0.002,  # from edm paper
                "sigma_max": 80.0,
            }
        )
        # if sde == "EDM":
        #     config.update(
        #         {
        #             "sigma_min": 0.002,  # from edm paper
        #             "sigma_max": 80.0,
        #         }
        #     )
        # elif sde == "VE":
        #     config.update(
        #         {
        #             "sigma_min": 0.001,  # from edm paper
        #             "sigma_max": 15.0,
        #         }
        #     )
        # elif sde == "VP":
        #     config.update(
        #         {
        #             "beta_min": 0.1,
        #             "beta_max": 20.0,
        #         }
        #     )
        return config

    def get_loss_fn(
        self,
    ):
        def loss_fn(model, batch, key: jax.random.PRNGKey):
            rng_x0, rng_sigma = jax.random.split(key, 2)

            x_1 = batch
            # sigma = self.path.sample_sigma(rng_sigma, (x_1.shape[0], ))
            sigma = self.path.sample_sigma(rng_sigma, (x_1.shape[0], 1, 1))
            # sigma = repeat(sigma, f"b -> b {'1 ' * (x_1.ndim - 1)}")  # TODO fixme

            batch = (x_1, sigma)
            loss = self.loss_fn(rng_x0, model, batch, node_ids=self.obs_ids)
            return loss

        return loss_fn

    def _wrap_model(self):
        self.model_wrapped = UnconditionalWrapper(self.model)
        self.ema_model_wrapped = UnconditionalWrapper(self.ema_model)
        return

    def get_sampler(
        self,
        nsteps=18,
        use_ema=True,
        return_intermediates=False,
        **model_extras,
    ):
        if use_ema:
            model = self.ema_model_wrapped
        else:
            model = self.model_wrapped

        solver = SDESolver(score_model=model, path=self.path)

        model_extras = {"obs_ids": self.obs_ids, **model_extras}

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
        nsamples=10_000,
        nsteps=18,
        use_ema=True,
        return_intermediates=False,
        **model_extras,
    ):
        sampler = self.get_sampler(
            nsteps=nsteps,
            use_ema=use_ema,
            return_intermediates=return_intermediates,
            **model_extras,
        )
        samples = sampler(key, nsamples)

        return samples

    def sample_batched(
        self,
        *args,
        **kwargs,
    ):
        raise NotImplementedError(
            "Batched sampling not implemented for UnconditionalDiffusionPipeline."
        )
