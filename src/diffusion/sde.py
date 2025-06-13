# we define a base class for the forward and reverse sde
# to avoid code duplication

# for each sde, we define the drift, diffusion and marginals
# for the reverse sde, we implement a sampler given a solver using diffrax
# the reverse sde is defined from the forward sde

import abc
import jax.numpy as jnp
import jax
from jax import grad, jit, vmap
import jax.random as random
from functools import partial
from diffrax import (
    diffeqsolve,
    ControlTerm,
    MultiTerm,
    ODETerm,
    VirtualBrownianTree,
    UnsafeBrownianPath,
    DirectAdjoint,
    Euler,
)

import numpyro.distributions as dist

from typing import Callable


class BaseSDE(abc.ABC):
    def __init__(self, dim, prior_dist: dist.Distribution, diff_steps: int = 1000):
        self.dim = dim
        self._prior_dist = prior_dist
        self._diff_steps = diff_steps
        self.scale_min = None
        return

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @property
    def diff_steps(self):
        """Diffusion steps of the forward SDE."""
        return self._diff_steps

    @abc.abstractmethod
    def drift(self, x, t):
        """
        Drift function of the SDE.
        """
        pass

    @abc.abstractmethod
    def diffusion(self, x, t):
        """
        Diffusion function of the SDE.
        """
        pass

    def marginal_dist(self, x0, t):
        """
        Marginal probability p(x_t | x_0)
        """
        mean_t = self.marginal_mean(x0, t)
        std_t = self.marginal_stddev(x0, t)
        covar = jnp.diag(std_t)
        return dist.MultivariateNormal(loc=mean_t, covariance_matrix=covar)

    @abc.abstractmethod
    def marginal_mean(self, x0, t):
        """
        Mean of the marginal distribution.
        """
        pass

    @abc.abstractmethod
    def marginal_stddev(self, x0, t):
        """
        Mean of the marginal distribution.
        """
        pass

    @property
    def prior_dist(self) -> dist.Distribution:
        """The prior distribution."""
        return self._prior_dist

    def prior_sample(self, rng, shape):
        """
        Sample from the prior distribution.
        """
        return self._prior_dist.sample(rng, shape)

    @abc.abstractmethod
    def reverse(self, score):
        """
        Return the reverse SDE.
        """
        pass

    def get_loss_function(self, weight_fn: Callable = None):
        """
        Get the loss function for denoising score matching.
        """
        if weight_fn is None:
            # according to https://arxiv.org/abs/2101.09258 , for MLE we use g(t)**2 as weight function
            weight_fn = lambda t: self.diffusion(1.0, t) ** 2

        def loss_fn(score_model, x0, loss_mask=None, *args, rng, **kwargs):
            N_batch = x0.shape[0]
            t_shape = (N_batch,) + (1,) * (x0.ndim - 1)
            # t needs to have the same number of dimensions as x0, so we broadcast it
            rng, step_key = random.split(rng)
            T_min = 1 / self.diff_steps
            t = jax.random.uniform(step_key, t_shape, minval=T_min, maxval=1.0)

            mean_t = self.marginal_mean(x0, t)
            std_t = self.marginal_stddev(x0, t)

            rng, step_key = random.split(rng)
            noise = random.normal(step_key, x0.shape)
            xt = mean_t + noise * std_t

            # if the loss_mask is not None, we use it to mask some features, effectively conditioning on them
            if loss_mask is not None:
                loss_mask = loss_mask.reshape(x0.shape)
                xt = jnp.where(loss_mask, x0, xt)

            score_pred = score_model(xt, t, *args, **kwargs)
            score_target = -noise / std_t
            weight = weight_fn(t)

            loss = weight * jnp.sum((score_pred - score_target) ** 2, axis=-1, keepdims=True)
            if loss_mask is not None:
                loss = jnp.where(loss_mask, 0.0, loss)
            loss = jnp.mean(loss)

            return loss

        return loss_fn

    def get_output_scale_fn(self, scale_min=None):
        if scale_min is None:
            assert not self.scale_min is None, "scale_min must be implemented in the SDE class or provided as an argument"
            scale_min = self.scale_min
        def output_scale_fn(t, x):
            scale = jnp.clip(jnp.sqrt(jnp.sum(self.marginal_variance(t[..., None])*jnp.ones(x.shape), -1)), scale_min)
            return 1/scale[..., None] * x
        
        return output_scale_fn
    
    @abc.abstractmethod
    def _tree_flatten(self):
        pass

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


class ReverseSDE(abc.ABC):
    def __init__(self, forward_sde: BaseSDE, score: Callable, dim, **kwargs):
        self.dim = dim
        self.forward_sde = forward_sde
        self.score = score

    def drift(self, x, t, condition_mask=0, replace_conditioned=True, score_args={}):
        """
        Drift function of the reverse SDE.
        """
        x = jnp.atleast_2d(x)
        t = jnp.atleast_1d(t)
        forward_drift = self.forward_sde.drift
        diffusion = self.forward_sde.diffusion
        # to obtain a process for the reverse time, we need to reverse the drift and diffusion of the forward sde
        res = forward_drift(x, t) - diffusion(x, t)**2 *self.score(jnp.atleast_2d(x), t, **score_args)
        if replace_conditioned:
            res = res * (1 - condition_mask)
        return jnp.squeeze(res, axis=0) # remove the extra dimension

    def diffusion(self, x, t, condition_mask=0, replace_conditioned=True):
        """
        Diffusion function of the reverse SDE.
        """
        res = self.forward_sde.diffusion(x, t)
        if replace_conditioned:
            res = res * (1 - condition_mask)
        return res

    def sample(
        self,
        rng,
        n_samples,
        condition_mask=None,
        condition_value=None,
        solver=Euler(),
        safe=True,
        n_steps=1000,
        replace_conditioned=True,
        score_args={},
    ):
        """
        Sample from the reverse SDE.
        """
        t0 = self.forward_sde.T
        t1 = 1 / n_steps
        dt = -t0 / n_steps

        if condition_mask is not None:
            assert (
                condition_value is not None
            ), "Condition value must be provided if condition mask is provided"
        else:
            condition_mask = 0
            condition_value = 0

        def diff(t, y, args):
            return jnp.diag(
                jnp.broadcast_to(self.diffusion(y, t, condition_mask,replace_conditioned), (self.dim,))
            )

        def drift(t, y, args):
            return self.drift(y, t, condition_mask, replace_conditioned, score_args=score_args)

        rng, key = jax.random.split(rng)

        y0s = self.forward_sde.prior_sample(key, (n_samples,))
        if replace_conditioned:
            y0s = y0s * (1 - condition_mask) + condition_value * condition_mask

        keys = jax.random.split(rng, n_samples)

        @jit
        def sample_one(key, y0):
            brownian_motion = VirtualBrownianTree(
                t1, t0, tol=1e-5, shape=(self.dim,), key=key
            )
            terms = MultiTerm(ODETerm(drift), ControlTerm(diff, brownian_motion))
            sol = diffeqsolve(terms, solver, t0, t1, dt0=dt, y0=y0)
            return sol.ys

        @jit
        def sample_one_unsafe(key, y0):
            brownian_motion = UnsafeBrownianPath(shape=(self.dim,), key=key)

            terms = MultiTerm(ODETerm(drift), ControlTerm(diff, brownian_motion))
            sol = diffeqsolve(
                terms, solver, t0, t1, dt0=dt, y0=y0, adjoint=DirectAdjoint()
            )
            return sol.ys

        if safe:
            res = vmap(sample_one, in_axes=(0, 0))(keys, y0s)
        else:
            res = vmap(sample_one_unsafe, in_axes=(0, 0))(keys, y0s)

        return jnp.squeeze(res)

    @abc.abstractmethod
    def _tree_flatten(self):
        pass

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


class VPSDE(BaseSDE):
    """
    Variance Preseverving SDE, also known as the Ornstein-Uhlenbeck SDE.
    """

    def __init__(
        self, dim, beta_min: float = 0.001, beta_max: float = 3.0, diff_steps=1000
    ):
        prior_dist = dist.MultivariateNormal(
            loc=jnp.zeros(dim), covariance_matrix=jnp.eye(dim)
        )
        super().__init__(dim, prior_dist, diff_steps)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.scale_min = 0.
        return

    @property
    def T(self):
        return 1

    def beta_t(self, t):
        """
        Beta function of the VPSDE.
        """
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def alpha_t(self, t):
        """
        Alpha function of the VPSDE.
        """
        return t * self.beta_min + 0.5 * t**2 * (self.beta_max - self.beta_min)

    def drift(self, x, t):
        """
        Drift function of the VPSDE.
        """
        return -0.5 * self.beta_t(t) * x

    def diffusion(self, x, t):
        """
        Diffusion function of the VPSDE.
        """
        return jnp.sqrt(self.beta_t(t))

    def mean_coeff(self, t):
        """
        t: time (number)
        returns m_t as above
        """
        return jnp.exp(-0.5 * self.alpha_t(t))

    def marginal_mean(self, x0, t):
        """
        Mean of the marginal distribution.
        """
        return self.mean_coeff(t) * x0

    def marginal_stddev(self, x0, t):
        """
        Standard deviation of the marginal distribution.
        """
        return jnp.broadcast_to(jnp.sqrt(self.marginal_variance(t)), x0.shape)

    def marginal_variance(self, t):
        """
        t: time (number)
        returns v_t as above
        """
        return 1 - jnp.exp(-self.alpha_t(t))

    def reverse(self, score):
        """
        Return the reverse VPSDE.
        """
        return R_VPSDE(self, score, self.dim)

    def _tree_flatten(self):

        children = (None,)  # arrays / dynamic values

        aux_data = {
            "dim": self.dim,
            "prior_dist": self.prior_dist,
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
        }  # static values

        return (children, aux_data)


class R_VPSDE(ReverseSDE):
    def __init__(self, forward_sde, score, dim, **kwargs):
        super().__init__(forward_sde, score, dim, **kwargs)
        return

    def _tree_flatten(self):

        children = (None,)  # arrays / dynamic values

        aux_data = {
            "dim": self.dim,
            "forward_sde": self.forward_sde,
            "score": self.score,
        }  # static values

        return (children, aux_data)


def get_exponential_sigma_function(sigma_min, sigma_max):
    log_sigma_min = jnp.log(sigma_min)
    log_sigma_max = jnp.log(sigma_max)

    @jit
    def sigma(t):
        # return sigma_min * (sigma_max / sigma_min)**t  # Has large relative error close to zero compared to alternative, below
        return jnp.exp(log_sigma_min + t * (log_sigma_max - log_sigma_min))

    return sigma


class VESDE(BaseSDE):
    """Variance exploding (VE) SDE, a.k.a. diffusion process with a time dependent diffusion coefficient."""

    def __init__(self, dim, sigma_min=1e-3, sigma_max=15.0, diff_steps=1000):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        prior_dist = dist.MultivariateNormal(
            loc=jnp.zeros(dim), covariance_matrix=self.sigma_max * jnp.eye(dim)
        )
        super().__init__(dim, prior_dist, diff_steps)
        self.sigma = get_exponential_sigma_function(sigma_min, sigma_max)
        self.scale_min = 1e-3

    @property
    def T(self):
        """End time of the SDE."""
        return 1

    def drift(self, x, t):
        """
        Drift function of the SDE.
        """
        return jnp.zeros_like(x)

    def diffusion(self, x, t):
        """
        Diffusion function of the SDE.
        """
        sigma_t = self.sigma(t)
        return sigma_t * jnp.sqrt(
            2 * (jnp.log(self.sigma_max) - jnp.log(self.sigma_min))
        )

    def mean_coeff(self, t):
        return jnp.ones_like(t)

    def marginal_variance(self, t):
        return self.sigma(t) ** 2

    def marginal_mean(self, x0, t):
        """
        Mean of the marginal distribution.
        """
        return self.mean_coeff(t) * x0

    def marginal_stddev(self, x0, t):
        """
        Standard deviation of the marginal distribution.
        """
        return jnp.broadcast_to(self.sigma(t), x0.shape)

    def marginal_dist(self, x0, t):
        """
        Marginal probability of the VPSDE.
        """
        mean_coeff = self.mean_coeff(t)
        std = self.sigma(t)
        covar = jnp.diag(std)
        return dist.MultivariateNormal(loc=mean_coeff * x0, covariance_matrix=covar)
    

    def reverse(self, score):
        """
        Return the reverse VPSDE.
        """
        return R_VESDE(self, score, self.dim)

    def _tree_flatten(self):

        children = (None,)  # arrays / dynamic values
        aux_data = {
            "dim": self.dim,
            "prior_dist": self.prior_dist,
            "sigma_min": self.sigma_min,
            "sigma_max": self.sigma_max,
        }  # static values

        return (children, aux_data)


class R_VESDE(ReverseSDE):
    def __init__(self, forward_sde, score, dim, **kwargs):
        super().__init__(forward_sde, score, dim, **kwargs)
        return

    def _tree_flatten(self):

        children = (None,)  # arrays / dynamic values

        aux_data = {
            "dim": self.dim,
            "forward_sde": self.forward_sde,
            "score": self.score,
        }  # static values

        return (children, aux_data)
