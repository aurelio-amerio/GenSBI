"""
Flow Matching SDE Solvers.

This module provides stochastic sampling for flow matching models,
using SDE formulations from arXiv:2410.02217.

**Time direction convention:**
In flow matching, ``t=0`` is noise and ``t=1`` is data.
These SDE solvers integrate **forward** from ``t=eps`` to ``t=1`` to transform
noise into data, matching the ODE solver direction.
This is the opposite of standard score matching (reverse SDE: ``t=T→eps``).

**Input shape convention:**
All inputs must have shape ``(batch, features, channels)``.
If your data is 2D ``(batch, features)``, add a trailing dimension:
``x = x[..., None]``.

Based on: "Improving Flow Matching by Stochastic Sampling"
(arXiv:2410.02217)
"""

from typing import Callable
from functools import partial
from abc import abstractmethod
import math

import jax
from jax import jit
import jax.numpy as jnp
from jax import Array
import diffrax

from numpyro.distributions import Independent, Normal

from diffrax import (
    diffeqsolve,
    ControlTerm,
    MultiTerm,
    ODETerm,
    VirtualBrownianTree,
    SaveAt,
)

from gensbi.solver import Solver
from gensbi.utils.model_wrapping import ModelWrapper


class BaseFmSDESolver(Solver):
    """Base class for flow matching SDE solvers.

    Uses a velocity field model to construct drift and diffusion terms for an SDE
    that samples from the same distribution as the flow matching ODE, but with
    added stochasticity for improved sample quality.

    Parameters
    ----------
        velocity_model : ModelWrapper
            A velocity field model, properly wrapped.
        mu0 : Array
            Mean of the prior Gaussian distribution. Shape: ``(features, channels)``.
        sigma0 : Array
            Standard deviation of the prior Gaussian distribution. Shape: ``(features, channels)``.
        eps0 : float
            Minimum time value (to avoid singularities near t=0).
    """

    def __init__(
        self,
        velocity_model: ModelWrapper,
        mu0: Array,
        sigma0: Array,
        eps0: float = 1e-5,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.mu0 = mu0
        self.sigma0 = sigma0

        assert mu0.ndim == 2, (
            f"mu0 must have shape (features, channels), got shape {mu0.shape}. "
            "If your data is 1D, use mu0[:, None]."
        )
        assert (
            sigma0.shape == mu0.shape
        ), f"sigma0 shape {sigma0.shape} must match mu0 shape {mu0.shape}"

        self.sample_shape = mu0.shape  # (features, channels)
        self.flat_dim = math.prod(mu0.shape)

        self.prior_distribution = Independent(
            Normal(mu0.reshape(-1), sigma0.reshape(-1)),
            reinterpreted_batch_ndims=1,
        )

        self.eps0 = eps0

    @abstractmethod
    def get_f_tilde(self) -> Callable:
        r"""Get the drift function :math:`\tilde{f}` for the SDE.

        See arXiv:2410.02217. Also known as the "drift" term.
        """
        ...  # pragma: no cover

    @abstractmethod
    def get_g_tilde(self) -> Callable:
        r"""Get the diffusion function :math:`\tilde{g}` for the SDE.

        See arXiv:2410.02217. Also known as the "diffusion" term.
        Must return a ``(flat_dim, flat_dim)`` matrix for ``ControlTerm``.
        """
        ...  # pragma: no cover

    def get_score(self, **kwargs):
        """Obtain the score function given the velocity model. See arXiv:2410.02217."""

        vf = self.velocity_model.get_vector_field(**kwargs)

        def score(t, x, args):
            res = (-t * vf(t, x, args) + self.mu0 - x) / ((1 - t) * self.sigma0**2)
            return res

        return score

    def get_sampler(
        self,
        args=None,
        nsteps=300,
        method="SEA",
        adaptive=False,
        return_intermediates=False,
        **kwargs,
    ) -> Callable:
        """Stochastic sampler for the SDE.

        Parameters
        ----------
            args : optional
                Additional arguments to pass to the velocity model.
            nsteps : int
                Number of steps for the SDE solver.
            method : str
                Integration method. One of ``"Euler"``, ``"Heun"``, ``"SEA"``,
                ``"ShARK"``. Defaults to ``"SEA"``.
            adaptive : bool
                Whether to use adaptive stepsize control (only for ShARK).
            return_intermediates : bool
                Whether to return all intermediate time steps.

        Returns
        -------
            Callable
                A function ``sample(key, nsamples)`` that returns sampled trajectories
                of shape ``(nsamples, features, channels)``.
        """
        solvers = {
            "Euler": diffrax.Euler,
            "Heun": diffrax.Heun,
            "SEA": diffrax.SEA,
            "ShARK": diffrax.ShARK,
        }
        if method not in solvers:
            raise ValueError(
                f"Method {method} not supported. Choose from {list(solvers.keys())}."
            )

        solver = solvers[method]()

        if method in ["Euler", "Heun"]:
            levy_area = diffrax.BrownianIncrement
        else:
            levy_area = diffrax.SpaceTimeLevyArea

        drift = self.get_f_tilde(**kwargs)  # (t, x, args) -> drift
        diff = self.get_g_tilde()  # (t, x, args) -> diffusion matrix

        # Time direction: forward, from noise (t=eps) to data (t=1)
        t0 = self.eps0
        t1 = 1.0
        dt = (t1 - t0) / nsteps

        dtmin = min(2e-5, dt)
        tol = dtmin / 2

        if method in ["ShARK"] and adaptive:
            stepsize_controller = diffrax.PIDController(
                rtol=1e-5, atol=1e-5, dtmin=dtmin, dtmax=2 * dt
            )
        else:
            stepsize_controller = diffrax.ConstantStepSize()

        flat_dim = self.flat_dim
        sample_shape = self.sample_shape

        def sample_one(key_i, y0_flat):
            """Integrate one sample. State is flat (flat_dim,)."""
            brownian_motion = VirtualBrownianTree(
                t0,
                t1,
                tol=tol,
                shape=(flat_dim,),
                key=key_i,
                levy_area=levy_area,
            )

            # Wrap drift: unflatten state, add batch dim for model, reflatten
            def drift_flat(t, y_flat, drift_args):
                y = y_flat.reshape(sample_shape)
                y_batched = y[None, ...]  # (1, features, channels)
                result = drift(t, y_batched, drift_args)
                result = jnp.squeeze(result, axis=0)  # (features, channels)
                return result.reshape(flat_dim)

            # Wrap diffusion: returns (flat_dim, flat_dim) diagonal matrix
            def diff_flat(t, y_flat, diff_args):
                return diff(t, y_flat, diff_args)

            terms = MultiTerm(
                ODETerm(drift_flat),
                ControlTerm(diff_flat, brownian_motion),
            )

            if return_intermediates:
                saveat = SaveAt(ts=jnp.linspace(t0, t1, nsteps + 1))
            else:
                saveat = SaveAt(t1=True)

            sol = diffeqsolve(
                terms,
                solver,
                t0,
                t1,
                dt0=dt,
                y0=y0_flat,
                args=args,
                stepsize_controller=stepsize_controller,
                saveat=saveat,
            )
            return sol.ys  # (n_saves, flat_dim)

        @partial(jit, static_argnums=(1,))
        def sample(key, nsamples):
            key1, key2 = jax.random.split(key)

            # Sample from prior: flat (nsamples, flat_dim)
            y0s_flat = self.prior_distribution.sample(key1, (nsamples,))

            keys = jax.random.split(key2, nsamples)
            results = jax.vmap(sample_one)(keys, y0s_flat)

            if return_intermediates:
                # (nsamples, n_steps+1, flat_dim) -> (n_steps+1, nsamples, features, channels)
                results = results.reshape(nsamples, nsteps + 1, *sample_shape)
                perm = (1, 0) + tuple(range(2, 2 + len(sample_shape)))
                return jnp.transpose(results, perm)
            else:
                # (nsamples, 1, flat_dim) -> (nsamples, features, channels)
                return results.reshape(nsamples, *sample_shape)

        return sample

    def sample(
        self,
        key: jax.Array,
        nsamples: int,
        nsteps: int = 300,
        method="SEA",
        adaptive=True,
        return_intermediates: bool = False,
        **kwargs,
    ) -> jax.Array:
        """Sample from the SDE.

        Parameters
        ----------
            key : Array
                JAX random key.
            nsamples : int
                Number of samples to generate.
            nsteps : int
                Number of integration steps.
            method : str
                Integration method. One of ``"Euler"``, ``"Heun"``, ``"SEA"``, ``"ShARK"``.
            adaptive : bool
                Whether to use adaptive stepsize (ShARK only).
            return_intermediates : bool
                Whether to return intermediate time steps.

        Returns
        -------
            Array
                Samples of shape ``(nsamples, features, channels)``.
        """
        sampler = self.get_sampler(
            nsteps=nsteps,
            method=method,
            adaptive=adaptive,
            return_intermediates=return_intermediates,
            **kwargs,
        )
        return sampler(key, nsamples)


class ZeroEnds(BaseFmSDESolver):
    """
    ZeroEnds SDE solver for flow matching.

    From Tab. 1 of `arXiv:2410.02217 <http://arxiv.org/abs/2410.02217>`_,
    with change of variable for time: t -> 1-t to match flow matching time notation.

    Parameters
    ----------
        velocity_model : ModelWrapper
            Velocity field model.
        mu0 : Array
            Prior mean, shape ``(features, channels)``.
        sigma0 : Array
            Prior std, shape ``(features, channels)``.
        alpha : float
            Diffusion strength parameter.
        eps0 : float
            Minimum time value.
    """

    def __init__(
        self,
        velocity_model: ModelWrapper,
        mu0: Array,
        sigma0: Array,
        alpha: float,
        eps0: float = 1e-3,
    ):
        super().__init__(velocity_model, mu0, sigma0, eps0=eps0)
        self.alpha = alpha

    def get_f_tilde(self, **kwargs) -> Callable:
        score = self.get_score(**kwargs)
        vf = self.velocity_model.get_vector_field(**kwargs)

        def f_tilde(t, x, args):
            res = vf(t, x, args) + 0.5 * self.alpha**2 * t * (1 - t) * score(t, x, args)
            return res

        return f_tilde

    def get_g_tilde(self) -> Callable:
        flat_dim = self.flat_dim

        def g_tilde(t, y_flat, args):
            """Returns (flat_dim, flat_dim) diagonal diffusion matrix."""
            g = self.alpha * jnp.sqrt(t * (1 - t))
            return g * jnp.eye(flat_dim)

        return g_tilde


class NonSingular(BaseFmSDESolver):
    """
    NonSingular SDE solver for flow matching.

    From Tab. 1 of `arXiv:2410.02217 <http://arxiv.org/abs/2410.02217>`_,
    with change of variable for time: t -> 1-t to match flow matching time notation.

    Parameters
    ----------
        velocity_model : ModelWrapper
            Velocity field model.
        mu0 : Array
            Prior mean, shape ``(features, channels)``.
        sigma0 : Array
            Prior std, shape ``(features, channels)``.
        alpha : float
            Diffusion strength parameter.
    """

    def __init__(
        self, velocity_model: ModelWrapper, mu0: Array, sigma0: Array, alpha: float
    ):
        super().__init__(velocity_model, mu0, sigma0)
        self.alpha = alpha

    def get_f_tilde(self, **kwargs) -> Callable:
        score = self.get_score(**kwargs)
        vf = self.velocity_model.get_vector_field(**kwargs)

        def f_tilde(t, x, args):
            return vf(t, x, args) + 0.5 * self.alpha**2 * (1 - t) * score(t, x, args)

        return f_tilde

    def get_g_tilde(self) -> Callable:
        flat_dim = self.flat_dim

        def g_tilde(t, y_flat, args):
            """Returns (flat_dim, flat_dim) diagonal diffusion matrix."""
            g = self.alpha * jnp.sqrt(1 - t)
            return g * jnp.eye(flat_dim)

        return g_tilde
