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

from typing import Callable, Optional, Union, Sequence, Tuple
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
        step_size: Optional[float],
        method: Union[str, diffrax.AbstractERK] = "Euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Array = jnp.array([0.0, 1.0]),
        return_intermediates: bool = False,
        static_model_kwargs: dict = None,
    ) -> Callable:
        """Stochastic sampler for the SDE.

        Parameters
        ----------
            step_size : Optional[float]
                The step size. Must be None when using ``"ShARK"`` (adaptive step sizing).
            method : Union[str, diffrax.AbstractERK]
                Integration method. One of ``"Euler"``, ``"Heun"``, ``"SEA"``,
                ``"ShARK"``. Defaults to ``"Euler"``. ``"ShARK"`` automatically
                uses adaptive step sizing via ``PIDController``; the others use
                fixed step size.
            atol : float
                Absolute tolerance, used for adaptive step solvers.
            rtol : float
                Relative tolerance, used for adaptive step solvers.
            time_grid : Array
                The process is solved in the interval [min(time_grid), max(time_grid)] and if step_size is None then time discretization is set by the time grid. May specify a descending time_grid to solve in the reverse direction. Defaults to jnp.array([0.0, 1.0]).
            return_intermediates : bool, optional
                If True then return intermediate time steps according to time_grid. Defaults to False.
            static_model_kwargs : dict
                Static keyword arguments baked into the drift/diffusion
                at creation time.  Condition-dependent data should be
                passed at call time via ``model_extras``.

        Returns
        -------
            Callable
                ``sampler(x_init, key, model_extras=None)`` that returns
                sampled trajectories of shape ``(batch, features, channels)``.
        """
        solvers = {
            "Euler": diffrax.Euler,
            "Heun": diffrax.Heun,
            "SEA": diffrax.SEA,
            "ShARK": diffrax.ShARK,
        }

        if isinstance(method, str):
            if method not in solvers:
                raise ValueError(
                    f"Method {method} not supported. Choose from {list(solvers.keys())}."
                )
            solver = solvers[method]()
        else:
            solver = method

        if isinstance(solver, (diffrax.Euler, diffrax.Heun, diffrax.EulerHeun)):
            levy_area = diffrax.BrownianIncrement
        else:
            levy_area = diffrax.SpaceTimeLevyArea

        if static_model_kwargs is None:
            static_model_kwargs = {}

        drift = self.get_f_tilde()  # (t, x, args) -> drift; no extras baked in
        diff = self.get_g_tilde()  # (t, x, args) -> diffusion matrix

        # Time direction: forward, from noise (t=eps) to data (t=1)
        t0 = time_grid[0]
        t1 = time_grid[-1]

        # If step_size is provided, use it.
        dt0 = step_size

        # Adaptive step sizing
        if isinstance(solver, diffrax.ShARK):
            # dtmin logic:
            dtmin = 1e-5
            if step_size is not None:
                dtmin = min(2e-5, step_size)

            stepsize_controller = diffrax.PIDController(
                rtol=rtol, atol=atol, dtmin=dtmin
            )
        else:
            stepsize_controller = diffrax.ConstantStepSize()

        flat_dim = self.flat_dim
        sample_shape = self.sample_shape

        # We remove static_argnums because now nsamples is implicit in x_init shape
        @jit
        def sampler(x_init, key, model_extras=None):
            if model_extras is None:
                model_extras = {}
            # x_init shape: (batch, features, channels)
            nsamples = x_init.shape[0]

            def sample_one(key_i, y0_flat):
                """Integrate one sample. State is flat (flat_dim,)."""

                brownian_motion = VirtualBrownianTree(
                    t0,
                    t1,
                    tol=1e-3,
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
                    saveat = diffrax.SaveAt(ts=time_grid)
                else:
                    saveat = diffrax.SaveAt(t1=True)

                sol = diffeqsolve(
                    terms,
                    solver,
                    t0,
                    t1,
                    dt0=dt0,
                    y0=y0_flat,
                    args=model_extras,
                    stepsize_controller=stepsize_controller,
                    saveat=saveat,
                )
                val = sol.ys  # (n_saves, flat_dim)
                return val

            # flatten x_init
            # x_init: (B, F, C) -> (B, F*C)
            y0s_flat = x_init.reshape(nsamples, flat_dim)

            keys = jax.random.split(key, nsamples)
            results = jax.vmap(sample_one)(keys, y0s_flat)

            if return_intermediates:
                # results is (batch, time, flat)
                # reshape to (batch, time, features, channels)
                n_times = results.shape[1]
                results = results.reshape(nsamples, n_times, *sample_shape)
                # transpose to (time, batch, features, channels)
                # to match ODESolver which returns (time, batch, ...) for intermediates?
                # ODESolver: return solution.ys -> (n_steps, batch, features)
                # (actually ODESolver vmap logic might be slightly different or implicit via diffrax batching if used,
                # but ODESolver vmap is explicit? no, ODESolver uses diffrax.diffeqsolve inside vmap?
                # Wait, ODESolver.get_sampler returns a jitted sampler:
                # @jax.jit
                # def sampler(x_init):
                #   solution = diffrax.diffeqsolve(...)
                #   return solution.ys
                #
                # If x_init is batched, diffrax.diffeqsolve might auto-batch if configured,
                # OR ODESolver expects unbatched x_init and user vmaps it?
                # Let's check ODESolver again.
                perm = (1, 0) + tuple(range(2, 2 + len(sample_shape)))
                return jnp.transpose(results, perm)
            else:
                # results is (batch, 1, flat) -> (batch, features, channels)
                # sample_one returns (1, flat) if not intermediate.
                # vmap adds batch dim -> (batch, 1, flat)
                return results.reshape(nsamples, *sample_shape)

        return sampler

    def sample(
        self,
        x_init: Array,
        step_size: Optional[float],
        method: Union[str, diffrax.AbstractERK] = "Euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Array = jnp.array([0.0, 1.0]),
        return_intermediates: bool = False,
        model_extras: dict = None,
        key: Optional[Array] = None,
    ) -> Array:
        """Sample from the SDE.

        Parameters
        ----------
            x_init : Array
                Initial conditions. Shape: [batch, features, channels].
            step_size : Optional[float]
                Step size.
            method : Union[str, diffrax.AbstractERK]
                Integration method.
            atol : float
                Absolute tolerance.
            rtol : float
                Relative tolerance.
            time_grid : Array
                Time grid.
            return_intermediates : bool
                Return intermediates.
            model_extras : dict
                Runtime model extras (e.g. ``cond``, ``obs_ids``).
            key : jax.random.PRNGKey
                Random key. Required for SDE.

        Returns
        -------
            Array
                Samples.
        """
        if key is None:
            raise ValueError("key is required for SDE sampling.")

        sampler = self.get_sampler(
            step_size=step_size,
            method=method,
            atol=atol,
            rtol=rtol,
            time_grid=time_grid,
            return_intermediates=return_intermediates,
        )
        return sampler(x_init, key, model_extras=model_extras)


class ZeroEndsSolver(BaseFmSDESolver):
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


class NonSingularSolver(BaseFmSDESolver):
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
