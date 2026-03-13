"""
Core SDE solver.

Provides :class:`NewSDESolver`, an abstract base solver for stochastic
differential equations using diffrax.  Subclasses implement
:meth:`get_drift` and :meth:`get_diffusion` to define the SDE coefficients
(often called f̃ and g̃ in the literature, see arXiv:2410.02217).
"""

from abc import abstractmethod
from typing import Callable, Optional, Union
import math

import jax
from jax import jit
import jax.numpy as jnp
from jax import Array

import diffrax
from diffrax import (
    diffeqsolve,
    ControlTerm,
    MultiTerm,
    ODETerm,
    VirtualBrownianTree,
)

from numpyro.distributions import Independent, Normal

from gensbi.solver import Solver
from gensbi.utils.model_wrapping import ModelWrapper


class NewSDESolver(Solver):
    r"""Abstract SDE solver built on diffrax.

    Subclass and implement :meth:`get_drift` (f̃) and :meth:`get_diffusion`
    (g̃) to provide the SDE coefficients:

    .. math::
        dx = \tilde{f}(t, x)\, dt + \tilde{g}(t)\, dW

    The ``velocity_model`` must be a
    :class:`~gensbi.utils.model_wrapping.ModelWrapper` subclass.
    Conditioning is handled entirely by the wrapper layer — the solver
    never needs to know about it.

    Parameters
    ----------
    velocity_model : ModelWrapper
        A properly wrapped model.
    mu0 : Array
        Mean of the prior Gaussian, shape ``(features, channels)``.
    sigma0 : Array
        Std of the prior Gaussian, shape ``(features, channels)``.
    eps0 : float
        Minimum time value (to avoid singularities near t=0).

    Notes
    -----
    **Input shape convention:** all inputs must have shape
    ``(batch, features, channels)``.  If your data is 2-D
    ``(batch, features)``, add a trailing dimension: ``x = x[..., None]``.
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

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def get_drift(self, **kwargs) -> Callable:
        r"""Return the drift function f̃(t, x, args) for the SDE.

        Also known as f̃ (f-tilde) in the SDE literature.

        Returns
        -------
        Callable
            ``drift(t, x, args) -> Array``
        """
        ...  # pragma: no cover

    @abstractmethod
    def get_diffusion(self) -> Callable:
        r"""Return the diffusion function g̃(t, y_flat, args) for the SDE.

        Also known as g̃ (g-tilde) in the SDE literature.
        Must return a ``(flat_dim, flat_dim)`` matrix for
        ``diffrax.ControlTerm``.

        Returns
        -------
        Callable
            ``diffusion(t, y_flat, args) -> Array`` of shape
            ``(flat_dim, flat_dim)``
        """
        ...  # pragma: no cover

    # ------------------------------------------------------------------
    # Sampler
    # ------------------------------------------------------------------
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
        r"""Build a stochastic sampler for the SDE.

        Parameters
        ----------
        step_size : float or None
            Fixed step size.  ``None`` when using ``"ShARK"``
            (adaptive step sizing).
        method : str or AbstractERK
            ``"Euler"``, ``"Heun"``, ``"SEA"``, ``"ShARK"``, or a
            diffrax solver instance.
        atol, rtol : float
            Tolerances for adaptive solvers.
        time_grid : Array
            Integration interval.  Defaults to ``[0, 1]``.
        return_intermediates : bool
            Return solution at every point in *time_grid*.
        static_model_kwargs : dict
            Static keyword arguments baked into the drift at creation
            time.

        Returns
        -------
        Callable
            ``sampler(x_init, key, model_extras=None)``
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

        drift = self.get_drift(**static_model_kwargs)
        diff = self.get_diffusion()

        t0 = time_grid[0]
        t1 = time_grid[-1]
        dt0 = step_size

        # Brownian tree needs t_min < t_max, regardless of integration direction
        bt_t0 = jnp.minimum(t0, t1)
        bt_t1 = jnp.maximum(t0, t1)

        # Adaptive step sizing
        if isinstance(solver, diffrax.ShARK):
            dtmin = 1e-5
            if step_size is not None:
                dtmin = min(2e-5, abs(step_size))
            stepsize_controller = diffrax.PIDController(
                rtol=rtol, atol=atol, dtmin=dtmin
            )
        else:
            stepsize_controller = diffrax.ConstantStepSize()

        flat_dim = self.flat_dim
        sample_shape = self.sample_shape

        @jit
        def sampler(x_init, key, model_extras=None):
            if model_extras is None:
                model_extras = {}
            nsamples = x_init.shape[0]

            def sample_one(key_i, y0_flat):
                """Integrate one sample.  State is flat ``(flat_dim,)``."""

                brownian_motion = VirtualBrownianTree(
                    bt_t0,
                    bt_t1,
                    tol=1e-3,
                    shape=(flat_dim,),
                    key=key_i,
                    levy_area=levy_area,
                )

                # Wrap drift: unflatten → model call → reflatten
                def drift_flat(t, y_flat, drift_args):
                    y = y_flat.reshape(sample_shape)
                    y_batched = y[None, ...]  # (1, features, channels)
                    result = drift(t, y_batched, drift_args)
                    result = jnp.squeeze(result, axis=0)
                    return result.reshape(flat_dim)

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
                return sol.ys  # (n_saves, flat_dim)

            # Flatten x_init: (B, F, C) -> (B, F*C)
            y0s_flat = x_init.reshape(nsamples, flat_dim)

            keys = jax.random.split(key, nsamples)
            results = jax.vmap(sample_one)(keys, y0s_flat)

            if return_intermediates:
                n_times = results.shape[1]
                results = results.reshape(nsamples, n_times, *sample_shape)
                perm = (1, 0) + tuple(range(2, 2 + len(sample_shape)))
                return jnp.transpose(results, perm)
            else:
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
            Initial conditions, shape ``(batch, features, channels)``.
        step_size : float or None
            Step size.
        method : str or AbstractERK
            Integration method.
        atol, rtol : float
            Tolerances.
        time_grid : Array
            Integration interval.
        return_intermediates : bool
            Return intermediates.
        model_extras : dict
            Runtime model extras.
        key : PRNGKey
            Random key (required).

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
