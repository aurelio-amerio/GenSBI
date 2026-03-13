"""
Core ODE solver.

Provides :class:`NewODESolver`, an abstract base solver for ordinary
differential equations using diffrax.  Subclasses only need to implement
:meth:`get_drift` to define the vector field (often called :math:`\tilde{f}` in the SDE
literature).
"""

from abc import abstractmethod
from typing import Callable, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array

import diffrax
from diffrax import AbstractERK

from gensbi.solver import Solver
from gensbi.utils.model_wrapping import ModelWrapper


class NewODESolver(Solver):
    r"""Abstract ODE solver built on diffrax.

    Subclass and implement :meth:`get_drift` to provide the drift / velocity
    field for the ODE.

    The ``velocity_model`` must be a :class:`~gensbi.utils.model_wrapping.ModelWrapper`
    subclass.  Conditioning is handled entirely by the wrapper layer
    (``ConditionalWrapper``, ``JointWrapper``, etc.) — the solver never
    needs to know about it.

    Parameters
    ----------
    velocity_model : ModelWrapper
        A properly wrapped model providing ``get_vector_field`` and
        ``get_divergence`` methods.
    """

    def __init__(self, velocity_model: ModelWrapper):
        super().__init__()
        self.velocity_model = velocity_model

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------
    @abstractmethod
    def get_drift(self, **kwargs) -> Callable:
        r"""Return the drift function for the ODE.

        Also known as :math:`\tilde{f}` in the SDE/ODE literature.

        Returns
        -------
        Callable
            ``drift(t, x, args) -> Array``
        """
        ...  # pragma: no cover

    # ------------------------------------------------------------------
    # Sampler
    # ------------------------------------------------------------------
    def get_sampler(
        self,
        step_size: Optional[float],
        method: Union[str, AbstractERK] = "Euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Optional[Array] = None,
        return_intermediates: bool = False,
        static_model_kwargs: dict = None,
    ) -> Callable:
        r"""Obtain a sampler to solve the ODE.

        Parameters
        ----------
        step_size : float or None
            Fixed step size. ``None`` when using adaptive solvers
            (e.g. ``"Dopri5"``).
        method : str or AbstractERK
            Diffrax solver.  ``"Euler"``, ``"Dopri5"``, ``diffrax.Heun()``,
            ``diffrax.Midpoint()``, etc.
        atol : float
            Absolute tolerance (adaptive solvers).
        rtol : float
            Relative tolerance (adaptive solvers).
        time_grid : Array, optional
            Integration interval ``[time_grid[0], time_grid[-1]]``.
            Defaults to ``[0, 1]``.
        return_intermediates : bool
            If True, return solution at every point in *time_grid*.
        static_model_kwargs : dict
            Static keyword arguments baked into the drift at creation
            time.  Condition-dependent data should be passed at call time
            via ``model_extras``.

        Returns
        -------
        Callable
            ``sampler(x_init, model_extras=None)``
        """
        if static_model_kwargs is None:
            static_model_kwargs = {}

        if time_grid is None:
            time_grid = jnp.array([0.0, 1.0])

        term = diffrax.ODETerm(self.get_drift(**static_model_kwargs))

        if isinstance(method, str):
            solver = {
                "Euler": diffrax.Euler,
                "Dopri5": diffrax.Dopri5,
            }[method]()
        else:
            solver = method

        if isinstance(solver, diffrax.Dopri5):
            stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        else:
            stepsize_controller = diffrax.ConstantStepSize()

        @jax.jit
        def sampler(x_init, model_extras=None):
            if model_extras is None:
                model_extras = {}

            solution = diffrax.diffeqsolve(
                term,
                solver,
                t0=time_grid[0],
                t1=time_grid[-1],
                dt0=step_size,
                y0=x_init,
                args=model_extras,
                saveat=(
                    diffrax.SaveAt(ts=time_grid)
                    if return_intermediates
                    else diffrax.SaveAt(t1=True)
                ),
                stepsize_controller=stepsize_controller,
            )
            return solution.ys if return_intermediates else solution.ys[-1]  # type: ignore

        return sampler

    def sample(
        self,
        x_init: Array,
        step_size: Optional[float],
        method: Union[str, AbstractERK] = "Euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Array = jnp.array([0.0, 1.0]),
        return_intermediates: bool = False,
        model_extras: dict = None,
    ) -> Union[Array, Sequence[Array]]:
        r"""Sample from the ODE.

        Parameters
        ----------
        x_init : Array
            Initial conditions. Shape ``(batch, ...)``.
        step_size : float or None
            Step size.
        method : str or AbstractERK
            Integration method.
        atol, rtol : float
            Tolerances for adaptive solvers.
        time_grid : Array
            Integration interval.
        return_intermediates : bool
            Return intermediate steps.
        model_extras : dict
            Runtime model extras (e.g. ``cond``, ``obs_ids``).

        Returns
        -------
        Array
            Solution at final time or at all intermediate times.
        """
        sampler = self.get_sampler(
            step_size=step_size,
            method=method,
            atol=atol,
            rtol=rtol,
            time_grid=time_grid,
            return_intermediates=return_intermediates,
        )
        return sampler(x_init, model_extras=model_extras)

    # ------------------------------------------------------------------
    # Log-probability (continuous change of variables)
    # ------------------------------------------------------------------
    def get_log_prob(
        self,
        log_p0: Callable[[Array], Array],
        step_size: float = 0.01,
        method: Union[str, AbstractERK] = "Dopri5",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Optional[Array] = None,
        return_intermediates: bool = False,
        exact_divergence: bool = True,
        *,
        static_model_kwargs: dict = None,
    ) -> Callable:
        r"""Build a log-probability function via the change-of-variables formula.

        Parameters
        ----------
        log_p0 : Callable
            Log-probability of the source (base) distribution.
        step_size : float
            Step size for fixed-step solvers.
        method : str or AbstractERK
            Integration method.
        atol, rtol : float
            Tolerances for adaptive solvers.
        time_grid : Array, optional
            Must be descending (data → source). Defaults to ``[1, 0]``.
        return_intermediates : bool
            Return intermediate steps.
        exact_divergence : bool
            Use exact divergence (True) or Hutchinson estimator (False).
        static_model_kwargs : dict
            Static keyword arguments for the drift.

        Returns
        -------
        Callable
            ``log_prob_fn(x_1, model_extras=None, *, key=None)``
        """
        if time_grid is None:
            time_grid = jnp.array([1.0, 0.0])
        assert (
            time_grid[0] > time_grid[-1]
        ), f"Time grid must be descending for log-prob (source ← data). Got {time_grid}"

        if static_model_kwargs is None:
            static_model_kwargs = {}

        vector_field = self.get_drift(**static_model_kwargs)
        divergence = self.velocity_model.get_divergence(
            exact=exact_divergence, **static_model_kwargs
        )

        def dynamics_func(t, states, args):
            xt, _ = states
            ut = vector_field(t, xt, args)
            div = divergence(t, xt, args)
            return ut, div

        term = diffrax.ODETerm(dynamics_func)

        if isinstance(method, str):
            solver = {
                "Euler": diffrax.Euler(),
                "Dopri5": diffrax.Dopri5(),
            }[method]
        else:
            solver = method

        if isinstance(solver, diffrax.Dopri5):
            stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        else:
            stepsize_controller = diffrax.ConstantStepSize()

        def sampler(x_1, model_extras=None, *, key=None):
            if model_extras is None:
                model_extras = {}
            _extras = dict(model_extras)

            if not exact_divergence:
                if key is None:
                    raise ValueError(
                        "A PRNG key is required for Hutchinson divergence. "
                        "Pass key= when calling the log_prob function."
                    )
                from gensbi.utils.math import _expand_dims
                v = jax.random.rademacher(
                    key, shape=_expand_dims(x_1).shape, dtype=x_1.dtype
                )
                _extras["div_v"] = v

            y_init = (
                x_1,
                jnp.zeros(x_1.shape[0]),
            )
            solution = diffrax.diffeqsolve(
                term,
                solver,
                t0=time_grid[0],
                t1=time_grid[-1],
                dt0=-step_size,
                y0=y_init,
                args=_extras,
                saveat=(
                    diffrax.SaveAt(ts=time_grid)
                    if return_intermediates
                    else diffrax.SaveAt(t1=True)
                ),
                stepsize_controller=stepsize_controller,
            )

            x_source, log_det = solution.ys[0], solution.ys[1]  # type: ignore

            if not return_intermediates:
                x_source = x_source[-1]
                log_det = log_det[-1]

            source_log_p = log_p0(x_source)

            return source_log_p + log_det

        return sampler

    def compute_log_prob(
        self,
        x_1: Array,
        log_p0: Callable[[Array], Array],
        step_size: float = 0.01,
        method: Union[str, AbstractERK] = "Dopri5",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Optional[Array] = None,
        return_intermediates: bool = False,
        exact_divergence: bool = True,
        *,
        key: jax.random.PRNGKey = None,
        model_extras: dict = None,
    ) -> Union[Tuple[Array, Array], Tuple[Sequence[Array], Array]]:
        """Compute log-probability for given samples."""
        sampler = self.get_log_prob(
            log_p0=log_p0,
            step_size=step_size,
            method=method,
            atol=atol,
            rtol=rtol,
            time_grid=time_grid,
            return_intermediates=return_intermediates,
            exact_divergence=exact_divergence,
        )
        return sampler(x_1, model_extras=model_extras, key=key)
