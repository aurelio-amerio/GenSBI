from typing import Callable, Optional, Sequence, Tuple, Union


import jax
import jax.numpy as jnp
from jax import Array
import diffrax
from diffrax import AbstractERK

from gensbi.solver import Solver
from gensbi.utils.model_wrapping import ModelWrapper


class ODESolver(Solver):
    """A class to solve ordinary differential equations (ODEs) using a specified velocity model.

    This class utilizes a velocity field model to solve ODEs over a given time grid using numerical ode solvers.

    Parameters
    ----------
        velocity_model : Union[ModelWrapper, Callable]
            a velocity field model receiving :math:`(x,t)` and returning :math:`u_t(x)`

    Example:
        .. code-block:: python

            from gensbi.flow_matching.solver import ODESolver
            from gensbi.utils.model_wrapping import ModelWrapper
            import jax, jax.numpy as jnp

            class DummyModel:
                def __call__(self, obs, t, *args, **kwargs):
                    return jnp.squeeze(obs + t, axis=-1)

            vf_model = DummyModel() # replace with your actual velocity field model, Simformer or Flux1

            model_wrapped = ModelWrapper(vf_model) # you should use the appropriate ModelWrapper for your model, either ConditionalWrapper or JointWrapper, or a custom subclass of ModelWrapper
            solver = ODESolver(velocity_model=model_wrapped)
            x_init = jnp.zeros((10, 2))
            time_grid = jnp.linspace(0, 1, 5)
            sol = solver.sample(x_init=x_init, step_size=0.05, time_grid=time_grid)
            print(sol.shape)
            # (5, 10, 2)
    """

    def __init__(self, velocity_model: ModelWrapper):
        super().__init__()
        self.velocity_model = velocity_model

    def get_sampler(
        self,
        step_size: Optional[float],
        method: Union[str, AbstractERK] = "Euler",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Array = jnp.array([0.0, 1.0]),
        return_intermediates: bool = False,
        static_model_kwargs: dict = {},
    ) -> Callable:
        r"""Obtain a sampler to solve the ODE with the velocity field.

        **Time direction convention:**
        In flow matching, ``t=0`` is noise and ``t=1`` is data. Sampling
        integrates **forward** from ``t=0`` to ``t=1`` to transform noise
        into data. This is the opposite of standard score matching (reverse
        SDE: ``t=T→eps``) and different from EDM (which uses σ-space).

        Parameters
        ----------
            step_size : Optional[float]
                The step size. Must be None when using ``"Dopri5"`` (adaptive step sizing).
            method : Union[str, AbstractERK]
                A method supported by diffrax. Defaults to "Euler". Other commonly
                used solvers are ``"Dopri5"`` (adaptive), ``diffrax.Heun()``, and
                ``diffrax.Midpoint()``. ``"Dopri5"`` automatically uses adaptive
                step sizing via ``PIDController``; the others use fixed step size.
            atol : float
                Absolute tolerance, used for adaptive step solvers.
            rtol : float
                Relative tolerance, used for adaptive step solvers.
            time_grid : Array
                The process is solved in the interval [min(time_grid), max(time_grid)] and if step_size is None then time discretization is set by the time grid. May specify a descending time_grid to solve in the reverse direction. Defaults to jnp.array([0.0, 1.0]).
            return_intermediates : bool, optional
                If True then return intermediate time steps according to time_grid. Defaults to False.
            static_model_kwargs : dict
                Static keyword arguments baked into the vector field at
                creation time.  Use for genuinely static configuration;
                condition-dependent data should be passed at call time
                via the ``model_extras`` argument of the returned sampler.

        Returns
        -------
            Callable
                ``sampler(x_init, model_extras={})`` — a function that
                takes initial conditions and runtime model extras, and
                returns the solution at final time or intermediate times.
        """

        term = diffrax.ODETerm(self.velocity_model.get_vector_field(**static_model_kwargs))

        if isinstance(method, str):
            solver = {
                "Euler": diffrax.Euler,
                "Dopri5": diffrax.Dopri5,
            }[method]()
        else:
            solver = method

        # Adaptive step sizing: only Dopri5 uses PIDController (high-order
        # embedded error pair makes adaptivity worthwhile). Other solvers
        # (Euler, Heun, Midpoint) use fixed step size.
        if isinstance(solver, diffrax.Dopri5):
            stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        else:
            stepsize_controller = diffrax.ConstantStepSize()

        @jax.jit
        def sampler(x_init, model_extras={}):

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
        model_extras: dict = {},
    ) -> Union[Array, Sequence[Array]]:
        r"""Sample from the ODE defined by the velocity field.

        Parameters
        ----------
            x_init : Array
                Initial conditions (e.g., source samples :math:`X_0 \sim p`). Shape: [batch_size, ...].
            step_size : Optional[float]
                The step size. Must be None when using ``"Dopri5"`` (adaptive step sizing).
            method : Union[str, AbstractERK]
                A method supported by diffrax. Defaults to "Euler". Other commonly
                used solvers are ``"Dopri5"`` (adaptive), ``diffrax.Heun()``, and
                ``diffrax.Midpoint()``. ``"Dopri5"`` automatically uses adaptive
                step sizing via ``PIDController``; the others use fixed step size.
            atol : float
                Absolute tolerance, used for adaptive step solvers.
            rtol : float
                Relative tolerance, used for adaptive step solvers.
            time_grid : Array
                The process is solved in the interval [min(time_grid), max(time_grid)] and if step_size is None then time discretization is set by the time grid. May specify a descending time_grid to solve in the reverse direction. Defaults to jnp.array([0.0, 1.0]).
            return_intermediates : bool, optional
                If True then return intermediate time steps according to time_grid. Defaults to False.
            model_extras : dict
                Runtime model extras (e.g. ``cond``, ``obs_ids``).

        Returns
        -------
            Union[Array, Sequence[Array]]
                The final state or the states at all intermediate time steps.
        """

        sampler = self.get_sampler(
            step_size=step_size,
            method=method,
            atol=atol,
            rtol=rtol,
            time_grid=time_grid,
            return_intermediates=return_intermediates,
        )

        solution = sampler(x_init, model_extras=model_extras)

        return solution

    def get_log_prob(
        self,
        log_p0: Callable[[Array], Array],
        step_size: float = 0.01,
        method: Union[str, AbstractERK] = "Dopri5",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid=[1.0, 0.0],
        return_intermediates: bool = False,
        exact_divergence: bool = True,
        *,
        static_model_kwargs: dict = {},
    ) -> Callable:
        r"""Solve for log_prob given a target sample at :math:`t=0`.

        Parameters
        ----------
            x_1 : Array
                target sample (e.g., samples :math:`X_1 \sim p_1`).
            log_p0 : Callable[[Array], Array]
                Log probability function of source distribution.
            step_size : Optional[float]
                Step size for fixed-step solvers.
            method : str
                Integration method to use.
            atol : float
                Absolute tolerance for adaptive solvers.
            rtol : float
                Relative tolerance for adaptive solvers.
            time_grid : Array
                Must start at 1.0 and end at 0.0.
            return_intermediates : bool
                Whether to return intermediate steps.
            exact_divergence : bool
                Use exact divergence vs Hutchinson estimator.
            static_model_kwargs : dict
                Static keyword arguments baked into the vector field.

        Returns
        -------
            Union[Tuple[Array, Array], Tuple[Sequence[Array], Array]]: Samples and log prob values.
        """
        assert (
            time_grid[0] == 1.0 and time_grid[-1] == 0.0
        ), f"Time grid must start at 1.0 and end at 0.0. Got {time_grid}"

        vector_field = self.velocity_model.get_vector_field(**static_model_kwargs)
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

        # Adaptive step sizing: only Dopri5 uses PIDController
        if isinstance(solver, diffrax.Dopri5):
            stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        else:
            stepsize_controller = diffrax.ConstantStepSize()

        def sampler(x_1, model_extras={}, *, key=None):
            _extras = dict(model_extras)  # shallow copy

            # For Hutchinson: draw probe vector v once, fixed across ODE steps
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
            )  # the divergence is a scalar, so it has one less dimension than the vector field
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
        time_grid=[1.0, 0.0],
        return_intermediates: bool = False,
        exact_divergence: bool = True,
        *,
        key: jax.random.PRNGKey = None,
        model_extras: dict = {},
    ) -> Union[Tuple[Array, Array], Tuple[Sequence[Array], Array]]:

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
        solution = sampler(x_1, model_extras=model_extras, key=key)
        return solution
