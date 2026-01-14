from typing import Callable, Optional, Sequence, Tuple, Union


import jax
import jax.numpy as jnp
from jax import Array
import diffrax
from diffrax import AbstractERK

from gensbi.flow_matching.solver.solver import Solver
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
        method: Union[str, AbstractERK] = "Dopri5",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid: Array = jnp.array([0.0, 1.0]),
        return_intermediates: bool = False,
        model_extras: dict = {},
    ) -> Callable:
        r"""Obtain a sampler to solve the ODE with the velocity field.

        Parameters
        ----------
            step_size : Optional[float]
                The step size. Must be None for adaptive step solvers.
            method : Union[str, AbstractERK]
                A method supported by diffrax. Defaults to "Dopri5". Other commonly used solvers are "Euler", diffrax.Heun(), and diffrax.Midpoint(). For a complete list, see diffrax documentation.
            atol : float
                Absolute tolerance, used for adaptive step solvers.
            rtol : float
                Relative tolerance, used for adaptive step solvers.
            time_grid : Array
                The process is solved in the interval [min(time_grid), max(time_grid)] and if step_size is None then time discretization is set by the time grid. May specify a descending time_grid to solve in the reverse direction. Defaults to jnp.array([0.0, 1.0]).
            return_intermediates : bool, optional
                If True then return intermediate time steps according to time_grid. Defaults to False.
            model_extras : dict
                Additional input for the model.

        Returns
        -------
            Callable
                A function that takes initial conditions and returns the solution at final time or intermediate times.
        """

        term = diffrax.ODETerm(self.velocity_model.get_vector_field(**model_extras))

        if isinstance(method, str):
            solver = {
                "Euler": diffrax.Euler,
                "Dopri5": diffrax.Dopri5,
            }[method]()
        else:
            solver = method

        if isinstance(solver, AbstractERK):
            stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        else:
            stepsize_controller = diffrax.ConstantStepSize()

        @jax.jit
        def sampler(x_init):

            solution = diffrax.diffeqsolve(
                term,
                solver,
                t0=time_grid[0],
                t1=time_grid[-1],
                dt0=step_size,
                y0=x_init,
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
        method: Union[str, AbstractERK] = "Dopri5",
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
                The step size. Must be None for adaptive step solvers.
            method : Union[str, AbstractERK]
                A method supported by diffrax. Defaults to "Dopri5". Other commonly used solvers are "Euler", diffrax.Heun(), and diffrax.Midpoint(). For a complete list, see diffrax documentation.
            atol : float
                Absolute tolerance, used for adaptive step solvers.
            rtol : float
                Relative tolerance, used for adaptive step solvers.
            time_grid : Array
                The process is solved in the interval [min(time_grid), max(time_grid)] and if step_size is None then time discretization is set by the time grid. May specify a descending time_grid to solve in the reverse direction. Defaults to jnp.array([0.0, 1.0]).
            return_intermediates : bool, optional
                If True then return intermediate time steps according to time_grid. Defaults to False.
            model_extras : dict
                Additional input for the model.

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
            model_extras=model_extras,
        )

        solution = sampler(x_init)

        return solution

    def get_unnormalized_logprob(
        self,
        log_p0: Callable[[Array], Array],
        step_size: float = 0.01,
        method: Union[str, AbstractERK] = "Dopri5",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid=[1.0, 0.0],
        return_intermediates: bool = False,
        # exact_divergence: bool = True,
        *,
        # key: jax.random.PRNGKey = None,
        model_extras: dict = {},
    ) -> Callable:
        r"""Solve for log likelihood given a target sample at :math:`t=0`.

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
            **model_extras: Additional model inputs.

        Returns
        -------
            Union[Tuple[Array, Array], Tuple[Sequence[Array], Array]]: Samples and log likelihood values.
        """
        assert (
            time_grid[0] == 1.0 and time_grid[-1] == 0.0
        ), f"Time grid must start at 1.0 and end at 0.0. Got {time_grid}"

        vector_field = self.velocity_model.get_vector_field(**model_extras)
        divergence = self.velocity_model.get_divergence(**model_extras)

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

        if isinstance(solver, AbstractERK):
            stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        else:
            stepsize_controller = diffrax.ConstantStepSize()

        def sampler(x_1):
            # y_init = (x_1, jnp.ones(x_1.shape)) # the divergence is a scalar, so it has one less dimension than the vector field
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
                saveat=(
                    diffrax.SaveAt(ts=time_grid)
                    if return_intermediates
                    else diffrax.SaveAt(t1=True)
                ),
                stepsize_controller=stepsize_controller,
            )

            x_source, log_det = solution.ys[0], solution.ys[1]  # type: ignore

            source_log_p = log_p0(x_source)

            return source_log_p + log_det

        return sampler

    def unnormalized_logprob(
        self,
        x_1: Array,
        log_p0: Callable[[Array], Array],
        step_size: float = 0.01,
        method: Union[str, AbstractERK] = "Dopri5",
        atol: float = 1e-5,
        rtol: float = 1e-5,
        time_grid=[1.0, 0.0],
        return_intermediates: bool = False,
        # exact_divergence: bool = True,
        *,
        # key: jax.random.PRNGKey = None,
        model_extras: dict = {},
    ) -> Union[Tuple[Array, Array], Tuple[Sequence[Array], Array]]:

        sampler = self.get_unnormalized_logprob(
            log_p0=log_p0,
            step_size=step_size,
            method=method,
            atol=atol,
            rtol=rtol,
            time_grid=time_grid,
            return_intermediates=return_intermediates,
            model_extras=model_extras,
        )
        solution = sampler(x_1)
        return solution
