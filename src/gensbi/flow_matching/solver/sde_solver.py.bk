from typing import Callable, Optional, Sequence, Tuple, Union

from abc import abstractmethod

import jax
from jax import jit, vmap
import jax.numpy as jnp
from jax import Array
import diffrax

from functools import partial

from einops import repeat

from numpyro.distributions import Independent, Normal

from diffrax import (
    diffeqsolve,
    ControlTerm,
    MultiTerm,
    ODETerm,
    VirtualBrownianTree,
)

from gensbi.flow_matching.solver.solver import Solver
from gensbi.utils.model_wrapping import ModelWrapper


# TODO: we might want add support for intermediate sampling steps


class BaseSDESolver(Solver):
    """A class to solve ordinary differential equations (ODEs) using a specified velocity model.

    This class utilizes a velocity field model to solve ODEs over a given time grid using numerical ode solvers.

    Args:
        velocity_model (Union[ModelWrapper, Callable]): a velocity field model receiving :math:`(x,t)` and returning :math:`u_t(x)`
    """

    def __init__(
        self,
        velocity_model: ModelWrapper,
        mu0: Array,
        sigma0: Array,
        eps0: float = 1e-5,
    ):
        """
        Args:
            velocity_model (ModelWrapper): a velocity field model, propery wrapped
            mu0 (Array): mean of the prior gaussian distribution
            sigma0 (Array): standard deviation of the prior gaussian distribution
        """
        super().__init__()
        self.velocity_model = velocity_model
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.prior_distribution = Independent(
            Normal(mu0, sigma0), reinterpreted_batch_ndims=1
        )
        self.dim = mu0.shape[0]

        self.eps0 = eps0  # dafaults to 1e-5
        return

    @abstractmethod
    def get_f_tilde(self) -> Callable:
        """Get the function :math:`\tilde{f}` for the velocity model. See arXiv.2410.02217
        Also known as the "drift" term in the SDE context.
        """
        ...  # pragma: no cover

    @abstractmethod
    def get_g_tilde(self) -> Callable:
        """Get the function :math:`\tilde{g}` for the velocity model. See arXiv.2410.02217
        Also known as the "diffusion" term in the SDE context.
        """
        ...  # pragma: no cover

    def get_score(self, **kwargs):
        """Obtain the score function given the velocity model. See arXiv.2410.02217"""

        vf = self.velocity_model.get_vector_field(**kwargs)

        def score(t, x, args):
            res = (-t * vf(t, x, args) + self.mu0 - x) / ((1 - t) * self.sigma0**2)
            return res

        return score

    def get_sampler(
        self, args=None, nsteps=300, method="SEA", adaptive=False, **kwargs
    ) -> Callable:
        """Stochastic sampler for the SDE.
        Args:
            args: additional arguments to pass to the velocity model
            nsteps: number of steps for the SDE solver
            method: the method to use for the SDE solver, can be one of "Euler", "SEA", "ShARK". Defaults to "SEA". Euler is the simplest algorithm. SEA (Shifted Euler method) has a better constant factor in the global error and an improved local error. ShARK (Shifted Additive-noise Runge-Kutta) provides a more accurate solution with a higher computational cost, and implements adaptive stepsize control.
            adaptive: whether to use adaptive stepsize control (only for ShARK). Defaults to True.

        Returns:
            Callable: A function that takes initial conditions and returns the solution at final time.
        """

        solvers = {
            "Euler": diffrax.Euler,
            "SEA": diffrax.SEA,
            "ShARK": diffrax.ShARK,
        }
        if method not in solvers.keys():
            raise ValueError(
                f"Method {method} not supported. Choose from {solvers.keys()}."
            )

        solver = solvers[method]()

        if method in ["Euler"]:
            levy_area = diffrax.BrownianIncrement
        else:
            levy_area = diffrax.SpaceTimeLevyArea
            # levy_area = diffrax.SpaceTimeTimeLevyArea

        drift = self.get_f_tilde(**kwargs)  # drift term, (t,x,args) -> f_tilde
        diff = self.get_g_tilde()  # diffusion term, (t,x,args) -> g_tilde

        t0 = self.eps0
        t1 = 1

        dt = t1 / nsteps

        dtmin = min(2e-5, dt)  # minimum step size
        tol = dtmin / 2

        if method in ["ShARK"] and adaptive:
            stepsize_controller = diffrax.PIDController(
                rtol=1e-5, atol=1e-5, dtmin=dtmin, dtmax=2 * dt
            )
        else:
            stepsize_controller = diffrax.ConstantStepSize()

        @jit
        def sample_batch(key, y0):
            brownian_motion = VirtualBrownianTree(
                t0, t1, tol=tol, shape=(self.dim,), key=key, levy_area=levy_area
            )
            terms = MultiTerm(ODETerm(drift), ControlTerm(diff, brownian_motion))
            sol = diffeqsolve(
                terms,
                solver,
                t0,
                t1,
                dt0=dt,
                y0=y0,
                args=args,
                stepsize_controller=stepsize_controller,
            )
            return sol.ys[-1]

        @partial(jit, static_argnums=(1))
        def sample(key, nsamples):
            key1, key2 = jax.random.split(key)
            y0s = self.prior_distribution.sample(key1, (nsamples,))
            res = sample_batch(key2, y0s)
            return res

        return sample

    def sample(
        self,
        key: jax.Array,
        nsamples: int,
        nsteps: int = 300,
        method="SEA",
        adaptive=True,
        **kwargs,
    ) -> jax.Array:
        """Sample from the SDE using the provided key and number of samples.

        Args:
            key (jax.Array): JAX random key for sampling.
            nsamples (int): Number of samples to generate.
            nsteps (int): Number of steps for the SDE solver.
            **kwargs: Additional arguments to pass to the velocity model.

        Returns:
            Array: Sampled trajectories from the SDE.
        """
        sampler = self.get_sampler(
            nsteps=nsteps, method=method, adaptive=adaptive, **kwargs
        )
        return sampler(key, nsamples)


class ZeroEnds(BaseSDESolver):
    """
    ZeroEnds SDE solver.

    From tab 1 of `arXiv:2410.02217 <http://arxiv.org/abs/2410.02217>`_, with change of variable for time: t -> 1-t to match our time notation.
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
        def g_tilde(t, x, args):
            t = jnp.atleast_1d(t)
            assert t.ndim == 1, f"t must be a 1D array, got shape {t.shape}"
            assert t.shape[0] in [
                1,
                x.shape[0],
            ], f"t must have shape (1,) or ({x.shape[0]},), got {t.shape}"
            b, d = x.shape

            res = self.alpha * jnp.sqrt(t * (1 - t))  # scalar
            if t.shape[0] == 1:
                res = jnp.repeat(res, b)
            eye = jnp.eye(d)
            return res[..., None, None] * eye[None, :, :]

        return g_tilde


class NonSingular(BaseSDESolver):
    """
    NonSingular SDE solver.

    From tab 1 of `arXiv:2410.02217 <http://arxiv.org/abs/2410.02217>`_, with change of variable for time: t -> 1-t to match our time notation.
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
        def g_tilde(t, x, args):
            t = jnp.atleast_1d(t)
            assert t.ndim == 1, f"t must be a 1D array, got shape {t.shape}"
            assert t.shape[0] in [
                1,
                x.shape[0],
            ], f"t must have shape (1,) or ({x.shape[0]},), got {t.shape}"
            b, d = x.shape
            res = self.alpha * jnp.sqrt(1 - t)  # scalar
            if t.shape[0] == 1:
                res = jnp.repeat(res, b)
            eye = jnp.eye(d)
            return res[..., None, None] * eye[None, :, :]

        return g_tilde
