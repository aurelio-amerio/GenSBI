"""
Flow matching SDE solvers.

Provides :class:`NewFMSDESolver` (abstract base with score derivation),
:class:`NewZeroEndsSolver`, and :class:`NewNonSingularSolver`.

Based on: "Improving Flow Matching by Stochastic Sampling"
(arXiv:2410.02217)

**Time direction convention:**
In flow matching, ``t=0`` is noise and ``t=1`` is data.
These SDE solvers integrate **forward** from ``t=eps`` to ``t=1``.
"""

from abc import abstractmethod
from typing import Callable

import jax.numpy as jnp
from jax import Array

from gensbi.core.sde_solver import NewSDESolver
from gensbi.utils.model_wrapping import ModelWrapper


class NewFMSDESolver(NewSDESolver):
    r"""Base flow matching SDE solver.

    Provides :meth:`get_score` which derives the score from the velocity
    field (see arXiv:2410.02217).  Subclasses implement :meth:`get_drift`
    (:math:`\tilde{f}`) and :meth:`get_diffusion` (:math:`\tilde{g}`) using the score and velocity field.

    Parameters
    ----------
    velocity_model : ModelWrapper
        Wrapped velocity field model.
    mu0 : Array
        Prior mean, shape ``(features, channels)``.
    sigma0 : Array
        Prior std, shape ``(features, channels)``.
    eps0 : float
        Minimum time value.
    """

    def get_score(self, **kwargs) -> Callable:
        r"""Obtain the score function from the velocity model.

        See arXiv:2410.02217.

        Returns
        -------
        Callable
            ``score(t, x, args) -> Array``
        """
        vf = self.velocity_model.get_vector_field(**kwargs)

        def score(t, x, args):
            res = (-t * vf(t, x, args) + self.mu0 - x) / (
                (1 - t) * self.sigma0**2
            )
            return res

        return score

    @abstractmethod
    def get_drift(self, **kwargs) -> Callable:
        ...  # pragma: no cover

    @abstractmethod
    def get_diffusion(self) -> Callable:
        ...  # pragma: no cover


class NewZeroEndsSolver(NewFMSDESolver):
    r"""ZeroEnds SDE solver for flow matching.

    From Tab. 1 of `arXiv:2410.02217 <http://arxiv.org/abs/2410.02217>`_,
    with change of variable for time: t → 1−t to match flow matching
    time notation.

    The drift (:math:`\tilde{f}`) and diffusion (:math:`\tilde{g}`) are:

    .. math::
        \tilde{f}(t, x) = u_t(x) + \tfrac{1}{2}\alpha^2 t(1-t)\, s(t, x)

        \tilde{g}(t) = \alpha \sqrt{t(1-t)}

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

    def get_drift(self, **kwargs) -> Callable:
        r"""Return drift :math:`\tilde{f}(t, x, \text{args})` for ZeroEnds SDE."""
        score = self.get_score(**kwargs)
        vf = self.velocity_model.get_vector_field(**kwargs)

        def drift(t, x, args):
            res = (
                vf(t, x, args)
                + 0.5 * self.alpha**2 * t * (1 - t) * score(t, x, args)
            )
            return res

        return drift

    def get_diffusion(self) -> Callable:
        r"""Return diffusion :math:`\tilde{g}(t)` for ZeroEnds SDE."""
        flat_dim = self.flat_dim

        def g_tilde(t, y_flat, args):
            """Returns (flat_dim, flat_dim) diagonal diffusion matrix."""
            g = self.alpha * jnp.sqrt(t * (1 - t))
            return g * jnp.eye(flat_dim)

        return g_tilde


class NewNonSingularSolver(NewFMSDESolver):
    r"""NonSingular SDE solver for flow matching.

    From Tab. 1 of `arXiv:2410.02217 <http://arxiv.org/abs/2410.02217>`_,
    with change of variable for time: t → 1−t to match flow matching
    time notation.

    The drift (:math:`\tilde{f}`) and diffusion (:math:`\tilde{g}`) are:

    .. math::
        \tilde{f}(t, x) = u_t(x) + \tfrac{1}{2}\alpha^2 (1-t)\, s(t, x)

        \tilde{g}(t) = \alpha \sqrt{1-t}

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
        self,
        velocity_model: ModelWrapper,
        mu0: Array,
        sigma0: Array,
        alpha: float,
    ):
        super().__init__(velocity_model, mu0, sigma0)
        self.alpha = alpha

    def get_drift(self, **kwargs) -> Callable:
        r"""Return drift :math:`\tilde{f}(t, x, \text{args})` for NonSingular SDE."""
        score = self.get_score(**kwargs)
        vf = self.velocity_model.get_vector_field(**kwargs)

        def drift(t, x, args):
            return (
                vf(t, x, args)
                + 0.5 * self.alpha**2 * (1 - t) * score(t, x, args)
            )

        return drift

    def get_diffusion(self) -> Callable:
        r"""Return diffusion :math:`\tilde{g}(t)` for NonSingular SDE."""
        flat_dim = self.flat_dim

        def g_tilde(t, y_flat, args):
            """Returns (flat_dim, flat_dim) diagonal diffusion matrix."""
            g = self.alpha * jnp.sqrt(1 - t)
            return g * jnp.eye(flat_dim)

        return g_tilde
