"""
Score matching reverse SDE solver.

Provides :class:`NewSMSDESolver`, an SDE solver for score matching
diffusion models.

The reverse SDE is:

.. math::
    dx = \\left[ f(x,t) - g(t)^2\\, s_\\theta(x, t) \\right] dt
         + g(t)\\, d\\bar{W}

where :math:`s_\\theta` is the learned score, :math:`f` is the forward
SDE drift, and :math:`g` is the forward diffusion coefficient.

The drift (:math:`\tilde{f}`) and diffusion (:math:`\tilde{g}`) are defined directly inside the solver
class (same pattern as ``ZeroEndsSolver``), using the raw score model
accessed via ``velocity_model.get_vector_field()`` and the SDE scheduler
for the forward process coefficients.

**Time direction convention:**
Score matching integrates **backwards** from ``t=T`` (noise) to ``t=eps``
(near-clean data).  This is the opposite of flow matching (``t=0→1``).
"""

from typing import Callable

import jax.numpy as jnp
from jax import Array

from gensbi.core.sde_solver import NewSDESolver
from gensbi.utils.model_wrapping import ModelWrapper


class NewSMSDESolver(NewSDESolver):
    r"""Score matching reverse SDE solver.

    The drift and diffusion are computed inline from the raw score model
    and the forward SDE scheduler, analogous to how ``ZeroEndsSolver``
    computes its SDE coefficients from the velocity field.

    Conditioning is handled entirely by the ``ModelWrapper`` layer
    (``ConditionalWrapper``, ``JointWrapper``).

    Parameters
    ----------
    velocity_model : ModelWrapper
        Wrapped **score** model.  ``get_vector_field()`` returns the
        raw score function.
    sde
        Forward SDE scheduler (``VPSmScheduler``, ``VESmScheduler``, etc.)
        providing ``drift(x, t)`` and ``diffusion(t)``.
    eps0 : float
        Minimum time value.
    """

    def __init__(
        self,
        velocity_model: ModelWrapper,
        sde,
        eps0: float = 1e-3,
    ):
        super().__init__(velocity_model, eps0=eps0)
        self.sde = sde

    def get_drift(self, **kwargs) -> Callable:
        r"""Return the reverse SDE drift.

        .. math::
            \tilde{f}(t, x) = f(x, t) - g(t)^2\, s_\theta(x, t)

        where :math:`f` and :math:`g` are the forward SDE coefficients
        and :math:`s_\theta` is the score model.
        """
        score_fn = self.velocity_model.get_vector_field(**kwargs)
        sde = self.sde

        def drift(t, x, args):
            score = score_fn(t, x, args)
            t_bc = jnp.broadcast_to(t, x.shape)
            g_sq = sde.diffusion(t_bc) ** 2
            forward_drift = sde.drift(x, t_bc)
            return forward_drift - g_sq * score

        return drift

    def get_diffusion(self) -> Callable:
        r"""Return the reverse SDE diffusion.

        .. math::
            \tilde{g}(t) = g(t)

        Returns a ``(flat_dim, flat_dim)`` diagonal matrix.
        """
        sde = self.sde

        def g_tilde(t, y_flat, args):
            flat_dim = y_flat.shape[0]
            t_bc = jnp.broadcast_to(t, (flat_dim,))
            g = sde.diffusion(t_bc)
            return jnp.diag(g)

        return g_tilde
