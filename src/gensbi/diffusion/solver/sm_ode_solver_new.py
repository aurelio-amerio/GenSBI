"""
Score matching probability flow ODE solver.

Provides :class:`NewSMODESolver`, a thin subclass of
:class:`~gensbi.core.ode_solver.NewODESolver` used for dispatch in
the score matching pipeline.

The probability flow ODE is:

.. math::
    dx = \\left[ f(x,t) - \\tfrac{1}{2}\\, g(t)^2\\, s_\\theta(x, t) \\right] dt

The pipeline wraps the score model with
:class:`~gensbi.utils.model_wrapping.ScoreToODEDrift` before constructing
this solver, so the drift returned by ``get_drift()`` is already the
PF-ODE drift.

All integration and log-probability logic is inherited from
``NewODESolver``.
"""

from typing import Callable

from gensbi.core.ode_solver import NewODESolver


class NewSMODESolver(NewODESolver):
    r"""Score matching probability flow ODE solver.

    Uses the probability flow ODE formulation to sample deterministically
    from a score matching model.  The velocity model passed to this solver
    should already be wrapped with ``ScoreToODEDrift`` + ``ModelWrapper``
    by the pipeline.

    All integration and log-probability logic is inherited from
    :class:`~gensbi.core.ode_solver.NewODESolver`.

    See Also
    --------
    gensbi.utils.model_wrapping.ScoreToODEDrift
    gensbi.core.score_matching.ScoreMatchingMethod.build_solver
    """

    def get_drift(self, **kwargs) -> Callable:
        """Return the probability flow ODE drift (from ScoreToODEDrift adapter)."""
        return self.velocity_model.get_vector_field(**kwargs)
