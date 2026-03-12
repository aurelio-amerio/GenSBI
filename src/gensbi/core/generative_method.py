"""
Abstract base class for generative method strategies.

A ``GenerativeMethod`` encapsulates the mathematical framework used for
training and sampling (e.g., flow matching, EDM diffusion, or score matching).
It is composed into mode-specific pipelines (Conditional, Joint, Unconditional)
via the strategy pattern.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Tuple

from jax import Array


class GenerativeMethod(ABC):
    """Strategy that encapsulates a generative framework.

    Concrete implementations handle:

    - **Path construction** — the probability path defining the forward process
    - **Loss creation** — the training objective
    - **Batch preparation** — sampling noise / time for training batches
    - **Solver construction** — building the sampler for inference
    - **Initial sample generation** — drawing from the prior
    """

    @abstractmethod
    def build_path(self, config: dict):
        """Create the probability path from training config.

        Parameters
        ----------
        config : dict
            Training configuration dictionary (may contain scheduler
            hyperparameters such as ``sigma_min``, ``sigma_max``).

        Returns
        -------
        path
            A probability path object (e.g. ``AffineProbPath``, ``EDMPath``,
            ``SMPath``).
        """
        ...  # pragma: no cover

    @abstractmethod
    def build_loss(self, path, weights=None):
        """Create the loss callable for this method.

        Parameters
        ----------
        path
            The probability path returned by :meth:`build_path`.
        weights : Array, optional
            Per-dimension loss weights (e.g. for rebalancing joint
            parameter/observation dimensions).

        Returns
        -------
        loss
            A loss object whose ``__call__`` computes the training loss.
        """
        ...  # pragma: no cover

    @abstractmethod
    def prepare_batch(self, key, x_1, path):
        """Prepare a training batch from clean data.

        All methods return a uniform ``(x_0, x_1, t_or_sigma)`` tuple
        where ``x_0`` is noise sampled from the prior.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        x_1 : Array
            Clean data batch of shape ``(batch_size, dim, ch)``.
        path
            The probability path.

        Returns
        -------
        batch : tuple
            ``(x_0, x_1, t_or_sigma)`` prepared training batch.
        """
        ...  # pragma: no cover

    @abstractmethod
    def get_default_solver(self) -> Tuple[type, dict]:
        """Return the default ``(SolverClass, kwargs)`` for this method.

        Returns
        -------
        solver : tuple
            A ``(SolverClass, default_kwargs)`` pair.
        """
        ...  # pragma: no cover

    @abstractmethod
    def build_solver(self, model_wrapped, path, solver=None):
        """Instantiate a solver.

        Parameters
        ----------
        model_wrapped
            The wrapped model (velocity field or score model).
        path
            The probability path.
        solver : tuple of (type, dict), optional
            ``(SolverClass, kwargs)`` override. If ``None``, uses
            :meth:`get_default_solver`.

        Returns
        -------
        solver_instance
            An instantiated solver.
        """
        ...  # pragma: no cover

    @abstractmethod
    def sample_init(self, key, shape, path):
        """Sample initial noise for the generative process.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        shape : tuple
            Shape of the initial sample, e.g. ``(nsamples, dim, ch)``.
        path
            The probability path (needed for ``sample_prior`` in diffusion/SM).

        Returns
        -------
        x_init : Array
            Initial noise sample.
        """
        ...  # pragma: no cover

    @abstractmethod
    def build_sampler_fn(self, model_wrapped, path, model_extras, **kwargs):
        """Build a sampler closure for inference.

        The pipeline should use :meth:`sample_init` to generate the
        initial noise ``x_init`` before calling the returned sampler.

        Parameters
        ----------
        model_wrapped
            The wrapped model.
        path
            The probability path.
        model_extras : dict
            Mode-specific extras (``cond``, ``obs_ids``, ``cond_ids``, etc.).
        **kwargs
            Method-specific sampler arguments (``step_size``, ``nsteps``,
            ``time_grid``, ``solver``, etc.).

        Returns
        -------
        sampler_fn : Callable
            A function ``(key, x_init) -> samples``.
        """
        ...  # pragma: no cover

    def build_log_prob_fn(self, model_wrapped, path, model_extras, **kwargs):
        """Build a log-probability closure for inference.

        Only supported by methods whose solver can evaluate the continuous
        change-of-variables formula (e.g., flow matching + ODE solver).

        Parameters
        ----------
        model_wrapped
            The wrapped model.
        path
            The probability path.
        model_extras : dict
            Mode-specific extras (``cond``, ``obs_ids``, ``cond_ids``, etc.).
        **kwargs
            Method-specific arguments (``step_size``, ``method``, etc.).

        Returns
        -------
        log_prob_fn : Callable
            A function ``(x_1, model_extras) -> log_prob``.

        Raises
        ------
        NotImplementedError
            If the method does not support log-probability computation.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support log-probability computation."
        )

    def get_extra_training_config(self) -> dict:
        """Return method-specific training config defaults.

        Override in subclasses to supply extra defaults (e.g. ``sigma_min``).

        Returns
        -------
        dict
            Extra configuration parameters.
        """
        return {}
