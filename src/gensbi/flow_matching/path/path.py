from abc import ABC, abstractmethod
from jax import Array

import jax
import jax.numpy as jnp

from gensbi.flow_matching.path.path_sample import PathSample


class ProbPath(ABC):
    r"""
    Abstract class, representing a probability path.

    A probability path transforms the distribution :math:`p(X_0)` into :math:`p(X_1)` over :math:`t=0\rightarrow 1`.

    The ``ProbPath`` class is designed to support model training in the flow matching framework. It supports two key functionalities: (1) sampling the conditional probability path and (2) conversion between various training objectives.
    Here is a high-level example

    .. code-block:: python

        # Instantiate a probability path
        my_path = ProbPath(...)

        # Sets t to a random value in [0,1]
        key = jax.random.PRNGKey(0)
        t = jax.random.uniform(key)

        # Samples the conditional path X_t ~ p_t(X_t|X_0,X_1)
        path_sample = my_path.sample(x_0=x_0, x_1=x_1, t=t)
    """

    @abstractmethod
    def sample(self, x_0: Array, x_1: Array, t: Array) -> PathSample:
        r"""
        Sample from an abstract probability path.

        Given :math:`(X_0,X_1) \sim \pi(X_0,X_1)`.
        Returns :math:`X_0, X_1, X_t \sim p_t(X_t|X_0,X_1)`, and a conditional target :math:`Y`, all objects are under ``PathSample``.

        Parameters
        ----------
            x_0 : Array
                Source data point, shape (batch_size, ...).
            x_1 : Array
                Target data point, shape (batch_size, ...).
            t : Array
                Times in [0,1], shape (batch_size,).

        Returns
        -------
            PathSample
                A conditional sample.
        """
        ...  # pragma: no cover

    def assert_sample_shape(self, x_0: Array, x_1: Array, t: Array) -> None:
        """
        Checks that the shapes of x_0, x_1, and t are compatible for sampling.

        Parameters
        ----------
            x_0 : Array
                Source data point.
            x_1 : Array
                Target data point.
            t : Array
                Time vector.

        Raises
        ------
            AssertionError
                If the shapes are not compatible.
        """
        assert (
            t.ndim == 1
        ), f"The time vector t must have shape [batch_size]. Got {t.shape}."
        assert (
            t.shape[0] == x_0.shape[0] == x_1.shape[0]
        ), f"Time t dimension must match the batch size [{x_1.shape[0]}]. Got {t.shape}"
