from flax import nnx

from jax import Array

from einops import rearrange

from typing import Callable, Optional
from gensbi.recipes.pipeline import AbstractPipeline


class PosteriorWrapper:
    """
    Wrap a GenSBI pipeline into a distribution compatible with `sbi`.

    Parameters
    ----------
        pipeline: An instance of a Pipeline from GenSBI.
        rngs: A nnx.Rngs instance for random number generation.
        theta_shape: Optional shape of the parameters (theta) to be sampled.
        x_shape: Optional shape of the observations (x) to condition on.
        *args, **kwargs: Additional arguments to be passed to the pipeline during sampling.
    """

    def __init__(
        self,
        pipeline: AbstractPipeline,
        *args,
        rngs: nnx.Rngs,
        theta_shape=None,
        x_shape=None,
        **kwargs,
    ):

        self.pipeline = pipeline
        self.args = args
        self.kwargs = kwargs
        self.default_x = None
        self.rngs = rngs

        if theta_shape is not None:
            self.dim_theta = theta_shape[0]
            self.ch_theta = theta_shape[1]
        else:
            self.ch_theta = self.pipeline.ch_obs
            self.dim_theta = self.pipeline.dim_obs

        if x_shape is not None:
            self.dim_x = x_shape[0]
            self.ch_x = x_shape[1]
        else:
            if self.pipeline.ch_cond is None:
                self.ch_x = self.ch_theta
            else:
                self.ch_x = self.pipeline.ch_cond
            self.dim_x = self.pipeline.dim_cond

    def _ravel(self, x):
        return x.reshape(x.shape[0], -1)

    def _unravel_theta(self, x):
        return x.reshape(x.shape[0], self.dim_theta, self.ch_theta)

    def _unravel_xs(self, x):
        return x.reshape(x.shape[0], self.dim_x, self.ch_x)

    def _process_x(self, x):
        assert x.ndim in (2, 3), "x must be of shape (batch, dim) or (batch, dim, ch)"

        if x.ndim == 3:
            assert (
                x.shape[2] == self.ch_x
            ), f"Wrong number of channels, expected {self.ch_x}, got {x.shape[2]}"

        if x.ndim == 2:
            x = self._unravel_xs(x)

        return self._ravel(x)

    def set_default_x(self, x):

        self.default_x = self._process_x(x)

    def sample(
        self,
        sample_shape,
        x: Optional[Array] = None,
        **kwargs,  # does nothing, for compatibility
    ) -> Array:
        """
        Sample from the posterior distribution conditioned on x.

        Parameters
        ----------
            sample_shape : Tuple
                Shape of the samples to be drawn.
            x : Array
                Optional tensor of observations to condition on. If None, uses the default_x.
            
        Returns
        -------
            Array
                Samples from the posterior distribution of shape (sample_shape, dim_theta * ch_theta).
        """
        key = self.rngs.sample()
        if x is None:
            cond = self.default_x
        else:
            cond = x

        if cond.ndim == 2:
            cond = self._unravel_xs(cond)

        res = self.pipeline.sample(
            key, cond, sample_shape[0], *self.args, **self.kwargs
        )
        res = self._ravel(res)
        return res

    def sample_batched(
        self,
        sample_shape,
        x: Optional[Array] = None,
        chunk_size: Optional[int] = 50,
        show_progress_bars=True,
        **kwargs,  # does nothing, for compatibility
    ) -> Array:
        """
        Sample from the posterior distribution conditioned on x.

        Parameters
        ----------
            sample_shape : Tuple
                Shape of the samples to be drawn.
            x : Array
                Optional tensor of observations to condition on. If None, uses the default_x.
            chunk_size : int
                Size of the chunks to use for batched sampling.
            show_progress_bars : bool
                Whether to show progress bars during sampling.

        """
        if x is None:
            cond = self.default_x
        else:
            cond = x

        if cond.ndim == 2:
            cond = self._unravel_xs(cond)

        chunk_size = self.kwargs.pop("chunk_size", chunk_size)

        key = self.rngs.sample()
        res = self.pipeline.sample_batched(
            key,
            cond,
            sample_shape[0],
            chunk_size=chunk_size,
            show_progress_bars=show_progress_bars,
            **self.kwargs,
        )

        res = rearrange(res, "... f c -> ... (f c)")
        return res
