"""
Continuous flow matching loss functions.

This module implements loss functions for training continuous flow matching models,
computing the squared difference between predicted and target velocities.
"""
import jax.numpy as jnp
from flax import nnx
from typing import Callable, Tuple, Any
from jax import Array


class ContinuousFMLoss(nnx.Module):
    """
    ContinuousFMLoss is a class that computes the continuous flow matching loss.

    Parameters
    ----------
        path : MixtureDiscreteProbPath
            Probability path (x-prediction training).
        reduction : str, optional
            Specify the reduction to apply to the output ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction is applied to the output, ``'mean'``: the output is reduced by mean over sequence elements, ``'sum'``: the output is reduced by sum over sequence elements. Defaults to 'mean'.

    Example:
        .. code-block:: python

            from gensbi.flow_matching.loss import ContinuousFMLoss
            from gensbi.flow_matching.path import AffineProbPath
            from gensbi.flow_matching.path.scheduler import CondOTScheduler
            import jax, jax.numpy as jnp
            scheduler = CondOTScheduler()
            path = AffineProbPath(scheduler)
            loss_fn = ContinuousFMLoss(path)
            def vf(x, t, args=None):
                return x + t
            x_0 = jnp.zeros((8, 2))
            x_1 = jnp.ones((8, 2))
            t = jnp.linspace(0, 1, 8)
            batch = (x_0, x_1, t)
            loss = loss_fn(vf, batch)
            print(loss.shape)
            # ()

    """

    def __init__(self, path, reduction: str = "mean") -> None:
        """
        Initialize the continuous flow matching loss.
        
        Parameters
        ----------
            path: Probability path for x-prediction training.
            reduction: Reduction method for the loss. Options: 'none', 'mean', 'sum'. Defaults to 'mean'.
            
        Raises
        ------
            ValueError
                If reduction is not one of 'None', 'mean', or 'sum'.
        """
        self.path = path
        if reduction not in ["None", "mean", "sum"]:
            raise ValueError(f"{reduction} is not a valid value for reduction")

        if reduction == "mean":
            self.reduction = jnp.mean
        elif reduction == "sum":
            self.reduction = jnp.sum
        else:
            self.reduction = lambda x: x

    def __call__(
        self,
        vf: Callable,
        batch: Tuple[Array, Array, Array],
        args: Any = None,
        **kwargs,
    ) -> Array:
        """
        Evaluates the continuous flow matching loss.

        Parameters
        ----------
            vf : callable
                The vector field model to evaluate.
            batch : tuple
                A tuple containing the input data (x_0, x_1, t).
            args : optional
                Additional arguments for the function.
            condition_mask : optional
                A mask to apply to the input data.
            **kwargs: Additional keyword arguments for the function.

        Returns
        -------
            Array
                The computed loss.
        """

        path_sample = self.path.sample(*batch)

        x_t = path_sample.x_t

        model_output = vf(x_t, path_sample.t, args=args, **kwargs)

        loss = model_output - path_sample.dx_t

        return self.reduction(jnp.square(loss))
