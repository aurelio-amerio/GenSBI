import jax.numpy as jnp
from flax import nnx
from typing import Callable, Tuple, Optional
from jax.numpy import ndarray as Array

from gensbi.flow_matching.loss import ContinuousFMLoss


class SimformerCFMLoss(ContinuousFMLoss):
    def __init__(self, path, reduction: str = "mean"):
        """
        Initialize the Simformer Continuous Flow Matching Loss.

        Args:
            path: Probability path for training.
            reduction (str): Reduction method ('none', 'mean', 'sum').
        """
        super().__init__(path, reduction)

    def __call__(
        self, 
        vf: Callable, 
        batch: Tuple[Array, Array, Array], 
        args: Optional[dict] = None, 
        condition_mask: Optional[Array] = None, 
        **kwargs
    ) -> Array:
        """
        Evaluate the continuous flow matching loss.

        Args:
            vf (Callable): Vector field model.
            batch (Tuple[Array, Array, Array]): Input data (x_0, x_1, t).
            args (Optional[dict]): Additional arguments.
            condition_mask (Optional[Array]): Mask for conditioning.
            **kwargs: Additional keyword arguments.

        Returns:
            Array: Computed loss.
        """
        _, x_1, _ = batch
        path_sample = self.path.sample(*batch)

        if condition_mask is not None:
            kwargs["condition_mask"] = condition_mask

        x_t = path_sample.x_t

        if condition_mask is not None:
            condition_mask = condition_mask.reshape(x_t.shape)
            x_t = jnp.where(condition_mask, x_1, x_t)

        model_output = vf(x_t, path_sample.t, args=args, **kwargs)
        
        loss = model_output - path_sample.dx_t
        if condition_mask is not None:
            loss = jnp.where(condition_mask, 0.0, loss)

        return self.reduction(jnp.square(loss)) # type: ignore
