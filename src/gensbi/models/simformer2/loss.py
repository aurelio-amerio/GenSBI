import jax
import jax.numpy as jnp
from flax import nnx
from typing import Callable, Tuple, Optional
from jax.numpy import ndarray as Array

from gensbi.models.simformer import SimformerCFMLoss, SimformerDiffLoss

# the losses are identical to Simformer ones

class Simformer2CFMLoss(SimformerCFMLoss):
    """
    Simformer2CFMLoss is a class that computes the continuous flow matching loss for the Simformer model.

    Args:
        path: Probability path for training.
        reduction (str): Reduction method ('none', 'mean', 'sum').
    """

    def __init__(self, path, reduction: str = "mean"):
        super().__init__(path, reduction)


class Simformer2DiffLoss(SimformerDiffLoss):
    """
    Simformer2DiffLoss is a class that computes the diffusion score matching loss for the Simformer model.

    Args:
        path: Probability path for training.
    """

    def __init__(self, path):
        super().__init__(path)
