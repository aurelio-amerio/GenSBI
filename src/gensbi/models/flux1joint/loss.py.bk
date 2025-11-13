import jax
import jax.numpy as jnp
from flax import nnx
from typing import Callable, Tuple, Optional
from jax.numpy import ndarray as Array

from gensbi.models.simformer import SimformerCFMLoss, SimformerDiffLoss

# the losses are identical to Simformer ones

class Flux1JointCFMLoss(SimformerCFMLoss):
    """
    Flux1JointCFMLoss computes the continuous flow matching loss for the Flux1Joint model.
    """
    def __init__(self, path, reduction: str = "mean"):
        super().__init__(path, reduction)

class Flux1JointDiffLoss(SimformerDiffLoss):
    """
    Flux1JointDiffLoss computes the diffusion score matching loss for the Flux1Joint model.
    """
    def __init__(self, path):
        super().__init__(path)
