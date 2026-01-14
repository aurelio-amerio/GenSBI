import jax
from jax import Array
from jax import numpy as jnp

def l1(x: Array, y: Array, axis: int = -1) -> Array:
    """
    Calculates the L1 (Manhattan) distance between two tensors.

    Parameters
    ----------
        x : Array
            The first tensor.
        y : Array
            The second tensor.
        axis : int, optional
            The axis along which to calculate the L2 distance. Defaults to -1.

    Returns
    -------
        Array
            A tensor containing the L1 distance between x and y along the specified axis.
    """
    return jnp.mean(jnp.abs(x - y), axis=axis)


def l2(x: Array, y: Array, axis: int = -1) -> Array:
    """
    Calculates the L2 (Euclidean) distance between two tensors.

    Parameters
    ----------
        x : Array
            The first tensor.
        y : Array
            The second tensor.
        axis : int, optional
            The axis along which to calculate the L2 distance. Defaults to -1.

    Returns
    -------
        Array
            A tensor containing the L2 distance between x and y along the specified axis.
    """
    return jnp.sqrt(jnp.sum((x - y) ** 2, axis=axis))
