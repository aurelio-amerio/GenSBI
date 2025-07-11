import jax
import jax.numpy as jnp
from jax import Array


def get_nearest_times(time_grid: Array, t_discretization: Array) -> Array:
    """Find the nearest times in t_discretization for each time in time_grid.

    Args:
        time_grid (Array): Query times to find nearest neighbors for, shape (N,)
        t_discretization (Array): Reference time points to match against, shape (M,)

    Returns:
        Array: Nearest times from t_discretization for each point in time_grid, shape (N,)
    """
    # Expand dimensions for broadcasting
    time_grid_expanded = jnp.expand_dims(time_grid, axis=1)  # (N, 1)
    t_disc_expanded = jnp.expand_dims(t_discretization, axis=0)  # (1, M)
    
    # Compute pairwise distances
    distances = jnp.abs(time_grid_expanded - t_disc_expanded)
    
    # Find indices of minimum distances
    nearest_indices = jnp.argmin(distances, axis=1)
    
    # Get the corresponding times
    return t_discretization[nearest_indices]