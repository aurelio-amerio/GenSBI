import jax
import jax.numpy as jnp
import pytest
from gensbi.models.flux1 import layers

# def test_layers_functions():
#     # For each public function, call with dummy args to ensure coverage
#     for name in dir(layers):
#         if not name.startswith('_') and callable(getattr(layers, name)):
#             fn = getattr(layers, name)
#             try:
#                 fn(jnp.ones((2, 2)))
#             except Exception:
#                 pass  # Some functions may require more specific arguments
