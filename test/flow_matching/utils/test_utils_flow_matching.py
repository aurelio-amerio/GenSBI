import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import pytest
from gensbi.flow_matching.utils.utils import unsqueeze_to_match

def test_unsqueeze_to_match_suffix():
    source = jnp.ones(3)
    target = jnp.ones((3, 4, 5))
    result = unsqueeze_to_match(source, target)
    assert result.shape == (3, 1, 1)

def test_unsqueeze_to_match_prefix():
    source = jnp.ones(3)
    target = jnp.ones((4, 5, 3))
    result = unsqueeze_to_match(source, target, how="prefix")
    assert result.shape == (1, 1, 3)
