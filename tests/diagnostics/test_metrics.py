#%%
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax

import pytest

from gensbi.diagnostics.metrics import l1, l2, c2st
#%%
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (100, 5))
y = jax.random.normal(key, (100, 5)) + 1.0  # Shifted distribution

dist = l1(x, y)

#%%
dist.shape
#%%
def test_l1():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (100, 5))
    y = jax.random.normal(key, (100, 5)) + 1.0  # Shifted distribution

    dist = l1(x, y)
    assert dist.shape == (100,) 
    assert (dist > 0).all(), "L1 distance should be positive for different distributions"

def test_l2():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (100, 5))
    y = jax.random.normal(key, (100, 5)) + 1.0  # Shifted distribution

    dist = l2(x, y)
    assert dist.shape == (100,)
    assert (dist > 0).all(), "L2 distance should be positive for different distributions"
    
def test_c2st():
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (100, 5))
    y = jax.random.normal(key, (100, 5)) + 1.0  # Shifted distribution

    score = c2st(x, y, classifier="rf")
    assert score > 0.5, "C2ST score should be greater than 0.5 for different distributions"

    score = c2st(x, y, classifier="mlp")
    assert score > 0.5, "C2ST score should be greater than 0.5 for different distributions"



# %%
