
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import jax
import jax.numpy as jnp
import pytest
from gensbi.diagnostics.metrics.c2st import c2st, check_c2st

def test_c2st_basics():
    # Identical distributions -> accuracy ~ 0.5
    key = jax.random.PRNGKey(0)
    X = jax.random.normal(key, (100, 2))
    Y = jax.random.normal(key, (100, 2)) # Same distribution (same key for this test logic to be strictly same samples? No, c2st checks indistinguishability. 
    # If I use same key for sample generation, they are identical samples.
    # If I use different keys, they are samples from same distribution.
    # C2ST on limited samples from same distribution should be close to 0.5.
    
    # Let's generate samples from same distribution but different realizations
    key1, key2 = jax.random.split(key)
    X = jax.random.normal(key1, (200, 2))
    Y = jax.random.normal(key2, (200, 2))
    
    score = c2st(X, Y, seed=42)
    # Allow some variance, but should be around 0.5
    assert 0.4 <= score <= 0.6
    
    # Distinct distributions -> accuracy > 0.5
    Y_shifted = X + 5.0
    score_shifted = c2st(X, Y_shifted, seed=42)
    assert score_shifted > 0.8

def test_c2st_options():
    key = jax.random.PRNGKey(1)
    X = jax.random.normal(key, (50, 2))
    Y = jax.random.normal(key, (50, 2)) + 0.5
    
    # MLP classifier
    score_mlp = c2st(X, Y, classifier="mlp", seed=1)
    assert 0.0 <= score_mlp <= 1.0
    
    # Z-score false
    score_no_z = c2st(X, Y, z_score=False, seed=1)
    assert 0.0 <= score_no_z <= 1.0
    
    # Noise scale
    score_noise = c2st(X, Y, noise_scale=0.1, seed=1)
    assert 0.0 <= score_noise <= 1.0
    
    # Classifier kwargs
    score_kwargs = c2st(X, Y, classifier="rf", classifier_kwargs={"n_estimators": 10}, seed=1)
    assert 0.0 <= score_kwargs <= 1.0

def test_check_c2st():
    key = jax.random.PRNGKey(2)
    X = jax.random.normal(key, (100, 2))
    Y = jax.random.normal(key, (100, 2)) # same dist
    
    # Should pass for similar distributions
    # Re-generating Y with different key to be proper samples from same dist
    key1, key2 = jax.random.split(key)
    X = jax.random.normal(key1, (100, 2))
    Y = jax.random.normal(key2, (100, 2))
    
    check_c2st(X, Y, alg="test_alg", tol=0.2)
    
    # Should fail for distinct
    Y_dist = X + 100
    with pytest.raises(AssertionError, match="too far from the desired near-chance"):
        check_c2st(X, Y_dist, alg="test_alg", tol=0.1)
