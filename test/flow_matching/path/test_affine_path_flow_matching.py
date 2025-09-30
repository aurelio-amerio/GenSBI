import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from gensbi.flow_matching.path.affine import AffineProbPath, CondOTProbPath
from gensbi.flow_matching.path.scheduler import (
    CondOTScheduler,
    CosineScheduler,
    LinearVPScheduler,
    PolynomialConvexScheduler,
    VPScheduler,
)

@pytest.fixture
def affine_prob_path():
    scheduler = CondOTScheduler()
    return AffineProbPath(scheduler)

@pytest.mark.parametrize(
    "scheduler_cls",
    [
        CondOTScheduler,
        CosineScheduler,
        LinearVPScheduler,
        PolynomialConvexScheduler,
        VPScheduler,
    ],
)
def test_affine_prob_path_sample(scheduler_cls):
    if scheduler_cls is PolynomialConvexScheduler:
        scheduler = scheduler_cls(n=2)
    else:
        scheduler = scheduler_cls()
    affine_prob_path = AffineProbPath(scheduler)
    batch_size, dim = 10, 5
    x_0 = jnp.ones((batch_size, dim))
    x_1 = jnp.ones((batch_size, dim)) * 2
    t = jnp.ones((batch_size,)) * 0.5
    sample = affine_prob_path.sample(x_0, x_1, t)
    # Check all returned shapes
    assert sample.x_t.shape == (batch_size, dim)
    assert sample.dx_t.shape == (batch_size, dim)
    assert sample.x_0.shape == (batch_size, dim)
    assert sample.x_1.shape == (batch_size, dim)
    assert sample.t.shape == (batch_size,)
    # Check values
    assert jnp.all(sample.t == t)
    assert jnp.all(sample.x_0 == x_0)
    assert jnp.all(sample.x_1 == x_1)

@pytest.mark.parametrize(
    "scheduler_cls",
    [
        CondOTScheduler,
        CosineScheduler,
        LinearVPScheduler,
        PolynomialConvexScheduler,
        VPScheduler,
    ],
)
def test_assert_sample_shape(scheduler_cls):
    if scheduler_cls is PolynomialConvexScheduler:
        scheduler = scheduler_cls(n=2)
    else:
        scheduler = scheduler_cls()
    affine_prob_path = AffineProbPath(scheduler)
    batch_size, dim = 10, 5
    x_0 = jnp.ones((batch_size, dim))
    x_1 = jnp.ones((batch_size, dim))
    t = jnp.ones((batch_size,))
    # Should not raise
    affine_prob_path.assert_sample_shape(x_0, x_1, t)



# test affine class functions

def test_target_to_velocity():
    path =  AffineProbPath(CondOTScheduler())
    x_1 = jnp.array([[2.0, 2.0], [3.0, 3.0]])
    x_t = jnp.array([[1.5, 1.5], [2.5, 2.5]])
    t = jnp.array([0.5, 0.5])
    velocity = path.target_to_velocity(x_1, x_t, t)

    assert velocity.shape == x_t.shape

def test_epsilon_to_velocity():
    path =  AffineProbPath(CondOTScheduler())
    x_t = jnp.array([[1.5, 1.5], [2.5, 2.5]])
    epsilon = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    t = jnp.array([0.5, 0.5])
    velocity = path.epsilon_to_velocity(epsilon, x_t, t)

    assert velocity.shape == epsilon.shape

def test_velocity_to_target():
    path =  AffineProbPath(CondOTScheduler())
    x_t = jnp.array([[1.5, 1.5], [2.5, 2.5]])
    velocity = jnp.array([[0.1, 0.1], [0.1, 0.1]])
    t = jnp.array([0.5, 0.5])
    x_1 = path.velocity_to_target(velocity,  x_t, t)

    assert x_1.shape == x_t.shape

def test_epsilon_to_target():
    path =  AffineProbPath(CondOTScheduler())
    epsilon = jnp.array([[0.5, 0.5], [0.5, 0.5]])
    x_t = jnp.array([[1.5, 1.5], [2.5, 2.5]])
    t = jnp.array([0.5, 0.5])
    x_1 = path.epsilon_to_target(epsilon, x_t, t)

    assert x_1.shape == x_t.shape

def test_velocity_to_epsilon():
    path =  AffineProbPath(CondOTScheduler())
    velocity = jnp.array([[0.1, 0.1], [0.1, 0.1]])
    x_t = jnp.array([[1.5, 1.5], [2.5, 2.5]])
    t = jnp.array([0.5, 0.5])
    epsilon = path.velocity_to_epsilon(velocity, x_t, t)

    assert epsilon.shape == x_t.shape

def test_target_to_epsilon():   
    path =  AffineProbPath(CondOTScheduler())
    x_1 = jnp.array([[2.0, 2.0], [3.0, 3.0]])
    x_t = jnp.array([[1.5, 1.5], [2.5, 2.5]])
    t = jnp.array([0.5, 0.5])
    epsilon = path.target_to_epsilon(x_1, x_t, t)

    assert epsilon.shape == x_t.shape   


def test_cond_ot_instantiation():
    path = CondOTProbPath()
    assert isinstance(path, AffineProbPath)