import jax
import jax.numpy as jnp
import pytest
from gensbi.diffusion.path.path import ProbPath

class DummyScheduler:
    def __init__(self):
        self.name = "EDM"
    def sample_prior(self, key, shape):
        return jnp.zeros(shape)

def test_probpath_sample_prior(mocker):
    mocker.patch.object(ProbPath, "__abstractmethods__", set())
    scheduler = DummyScheduler()
    path = ProbPath(scheduler=scheduler)
    key = jax.random.PRNGKey(0)
    shape = (3, 2)
    prior = path.sample_prior(key, shape)
    assert prior.shape == shape
    assert jnp.all(prior == 0)

def test_probpath_name(mocker):
    mocker.patch.object(ProbPath, "__abstractmethods__", set())
    scheduler = DummyScheduler()
    path = ProbPath(scheduler=scheduler)
    assert path.name == "EDM"
