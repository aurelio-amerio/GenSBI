import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from gensbi.models.losses import ConditionalCFMLoss, ConditionalEDMLoss

from gensbi.flow_matching.path.scheduler import CondOTScheduler
from gensbi.flow_matching.path import AffineProbPath

from gensbi.diffusion.path import EDMPath
from gensbi.diffusion.path.scheduler import EDMScheduler


def test_flux_cfmloss_runs():
    path = AffineProbPath(scheduler=CondOTScheduler())
    loss = ConditionalCFMLoss(path)

    def vf(obs, obs_ids, cond, cond_ids, t, conditioned=True):
        return obs + 1

    x0 = jnp.ones((2, 2))
    x1 = jnp.ones((2, 2))
    t = jnp.ones((2,))
    cond = jnp.ones((2, 2))
    obs_ids = jnp.array([0, 1])
    cond_ids = jnp.array([2, 3])
    batch = (x0, x1, t)
    result = loss(vf, batch, cond, obs_ids, cond_ids)
    assert result is not None


def test_flux_diffloss_runs():

    scheduler = EDMScheduler()
    path = EDMPath(scheduler=scheduler)
    loss = ConditionalEDMLoss(path)

    def vf(obs, obs_ids, cond, cond_ids, t, conditioned=True):
        return obs + 1

    x1 = jnp.ones((2, 2))
    t = jnp.ones((2,))
    cond = jnp.ones((2, 2))
    obs_ids = jnp.array([0, 1])
    cond_ids = jnp.array([2, 3])
    batch = (x1, t)
    key = jax.random.PRNGKey(0)
    result = loss(key, vf, batch, cond, obs_ids, cond_ids)
    assert result is not None
