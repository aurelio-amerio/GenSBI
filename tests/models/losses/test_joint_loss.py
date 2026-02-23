import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from gensbi.models.losses import JointCFMLoss, JointEDMLoss

from gensbi.flow_matching.path.scheduler import CondOTScheduler
from gensbi.flow_matching.path import AffineProbPath

from gensbi.diffusion.path import EDMPath
from gensbi.diffusion.path.scheduler import EDMScheduler


def test_simformer_cfmloss_runs():
    path = AffineProbPath(scheduler=CondOTScheduler())
    loss = JointCFMLoss(path)

    def vf(obs, t, *args, **kwargs):
        return obs + t

    x0 = jnp.ones((2, 2))
    x1 = jnp.ones((2, 2))
    t = jnp.ones((2,))
    batch = (x0, x1, t)
    result = loss(vf, batch)
    assert result is not None


def test_simformer_diffloss_runs():
    scheduler = EDMScheduler()
    path = EDMPath(scheduler=scheduler)
    loss = JointEDMLoss(path)

    def vf(obs, t, *args, **kwargs):
        return obs + t

    x1 = jnp.ones((2, 2))
    t = jnp.ones((2,))
    batch = (x1, t)
    key = jax.random.PRNGKey(0)
    result = loss(key, vf, batch)
    assert result is not None


def test_simformer_smloss_runs():
    from gensbi.models.losses import JointSMLoss
    from gensbi.diffusion.path.sm_path import SMPath
    from gensbi.diffusion.path.scheduler.sm_sde import VPSmScheduler

    sde = VPSmScheduler()
    path = SMPath(sde)
    loss = JointSMLoss(path)

    def vf(obs, t, *args, **kwargs):
        return obs + t

    x1 = jnp.ones((2, 2))
    t = jnp.ones((2, 1))
    batch = (x1, t)
    key = jax.random.PRNGKey(0)
    result = loss(key, vf, batch)
    assert result is not None
