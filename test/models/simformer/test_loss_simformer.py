import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from gensbi.models.simformer.loss import SimformerCFMLoss

from gensbi.flow_matching.path.scheduler import CondOTScheduler
from gensbi.flow_matching.path import AffineProbPath

def test_simformer_cfmloss_runs():
    path = AffineProbPath(scheduler=CondOTScheduler())
    loss = SimformerCFMLoss(path)
    def vf(x, t, args=None, **kwargs):
        return x + t
    x0 = jnp.ones((2, 2))
    x1 = jnp.ones((2, 2))
    t = jnp.ones((2,))
    batch = (x0, x1, t)
    result = loss(vf, batch)
    assert result is not None
