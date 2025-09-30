import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import pytest

from gensbi.flow_matching.path.scheduler import CondOTScheduler, VPScheduler
from gensbi.flow_matching.path.scheduler import ScheduleTransformedModel

def test_schedule_transform():

    def vf_model(x,t):
        return x
    
    original_scheduler = CondOTScheduler()
    new_scheduler = VPScheduler()
    
    transform = ScheduleTransformedModel(
        vf_model,
        original_scheduler,
        new_scheduler,
    )
    x = jnp.array([[1.,2.],[3.,4.]])
    t = jnp.array([0.,0.5])

    res = transform(x, t)

    assert res.shape == x.shape, f"Expected shape {x.shape}, but got {res.shape}"