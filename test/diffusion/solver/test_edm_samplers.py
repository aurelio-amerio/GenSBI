import os
os.environ['JAX_PLATFORMS'] = "cpu"

import pytest
from gensbi.diffusion.solver.edm_samplers import EDMHeunSampler, EDMEulerSampler


def test_edm_heun_sampler_initialization():
    sampler = EDMHeunSampler()
    assert isinstance(sampler, EDMHeunSampler)


def test_edm_euler_sampler_initialization():
    sampler = EDMEulerSampler()
    assert isinstance(sampler, EDMEulerSampler)
