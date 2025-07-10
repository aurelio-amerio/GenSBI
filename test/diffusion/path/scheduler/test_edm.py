import os
os.environ['JAX_PLATFORMS'] = "cpu"

import pytest
from gensbi.diffusion.path.scheduler.edm import EDMScheduler


def test_edm_scheduler_initialization():
    scheduler = EDMScheduler()
    assert isinstance(scheduler, EDMScheduler)


def test_edm_scheduler_sigma_range():
    scheduler = EDMScheduler()
    sigma_min, sigma_max = scheduler.sigma_min, scheduler.sigma_max
    assert sigma_min < sigma_max
