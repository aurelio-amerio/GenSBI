import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
import pytest
from gensbi.flow_matching.path.scheduler import (
    CondOTScheduler,
    ConvexScheduler,
    CosineScheduler,
    LinearVPScheduler,
    PolynomialConvexScheduler,
    SchedulerOutput,
    VPScheduler,
)


class MockScheduler(ConvexScheduler):
    def __call__(self, t: jnp.ndarray) -> SchedulerOutput:
        alpha_t = t
        sigma_t = 1 - t
        d_alpha_t = jnp.ones_like(t)
        d_sigma_t = -jnp.ones_like(t)
        return SchedulerOutput(alpha_t, sigma_t, d_alpha_t, d_sigma_t)

    def snr_inverse(self, snr: jnp.ndarray) -> jnp.ndarray:
        return snr  # Mock implementation

    def kappa_inverse(self, kappa: jnp.ndarray) -> jnp.ndarray:
        return kappa  # Mock implementation


@pytest.mark.parametrize(
    "scheduler_cls",
    [
        CondOTScheduler,
        CosineScheduler,
        LinearVPScheduler,
        MockScheduler,
        PolynomialConvexScheduler,
        VPScheduler,
    ],
)
def test_scheduler_output_shapes(scheduler_cls):
    t = jnp.array([0.1, 0.5, 0.9])
    if scheduler_cls is PolynomialConvexScheduler:
        scheduler = scheduler_cls(n=2)
    else:
        scheduler = scheduler_cls()
    output = scheduler(t)
    expected_shape = t.shape
    assert output.alpha_t.shape == expected_shape
    assert output.sigma_t.shape == expected_shape
    assert output.d_alpha_t.shape == expected_shape
    assert output.d_sigma_t.shape == expected_shape


@pytest.mark.parametrize(
    "scheduler_cls",
    [
        CondOTScheduler,
        MockScheduler,
        PolynomialConvexScheduler,
    ],
)
def test_kappa_inverse(scheduler_cls):
    t = jnp.array([0.1, 0.5, 0.9])
    if scheduler_cls is PolynomialConvexScheduler:
        scheduler = scheduler_cls(n=2)
    else:
        scheduler = scheduler_cls()
    output = scheduler(t)
    t_recovered = scheduler.kappa_inverse(output.alpha_t)
    assert jnp.allclose(t, t_recovered, atol=1e-5)


@pytest.mark.parametrize(
    "scheduler_cls",
    [
        CondOTScheduler,
        CosineScheduler,
        LinearVPScheduler,
        MockScheduler,
        PolynomialConvexScheduler,
        VPScheduler,
    ],
)
def test_snr_inverse(scheduler_cls):
    snr = jnp.array([0.1, 0.5, 0.9])
    # if the scheduler is PolynomialConvexScheduler, we need to pass n to it
    if scheduler_cls is PolynomialConvexScheduler:
        scheduler = scheduler_cls(n=2)
    else:
        scheduler = scheduler_cls()
    snr_inv = scheduler.snr_inverse(snr)
    assert snr_inv.shape == snr.shape
