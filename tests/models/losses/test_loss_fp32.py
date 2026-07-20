import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

from gensbi.flow_matching.loss import FMLoss
from gensbi.flow_matching.path.affine import CondOTProbPath
from gensbi.diffusion.loss import EDMLoss, SMLoss
from gensbi.diffusion.path import EDMPath
from gensbi.diffusion.path.sm_path import SMPath
from gensbi.diffusion.path.scheduler import EDMScheduler
from gensbi.diffusion.path.scheduler.sm_sde import VPSmScheduler


class _Bf16Model:
    """Fake velocity/score model that emits bf16, violating the emit-fp32 contract."""

    def __call__(self, obs, t, **kwargs):
        return jnp.asarray(obs, jnp.bfloat16) * jnp.bfloat16(0.5)


def test_fmloss_returns_fp32_for_bf16_model():
    path = CondOTProbPath()
    loss = FMLoss(path)
    key = jax.random.PRNGKey(0)
    x0 = jax.random.normal(key, (4, 3))
    x1 = jax.random.normal(key, (4, 3))
    t = jnp.full((4,), 0.5)
    out = loss(_Bf16Model(), (x0, x1, t))
    assert out.dtype == jnp.float32
    assert jnp.isfinite(out)


def test_edmloss_returns_fp32_for_bf16_model():
    scheduler = EDMScheduler()
    path = EDMPath(scheduler=scheduler)
    loss = EDMLoss(path)
    key = jax.random.PRNGKey(0)
    x0 = jax.random.normal(key, (4, 3))
    x1 = jax.random.normal(key, (4, 3))
    sigma = jnp.ones((4, 1))
    out = loss(_Bf16Model(), (x0, x1, sigma))
    assert out.dtype == jnp.float32
    assert jnp.isfinite(out)


def test_smloss_returns_fp32_for_bf16_model():
    scheduler = VPSmScheduler()
    path = SMPath(scheduler)
    loss = SMLoss(path)
    key = jax.random.PRNGKey(0)
    x0 = jax.random.normal(key, (4, 3))
    x1 = jax.random.normal(key, (4, 3))
    t = jnp.ones((4, 1))
    out = loss(_Bf16Model(), (x0, x1, t))
    assert out.dtype == jnp.float32
    assert jnp.isfinite(out)


# --- stronger regression tests -------------------------------------------
#
# When the batch itself is fp32 (as in the tests above), JAX's automatic
# dtype-promotion rules already upcast a bf16 model output to fp32 the
# moment it's combined with an fp32 target/weights array, so those tests
# pass even without the explicit casts below. To actually exercise the
# defense-in-depth cast (and get a genuine RED before / GREEN after), the
# batch here is bf16 end-to-end, so nothing upstream forces promotion —
# only the explicit `jnp.asarray(..., jnp.float32)` calls do.


def test_fmloss_returns_fp32_for_all_bf16_batch():
    path = CondOTProbPath()
    loss = FMLoss(path)
    key = jax.random.PRNGKey(0)
    x0 = jax.random.normal(key, (4, 3), dtype=jnp.bfloat16)
    x1 = jax.random.normal(key, (4, 3), dtype=jnp.bfloat16)
    t = jnp.full((4,), 0.5, dtype=jnp.bfloat16)
    out = loss(_Bf16Model(), (x0, x1, t))
    assert out.dtype == jnp.float32
    assert jnp.isfinite(out)


def test_edmloss_returns_fp32_for_all_bf16_batch():
    scheduler = EDMScheduler()
    path = EDMPath(scheduler=scheduler)
    loss = EDMLoss(path)
    key = jax.random.PRNGKey(0)
    x0 = jax.random.normal(key, (4, 3), dtype=jnp.bfloat16)
    x1 = jax.random.normal(key, (4, 3), dtype=jnp.bfloat16)
    sigma = jnp.ones((4, 1), dtype=jnp.bfloat16)
    out = loss(_Bf16Model(), (x0, x1, sigma))
    assert out.dtype == jnp.float32
    assert jnp.isfinite(out)


def test_smloss_returns_fp32_for_all_bf16_batch():
    scheduler = VPSmScheduler()
    path = SMPath(scheduler)
    loss = SMLoss(path)
    key = jax.random.PRNGKey(0)
    x0 = jax.random.normal(key, (4, 3), dtype=jnp.bfloat16)
    x1 = jax.random.normal(key, (4, 3), dtype=jnp.bfloat16)
    t = jnp.ones((4, 1), dtype=jnp.bfloat16)
    out = loss(_Bf16Model(), (x0, x1, t))
    assert out.dtype == jnp.float32
    assert jnp.isfinite(out)
