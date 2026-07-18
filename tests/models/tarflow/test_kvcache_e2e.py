"""End-to-end gate: TarFlow.sample (KV-cached) == reference full-recompute.

Covers the spec matrix {rope on/off} x {bias, vector, image conditioner}
x {vector, image modeled} (rope requires image-modeled, so rope-on vector
configs are excluded by construction).
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.models.tarflow.model import TarFlow, TarFlowParams


def _params(modeled, cond, use_rope):
    base = dict(rngs=nnx.Rngs(0), cond=cond, head_dim=8, num_heads=2,
                num_blocks=2, layers_per_block=1, zero_init=False,
                use_rope=use_rope)
    if modeled == "vector":
        base.update(dim=4)
    else:
        base.update(modeled="image", img_size=4, patch_size=2, img_channels=1)
    if cond in ("bias", "vector"):
        base.update(cond_dim=3)
    else:
        base.update(cond_img_size=4, cond_patch_size=2)
    return TarFlowParams(**base)


def _cond(cond, key):
    if cond == "bias":
        return jax.random.normal(key, (2, 3))
    if cond == "vector":
        return jax.random.normal(key, (2, 3, 1))
    return jax.random.normal(key, (2, 4, 4, 1))


def _reference_sample(model, key, cond):
    """TarFlow.sample with every block routed through _forward_reference."""
    nsamples = cond.shape[0]
    z = jax.random.normal(key, (nsamples, model.T, model.F))
    x = z
    for blk in reversed(model.blocks):
        x, _ = blk._forward_reference(x, cond)
    x = model.tokenizer.detokenize(x)
    return x * model.std[...] + model.mean[...]


CONFIGS = [
    ("vector", "bias", False), ("vector", "vector", False),
    ("vector", "image", False),
    ("image", "bias", False), ("image", "vector", False),
    ("image", "image", False),
    ("image", "bias", True), ("image", "vector", True),
    ("image", "image", True),
]


@pytest.mark.parametrize("modeled,cond,use_rope", CONFIGS)
def test_sample_cached_equals_reference(modeled, cond, use_rope):
    model = TarFlow(_params(modeled, cond, use_rope))
    key = jax.random.PRNGKey(42)
    c = _cond(cond, jax.random.PRNGKey(7))
    x_cached = model.sample(key, cond=c)
    x_ref = _reference_sample(model, key, c)
    assert x_cached.shape == x_ref.shape
    assert jnp.allclose(x_cached, x_ref, atol=1e-4), \
        jnp.abs(x_cached - x_ref).max()


@pytest.mark.parametrize("modeled,cond,use_rope", CONFIGS)
def test_log_prob_of_samples_is_finite(modeled, cond, use_rope):
    """Round-trip sanity: samples from the cached path score finite NLL."""
    model = TarFlow(_params(modeled, cond, use_rope))
    c = _cond(cond, jax.random.PRNGKey(8))
    x = model.sample(jax.random.PRNGKey(9), cond=c)
    lp = model.log_prob(x, c)
    assert lp.shape == (2,) and jnp.all(jnp.isfinite(lp))
