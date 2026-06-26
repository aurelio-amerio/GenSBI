import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.models import TarFlow, TarFlowParams


def test_params_channels_derived_and_validation():
    p = TarFlowParams(rngs=nnx.Rngs(0), dim=4, head_dim=16, num_heads=4)
    assert p.channels == 64                              # head_dim * num_heads
    with pytest.raises(ValueError):
        TarFlowParams(rngs=nnx.Rngs(0), modeled="image")  # img_size/patch_size missing
    with pytest.raises(ValueError):
        TarFlowParams(rngs=nnx.Rngs(0), dim=4, cond="image_prefix")  # cond img args missing


def test_default_head_gives_channels_64():
    p = TarFlowParams(rngs=nnx.Rngs(0), dim=4)
    assert (p.head_dim, p.num_heads, p.channels) == (16, 4, 64)


def test_tarflow_log_prob_and_sample_shapes():
    flow = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), dim=4, cond_dim=2,
                                 head_dim=8, num_heads=2, num_blocks=2,
                                 layers_per_block=1))
    x = jax.random.normal(jax.random.key(1), (3, 4))
    c = jax.random.normal(jax.random.key(2), (3, 2))
    lp = flow.log_prob(x, c)
    assert lp.shape == (3,) and bool(jnp.all(jnp.isfinite(lp)))
    s = flow.sample(jax.random.key(3), cond=c)
    assert s.shape == (3, 4)
