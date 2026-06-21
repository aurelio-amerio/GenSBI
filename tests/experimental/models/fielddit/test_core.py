import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.recipes.utils import init_ids_2d, init_ids_1d
from gensbi.experimental.models.fielddit.core import MMDiTCore


def _make_core():
    # hidden = sum(axes_dim) * num_heads = 8 * 2 = 16; head_dim = 8
    return MMDiTCore(
        hidden_size=16, num_heads=2, mlp_ratio=4.0, depth=1, depth_single_blocks=1,
        axes_dim=[2, 2, 4], theta=10000, n_cond_tokens=3, qkv_bias=False,
        rngs=nnx.Rngs(0), param_dtype=jnp.float32,
    )


def test_mmdit_core_forward_shape_and_finite():
    core = _make_core()
    obs_ids, n_obs = init_ids_2d((8, 8), semantic_id=0, size=2)  # 16 tokens
    cond_ids, _ = init_ids_1d(3)

    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 16))
    vec = jnp.ones((2, 16))

    out = core(obs, cond, vec, obs_ids, cond_ids)
    # returns only the obs tokens, same shape as obs input
    assert out.shape == (2, 16, 16)
    assert jnp.all(jnp.isfinite(out))


def test_mmdit_core_batch1_ids_broadcast():
    """obs_ids/cond_ids have batch dim 1 but obs has batch 4."""
    core = _make_core()
    obs_ids, _ = init_ids_2d((8, 8), semantic_id=0, size=2)
    cond_ids, _ = init_ids_1d(3)
    obs = jax.random.normal(jax.random.PRNGKey(1), (4, 16, 16))
    cond = jax.random.normal(jax.random.PRNGKey(2), (4, 3, 16))
    vec = jnp.ones((4, 16))
    out = core(obs, cond, vec, obs_ids, cond_ids)
    assert out.shape == (4, 16, 16)


def test_mmdit_core_prebatched_obs_ids():
    """obs_ids already at batch B -> the repeat-guard branch is a no-op (real-usage path)."""
    core = _make_core()
    obs_ids, _ = init_ids_2d((8, 8), semantic_id=0, size=2)
    obs_ids = jnp.repeat(obs_ids, 2, axis=0)  # batch 2, matching obs
    cond_ids, _ = init_ids_1d(3)
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 16))
    vec = jnp.ones((2, 16))
    out = core(obs, cond, vec, obs_ids, cond_ids)
    assert out.shape == (2, 16, 16)
    assert jnp.all(jnp.isfinite(out))


def test_mmdit_core_bfloat16_default_runs():
    """Smoke test the bfloat16 default param_dtype (the Task-9 / production config)."""
    core = MMDiTCore(
        hidden_size=16, num_heads=2, mlp_ratio=4.0, depth=1, depth_single_blocks=1,
        axes_dim=[2, 2, 4], theta=10000, n_cond_tokens=3, qkv_bias=False,
        rngs=nnx.Rngs(0),  # param_dtype defaults to bfloat16
    )
    obs_ids, _ = init_ids_2d((8, 8), semantic_id=0, size=2)
    cond_ids, _ = init_ids_1d(3)
    obs = jax.random.normal(jax.random.PRNGKey(1), (2, 16, 16))
    cond = jax.random.normal(jax.random.PRNGKey(2), (2, 3, 16))
    vec = jnp.ones((2, 16))
    out = core(obs, cond, vec, obs_ids, cond_ids)
    assert out.shape == (2, 16, 16)
    assert jnp.all(jnp.isfinite(out))


def _randomized_core(seed=0):
    """MMDiTCore with all float params replaced by small random values, so the
    AdaLN-zero gates are open and attention/rope/cond paths actually execute."""
    core = MMDiTCore(
        hidden_size=16, num_heads=2, mlp_ratio=2.0, depth=1, depth_single_blocks=1,
        axes_dim=[2, 2, 4], theta=100, n_cond_tokens=3, qkv_bias=False,
        rngs=nnx.Rngs(seed), param_dtype=jnp.float32,
    )
    graphdef, state = nnx.split(core)
    counter = iter(range(100_000))

    def _rand(x):
        if jnp.issubdtype(x.dtype, jnp.floating):
            k = jax.random.fold_in(jax.random.PRNGKey(42), next(counter))
            return 0.05 * jax.random.normal(k, x.shape, x.dtype)
        return x

    state = jax.tree_util.tree_map(_rand, state)
    return nnx.merge(graphdef, state)


def test_core_with_open_gates_is_cond_sensitive_and_rope_active():
    core = _randomized_core()
    B, hid = 2, 16
    obs_ids, n_obs = init_ids_2d((8, 8), semantic_id=0, size=2)  # 16 tokens
    cond_ids, _ = init_ids_1d(3, semantic_id=None)
    obs_tokens = jax.random.normal(jax.random.PRNGKey(1), (B, n_obs, hid))
    vec = jax.random.normal(jax.random.PRNGKey(2), (B, hid))
    cond_a = jax.random.normal(jax.random.PRNGKey(3), (B, 3, hid))
    # cond_b must differ STRUCTURALLY, not by a uniform affine shift: the cond
    # tokens are LayerNorm'd (use_scale/use_bias=False) before the QKV
    # projection, so a uniform shift like `cond_a + 1.0` (and likewise a uniform
    # scale) lies in LayerNorm's null space and is annihilated before reaching
    # attention — it would make the live cond path look dead. An independent
    # draw changes the per-feature pattern and survives the norm.
    cond_b = jax.random.normal(jax.random.PRNGKey(99), (B, 3, hid))

    out_a = core(obs_tokens, cond_a, vec, obs_ids, cond_ids)
    out_b = core(obs_tokens, cond_b, vec, obs_ids, cond_ids)
    assert jnp.all(jnp.isfinite(out_a))
    # the cond value path must reach the obs stream
    assert not jnp.allclose(out_a, out_b)

    # rope must break obs-token permutation equivariance: permuting the input
    # tokens (with FIXED position ids) must NOT just permute the output
    perm = jax.random.permutation(jax.random.PRNGKey(4), n_obs)
    out_perm = core(obs_tokens[:, perm, :], cond_a, vec, obs_ids, cond_ids)
    assert not jnp.allclose(out_perm, out_a[:, perm, :], atol=1e-5)
