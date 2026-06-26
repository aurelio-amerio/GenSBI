import jax
import jax.numpy as jnp
from flax import nnx
from gensbi.models.tarflow.blocks import AttentionBlock


def test_output_shape():
    blk = AttentionBlock(channels=8, num_heads=2, expansion=2, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 8))
    assert blk(x).shape == (2, 5, 8)


def test_attention_is_causal():
    """output[i] must not depend on input[j] for j > i (causal mask)."""
    blk = AttentionBlock(channels=8, num_heads=2, expansion=2, rngs=nnx.Rngs(0))
    T, C = 4, 8
    x0 = jax.random.normal(jax.random.PRNGKey(2), (1, T, C))

    def f(x):
        return blk(x[None])[0]            # (T, C)

    J = jax.jacrev(f)(x0[0])              # (T, C, T, C)
    for i in range(T):
        for j in range(i + 1, T):
            assert jnp.allclose(J[i, :, j, :], 0.0, atol=1e-6), (i, j)


def test_explicit_tril_mask_matches_is_causal():
    blk = AttentionBlock(channels=8, num_heads=2, expansion=2, rngs=nnx.Rngs(0))
    T = 5
    x = jax.random.normal(jax.random.PRNGKey(3), (2, T, 8))
    tril = jnp.tril(jnp.ones((T, T), dtype=bool))
    assert jnp.allclose(blk(x), blk(x, tril), atol=1e-6)


def test_prefix_mask_blocks_prefix_from_seeing_modeled():
    """With a prefix-LM mask, prefix-row outputs must not depend on modeled inputs."""
    blk = AttentionBlock(channels=8, num_heads=2, expansion=2, rngs=nnx.Rngs(0))
    M, T = 2, 3
    S = M + T
    idx = jnp.arange(S)
    is_modeled_q = idx[:, None] >= M
    is_prefix_k = idx[None, :] < M
    causal = idx[None, :] <= idx[:, None]
    mask = jnp.where(is_modeled_q, is_prefix_k | causal, is_prefix_k)
    x0 = jax.random.normal(jax.random.PRNGKey(4), (S, 8))

    def f(x):
        return blk(x[None], mask)[0]            # (S, C)

    J = jax.jacrev(f)(x0)                        # (S, C, S, C)
    # prefix rows (i < M) must be invariant to modeled inputs (j >= M)
    for i in range(M):
        for j in range(M, S):
            assert jnp.allclose(J[i, :, j, :], 0.0, atol=1e-6), (i, j)
