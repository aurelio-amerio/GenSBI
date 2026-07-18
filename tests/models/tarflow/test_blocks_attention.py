import jax
import jax.numpy as jnp
from flax import nnx
from gensbi.models.tarflow.blocks import AttentionBlock
from gensbi.models.tarflow.pe import VisionRotaryEmbedding, get_positions


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


def _freqs(T=4, head_dim=4):
    rope = VisionRotaryEmbedding(dim=head_dim // 2, pt_seq_len=2)
    pos = get_positions(h=2, w=T // 2, pt_seq_len=2)
    return rope(pos)                              # (T, head_dim)


def test_freqs_cis_none_is_default_behavior():
    blk = AttentionBlock(channels=8, num_heads=2, expansion=2, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(5), (2, 4, 8))
    assert jnp.allclose(blk(x), blk(x, None, None), atol=0)


def test_zero_freqs_is_identity_rotation():
    """All-zero angles must reproduce the unrotated output exactly."""
    blk = AttentionBlock(channels=8, num_heads=2, expansion=2, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(6), (2, 4, 8))
    zero = jnp.zeros((4, 4))                      # (T, head_dim)
    assert jnp.allclose(blk(x), blk(x, None, zero), atol=1e-6)


def test_nonzero_freqs_change_output():
    blk = AttentionBlock(channels=8, num_heads=2, expansion=2, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(7), (2, 4, 8))
    assert not jnp.allclose(blk(x), blk(x, None, _freqs()), atol=1e-4)


def test_attention_stays_causal_with_freqs():
    blk = AttentionBlock(channels=8, num_heads=2, expansion=2, rngs=nnx.Rngs(0))
    T = 4
    x0 = jax.random.normal(jax.random.PRNGKey(8), (1, T, 8))
    freqs = _freqs(T=T)

    def f(x):
        return blk(x[None], None, freqs)[0]

    J = jax.jacrev(f)(x0[0])
    for i in range(T):
        for j in range(i + 1, T):
            assert jnp.allclose(J[i, :, j, :], 0.0, atol=1e-6), (i, j)


def _decode_full_equivalence(freqs):
    """Token-by-token decode must match the parallel causal pass."""
    blk = AttentionBlock(channels=8, num_heads=2, expansion=2, rngs=nnx.Rngs(0))
    B, T, C = 2, 5, 8
    x = jax.random.normal(jax.random.PRNGKey(9), (B, T, C))
    full = blk(x, None, freqs)                        # (B, T, C)

    k_cache = jnp.zeros((B, T, blk.num_heads, blk.head_dim))
    v_cache = jnp.zeros_like(k_cache)
    outs = []
    for t in range(T):
        out, k_cache, v_cache = blk.decode(x[:, t:t + 1], k_cache, v_cache,
                                           t, freqs)
        outs.append(out)
    dec = jnp.concatenate(outs, axis=1)               # (B, T, C)
    assert jnp.allclose(dec, full, atol=1e-5), jnp.abs(dec - full).max()


def test_decode_matches_full_causal_pass_no_rope():
    _decode_full_equivalence(None)


def test_decode_matches_full_causal_pass_with_rope():
    rope = VisionRotaryEmbedding(dim=2, pt_seq_len=2)
    # 5 slots: 1 prefix-at-identity + 2x2 grid
    pos = jnp.concatenate([jnp.zeros((1, 2)),
                           get_positions(h=2, w=2, pt_seq_len=2)], axis=0)
    _decode_full_equivalence(rope(pos))


def test_decode_with_prefill_matches_prefix_masked_pass():
    """Prefill M prefix slots via return_kv, then decode the modeled tokens."""
    blk = AttentionBlock(channels=8, num_heads=2, expansion=2, rngs=nnx.Rngs(0))
    B, M, T, C = 2, 3, 4, 8
    S = M + T
    x = jax.random.normal(jax.random.PRNGKey(10), (B, S, C))

    # reference: one parallel pass under the training prefix-LM mask
    idx = jnp.arange(S)
    is_modeled_q = idx[:, None] >= M
    is_prefix_k = idx[None, :] < M
    causal = idx[None, :] <= idx[:, None]
    mask = jnp.where(is_modeled_q, is_prefix_k | causal, is_prefix_k)
    full = blk(x, mask)

    # cached: prefill prefix (bidirectional among itself), then decode
    k_cache = jnp.zeros((B, S, blk.num_heads, blk.head_dim))
    v_cache = jnp.zeros_like(k_cache)
    prefix_mask = jnp.ones((M, M), dtype=bool)
    _, k, v = blk(x[:, :M], prefix_mask, None, return_kv=True)
    k_cache = k_cache.at[:, :M].set(k)
    v_cache = v_cache.at[:, :M].set(v)
    outs = []
    for t in range(T):
        out, k_cache, v_cache = blk.decode(x[:, M + t:M + t + 1],
                                           k_cache, v_cache, M + t)
        outs.append(out)
    dec = jnp.concatenate(outs, axis=1)
    assert jnp.allclose(dec, full[:, M:], atol=1e-5)


def test_return_kv_shapes_and_unrotated():
    blk = AttentionBlock(channels=8, num_heads=2, expansion=2, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(11), (2, 4, 8))
    rope = VisionRotaryEmbedding(dim=2, pt_seq_len=2)
    freqs = rope(get_positions(h=2, w=2, pt_seq_len=2))
    _, k_rot, _ = blk(x, None, freqs, return_kv=True)
    _, k_plain, _ = blk(x, None, None, return_kv=True)
    assert k_rot.shape == (2, 4, 2, 4)
    # cache contract: returned k is UNROTATED regardless of freqs
    assert jnp.allclose(k_rot, k_plain, atol=0)
