import jax
import jax.numpy as jnp
from flax import nnx
from gensbi.normalizing_flows.transformer_flow.blocks import AttentionBlock


def test_output_shape():
    blk = AttentionBlock(channels=8, head_dim=4, expansion=2, rngs=nnx.Rngs(0))
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, 8))
    assert blk(x).shape == (2, 5, 8)


def test_attention_is_causal():
    """output[i] must not depend on input[j] for j > i (causal mask)."""
    blk = AttentionBlock(channels=8, head_dim=4, expansion=2, rngs=nnx.Rngs(0))
    T, C = 4, 8
    x0 = jax.random.normal(jax.random.PRNGKey(2), (1, T, C))

    def f(x):
        return blk(x[None])[0]            # (T, C)

    J = jax.jacrev(f)(x0[0])              # (T, C, T, C)
    for i in range(T):
        for j in range(i + 1, T):
            assert jnp.allclose(J[i, :, j, :], 0.0, atol=1e-6), (i, j)
