import jax.numpy as jnp
from flax import nnx

from gensbi.models.embedding import FeatureEmbedder
from gensbi.models.embedding.embedding import GaussianFourierEmbedding, MLPEmbedder
from tests.precision_utils import assert_tree_dtype


def test_mlpembedder_bf16_compute_fp32_params():
    m = MLPEmbedder(4, 8, rngs=nnx.Rngs(0), dtype=jnp.bfloat16,
                    param_dtype=jnp.float32)
    assert_tree_dtype(nnx.to_pure_dict(nnx.state(m, nnx.Param)), jnp.float32)
    out = m(jnp.ones((2, 3, 4), jnp.float32))
    assert out.dtype == jnp.bfloat16


def test_featureembedder_absolute_bf16():
    m = FeatureEmbedder(num_embeddings=5, hidden_size=8, kind="absolute",
                        dtype=jnp.bfloat16, param_dtype=jnp.float32,
                        rngs=nnx.Rngs(0))
    assert_tree_dtype(nnx.to_pure_dict(nnx.state(m, nnx.Param)), jnp.float32)
    out = m(jnp.zeros((2, 3, 1), jnp.int32))
    assert out.dtype == jnp.bfloat16


def test_default_dtype_is_neutral_fp32():
    m = MLPEmbedder(4, 8, rngs=nnx.Rngs(0))
    out = m(jnp.ones((2, 3, 4), jnp.float32))
    assert out.dtype == jnp.float32


def test_gaussianfourierembedding_bf16_computes_in_fp32():
    # Same seed => identical B across both instances, so any numerical
    # difference between the two outputs comes only from where the
    # cos/sin math happened (fp32 internally vs. bf16 internally), not
    # from different random init.
    t = jnp.linspace(-5.0, 5.0, 11).reshape(-1, 1)

    m_fp32 = GaussianFourierEmbedding(output_dim=8, rngs=nnx.Rngs(0))
    out_fp32 = m_fp32(t)
    assert out_fp32.dtype == jnp.float32

    m_bf16 = GaussianFourierEmbedding(output_dim=8, rngs=nnx.Rngs(0),
                                      dtype=jnp.bfloat16, param_dtype=jnp.float32)
    out_bf16 = m_bf16(t)
    assert out_bf16.dtype == jnp.bfloat16

    # If the internal dot/cos/sin had run in bf16, phase errors from B's
    # ~O(1) values times t up to 5 would blow well past bf16's own
    # rounding tolerance. Requiring closeness at bf16's rounding level
    # proves the trig itself ran in fp32 and only the output was cast.
    out_fp32_as_bf16 = out_fp32.astype(jnp.bfloat16)
    assert jnp.allclose(
        out_bf16.astype(jnp.float32),
        out_fp32_as_bf16.astype(jnp.float32),
        atol=1e-2,
        rtol=1e-2,
    )
