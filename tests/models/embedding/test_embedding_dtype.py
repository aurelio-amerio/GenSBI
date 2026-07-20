import jax.numpy as jnp
from flax import nnx

from gensbi.models.embedding import FeatureEmbedder
from gensbi.models.embedding.embedding import MLPEmbedder
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
