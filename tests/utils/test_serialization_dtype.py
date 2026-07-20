import jax.numpy as jnp
from flax import nnx

from gensbi.utils.serialization import save_safetensors, load_safetensors


def test_bf16_checkpoint_loads_into_fp32_model(tmp_path):
    # Old checkpoints (bf16 master weights) must load into new fp32 models.
    old = nnx.Linear(2, 3, rngs=nnx.Rngs(0), param_dtype=jnp.bfloat16)
    p = tmp_path / "old.safetensors"
    save_safetensors(old, p)
    new = nnx.Linear(2, 3, rngs=nnx.Rngs(1), param_dtype=jnp.float32)
    load_safetensors(new, p)
    assert new.kernel[...].dtype == jnp.float32
    assert jnp.allclose(
        new.kernel[...], old.kernel[...].astype(jnp.float32)
    )
