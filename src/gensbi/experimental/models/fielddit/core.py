"""The MMDiT bottleneck for FieldDiT — Flux1 joint-attention over obs+cond.

obs tokens carry rope2d positional ids; the few cond tokens are absolute
(order-free), embedded with a learned id embedding and given dummy zero rope
ids so the rotary encoding is identity on them. Block order matches Flux1:
cond is concatenated before obs.
"""

import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike

from gensbi.models.flux1.layers import DoubleStreamBlock, SingleStreamBlock, EmbedND
from gensbi.models.embedding import FeatureEmbedder


class MMDiTCore(nnx.Module):
    """Flux1 double-stream + single-stream transformer over obs+cond tokens.

    Parameters mirror the relevant subset of ``Flux1Params``. ``vec`` (the
    time (+cond summary, +guidance) modulation vector) is supplied externally
    so the same vector can drive the conv codec's AdaGN-zero modulation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        depth: int,
        depth_single_blocks: int,
        axes_dim,
        theta: int,
        n_cond_tokens: int,
        qkv_bias: bool,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        assert sum(axes_dim) == head_dim, (
            f"sum(axes_dim)={sum(axes_dim)} must equal head_dim={head_dim}"
        )
        self.pe_embedder = EmbedND(dim=head_dim, theta=theta, axes_dim=tuple(axes_dim))
        # absolute (order-free) id embedding for the few cond tokens
        self.cond_ids_embedder = FeatureEmbedder(
            num_embeddings=n_cond_tokens,
            hidden_size=hidden_size,
            kind="absolute",
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.double_blocks = nnx.Sequential(
            *[
                DoubleStreamBlock(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio,
                    qkv_features=hidden_size, qkv_bias=qkv_bias,
                    rngs=rngs, param_dtype=param_dtype,
                )
                for _ in range(depth)
            ]
        )
        self.single_blocks = nnx.Sequential(
            *[
                SingleStreamBlock(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio,
                    qkv_features=hidden_size, rngs=rngs, param_dtype=param_dtype,
                )
                for _ in range(depth_single_blocks)
            ]
        )

    def __call__(self, obs_tokens, cond_tokens, vec, obs_ids, cond_ids):
        B = obs_tokens.shape[0]
        if obs_ids.shape[0] == 1 and B > 1:
            obs_ids = jnp.repeat(obs_ids, B, axis=0)

        # absolute id embedding added to the cond value embedding (Flux1 pattern)
        cond_tokens = cond_tokens * jnp.sqrt(self.hidden_size) + self.cond_ids_embedder(cond_ids)

        # dummy zero rope ids for cond so rope is identity on cond positions
        cond_ids_rope = jnp.zeros(
            (obs_ids.shape[0], cond_tokens.shape[1], obs_ids.shape[2]), dtype=obs_ids.dtype
        )
        ids = jnp.concatenate((cond_ids_rope, obs_ids), axis=1)
        pe = self.pe_embedder(ids)

        for blk in self.double_blocks.layers:
            obs_tokens, cond_tokens = blk(obs=obs_tokens, cond=cond_tokens, vec=vec, pe=pe)

        x = jnp.concatenate((cond_tokens, obs_tokens), axis=1)
        for blk in self.single_blocks.layers:
            x = blk(x, vec=vec, pe=pe)

        return x[:, cond_tokens.shape[1]:, ...]
