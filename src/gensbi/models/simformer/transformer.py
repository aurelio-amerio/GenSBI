import jax
from jax import numpy as jnp
from jax import jit, vmap
from flax import nnx
from typing import Callable, Optional
from jaxtyping import Array, PyTree
from jax.typing import DTypeLike


# layer = nnx.MultiHeadAttention(
#     num_heads=8, in_features=5, qkv_features=16, decode=False, rngs=nnx.Rngs(0)
# )


class AttentionBlock(nnx.Module):
    def __init__(
        self,
        din: int,
        num_heads: int,
        features: int,
        skip_connection: bool,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.float32,
    ):
        self.skip_connection = skip_connection

        self.layer_norm = nnx.LayerNorm(din, rngs=rngs, param_dtype=param_dtype)
        self.attn = nnx.MultiHeadAttention(
            in_features=din,
            num_heads=num_heads,
            qkv_features=features,
            decode=False,
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray | None) -> jnp.ndarray:
        x = self.layer_norm(x)
        x_in = x
        x = self.attn(x, mask=mask)

        if self.skip_connection:
            x = x + x_in
        return x


class DenseBlock(nnx.Module):
    def __init__(
        self,
        din,
        dcontext,
        num_hidden_layers,
        widening_factor: int,
        act: Callable,
        skip_connection: bool,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.float32,
    ):
        self.skip_connection = skip_connection
        n_features = din
        self.layer_norm = nnx.LayerNorm(din, rngs=rngs, param_dtype=param_dtype)
        hidden_blocks = []
        hidden_blocks.append(
            nnx.Linear(
                n_features,
                widening_factor * n_features,
                rngs=rngs,
                param_dtype=param_dtype,
            )
        )

        n_features *= widening_factor

        for i in range(1, num_hidden_layers):
            hidden_blocks.append(
                nnx.Linear(
                    n_features, n_features, rngs=rngs, param_dtype=param_dtype
                )
            )

        hidden_blocks.append(
            nnx.Linear(n_features, din, rngs=rngs, param_dtype=param_dtype)
        )

        self.hidden_blocks = nnx.List(hidden_blocks)

        self.act = act
        self.context_block = nnx.Linear(
            dcontext, din, rngs=rngs, param_dtype=param_dtype
        )
        return

    def __call__(self, x, context):
        x = self.layer_norm(x)
        x_in = x

        for i in range(len(self.hidden_blocks) - 1):
            x = self.hidden_blocks[i](x)
            x = self.act(x)

        x = self.hidden_blocks[-1](x)

        if context is not None:
            context_emb = self.context_block(context)
            context_emb = self.act(context_emb)
            while context_emb.ndim < x.ndim:
                context_emb = context_emb[..., None, :]

            x = x + context_emb

        if self.skip_connection:
            x = x + x_in

        return x


class Transformer(nnx.Module):
    """A transformer stack."""

    def __init__(
        self,
        din: int,
        dcontext: int,
        num_heads: int,
        num_layers: int,
        features: int,
        widening_factor: int = 4,
        num_hidden_layers: int = 1,
        act: Callable = jax.nn.gelu,
        skip_connection_attn: bool = True,
        skip_connection_mlp: bool = True,
        *,  # Enforce keyword arguments
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.float32,
    ):
        self.din = din
        self.dcontext = dcontext
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.widening_factor = widening_factor
        self.num_hidden_layers = num_hidden_layers
        self.act = act
        self.skip_connection_attn = skip_connection_attn
        self.skip_connection_mlp = skip_connection_mlp
        self.rngs = rngs
        self.param_dtype = param_dtype

        # now we define attention and dense blocks
        attention_blocks = []
        dense_blocks = []
        self.layer_norm = nnx.LayerNorm(din, rngs=rngs, param_dtype=param_dtype)

        for _ in range(num_layers):
            attention_blocks.append(
                AttentionBlock(
                    din=self.din,
                    num_heads=num_heads,
                    features=features,
                    skip_connection=skip_connection_attn,
                    rngs=rngs,
                    param_dtype=param_dtype,
                )
            )
            dense_blocks.append(
                DenseBlock(
                    din,
                    dcontext,
                    num_hidden_layers,
                    widening_factor,
                    act=self.act,
                    skip_connection=skip_connection_mlp,
                    rngs=rngs,
                    param_dtype=param_dtype,
                )
            )

        self.attention_blocks = nnx.List(attention_blocks)
        self.dense_blocks = nnx.List(dense_blocks)

        return

    def __call__(
        self,
        inputs: Array,  # [B, T, D]
        context: Optional[Array] = None,  # [B, D_context]
        mask: Array | None = None,  # [T, T] or [B, T, T]
    ) -> jax.Array:  # [B, T, D]
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[None, None, :, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            else:
                raise ValueError(f"Mask must have ndim 2 or 3, got {mask.ndim}.")

        x = inputs
        for i in range(self.num_layers):
            x = self.attention_blocks[i](x, mask)
            x = self.dense_blocks[i](x, context)

        out = self.layer_norm(x)
        return out
