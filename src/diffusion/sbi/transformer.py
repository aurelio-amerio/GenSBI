import jax
from jax import numpy as jnp
from jax import jit, vmap
from flax import nnx
from typing import Callable, Optional
from jaxtyping import Array, PyTree


layer = nnx.MultiHeadAttention(
    num_heads=8, in_features=5, qkv_features=16, decode=False, rngs=nnx.Rngs(0)
)


class AttentionBlock(nnx.Module):
    def __init__(
        self,
        din: int,
        num_heads: int,
        features: int,
        dropout_rate: float,
        skip_connection: bool,
        rngs: nnx.Rngs,
    ):
        self.skip_connection = skip_connection
        self.dropout_rate = dropout_rate

        self.layer_norm = nnx.LayerNorm(din, rngs=rngs)
        self.attn = nnx.MultiHeadAttention(
            in_features=din,
            num_heads=num_heads,
            qkv_features=features,
            dropout_rate=dropout_rate,
            decode=False,
            rngs=rngs,
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
        dropout_rate: float,
        act: Callable,
        skip_connection: bool,
        rngs: nnx.Rngs,
    ):
        self.skip_connection = skip_connection
        n_features = din
        self.layer_norm = nnx.LayerNorm(din, rngs=rngs)
        self.hidden_blocks = []
        self.hidden_blocks.append(
                    nnx.Linear(n_features, widening_factor * n_features, rngs=rngs)
                )
        n_features *= widening_factor
        
        for i in range(1, num_hidden_layers):
            self.hidden_blocks.append(
                    nnx.Linear(n_features, n_features, rngs=rngs)
                )

        self.hidden_blocks.append(nnx.Linear(n_features, din, rngs=rngs))
        self.act = act
        self.dropout_rate = dropout_rate
        self.dropout = nnx.Dropout(rate=dropout_rate)
        self.context_block = nnx.Linear(dcontext, din, rngs=rngs)
        return

    def __call__(self, x, context):
        x = self.layer_norm(x)
        x_in = x


        for i in range(len(self.hidden_blocks) - 1):
            x = self.hidden_blocks[i](x)
            x = self.act(x)

        x = self.hidden_blocks[-1](x)
        if self.dropout_rate > 0:
            x = self.dropout(x)

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
        dropout_rate: float = 0,
        widening_factor: int = 4,
        num_hidden_layers: int = 1,
        act: Callable = jax.nn.gelu,
        skip_connection_attn: bool = True,
        skip_connection_mlp: bool = True,
        *, # Enforce keyword arguments
        rngs: nnx.Rngs,
    ):
        self.din = din
        self.dcontext = dcontext
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.dropout_rate = dropout_rate
        self.widening_factor = widening_factor
        self.num_hidden_layers = num_hidden_layers
        self.act = act
        self.skip_connection_attn = skip_connection_attn
        self.skip_connection_mlp = skip_connection_mlp
        self.rngs = rngs

        # now we define attention and dense blocks
        self.attention_blocks = []
        self.dense_blocks = []
        self.layer_norm = nnx.LayerNorm(din, rngs=rngs)

        for _ in range(num_layers):
            self.attention_blocks.append(
                AttentionBlock(
                    din=self.din,
                    num_heads=num_heads,
                    features=features,
                    dropout_rate=dropout_rate,
                    skip_connection=skip_connection_attn,
                    rngs=rngs,
                )
            )
            self.dense_blocks.append(
                DenseBlock(
                    din,
                    dcontext,
                    num_hidden_layers,
                    widening_factor,
                    dropout_rate,
                    act=self.act,
                    skip_connection=skip_connection_mlp,
                    rngs=rngs,
                )
            )

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

