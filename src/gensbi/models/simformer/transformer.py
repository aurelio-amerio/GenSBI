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
    """Self-attention block with a fixed fp32 compute island.

    ``flax.nnx.MultiHeadAttention`` (flax 0.12.7) computes its softmax in
    whatever ``dtype`` it is given -- there is no internal fp32 upcast for
    the attention logits/softmax the way some other implementations provide.
    Rather than bolt on a custom ``attention_fn`` just to force fp32 softmax,
    this block keeps *all* of its compute (LayerNorm + MultiHeadAttention) at
    ``dtype=jnp.float32``, ignoring the ``dtype`` compute-precision knob
    entirely for this block. This is a deliberately larger fp32 island than
    a single softmax op, but the bulk of the model's FLOPs live in the
    ``DenseBlock`` MLP stack (``widening_factor`` x wider), which does honor
    the bf16 ``dtype`` knob, so this island has a small cost in practice.
    ``param_dtype`` (master-weight storage) is unaffected and still threads
    through normally.
    """

    def __init__(
        self,
        din: int,
        num_heads: int,
        features: int,
        skip_connection: bool,
        rngs: nnx.Rngs,
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
    ):
        self.skip_connection = skip_connection

        # fp32 island: dtype is intentionally fixed to float32 regardless of
        # the dtype knob above (see class docstring).
        self.layer_norm = nnx.LayerNorm(
            din, rngs=rngs, dtype=jnp.float32, param_dtype=param_dtype
        )
        self.attn = nnx.MultiHeadAttention(
            in_features=din,
            num_heads=num_heads,
            qkv_features=features,
            decode=False,
            rngs=rngs,
            dtype=jnp.float32,
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
        dtype: DTypeLike = jnp.float32,
        param_dtype: DTypeLike = jnp.float32,
    ):
        self.skip_connection = skip_connection
        n_features = din
        # fp32 island: LayerNorm stays fp32 regardless of the compute dtype
        # knob (mirrors AttentionBlock's normalization treatment).
        self.layer_norm = nnx.LayerNorm(
            din, rngs=rngs, dtype=jnp.float32, param_dtype=param_dtype
        )
        hidden_blocks = []
        hidden_blocks.append(
            nnx.Linear(
                n_features,
                widening_factor * n_features,
                rngs=rngs,
                dtype=dtype,
                param_dtype=param_dtype,
            )
        )

        n_features *= widening_factor

        for i in range(1, num_hidden_layers):
            hidden_blocks.append(
                nnx.Linear(
                    n_features,
                    n_features,
                    rngs=rngs,
                    dtype=dtype,
                    param_dtype=param_dtype,
                )
            )

        hidden_blocks.append(
            nnx.Linear(n_features, din, rngs=rngs, dtype=dtype, param_dtype=param_dtype)
        )

        self.hidden_blocks = nnx.List(hidden_blocks)

        self.act = act
        self.context_block = nnx.Linear(
            dcontext, din, rngs=rngs, dtype=dtype, param_dtype=param_dtype
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
        dtype: DTypeLike = jnp.float32,
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
        self.dtype = dtype
        self.param_dtype = param_dtype

        # now we define attention and dense blocks
        attention_blocks = []
        dense_blocks = []
        # fp32 island: the final norm feeds directly into the model's output
        # projection, so it stays fp32 regardless of the compute dtype knob.
        self.layer_norm = nnx.LayerNorm(
            din, rngs=rngs, dtype=jnp.float32, param_dtype=param_dtype
        )

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
                    dtype=dtype,
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
