import jax
from jax import numpy as jnp
from jax import jit, vmap
from flax import nnx
from typing import Callable, Optional
from jaxtyping import Array, PyTree

from ..flux1.layers import SingleStreamBlock, LastLayer


class Transformer(nnx.Module):
    """A transformer stack."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_layers: int,
        widening_factor: int = 4,
        *,  # Enforce keyword arguments
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.bfloat16,
    ):
        self.hidden_size = hidden_size

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.widening_factor = widening_factor
        self.rngs = rngs

        # now we define attention and dense blocks
        self.singlestreak_blocks = []

        for _ in range(num_layers):
            self.singlestreak_blocks.append(
                SingleStreamBlock(
                    hidden_size=self.hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=widening_factor,
                    rngs=rngs,
                    param_dtype=param_dtype,
                )
            )

        return

    def __call__(
        self,
        inputs: Array,  # [B, T, D]
        vec: Array,  # [B, D_context], it includes the time embedding and the conditioning embedding
        pe: Array,
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
            x = self.singlestreak_blocks[i](x, vec=vec, pe=pe, mask=mask)

        return x
