import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from einops import rearrange
from flax import nnx

from functools import partial

from dataclasses import dataclass

from .transformer import Transformer
from ..flux1.layers import EmbedND, timestep_embedding, MLPEmbedder, LastLayer


@dataclass
class SimformerParams2:
    rngs: nnx.Rngs
    dim_joint: int
    hidden_dim: int
    axes_dim: list[int]
    fourier_features: int = 128
    theta: int = 10000
    num_heads: int = 4
    num_layers: int = 6
    widening_factor: int = 3
    param_dtype: DTypeLike = jnp.bfloat16


class Simformer2(nnx.Module):
    def __init__(
        self,
        params: SimformerParams2,
    ):
        """
        Simformer model for time series forecasting.
        Args:
            params (SimformerParams): Parameters for the Simformer model.
        """
        self.params = params
        self.hidden_dim = params.hidden_dim
        rngs = params.rngs

        self.embedding_net_value = MLPEmbedder(
            in_dim=1,
            hidden_dim=self.hidden_dim,
            rngs=rngs,
            param_dtype=params.param_dtype,
        )
        # self.embedding_net_value = lambda x: jnp.repeat(x, dim_value, axis=-1)

        self.fourier_features = params.fourier_features
        self.embedding_time = MLPEmbedder(
            in_dim=self.fourier_features,
            hidden_dim=self.hidden_dim,
            rngs=rngs,
            param_dtype=params.param_dtype,
        )

        self.condition_embedding = MLPEmbedder(
            in_dim=params.dim_joint,
            hidden_dim=self.hidden_dim,
            rngs=rngs,
            param_dtype=params.param_dtype,
        )

        pe_dim = params.hidden_dim // params.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
        )

        self.transformer = Transformer(
            hidden_size=self.hidden_dim,
            num_heads=params.num_heads,
            num_layers=params.num_layers,
            widening_factor=params.widening_factor,
            rngs=params.rngs,
            param_dtype=params.param_dtype,
        )

        self.final_layer = LastLayer(
            self.hidden_dim,
            1,
            1,
            rngs=params.rngs,
            param_dtype=params.param_dtype,
        )
        return

    def __call__(self, x, t, args=None, *, node_ids, condition_mask, edge_mask=None):
        x = jnp.atleast_1d(x)
        t = jnp.atleast_1d(t)

        if x.ndim < 3:
            x = rearrange(x, "... -> 1 ... 1" if x.ndim == 1 else "... -> ... 1")
        # t = t.reshape(-1, 1)

        batch_size, seq_len, _ = x.shape

        # embed the node ids
        pe = self.pe_embedder(node_ids.reshape(1, -1, 1))

        # embed time
        vec = self.embedding_time(timestep_embedding(t, self.fourier_features))

        # broadcast the condition mask
        condition_mask = condition_mask.astype(jnp.bool_)

        if condition_mask.ndim == 1:
            condition_mask = condition_mask[None, ...]
            condition_mask = jnp.broadcast_to(condition_mask, (batch_size, seq_len))

        # embed the condition mask
        condition_embedding = self.condition_embedding(condition_mask)

        vec = (
            vec + condition_embedding
        )  # add the condition embedding to the time embedding

        # Embed inputs and broadcast
        x_encoded = self.embedding_net_value(x)

        h = self.transformer(x_encoded, vec=vec, pe=pe, mask=edge_mask)

        out = self.final_layer(h, vec)
        out = jnp.squeeze(out, axis=-1)
        return out
