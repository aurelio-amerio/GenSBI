import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from einops import rearrange
from flax import nnx

import numpy as np
from functools import partial
from typing import Optional

from dataclasses import dataclass

from gensbi.models.flux1.layers import (
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
    Identity,
)

from gensbi.models.embedding import FeatureEmbedder

import warnings

from typing import Union, Callable, Optional


@dataclass
class Flux1JointParams:
    """Parameters for the Flux1Joint model.

    GenSBI uses the tensor convention `(batch, dim, channels)`.

    For joint density estimation, the model consumes a *single* sequence `obs` that
    mixes all variables you want to model jointly. In this case:

    - `dim_joint` is the number of tokens in that joint sequence.
    - `in_channels` is the number of channels/features per token.

    In many SBI-style problems you will still use `in_channels = 1` (one scalar per token),
    but for some datasets a token may carry multiple features.

    Parameters
    ----------
        in_channels : int
            Number of channels/features per token.
        vec_in_dim : Union[int, None]
            Dimension of the vector input, if applicable.
        mlp_ratio : float
            Ratio for the MLP layers.
        num_heads : int
            Number of attention heads.
        depth_single_blocks : int
            Number of single stream blocks.
        value_emb_dim : int
            Number of features per head used to embed the data.
        cond_emb_dim : int
            Number of features per head used to encode the condition mask, which determines the features on which we are conditioning.
        id_emb_dim : int
            Number of features per head used to encode the token ids.
        qkv_bias : bool
            Whether to use bias in QKV layers.
        rngs : nnx.Rngs
            Random number generators for initialization.
        dim_joint : int
            Number of tokens in the joint sequence.
        id_embedding_strategy : str
            Kind of embedding for token ids ('sum', 'concat').
        guidance_embed : bool
            Whether to use guidance embedding.
        param_dtype : DTypeLike
            Data type for model parameters.

    """

    in_channels: int
    vec_in_dim: Union[int, None]
    mlp_ratio: float
    num_heads: int
    depth_single_blocks: int
    value_emb_dim: int
    cond_emb_dim: int
    id_emb_dim: int
    qkv_bias: bool
    rngs: nnx.Rngs
    dim_joint: int  # joint dimension
    id_embedding_strategy: str = "sum"
    guidance_embed: bool = False
    param_dtype: DTypeLike = jnp.bfloat16

    def __post_init__(self):
        available_embeddings = ["sum", "concat"]

        assert (
            self.id_embedding_strategy in available_embeddings
        ), f"Unknown id embedding kind {self.id_embedding_strategy} for obs."

        self.input_token_dim = int(self.value_emb_dim * self.num_heads)
        if self.id_embedding_strategy == "sum":
            self.cond_emb_dim = 0
            self.id_emb_dim = 0

        self.condition_token_dim = int(self.cond_emb_dim * self.num_heads)
        self.id_token_dim = int(self.id_emb_dim * self.num_heads)

        self.hidden_size = int(
            self.input_token_dim + self.condition_token_dim + self.id_token_dim
        )
        self.qkv_features = self.hidden_size


class Flux1Joint(nnx.Module):
    """
    Flux1Joint model for joint density estimation.

    Parameters
    ----------
        params : Flux1JointParams
            Parameters for the Flux1Joint model.
    """

    def __init__(self, params: Flux1JointParams):
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.in_channels
        self.hidden_size = params.hidden_size
        self.qkv_features = params.qkv_features

        self.num_heads = params.num_heads

        if params.id_embedding_strategy == "sum":
            self.ids_embedder = FeatureEmbedder(
                num_embeddings=params.dim_joint,
                hidden_size=self.hidden_size,
                kind="absolute",
                param_dtype=params.param_dtype,
                rngs=params.rngs,
            )
        elif params.id_embedding_strategy == "concat":
            self.ids_embedder = FeatureEmbedder(
                num_embeddings=params.dim_joint,
                hidden_size=self.params.id_token_dim,
                kind="absolute",
                param_dtype=params.param_dtype,
                rngs=params.rngs,
            )
        else:
            raise ValueError(
                f"Unknown id embedding strategy: {params.id_embedding_strategy}"
            )

        self.obs_in = nnx.Linear(
            in_features=self.in_channels,
            out_features=self.params.input_token_dim,
            use_bias=True,
            rngs=params.rngs,
            param_dtype=params.param_dtype,
        )
        self.time_in = MLPEmbedder(
            in_dim=256,
            hidden_dim=self.hidden_size,
            rngs=params.rngs,
            param_dtype=params.param_dtype,
        )
        self.vector_in = (
            MLPEmbedder(
                params.vec_in_dim,
                self.hidden_size,
                rngs=params.rngs,
                param_dtype=params.param_dtype,
            )
            if params.guidance_embed
            else Identity()
        )

        if params.id_embedding_strategy == "sum":
            self.condition_embedding = nnx.Param(
                0.01
                * jnp.ones((1, 1, self.params.hidden_size), dtype=params.param_dtype)
            )
        elif params.id_embedding_strategy == "concat":
            self.condition_embedding = nnx.Param(
                0.01
                * jnp.ones(
                    (1, 1, self.params.condition_token_dim), dtype=params.param_dtype
                )
            )
        else:
            raise ValueError(
                f"Unknown id embedding strategy: {params.id_embedding_strategy}"
            )

        self.single_blocks = nnx.Sequential(
            *[
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_features=self.qkv_features,
                    rngs=params.rngs,
                    param_dtype=params.param_dtype,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(
            self.hidden_size,
            1,
            self.out_channels,
            rngs=params.rngs,
            param_dtype=params.param_dtype,
        )

    def __call__(
        self,
        t: Array,
        obs: Array,
        node_ids: Array,
        condition_mask: Array,
        guidance: Array | None = None,
        edge_mask: Optional[Array] = None,
    ) -> Array:
        batch_size, seq_len, _ = obs.shape
        obs = jnp.asarray(obs, dtype=self.params.param_dtype)
        t = jnp.asarray(t, dtype=self.params.param_dtype)
        if obs.ndim != 3:
            raise ValueError(
                "Input obs tensor must have 3 dimensions, got {}".format(obs.ndim)
            )
        obs = self.obs_in(obs)
        condition_mask = condition_mask.astype(
            jnp.bool_
        )  # .reshape(batch_size, seq_len, -1)
        if condition_mask.shape[0] == 1:
            condition_mask = jnp.repeat(condition_mask, repeats=batch_size, axis=0)

        if node_ids.shape[0] == 1:
            node_ids = jnp.repeat(node_ids, repeats=batch_size, axis=0)

        condition_embedding = self.condition_embedding * condition_mask
        ids_embedding = self.ids_embedder(node_ids)

        if self.params.id_embedding_strategy == "sum":
            obs = obs * jnp.sqrt(self.hidden_size) + ids_embedding + condition_embedding
        elif self.params.id_embedding_strategy == "concat":
            obs = jnp.concatenate([obs, condition_embedding, ids_embedding], axis=-1)
        else:
            raise ValueError(
                f"Unknown id embedding strategy: {self.params.id_embedding_strategy}"
            )

        vec = self.time_in(timestep_embedding(t, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.vector_in(guidance)

        for block in self.single_blocks.layers:
            obs = block(obs, vec=vec, pe=None)

        obs = self.final_layer(obs, vec)
        return obs


# the wrapper is the same as the Simformer one, we reuse the class
# class JointWrapper(JointWrapper):
#     """
#     Module to handle conditioning in the Flux1Joint model.

#     Args:
#         model (Flux1Joint): Flux1Joint model instance.
#     """
#     def __init__(self, model):
#         super().__init__(model)
#     def __call__(
#         self,
#         t: Array,
#         obs: Array,
#         obs_ids: Array,
#         cond: Array,
#         cond_ids: Array,
#         conditioned: bool = True,
#     ) -> Array:
#         return super().__call__(t, obs, obs_ids, cond, cond_ids, conditioned)
