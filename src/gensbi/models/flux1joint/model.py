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
        axes_dim : list[int]
            Dimensions of the axes for positional encoding.
        condition_dim : list[int]
            Number of features used to encode the condition mask, which determines the features on which we are conditioning.
        qkv_bias : bool
            Whether to use bias in QKV layers.
        rngs : nnx.Rngs
            Random number generators for initialization.
        dim_joint : int
            Number of tokens in the joint sequence.
        theta : int
            Scaling factor for positional encoding.
        id_embedding_strategy : str
            Kind of embedding for token ids ('absolute', 'pos1d', 'pos2d', 'rope').
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
    axes_dim: list[int]
    condition_dim: list[int]
    qkv_bias: bool
    rngs: nnx.Rngs
    dim_joint: int  # joint dimension
    theta: int = 500
    id_embedding_strategy: str = "absolute"
    guidance_embed: bool = False
    param_dtype: DTypeLike = jnp.bfloat16

    def __post_init__(self):
        availabel_embeddings = ["absolute", "pos1d", "pos2d", "rope"]

        assert (
            self.id_embedding_strategy in availabel_embeddings
        ), f"Unknown id embedding kind {self.id_embedding_strategy} for obs."

        if self.id_embedding_strategy == "rope":

            # raise a warning tha using rope for joint modeling is not recommended

            warnings.warn(
                "Using RoPE embedding for joint density estimation is not recommended. Consider using 'absolute' embeddings instead.",
                UserWarning,
            )

        self.input_token_dim = (
            np.sum(jnp.asarray(self.axes_dim, dtype=jnp.int32)) * self.num_heads
        )
        self.condition_token_dim = (
            np.sum(jnp.asarray(self.condition_dim, dtype=jnp.int32)) * self.num_heads
        )
        self.hidden_size = int(self.input_token_dim + self.condition_token_dim)
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

        pe_dim = self.qkv_features // params.num_heads
        if sum(params.axes_dim) + sum(params.condition_dim) != pe_dim:
            raise ValueError(
                f"Got axes_dim:{params.axes_dim} + condition_dim:{params.condition_dim} but expected positional dim {pe_dim}"
            )
        self.num_heads = params.num_heads

        assert (
            np.array(params.axes_dim).ndim == np.array(params.condition_dim).ndim
        ), "axes_dim and condition_dim must have the same dimension, got {} and {}".format(
            params.axes_dim, params.condition_dim
        )

        axes_dim = [a + b for a, b in zip(params.axes_dim, params.condition_dim)]

        if "rope" in params.id_embedding_strategy:

            self.use_rope = True
            self.pe_embedder = EmbedND(
                dim=pe_dim, theta=params.theta, axes_dim=axes_dim
            )
            self.ids_embedder = None

        else:

            self.use_rope = False
            self.pe_embedder = None
            self.ids_embedder = FeatureEmbedder(
                num_embeddings=params.dim_joint,
                hidden_size=self.hidden_size,
                kind=params.id_embedding_strategy,
                param_dtype=params.param_dtype,
                rngs=params.rngs,
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

        self.condition_embedding = nnx.Param(
            0.01
            * jnp.ones(
                (1, 1, self.params.condition_token_dim), dtype=params.param_dtype
            )
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
        condition_embedding = self.condition_embedding * condition_mask
        obs = jnp.concatenate([obs, condition_embedding], axis=-1)

        if self.use_rope:
            pe = self.pe_embedder(node_ids)  # enable rope positional embeddings
        else:
            # we add the positional embeddings
            obs = obs * jnp.sqrt(self.hidden_size) + self.ids_embedder(node_ids)
            pe = None

        vec = self.time_in(timestep_embedding(t, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.vector_in(guidance)

        for block in self.single_blocks.layers:
            obs = block(obs, vec=vec, pe=pe)

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
