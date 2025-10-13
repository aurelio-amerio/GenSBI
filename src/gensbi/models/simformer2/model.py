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

from gensbi.models.simformer import SimformerWrapper

from gensbi.models.flux1.layers import (
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
    Identity,
)

from typing import Union, Callable, Optional


@dataclass
class Simformer2Params:
    """Parameters for the Simformer2 model.

    Args:
        in_channels (int): Number of input channels.
        vec_in_dim (Union[int, None]): Dimension of the vector input, if applicable.
        context_in_dim (int): Dimension of the context input.
        mlp_ratio (float): Ratio for the MLP layers.
        num_heads (int): Number of attention heads.
        depth (int): Number of double stream blocks.
        depth_single_blocks (int): Number of single stream blocks.
        axes_dim (list[int]): Dimensions of the axes for positional encoding.
        qkv_bias (bool): Whether to use bias in QKV layers.
        rngs (nnx.Rngs): Random number generators for initialization.
        obs_dim (int): Observation dimension.
        cond_dim (int): Condition dimension.
        theta (int): Scaling factor for positional encoding.
        guidance_embed (bool): Whether to use guidance embedding.
        qkv_multiplier (int): Multiplier for QKV features.
        param_dtype (DTypeLike): Data type for model parameters.

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
    joint_dim: int  # joint dimension
    theta: int = 10_000
    guidance_embed: bool = False
    param_dtype: DTypeLike = jnp.bfloat16

    def __post_init__(self):

        self.input_token_dim = np.sum(jnp.asarray(self.axes_dim, dtype=jnp.int32))*self.num_heads
        self.condition_token_dim = np.sum(jnp.asarray(self.condition_dim, dtype=jnp.int32))*self.num_heads
        self.hidden_size = int(self.input_token_dim + self.condition_token_dim)

        self.qkv_features = self.hidden_size



class Simformer2(nnx.Module):
    """
    Simformer model for joint density estimation.

    Args:
        params (Simformer2Params): Parameters for the Simformer2 model.
    """

    def __init__(self, params: Simformer2Params):
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

        assert np.array(params.axes_dim).ndim == np.array(params.condition_dim).ndim, "axes_dim and condition_dim must have the same dimension, got {} and {}".format(params.axes_dim, params.condition_dim)

        axes_dim = [a + b for a, b in zip(params.axes_dim, params.condition_dim)]
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=params.theta, axes_dim=axes_dim
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

        # need to check this
        self.condition_embedding = nnx.Param(
            0.01 * jnp.ones((1, 1, self.params.condition_token_dim), dtype=params.param_dtype)
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
        edge_mask: Optional[Array] = None,  # for compatibility, does nothing
    ) -> Array:

        batch_size, seq_len, _ = obs.shape

        obs = jnp.asarray(obs, dtype=self.params.param_dtype)
        t = jnp.asarray(t, dtype=self.params.param_dtype)

        if obs.ndim != 3:
            raise ValueError(
                "Input obs tensor must have 3 dimensions, got {}".format(obs.ndim)
            )

        # running on sequences obs
        obs = self.obs_in(obs)

        condition_mask = condition_mask.astype(jnp.bool_).reshape(-1, seq_len, 1)
        condition_mask = jnp.broadcast_to(condition_mask, (batch_size, seq_len, 1))
        condition_embedding = self.condition_embedding * condition_mask

        obs = jnp.concatenate([obs, condition_embedding], axis=-1) # add the condition information to the token dimension

        vec = self.time_in(timestep_embedding(t, 256))

        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.vector_in(guidance)

        pe = self.pe_embedder(node_ids)

        for block in self.single_blocks.layers:
            obs = block(obs, vec=vec, pe=pe)

        obs = self.final_layer(obs, vec)  # (N, T, patch_size ** 2 * out_channels)
        return obs


# the wrapper is the same as the Simformer one, we reuse the class
class Simformer2Wrapper(SimformerWrapper):
    """
    Module to handle conditioning in the Simformer2 model.

    Args:
        model (Simformer2): Simformer2 model instance.
    """

    def __init__(self, model):
        super().__init__(model)

    def __call__(
        self,
        t: Array,
        obs: Array,
        obs_ids: Array,
        cond: Array,
        cond_ids: Array,
        conditioned: bool = True,
    ) -> Array:
        """
        Perform inference based on conditioning.

        Args:
            obs (Array): Observations.
            obs_ids (Array): Observation identifiers.
            cond (Array): Conditioning values.
            cond_ids (Array): Conditioning identifiers.
            timesteps (Array): Time steps.
            conditioned (bool): Whether to perform conditioned inference.

        Returns:
            Array: Model output.
        """
        return super().__call__(t, obs, obs_ids, cond, cond_ids, conditioned)
