from dataclasses import dataclass

from typing import Union

import jax
import jax.numpy as jnp
from jax import Array
from flax import nnx
from jax.typing import DTypeLike

from gensbi.models.flux1.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
    Identity,
)

from gensbi.models.embedding import FeatureEmbedder


@dataclass
class Flux1Params:
    """Parameters for the Flux1 model.

    GenSBI uses the tensor convention `(batch, dim, channels)`.

    - `dim_*` counts **tokens** (how many distinct observables/variables you have).
    - `channels` counts **features per token** (how many values each observable carries).

    For conditional SBI with Flux1:

    - Parameters to infer (often denoted $\theta$) have shape `(batch, dim_obs, in_channels)`.
        In most SBI problems `in_channels = 1` (one scalar per parameter token).
    - Conditioning data (often denoted $x$) has shape `(batch, dim_cond, context_in_dim)`.
        `context_in_dim` can be > 1 (e.g., multiple detectors or multiple features per measured token).

    Example: 2 parameters, frequency grid with 2 detectors

    - `dim_obs = 2`, `in_channels = 1`  -> `(batch, 2, 1)`
    - `dim_cond = n_freq`, `context_in_dim = 2` -> `(batch, n_freq, 2)`

    Parameters
    ----------
        in_channels : int
            Number of channels per observation/parameter token.
        vec_in_dim : Union[int, None]
            Dimension of the vector input, if applicable.
        context_in_dim : int
            Number of channels per conditioning token.
        mlp_ratio : float
            Ratio for the MLP layers.
        num_heads : int
            Number of attention heads.
        depth : int
            Number of double stream blocks.
        depth_single_blocks : int
            Number of single stream blocks.
        axes_dim : list[int]
            Dimensions of the axes for positional encoding.
        qkv_bias : bool
            Whether to use bias in QKV layers.
        rngs : nnx.Rngs
            Random number generators for initialization.
        dim_obs : int
            Number of observation/parameter tokens.
        dim_cond : int
            Number of conditioning tokens.
        theta : int
            Scaling factor for positional encoding.
        id_embedding_kind : tuple[str, str]
            Kind of ID embedding for obs and cond respectively. Options are "absolute", "pos1d", "pos2d" or "rope".
        guidance_embed : bool
            Whether to use guidance embedding.
        param_dtype : DTypeLike
            Data type for model parameters.

    """

    in_channels: int
    vec_in_dim: Union[int, None]
    context_in_dim: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    qkv_bias: bool
    rngs: nnx.Rngs
    dim_obs: int  # observation dimension
    dim_cond: int  # condition dimension
    theta: int = 500
    id_embedding_kind: tuple[str, str] = (
        "absolute",
        "absolute",
    )  # "absolute", "pos1d", "pos2d" or "rope" - for obs and cond respectively
    guidance_embed: bool = False
    param_dtype: DTypeLike = jnp.bfloat16

    def __post_init__(self):
        availabel_embeddings = [
            "absolute",
            "pos1d",
            "pos2d",
            "rope",
            "rope1d",
            "rope2d",
        ]
        assert (
            self.id_embedding_kind[0] in availabel_embeddings
        ), f"Unknown id embedding kind {self.id_embedding_kind[0]} for obs."
        assert (
            self.id_embedding_kind[1] in availabel_embeddings
        ), f"Unknown id embedding kind {self.id_embedding_kind[1]} for cond."

        self.hidden_size = int(
            jnp.sum(jnp.asarray(self.axes_dim, dtype=jnp.int32)) * self.num_heads
        )
        self.qkv_features = self.hidden_size


class Flux1(nnx.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: Flux1Params):
        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.in_channels
        self.hidden_size = params.hidden_size
        self.qkv_features = params.qkv_features

        pe_dim = self.qkv_features // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {params.axes_dim} but expected positional dim {pe_dim}"
            )
        self.num_heads = params.num_heads

        self.id_embedding_kind_obs, self.id_embedding_kind_cond = (
            params.id_embedding_kind
        )

        # rope1d and rope2d are all equivalent to rope
        if self.id_embedding_kind_obs in ["rope", "rope1d", "rope2d"]:
            self.id_embedding_kind_obs = "rope"
        if self.id_embedding_kind_cond in ["rope", "rope1d", "rope2d"]:
            self.id_embedding_kind_cond = "rope"

        if (
            self.id_embedding_kind_obs == "rope"
            or self.id_embedding_kind_cond == "rope"
        ):
            self.pe_embedder = EmbedND(
                dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim
            )
        else:
            self.pe_embedder = None

        if self.id_embedding_kind_obs == "rope":
            self.use_rope_obs = True
            self.obs_ids_embedder = None
        else:
            self.use_rope_obs = False
            self.obs_ids_embedder = FeatureEmbedder(
                num_embeddings=params.dim_obs,
                hidden_size=self.hidden_size,
                kind=params.id_embedding_kind[0],
                param_dtype=params.param_dtype,
                rngs=params.rngs,
            )

        if self.id_embedding_kind_cond == "rope":
            self.use_rope_cond = True
            self.cond_ids_embedder = None
        else:
            self.use_rope_cond = False
            self.cond_ids_embedder = FeatureEmbedder(
                num_embeddings=params.dim_cond,
                hidden_size=self.hidden_size,
                kind=params.id_embedding_kind[1],
                param_dtype=params.param_dtype,
                rngs=params.rngs,
            )

        self.obs_in = nnx.Linear(
            in_features=self.in_channels,
            out_features=self.hidden_size,
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

        self.cond_in = nnx.Linear(
            in_features=params.context_in_dim,
            out_features=self.hidden_size,
            use_bias=True,
            rngs=params.rngs,
            param_dtype=params.param_dtype,
        )

        # self.condition_embedding = nnx.Param(
        #     0.01 * jnp.ones((1, self.hidden_size), dtype=params.param_dtype)
        # )
        # self.condition_null = nnx.Param(
        #     jax.random.normal(
        #         params.rngs.cond(),
        #         (1, params.dim_cond, self.hidden_size),
        #         dtype=params.param_dtype,
        #     )
        # )

        self.double_blocks = nnx.Sequential(
            *[
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_features=self.qkv_features,
                    qkv_bias=params.qkv_bias,
                    rngs=params.rngs,
                    param_dtype=params.param_dtype,
                )
                for _ in range(params.depth)
            ]
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
        obs_ids: Array,
        cond: Array,
        cond_ids: Array,
        conditioned: bool | Array = True,  # does nothing
        guidance: Array | None = None,
    ) -> Array:

        # assumes obs, cond, obs_ids, cond_ids have shape (B, F, C)
        # assumes t has shape (B,) or (B, 1)

        obs = jnp.asarray(obs, dtype=self.params.param_dtype)
        cond = jnp.asarray(cond, dtype=self.params.param_dtype)
        t = jnp.asarray(t, dtype=self.params.param_dtype)

        # obs = _expand_dims(obs)
        # cond = _expand_dims(cond)

        if obs.ndim != 3 or cond.ndim != 3:
            raise ValueError(
                "Input obs and cond tensors must have 3 dimensions, got {} and {}".format(
                    obs.ndim, cond.ndim
                )
            )

        # running on sequences obs
        obs = self.obs_in(obs)
        vec = self.time_in(timestep_embedding(t, 256))

        # conditioned = jnp.asarray(conditioned, dtype=jnp.bool_)  # type: ignore
        # conditioned_int = jnp.asarray(conditioned, dtype=jnp.int32)[..., None]  # type: ignore

        # condition_embedding = self.condition_embedding * (1 - conditioned_int)

        # vec = vec + condition_embedding  # we add the condition embedding to the vector

        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.vector_in(guidance)

        # cond_processed = self.cond_in(cond)  # (B, F, H)
        # cond_null = jnp.repeat(
        #     self.condition_null, repeats=obs.shape[0], axis=0
        # )  # (B, F, H)
        # cond = jnp.where(
        #     conditioned[..., None, None], cond_processed, cond_null
        # )  # we replace the condition with a null vector if not conditioned

        cond = self.cond_in(cond)

        # if not using rope for a dimension, perform id embedding and add it to the input
        if self.obs_ids_embedder is not None:
            obs = obs * jnp.sqrt(self.hidden_size) + self.obs_ids_embedder(obs_ids)
            obs_ids_rope = jnp.zeros(
                (obs_ids.shape[0], obs_ids.shape[1], cond_ids.shape[2]),
                dtype=obs_ids.dtype,
            )
        else:
            obs_ids_rope = obs_ids
        if self.cond_ids_embedder is not None:
            cond = cond * jnp.sqrt(self.hidden_size) + self.cond_ids_embedder(cond_ids)
            cond_ids_rope = jnp.zeros(
                (cond_ids.shape[0], cond_ids.shape[1], obs_ids.shape[2]),
                dtype=cond_ids.dtype,
            )
        else:
            cond_ids_rope = cond_ids

        if self.use_rope_obs or self.use_rope_cond:
            # we use rope embeddings
            ids = jnp.concatenate((cond_ids_rope, obs_ids_rope), axis=1)
            pe = self.pe_embedder(ids)
        else:
            pe = None

        for block in self.double_blocks.layers:
            obs, cond = block(obs=obs, cond=cond, vec=vec, pe=pe)

        obs = jnp.concatenate((cond, obs), axis=1)
        for block in self.single_blocks.layers:
            obs = block(obs, vec=vec, pe=pe)
        obs = obs[:, cond.shape[1] :, ...]

        obs = self.final_layer(obs, vec)  # (N, T, patch_size ** 2 * out_channels)
        return obs
