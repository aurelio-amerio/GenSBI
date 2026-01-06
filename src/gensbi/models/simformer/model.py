import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from einops import rearrange
from flax import nnx

from functools import partial
from typing import Optional

from dataclasses import dataclass

from .transformer import Transformer
from gensbi.models.embedding import GaussianFourierEmbedding, MLPEmbedder

from gensbi.utils.model_wrapping import ModelWrapper, _expand_dims, _expand_time


@dataclass
class SimformerParams:
    """Parameters for the Simformer model.

    GenSBI uses the tensor convention `(batch, dim, channels)`.

    For Simformer (joint modeling), the input `obs` is a single sequence with:

    - `dim_joint`: number of tokens in the sequence (how many variables / measured points).
    - `in_channels`: number of channels/features per token.

    Conditioning is controlled via `condition_mask` at call time (mask is over **tokens**,
    not channels): tokens with mask=1 are treated as conditioned.

    Parameters
    ----------
        rngs : nnx.Rngs
            Random number generators for initialization.
        in_channels : int
            Number of channels/features per token.
        dim_value : int
            Dimension of the value embeddings.
        dim_id : int
            Dimension of the ID embeddings.
        dim_condition : int
            Dimension of the condition embeddings.
        dim_joint : int
            Number of tokens in the joint sequence.
        fourier_features : int
            Number of Fourier features for time embedding.
        num_heads : int
            Number of attention heads.
        num_layers : int
            Number of transformer layers.
        widening_factor : int
            Widening factor for the transformer.
        qkv_features : int
            Number of features for QKV layers.
        num_hidden_layers : int
            Number of hidden layers in the transformer.
        param_dtype : DTypeLike
            Data type for model parameters.

    """

    rngs: nnx.Rngs
    in_channels: int
    dim_value: int
    dim_id: int
    dim_condition: int
    dim_joint: int
    num_heads: int
    num_layers: int
    num_hidden_layers: int = 1
    fourier_features: int = 128
    widening_factor: int = 3
    qkv_features: int | None = None
    param_dtype: DTypeLike = jnp.float32

    def __post_init__(self):
        if self.qkv_features is None:
            self.qkv_features = self.dim_value + self.dim_id + self.dim_condition


class Simformer(nnx.Module):
    """
    Simformer model for joint density estimation.

    Parameters
    ----------
        params : SimformerParams
            Parameters for the Simformer model.
    """

    def __init__(
        self,
        params: SimformerParams,
        embedding_net_value: Optional[nnx.Module] = None,
    ):
        self.params = params
        self.in_channels = params.in_channels
        self.dim_value = params.dim_value
        self.dim_id = params.dim_id
        self.dim_condition = params.dim_condition

        if embedding_net_value is not None:
            self.embedding_net_value = embedding_net_value
        else:
            self.embedding_net_value = MLPEmbedder(
                in_dim=self.in_channels,
                hidden_dim=params.dim_value,
                rngs=params.rngs,
                param_dtype=params.param_dtype,
            )
        # self.embedding_net_value = lambda obs: jnp.repeat(obs, dim_value, axis=-1)

        fourier_features = params.fourier_features
        self.embedding_time = GaussianFourierEmbedding(
            fourier_features,
            rngs=params.rngs,
            learnable=True,
            param_dtype=params.param_dtype,
        )
        self.embedding_net_id = nnx.Embed(
            num_embeddings=params.dim_joint,
            features=params.dim_id,
            rngs=params.rngs,
            param_dtype=params.param_dtype,
        )
        self.condition_embedding = nnx.Param(
            0.01 * jnp.ones((1, 1, params.dim_condition), dtype=params.param_dtype)
        )

        self.total_tokens = params.dim_value + params.dim_id + params.dim_condition

        self.transformer = Transformer(
            din=self.total_tokens,
            dcontext=fourier_features,
            num_heads=params.num_heads,
            num_layers=params.num_layers,
            features=params.qkv_features,
            widening_factor=params.widening_factor,
            num_hidden_layers=params.num_hidden_layers,
            act=jax.nn.gelu,
            skip_connection_attn=True,
            skip_connection_mlp=True,
            rngs=params.rngs,
            param_dtype=params.param_dtype,
        )

        self.output_fn = nnx.Linear(
            self.total_tokens,
            self.in_channels,
            rngs=params.rngs,
            param_dtype=params.param_dtype,
        )
        return

    def __call__(
        self,
        t: Array,
        obs: Array,
        node_ids: Array,
        condition_mask: Array,
        edge_mask: Optional[Array] = None,
    ) -> Array:
        """
        Forward pass of the Simformer model.

        Parameters
        ----------
            t : Array
                Time steps.
            obs : Array
                Input data.
            args : Optional[dict]
                Additional arguments.
            node_ids : Array
                Node identifiers.
            condition_mask : Array
                Mask for conditioning.
            edge_mask : Optional[Array]
                Mask for edges.

        Returns
        -------
            Array
                Model output.
        """

        obs = jnp.asarray(obs, dtype=self.params.param_dtype)
        t = jnp.asarray(jnp.atleast_1d(t), dtype=self.params.param_dtype)

        assert (
            obs.ndim == 3
        ), f"Input obs must be of shape (batch_size, seq_len, 1), got {obs.shape}"
        assert (
            len(t.ravel()) == obs.shape[0] or len(t.ravel()) == 1
        ), "t must have the same batch size as obs or size 1, got {} and {}".format(
            t.shape, obs.shape
        )

        t = t.reshape(-1, 1, 1)

        batch_size, seq_len, _ = obs.shape
        condition_mask = condition_mask.astype(jnp.bool_)  # .reshape(-1, seq_len, 1)
        # condition_mask = jnp.broadcast_to(condition_mask, (batch_size, seq_len, 1))

        if node_ids.ndim == 1:
            node_ids = node_ids.reshape(-1, seq_len)
        elif node_ids.ndim == 2:
            assert (
                node_ids.shape[1] == seq_len
            ), f"node_ids must have shape (-1, {seq_len}), got {node_ids.shape}"
        elif node_ids.ndim == 3:
            assert (
                node_ids.shape[1] == seq_len and node_ids.shape[2] == 1
            ), f"node_ids must have shape (-1, {seq_len}, 1), got {node_ids.shape}"
            node_ids = jnp.squeeze(node_ids, axis=-1)
        else:
            raise ValueError(f"node_ids must have ndim <=3, got {node_ids.ndim}")

        time_embeddings = self.embedding_time(t)

        condition_embedding = (
            self.condition_embedding * condition_mask
        )  # If condition_mask is 0, then the embedding is 0, otherwise it is the condition_embedding vector
        condition_embedding = jnp.broadcast_to(
            condition_embedding, (batch_size, seq_len, self.dim_condition)
        )

        # Embed inputs and broadcast
        value_embeddings = self.embedding_net_value(obs)
        id_embeddings = self.embedding_net_id(node_ids)
        id_embeddings = jnp.broadcast_to(
            id_embeddings, (batch_size, seq_len, self.dim_id)
        )

        # Concatenate embeddings (alternatively you can also add instead of concatenating)
        x_encoded = jnp.concatenate(
            [value_embeddings, id_embeddings, condition_embedding], axis=-1
        )

        h = self.transformer(x_encoded, context=time_embeddings, mask=edge_mask)

        out = self.output_fn(h)
        # out = jnp.squeeze(out, axis=-1)
        return out


# class JointWrapper(ModelWrapper):
#     """
#     Module to handle conditioning in the Simformer model.

#     Args:
#         model (Simformer): Simformer model instance.
#     """
#     def __init__(self, model):
#         super().__init__(model)

#     def conditioned(
#         self,
#         obs: Array,
#         obs_ids: Array,
#         cond: Array,
#         cond_ids: Array,
#         t: Array,
#         edge_mask: Optional[Array] = None
#     ) -> Array:
#         """
#         Perform conditioned inference.

#         Args:
#             obs (Array): Observations.
#             obs_ids (Array): Observation identifiers.
#             cond (Array): Conditioning values.
#             cond_ids (Array): Conditioning identifiers.
#             t (Array): Time steps.
#             edge_mask (Optional[Array]): Mask for edges.

#         Returns:
#             Array: Conditioned output.
#         """

#         dim_obs = obs.shape[1]
#         dim_cond = cond.shape[1]
#         # repeat cond on the first dimension to match obs
#         cond = jnp.broadcast_to(
#             cond, (obs.shape[0], *cond.shape[1:])
#         )

#         condition_mask_dim = dim_obs + dim_cond

#         condition_mask = jnp.zeros((condition_mask_dim,), dtype=jnp.bool_)
#         condition_mask = condition_mask.at[dim_obs:].set(True)

#         x = jnp.concatenate([obs, cond], axis=1)
#         node_ids = jnp.concatenate([obs_ids, cond_ids], axis=1)

#         res = self.model(
#             obs=x,
#             t=t,
#             node_ids=node_ids,
#             condition_mask=condition_mask,
#             edge_mask=edge_mask,
#         )
#         # now return only the values on which we are not conditioning
#         res = res[:, :dim_obs]
#         return res

#     def unconditioned(
#         self,
#         obs: Array,
#         obs_ids: Array,
#         t: Array,
#         edge_mask: Optional[Array] = None
#     ) -> Array:
#         """
#         Perform unconditioned inference.

#         Args:
#             obs (Array): Observations.
#             obs_ids (Array): Observation identifiers.
#             t (Array): Time steps.
#             edge_mask (Optional[Array]): Mask for edges.

#         Returns:
#             Array: Unconditioned output.
#         """

#         condition_mask = jnp.zeros((obs.shape[1],), dtype=jnp.bool_)

#         node_ids = obs_ids

#         res = self.model(
#             obs=obs,
#             t=t,
#             node_ids=node_ids,
#             condition_mask=condition_mask,
#             edge_mask=edge_mask,
#         )

#         return res

#     def __call__(
#         self,
#         t: Array,
#         obs: Array,
#         obs_ids: Array,
#         cond: Array,
#         cond_ids: Array,
#         conditioned: bool = True,
#         edge_mask: Optional[Array] = None
#     ) -> Array:
#         """
#         Perform inference based on conditioning.

#         Args:
#             obs (Array): Observations.
#             obs_ids (Array): Observation identifiers.
#             cond (Array): Conditioning values.
#             cond_ids (Array): Conditioning identifiers.
#             timesteps (Array): Time steps.
#             conditioned (bool): Whether to perform conditioned inference.
#             edge_mask (Optional[Array]): Mask for edges.

#         Returns:
#             Array: Model output.
#         """
#         t = _expand_time(t)
#         obs = _expand_dims(obs)
#         cond = _expand_dims(cond)

#         obs_ids = _expand_dims(obs_ids)
#         cond_ids = _expand_dims(cond_ids)

#         if conditioned:
#             return self.conditioned(
#                 obs, obs_ids, cond, cond_ids, t, edge_mask=edge_mask
#             )
#         else:
#             return self.unconditioned(obs, obs_ids, t, edge_mask=edge_mask)
