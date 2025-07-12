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
from .embedding import GaussianFourierEmbedding, MLPEmbedder


@dataclass
class SimformerParams:
    """Parameters for the Simformer model.
    
    Args:
        rngs (nnx.Rngs): Random number generators for initialization.
        dim_value (int): Dimension of the value embeddings.
        dim_id (int): Dimension of the ID embeddings.
        dim_condition (int): Dimension of the condition embeddings.
        dim_joint (int): Total dimension of the joint embeddings.
        fourier_features (int): Number of Fourier features for time embedding.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        widening_factor (int): Widening factor for the transformer.
        qkv_features (int): Number of features for QKV layers.
        num_hidden_layers (int): Number of hidden layers in the transformer.

    """
    rngs: nnx.Rngs
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
    # param_dtype: DTypeLike = jnp.float32

    def __post_init__(self):
        if self.qkv_features is None:
            self.qkv_features = self.dim_value + self.dim_id + self.dim_condition


class Simformer(nnx.Module):
    """
    Simformer model for joint density estimation.

    Args:
        params (SimformerParams): Parameters for the Simformer model.
    """
    def __init__(
        self,
        params: SimformerParams,
    ):
        self.params = params
        self.dim_value = params.dim_value
        self.dim_id = params.dim_id
        self.dim_condition = params.dim_condition

        self.embedding_net_value = MLPEmbedder(
            in_dim=1, hidden_dim=params.dim_value, rngs=params.rngs
        )
        # self.embedding_net_value = lambda x: jnp.repeat(x, dim_value, axis=-1)

        fourier_features = params.fourier_features
        self.embedding_time = GaussianFourierEmbedding(
            fourier_features, rngs=params.rngs
        )
        self.embedding_net_id = nnx.Embed(
            num_embeddings=params.dim_joint, features=params.dim_id, rngs=params.rngs
        )
        self.condition_embedding = nnx.Param(
            0.01 * jnp.ones((1, 1, params.dim_condition))
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
        )

        self.output_fn = nnx.Linear(self.total_tokens, 1, rngs=params.rngs)
        return

    def __call__(
        self, 
        x: Array, 
        t: Array, 
        args: Optional[dict] = None, 
        *, 
        node_ids: Array, 
        condition_mask: Array, 
        edge_mask: Optional[Array] = None
    ) -> Array:
        """
        Forward pass of the Simformer model.

        Args:
            x (Array): Input data.
            t (Array): Time steps.
            args (Optional[dict]): Additional arguments.
            node_ids (Array): Node identifiers.
            condition_mask (Array): Mask for conditioning.
            edge_mask (Optional[Array]): Mask for edges.

        Returns:
            Array: Model output.
        """
        x = jnp.atleast_1d(x)
        t = jnp.atleast_1d(t)

        if x.ndim < 3:
            x = rearrange(x, "... -> 1 ... 1" if x.ndim == 1 else "... -> ... 1")
        t = t.reshape(-1, 1, 1)

        batch_size, seq_len, _ = x.shape
        condition_mask = condition_mask.astype(jnp.bool_).reshape(-1, seq_len, 1)
        condition_mask = jnp.broadcast_to(condition_mask, (batch_size, seq_len, 1))

        node_ids = node_ids.reshape(-1, seq_len)

        time_embeddings = self.embedding_time(t)

        condition_embedding = (
            self.condition_embedding * condition_mask
        )  # If condition_mask is 0, then the embedding is 0, otherwise it is the condition_embedding vector
        condition_embedding = jnp.broadcast_to(
            condition_embedding, (batch_size, seq_len, self.dim_condition)
        )

        # Embed inputs and broadcast
        value_embeddings = self.embedding_net_value(x)
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
        out = jnp.squeeze(out, axis=-1)
        return out


class SimformerConditioner(nnx.Module):
    """
    Module to handle conditioning in the Simformer model.

    Args:
        model (Simformer): Simformer model instance.
    """
    def __init__(self, model: Simformer):
        self.model = model
        self.dim_joint = model.params.dim_joint

    def conditioned(
        self, 
        obs: Array, 
        obs_ids: Array, 
        cond: Array, 
        cond_ids: Array, 
        t: Array, 
        edge_mask: Optional[Array] = None
    ) -> Array:
        """
        Perform conditioned inference.

        Args:
            obs (Array): Observations.
            obs_ids (Array): Observation identifiers.
            cond (Array): Conditioning values.
            cond_ids (Array): Conditioning identifiers.
            t (Array): Time steps.
            edge_mask (Optional[Array]): Mask for edges.

        Returns:
            Array: Conditioned output.
        """
        obs = jnp.atleast_1d(obs)
        cond = jnp.atleast_1d(cond)
        t = jnp.atleast_1d(t)

        if obs.ndim < 3:
            obs = rearrange(obs, "... -> 1 ... 1" if obs.ndim == 1 else "... -> ... 1")

        if cond.ndim < 3:
            cond = rearrange(
                cond, "... -> 1 ... 1" if cond.ndim == 1 else "... -> ... 1"
            )
        
        # repeat cond on the first dimension to match obs
        cond = jnp.broadcast_to(
            cond, (obs.shape[0], *cond.shape[1:])
        )

        condition_mask_dim = len(obs_ids) + len(cond_ids)

        condition_mask = jnp.zeros((condition_mask_dim,), dtype=jnp.bool_)
        condition_mask = condition_mask.at[cond_ids].set(True)

        x = jnp.concatenate([obs, cond], axis=1)
        node_ids = jnp.concatenate([obs_ids, cond_ids])

        # Sort the nodes and the corresponding values
        # nodes_sort = jnp.argsort(node_ids)
        # x = x[:, nodes_sort]
        # node_ids = node_ids[nodes_sort]

        res = self.model(
            x=x,
            t=t,
            node_ids=node_ids,
            condition_mask=condition_mask,
            edge_mask=edge_mask,
        )
        # now return only the values on which we are not conditioning
        res = res[:, :len(obs_ids)]
        return res

    def unconditioned(
        self, 
        obs: Array, 
        obs_ids: Array, 
        t: Array, 
        edge_mask: Optional[Array] = None
    ) -> Array:
        """
        Perform unconditioned inference.

        Args:
            obs (Array): Observations.
            obs_ids (Array): Observation identifiers.
            t (Array): Time steps.
            edge_mask (Optional[Array]): Mask for edges.

        Returns:
            Array: Unconditioned output.
        """
        obs = jnp.atleast_1d(obs)
        t = jnp.atleast_1d(t)

        if obs.ndim < 3:
            obs = rearrange(obs, "... -> 1 ... 1" if obs.ndim == 1 else "... -> ... 1")

        condition_mask = jnp.zeros((obs.shape[1],), dtype=jnp.bool_)

        node_ids = obs_ids
        x = obs

        res = self.model(
            x=x,
            t=t,
            node_ids=node_ids,
            condition_mask=condition_mask,
            edge_mask=edge_mask,
        )

        return res

    def __call__(
        self, 
        obs: Array, 
        obs_ids: Array, 
        cond: Array, 
        cond_ids: Array, 
        timesteps: Array, 
        conditioned: bool = True, 
        edge_mask: Optional[Array] = None
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
            edge_mask (Optional[Array]): Mask for edges.

        Returns:
            Array: Model output.
        """
        if conditioned:
            return self.conditioned(
                obs, obs_ids, cond, cond_ids, timesteps, edge_mask=edge_mask
            )
        else:
            return self.unconditioned(obs, obs_ids, timesteps, edge_mask=edge_mask)
