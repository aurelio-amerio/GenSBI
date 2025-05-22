import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike 

from einops import rearrange, repeat
from flax import nnx

from sbi_utils import Transformer
from sbi_utils.embedding import GaussianFourierEmbedding

class MLPEmbedder(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ):
        self.p_skip = nnx.Param(0.01*jnp.ones((1, 1, hidden_dim)))
        self.in_layer = nnx.Linear(
            in_features=in_dim,
            out_features=hidden_dim,
            use_bias=True,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.silu = nnx.silu
        self.out_layer = nnx.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim,
            use_bias=True,
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, x: Array) -> Array:
        x = jnp.atleast_1d(x)
        out =  self.out_layer(self.silu(self.in_layer(x)))
        x_repeated, out = jnp.broadcast_arrays(x, out)
        out = x_repeated * self.p_skip + (1-self.p_skip)*out
        return out

class Simformer(nnx.Module):
    def __init__(self, dim_value, dim_id, dim_condition, dim_joint, *, num_heads=2, num_layers=6, widening_factor=4, transformer_features=10, num_hidden_layers=1,dropout_rate=0.1,rngs):
        """
        Simformer model for time series forecasting.
        Args:
            dim_value: Dimension of the value embedding.
            dim_id: Dimension of the id embedding.
            dim_condition: Dimension of the condition embedding.
            dim_joint: Dimensionality of the data + parameters. For example, if the data is 1D and the parameters are 2D, then dim_joint = 3.
            num_heads: Number of attention heads.
            transformer_features: Number of features in the transformer.
            
            rngs: Random number generator keys.
        """

        self.dim_value = dim_value
        self.dim_id = dim_id
        self.dim_condition = dim_condition

        self.embedding_net_value = MLPEmbedder(in_dim=1, hidden_dim=dim_value, rngs=rngs)
        # self.embedding_net_value = lambda x: jnp.repeat(x, dim_value, axis=-1)

        fourier_features=128
        self.embedding_time = GaussianFourierEmbedding(fourier_features, rngs=rngs)
        self.embedding_net_id = nnx.Embed(
            num_embeddings=dim_joint, features=dim_id, rngs=rngs
        ) 
        self.condition_embedding = nnx.Param(0.01*jnp.ones((1, 1, dim_condition)))

        self.total_tokens = dim_value + dim_id + dim_condition

        self.transformer = Transformer(
            din=self.total_tokens,
            dcontext=fourier_features,
            num_heads=num_heads,
            num_layers=num_layers,
            features=transformer_features,
            widening_factor=widening_factor,
            dropout_rate=dropout_rate,
            num_hidden_layers=num_hidden_layers,
            act=jax.nn.gelu,
            skip_connection_attn=True,
            skip_connection_mlp=True,
            rngs=rngs,
        )

        self.output_fn = nnx.Linear(self.total_tokens,1,rngs=rngs)
        return
    
    def __call__(self, x, t, args=None,*, node_ids, condition_mask, edge_mask=None):
        x = jnp.atleast_1d(x)
        t = jnp.atleast_1d(t)

        if x.ndim < 3:
            x = rearrange(x, '... -> 1 ... 1' if x.ndim == 1 else '... -> ... 1')
        t = t.reshape(-1,1, 1)

        batch_size, seq_len, _ = x.shape
        condition_mask = condition_mask.astype(jnp.bool_).reshape(-1,seq_len,1)
        condition_mask = jnp.broadcast_to(condition_mask, (batch_size, seq_len, 1))
        
        node_ids = node_ids.reshape(-1,seq_len)

        time_embeddings = self.embedding_time(t)

        condition_embedding = self.condition_embedding * condition_mask # If condition_mask is 0, then the embedding is 0, otherwise it is the condition_embedding vector
        condition_embedding = jnp.broadcast_to(condition_embedding, (batch_size, seq_len, self.dim_condition))

        # Embed inputs and broadcast
        value_embeddings = self.embedding_net_value(x)
        id_embeddings = self.embedding_net_id(node_ids)
        id_embeddings = jnp.broadcast_to(id_embeddings, (batch_size, seq_len, self.dim_id))

        # Concatenate embeddings (alternatively you can also add instead of concatenating)
        x_encoded = jnp.concatenate([value_embeddings, id_embeddings, condition_embedding], axis=-1)

        h = self.transformer(x_encoded, context=time_embeddings, mask=edge_mask)

        out = self.output_fn(h)
        out = jnp.squeeze(out, axis=-1)
        return out
    
    