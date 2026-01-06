import math
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
from einops import rearrange
from flax import nnx
from jax.typing import DTypeLike
import jax

from .math import attention, rope

class Identity(nnx.Module):
    def __call__(self, x: Array) -> Array:
        return x

class EmbedND(nnx.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, ids: Array) -> Array:
        n_axes = ids.shape[-1]
        emb = jnp.concatenate(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            axis=-3,
        )

        return jnp.expand_dims(emb, axis=1)


def timestep_embedding(
    t: Array, dim: int, max_period=10000, time_factor: float = 1000.0
) -> Array:
    """
    Generate timestep embeddings.

    Parameters
    ----------
        t: a 1-D Tensor of N indices, one per batch element.
            These may be fractional.
        dim: the dimension of the output.
        max_period: controls the minimum frequency of the embeddings.
        time_factor: Tensor of positional embeddings.

    Returns
    -------
        timestep embeddings.
    """
    t = jnp.atleast_1d(t)

    t = (
        t.ravel()
    )  # FIXME will return an error later on in case the shape is not (N,1...), we should find a better way to handle this

    t = time_factor * t
    half = dim // 2

    freqs = jnp.exp(
        -math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half
    ).astype(dtype=t.dtype)

    args = t[:, None].astype(jnp.float32) * freqs[None]
    embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)

    if dim % 2:
        embedding = jnp.concatenate(
            [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1
        )

    if jnp.issubdtype(t.dtype, jnp.floating):
        embedding = embedding.astype(t.dtype)

    return embedding


class MLPEmbedder(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ):
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
        return self.out_layer(self.silu(self.in_layer(x)))


class QKNorm(nnx.Module):
    def __init__(
        self,
        dim: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ):
        self.query_norm = nnx.RMSNorm(dim, rngs=rngs, param_dtype=param_dtype)
        self.key_norm = nnx.RMSNorm(dim, rngs=rngs, param_dtype=param_dtype)

    def __call__(self, q: Array, k: Array, v: Array) -> tuple[Array, Array]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q, k


class SelfAttention(nnx.Module):
    def __init__(
        self,
        dim: int,
        rngs: nnx.Rngs,
        qkv_features: int | None = None,
        param_dtype: DTypeLike = jnp.bfloat16,
        num_heads: int = 8,
        qkv_bias: bool = False,
    ):
        if qkv_features is None:
            qkv_features = dim

        self.num_heads = num_heads
        head_dim = qkv_features // num_heads

        self.qkv = nnx.Linear(
            in_features=dim,
            out_features=qkv_features * 3,
            use_bias=qkv_bias,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.norm = QKNorm(dim=head_dim, rngs=rngs, param_dtype=param_dtype)
        self.proj = nnx.Linear(
            in_features=qkv_features,
            out_features=dim,
            use_bias=True,
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, x: Array, pe: Array, mask: Array | None = None) -> Array:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe, mask=mask)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Array
    scale: Array
    gate: Array


# includes AdaLN-zero initialization
class Modulation(nnx.Module):
    def __init__(
        self,
        dim: int,
        double: bool,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ):
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nnx.Linear(
            in_features=dim,
            out_features=self.multiplier * dim,
            use_bias=True,
            rngs=rngs,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.zeros,  # this ensures that the initial modulation is neutral
            bias_init=jax.nn.initializers.zeros,  # this ensures that the initial modulation is neutral
        )

    def __call__(self, vec: Array) -> tuple[ModulationOut, ModulationOut | None]:
        out = jnp.split(self.lin(nnx.silu(vec))[:, None, :], self.multiplier, axis=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        rngs: nnx.Rngs,
        qkv_features: int | None = None,
        param_dtype: DTypeLike = jnp.bfloat16,
        qkv_bias: bool = False,
    ):
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.qkv_features = qkv_features if qkv_features is not None else hidden_size
        self.obs_mod = Modulation(
            dim=hidden_size, double=True, rngs=rngs, param_dtype=param_dtype
        )
        self.obs_norm1 = nnx.LayerNorm(
            num_features=hidden_size,
            use_scale=False,
            use_bias=False,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.obs_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_features=self.qkv_features,
            qkv_bias=qkv_bias,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        self.obs_norm2 = nnx.LayerNorm(
            num_features=hidden_size,
            use_scale=False,
            use_bias=False,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.obs_mlp = nnx.Sequential(
            nnx.Linear(
                in_features=hidden_size,
                out_features=mlp_hidden_dim,
                use_bias=True,
                rngs=rngs,
                param_dtype=param_dtype,
            ),
            nnx.gelu,
            nnx.Linear(
                in_features=mlp_hidden_dim,
                out_features=hidden_size,
                use_bias=True,
                rngs=rngs,
                param_dtype=param_dtype,
            ),
        )

        self.cond_mod = Modulation(
            dim=hidden_size, double=True, rngs=rngs, param_dtype=param_dtype
        )
        self.cond_norm1 = nnx.LayerNorm(
            num_features=hidden_size,
            use_scale=False,
            use_bias=False,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.cond_attn = SelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_features=self.qkv_features,
            qkv_bias=qkv_bias,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        self.cond_norm2 = nnx.LayerNorm(
            num_features=hidden_size,
            use_scale=False,
            use_bias=False,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.cond_mlp = nnx.Sequential(
            nnx.Linear(
                in_features=hidden_size,
                out_features=mlp_hidden_dim,
                use_bias=True,
                rngs=rngs,
                param_dtype=param_dtype,
            ),
            nnx.gelu,
            nnx.Linear(
                in_features=mlp_hidden_dim,
                out_features=hidden_size,
                use_bias=True,
                rngs=rngs,
                param_dtype=param_dtype,
            ),
        )

    def __call__(
        self,
        obs: Array,
        cond: Array,
        vec: Array,
        pe: Array | None = None,
        mask: Array | None = None,
    ) -> tuple[Array, Array]:
        obs_mod1, obs_mod2 = self.obs_mod(vec)
        cond_mod1, cond_mod2 = self.cond_mod(vec)

        # prepare image for attention
        obs_modulated = self.obs_norm1(obs)
        obs_modulated = (1 + obs_mod1.scale) * obs_modulated + obs_mod1.shift
        obs_qkv = self.obs_attn.qkv(obs_modulated)
        obs_q, obs_k, obs_v = rearrange(
            obs_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        obs_q, obs_k = self.obs_attn.norm(obs_q, obs_k, obs_v)

        # prepare cond for attention
        cond_modulated = self.cond_norm1(cond)
        cond_modulated = (1 + cond_mod1.scale) * cond_modulated + cond_mod1.shift
        cond_qkv = self.cond_attn.qkv(cond_modulated)
        cond_q, cond_k, cond_v = rearrange(
            cond_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        cond_q, cond_k = self.cond_attn.norm(cond_q, cond_k, cond_v)

        # run actual attention
        q = jnp.concatenate((cond_q, obs_q), axis=2)
        k = jnp.concatenate((cond_k, obs_k), axis=2)
        v = jnp.concatenate((cond_v, obs_v), axis=2)

        attn = attention(q, k, v, pe=pe, mask=mask)
        cond_attn, obs_attn = attn[:, : cond.shape[1]], attn[:, cond.shape[1] :]

        # calculate the obs bloks
        obs = obs + obs_mod1.gate * self.obs_attn.proj(obs_attn)
        obs = obs + obs_mod2.gate * self.obs_mlp(
            (1 + obs_mod2.scale) * self.obs_norm2(obs) + obs_mod2.shift
        )

        # calculate the cond bloks
        cond = cond + cond_mod1.gate * self.cond_attn.proj(cond_attn)
        cond = cond + cond_mod2.gate * self.cond_mlp(
            (1 + cond_mod2.scale) * self.cond_norm2(cond) + cond_mod2.shift
        )
        return obs, cond


class SingleStreamBlock(nnx.Module):
    """
    A DiT block with parallel linear layers as described in
    `arXiv:2302.05442 <https://arxiv.org/abs/2302.05442>`_ and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rngs: nnx.Rngs,
        qkv_features: int | None = None,
        param_dtype: DTypeLike = jnp.bfloat16,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        self.hidden_dim = hidden_size
        if qkv_features is None:
            self.qkv_features = hidden_size
        else:
            self.qkv_features = qkv_features
        self.num_heads = num_heads
        head_dim = qkv_features // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nnx.Linear(
            in_features=hidden_size,
            out_features=self.qkv_features * 3 + self.mlp_hidden_dim,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        # proj and mlp_out
        self.linear2 = nnx.Linear(
            in_features=self.qkv_features + self.mlp_hidden_dim,
            out_features=hidden_size,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        self.norm = QKNorm(dim=head_dim, rngs=rngs, param_dtype=param_dtype)

        self.hidden_size = hidden_size
        self.pre_norm = nnx.LayerNorm(
            num_features=hidden_size,
            use_scale=False,
            use_bias=False,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        self.mlp_act = nnx.gelu
        self.modulation = Modulation(
            hidden_size, double=False, rngs=rngs, param_dtype=param_dtype
        )

    def __call__(
        self, x: Array, vec: Array, pe: Array | None = None, mask: Array | None = None
    ) -> Array:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = jnp.split(self.linear1(x_mod), [3 * self.qkv_features], axis=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        attn = attention(q, k, v, pe=pe, mask=mask)
        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(jnp.concatenate((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: int,
        out_channels: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ):
        self.norm_final = nnx.LayerNorm(
            num_features=hidden_size,
            use_scale=False,
            use_bias=False,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.linear = nnx.Linear(
            in_features=hidden_size,
            out_features=patch_size * patch_size * out_channels,
            use_bias=True,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.adaLN_modulation = nnx.Sequential(
            nnx.silu,
            nnx.Linear(
                in_features=hidden_size,
                out_features=2 * hidden_size,
                use_bias=True,
                rngs=rngs,
                param_dtype=param_dtype,
            ),
        )

    def __call__(self, x: Array, vec: Array) -> Array:
        shift, scale = jnp.split(
            self.adaLN_modulation(vec),
            2,
            axis=1,
        )
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
