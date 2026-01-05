import jax
from jax import numpy as jnp
from flax import nnx
import numpy as np
from jax.typing import DTypeLike
from jax import Array


class MLPEmbedder(nnx.Module):
    """
    MLP-based embedder with skip connections.

    Parameters
    ----------
    in_dim : int
        Input dimension.
    hidden_dim : int
        Hidden dimension, must be a multiple of in_dim.
    rngs : nnx.Rngs
        Random number generators for initialization.
    param_dtype : DTypeLike, optional
        Data type for parameters. Defaults to jnp.float32.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.float32,
    ):
        assert (
            hidden_dim % in_dim == 0
        ), "hidden_dim must be multiple of in_dim, got {} and {}".format(
            hidden_dim, in_dim
        )
        self.repeats = hidden_dim // in_dim
        self.p_skip = nnx.Param(0.01 * jnp.ones((1, 1, hidden_dim), dtype=param_dtype))
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
        """
        Forward pass of the MLP embedder.

        Parameters
        ----------
        x : Array
            Input array.

        Returns
        -------
        Array
            Embedded output with skip connections.
        """
        x = jnp.atleast_1d(x)
        out = self.out_layer(self.silu(self.in_layer(x)))
        x_repeated = jnp.repeat(x, self.repeats, axis=-1)
        out = x_repeated * self.p_skip + (1 - self.p_skip) * out
        return out


class SimpleTimeEmbedding(nnx.Module):
    """Simple time embedding module using cosine and sine transformations."""
    
    def __init__(self):
        """Initialize simple time embedding module."""
        return

    def __call__(self, t):
        """
        Compute time embedding.

        Parameters
        ----------
        t : Array
            Time values.

        Returns
        -------
        Array
            Time embeddings.
        """
        t = jnp.atleast_1d(t)
        if t.ndim == 1:
            t = jnp.expand_dims(t, axis=1)
        out = jnp.concatenate(
            [
                t - 0.5,
                jnp.cos(2 * jnp.pi * t),
                jnp.sin(2 * jnp.pi * t),
                -jnp.cos(4 * jnp.pi * t),
            ],
            axis=-1,
        )
        return out


class SinusoidalTimeEmbedding(nnx.Module):
    def __init__(self, output_dim: int = 128):
        """Sinusoidal embedding module. Mostly used to embed time.

        Parameters
        ----------
        output_dim : int, optional
            Output dimension. Defaults to 128.
        """
        self.output_dim = output_dim
        return

    def __call__(self, t):
        """
        Compute sinusoidal time embedding.

        Parameters
        ----------
        t : Array
            Time values.

        Returns
        -------
        Array
            Sinusoidal time embeddings.
        """
        t = jnp.atleast_1d(t)
        if t.ndim == 1:
            t = jnp.expand_dims(t, axis=1)
        half_dim = self.output_dim // 2 + 1
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = jnp.expand_dims(emb, 0)
        # emb = t[..., None] * emb[None, ...]
        emb = jnp.dot(t, emb)
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
        return emb[..., : self.output_dim]


class GaussianFourierEmbedding(nnx.Module):
    def __init__(
        self,
        output_dim: int = 128,
        learnable: bool = False,
        *,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.float32,
    ):
        """Gaussian Fourier embedding module. Mostly used to embed time.

        Parameters
        ----------
        output_dim : int, optional
            Output dimension. Defaults to 128.
        learnable : bool, optional
            Whether parameters are learnable. Defaults to False.
        rngs : nnx.Rngs
            Random number generators for initialization.
        param_dtype : DTypeLike, optional
            Data type for parameters. Defaults to jnp.float32.
        """
        self.output_dim = output_dim
        half_dim = self.output_dim // 2 + 1
        self.B = nnx.Param(
            jax.random.normal(rngs.params(), [half_dim, 1], dtype=param_dtype)
        )
        if not learnable:
            self.B = jax.lax.stop_gradient(jnp.asarray(self.B, dtype=param_dtype))

        return

    def __call__(self, t):
        """
        Compute Gaussian Fourier time embedding.

        Parameters
        ----------
        t : Array
            Time values.

        Returns
        -------
        Array
            Gaussian Fourier time embeddings.
        """
        t = jnp.atleast_1d(t)
        if t.ndim == 1:
            t = jnp.expand_dims(t, axis=1)

        B = self.B

        arg = 2 * jnp.pi * jnp.dot(t, B.T)
        term1 = jnp.cos(arg)
        term2 = jnp.sin(arg)
        out = jnp.concatenate([term1, term2], axis=-1)
        return out[..., : self.output_dim]


class PEMatrix(nnx.Variable):
    """Variable type for storing pre-computed position embedding matrices."""
    pass


class SinusoidalPosEmbed1D(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        max_len: int = 5000,
        param_dtype: DTypeLike = jnp.float32,
    ):
        """
        Fast 1D Sinusoidal Embedding (Hugging Face Style).
        Pre-computes the matrix to avoid re-calculating sines/cosines.

        Parameters
        ----------
        hidden_size : int
            Hidden size, must be divisible by 2.
        max_len : int, optional
            Maximum sequence length. Defaults to 5000.
        param_dtype : DTypeLike, optional
            Data type for parameters. Defaults to jnp.float32.
        """

        if hidden_size % 2 != 0:
            raise ValueError(f"Hidden size ({hidden_size}) must be divisible by 2.")

        self.hidden_size = hidden_size

        # --- Hugging Face Logic ---
        # Omega: 1 / 10000^(i / (dim/2))
        # Note: We use dim/2 because we concat sin + cos blocks
        dim_half = hidden_size // 2
        omega = jnp.arange(dim_half, dtype=jnp.float32)
        omega /= dim_half
        omega = 1.0 / 10000**omega  # (D/2,)

        # Positions: 0, 1, 2, ... max_len
        pos = jnp.arange(max_len, dtype=jnp.float32)  # (MaxLen,)

        # Outer Product: pos * omega
        out = jnp.einsum("m,d->md", pos, omega)  # (MaxLen, D/2)

        # Block Concatenation: [Sin Block | Cos Block]
        emb_sin = jnp.sin(out)
        emb_cos = jnp.cos(out)
        pe = jnp.concatenate(
            [emb_sin, emb_cos], axis=1
        )  # (MaxLen, D)

        # Register as a constant (frozen state)
        self.pe = PEMatrix(jnp.asarray(pe, dtype=param_dtype))

    def __call__(self, ids):
        """
        Forward pass of the 1D sinusoidal position embedder.

        Parameters
        ----------
        ids : Array
            Input IDs with shape (batch, seq_len).

        Returns
        -------
        Array
            Position embeddings of shape (1, seq_len, hidden_size).
        """
        seq_len = ids.shape[1]
        # Slice the pre-computed matrix
        # This is extremely fast (just a memory pointer offset)
        return self.pe[None, :seq_len, :]


class SinusoidalPosEmbed2D(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        max_h: int = 128,
        max_w: int = 128,
        param_dtype: DTypeLike = jnp.float32,
    ):
        """
        Fast 2D Sinusoidal Embedding (Hugging Face / MAE Style).

        Parameters
        ----------
        hidden_size : int
            Hidden size, must be divisible by 2.
        max_h : int, optional
            Maximum height. Defaults to 128.
        max_w : int, optional
            Maximum width. Defaults to 128.
        param_dtype : DTypeLike, optional
            Data type for parameters. Defaults to jnp.float32.
        """

        if hidden_size % 2 != 0:
            raise ValueError(f"Hidden size ({hidden_size}) must be divisible by 2.")

        self.hidden_size = hidden_size
        dim_each = hidden_size // 2  # Half features for H, half for W

        # --- Internal Helper: Exact HF 1D Logic ---
        def _get_1d_block(length, dim):
            dim_half = dim // 2
            omega = jnp.arange(dim_half, dtype=jnp.float32)
            omega /= dim_half
            omega = 1.0 / 10000**omega

            pos = jnp.arange(length, dtype=jnp.float32)
            out = jnp.einsum("m,d->md", pos, omega)

            return jnp.concatenate(
                [jnp.sin(out), jnp.cos(out)], axis=1)  # (Length, D)

        # --- Pre-computation ---
        # 1. Height Embeddings (Y-axis)
        pe_h = _get_1d_block(max_h, dim_each)  # (MaxH, D/2)

        # 2. Width Embeddings (X-axis)
        pe_w = _get_1d_block(max_w, dim_each)  # (MaxW, D/2)

        # Register constants
        self.pe_h = PEMatrix(jnp.asarray(pe_h, dtype=param_dtype))
        self.pe_w = PEMatrix(jnp.asarray(pe_w, dtype=param_dtype))

    def __call__(self, ids):
        """
        Compute 2D sinusoidal position embeddings.

        Parameters
        ----------
        ids : Array
            Input IDs with shape (batch, h, w).

        Returns
        -------
        Array
            2D position embeddings of shape (batch, h*w, hidden_size).
        """
        h, w = ids.shape[1], ids.shape[2]
        # 1. Slice
        row_embed = self.pe_h[:h, None, :]  # (h, 1, D/2)
        col_embed = self.pe_w[None, :w, :]  # (1, w, D/2)

        # 2. Broadcast to Grid
        # Repeat row vector 'w' times across columns
        row_embed = jnp.repeat(row_embed, w, axis=1)  # (h, w, D/2)
        # Repeat col vector 'h' times across rows
        col_embed = jnp.repeat(col_embed, h, axis=0)  # (h, w, D/2)

        # 3. Concatenate
        pe_2d = jnp.concatenate([row_embed, col_embed], axis=-1)  # (h, w, D)

        # 4. Flatten
        res = pe_2d.reshape(1, h * w, self.hidden_size)
        return jnp.broadcast_to(res, (ids.shape[0], h * w, self.hidden_size))


class Embed(nnx.Module):
    """
    Wrapper around nnx.Embed that handles 3D input by removing the last dimension.
    
    Parameters
    ----------
    *args
        Positional arguments passed to nnx.Embed.
    **kwargs
        Keyword arguments passed to nnx.Embed.
    """
    def __init__(self, *args, **kwargs):
        self.embed = nnx.Embed(*args, **kwargs)

    def __call__(self, ids):
        """
        Apply embedding to input IDs.

        Parameters
        ----------
        ids : Array
            Input IDs with shape (batch, seq_len, 1).

        Returns
        -------
        Array
            Embedded output.
        """
        assert ids.ndim == 3, f"ids must have 3 dimensions, got {ids.ndim}"
        return self.embed(ids[..., 0])  # remove last dimension


class FeatureEmbedder(nnx.Module):
    """
    General Feature Embedder supporting learned, 1D sinusoidal, and 2D sinusoidal embeddings.
    1D sinusoidal embeddings are suitable for sequences, while 2D sinusoidal embeddings are ideal for grid-like data (e.g., images).

    Parameters
    ----------
    num_embeddings : int
        Number of embeddings.
    hidden_size : int
        Hidden size/embedding dimension.
    kind : str, optional
        Type of embedding: 'absolute', 'pos1d', or 'pos2d'. Defaults to 'absolute'.
    param_dtype : DTypeLike, optional
        Data type for parameters. Defaults to jnp.float32.
    rngs : nnx.Rngs, optional
        Random number generators for initialization.
    **kwargs
        Additional keyword arguments specific to the embedding type.
    """

    def __init__(
        self,
        num_embeddings: int,
        hidden_size: int,
        *,
        kind="absolute",
        param_dtype=jnp.float32,
        rngs: nnx.Rngs = None,
        **kwargs,
    ):
        if kind == "absolute":
            self.embedder = Embed(
                num_embeddings=num_embeddings,
                features=hidden_size,
                rngs=rngs,
                param_dtype=param_dtype,
                **kwargs,
            )
        elif kind == "pos1d":
            max_len = kwargs.pop("max_len", 5000)
            max_len = jnp.maximum(num_embeddings, max_len)
            self.embedder = SinusoidalPosEmbed1D(
                hidden_size=hidden_size,
                param_dtype=param_dtype,
                max_len=max_len,
                **kwargs,
            )
        elif kind == "pos2d":
            max_h = kwargs.pop("max_h", 256)
            max_w = kwargs.pop("max_w", 256)
            max_h = jnp.maximum(num_embeddings, max_h)
            max_w = jnp.maximum(num_embeddings, max_w)
            self.embedder = SinusoidalPosEmbed2D(
                hidden_size=hidden_size,
                max_h=max_h,
                max_w=max_w,
                param_dtype=param_dtype,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Unknown embedding kind {kind}, expected 'learned', 'pos1d' or 'pos2d'"
            )
        return

    def __call__(self, ids):
        """
        Apply feature embedding to input IDs.

        Parameters
        ----------
        ids : Array
            Input IDs.

        Returns
        -------
        Array
            Embedded features.
        """
        return self.embedder(ids)
