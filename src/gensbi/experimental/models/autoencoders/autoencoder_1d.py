import jax
import jax.numpy as jnp

# from chex import Array
from jax import Array
from einops import rearrange
from flax import nnx
from jax.typing import DTypeLike, ArrayLike

from gensbi.experimental.models.autoencoders.commons import AutoEncoderParams, DiagonalGaussian
from flax.nnx import swish



class AttnBlock1D(nnx.Module):
    """
    1D Self-attention block for sequence data.

    Parameters
    ----------
        in_channels : int
            Number of input channels.
        rngs : nnx.Rngs
            Random number generators for parameter initialization.
        param_dtype : DTypeLike
            Data type for parameters (default: jnp.bfloat16).
    """
    def __init__(
        self,
        in_channels: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ) -> None:
        self.in_channels = in_channels

        self.norm = nnx.GroupNorm(
            num_groups=32,
            num_features=in_channels,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        self.q = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1,),
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.k = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1,),
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.v = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1,),
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.proj_out = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(1,),
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def attention(self, h_: Array) -> Array:
        """
        Compute self-attention for 1D input.

        Parameters
        ----------
            h_ : Array
                Input tensor of shape (batch, length, channels).

        Returns
        -------
            Array
                Output tensor after attention.
        """
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, n, c = q.shape
        q = rearrange(q, "b n c -> b n 1 c")
        k = rearrange(k, "b n c -> b n 1 c")
        v = rearrange(v, "b n c -> b n 1 c")

        # Calculate Attention
        h_ = jax.nn.dot_product_attention(q, k, v)

        h_ = rearrange(h_, "b n 1 c -> b n c", n=n, c=c, b=b)
        return h_

    def __call__(self, x: Array) -> Array:
        """
        Forward pass for the attention block.

        Parameters
        ----------
            x : Array
                Input tensor.

        Returns
        -------
            Array
                Output tensor after residual attention.
        """
        return x + self.proj_out(self.attention(x))


class ResnetBlock1D(nnx.Module):
    """
    1D Residual block with optional channel up/downsampling.

    Parameters
    ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        rngs : nnx.Rngs
            Random number generators for parameter initialization.
        param_dtype : DTypeLike
            Data type for parameters (default: jnp.bfloat16).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = nnx.GroupNorm(
            num_groups=32,
            num_features=in_channels,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3,),
            strides=(1,),
            padding=(1,),
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.norm2 = nnx.GroupNorm(
            num_groups=32,
            num_features=out_channels,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.conv2 = nnx.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(3,),
            strides=(1,),
            padding=(1,),
            rngs=rngs,
            param_dtype=param_dtype,
        )
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(1,),
                strides=(1,),
                padding=(0,),
                rngs=rngs,
                param_dtype=param_dtype,
            )

    def __call__(self, x: Array) -> Array:
        """
        Forward pass for the residual block.

        Parameters
        ----------
            x : Array
                Input tensor.

        Returns
        -------
            Array
                Output tensor after residual connection.
        """
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample1D(nnx.Module):
    """
    1D Downsampling block using strided convolution.

    Parameters
    ----------
        in_channels : int
            Number of input channels.
        rngs : nnx.Rngs
            Random number generators for parameter initialization.
        param_dtype : DTypeLike
            Data type for parameters (default: jnp.bfloat16).
    """
    def __init__(
        self,
        in_channels: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ):
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3,),
            strides=(2,),
            padding=(0,),
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, x: Array) -> Array:
        """
        Downsample the input tensor by a factor of 2.

        Parameters
        ----------
            x : Array
                Input tensor of shape (batch, length, channels).

        Returns
        -------
            Array
                Downsampled tensor.
        """
        # Pad feature dimension (axis 1)
        pad_width = ((0, 0), (0, 1), (0, 0))
        x = jnp.pad(array=x, pad_width=pad_width, mode="constant", constant_values=0)
        x = self.conv(x)
        return x


class Upsample1D(nnx.Module):
    """
    1D Upsampling block using nearest-neighbor interpolation and convolution.

    Parameters
    ----------
        in_channels : int
            Number of input channels.
        rngs : nnx.Rngs
            Random number generators for parameter initialization.
        param_dtype : DTypeLike
            Data type for parameters (default: jnp.bfloat16).
    """
    def __init__(
        self,
        in_channels: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ):
        self.conv = nnx.Conv(
            in_features=in_channels,
            out_features=in_channels,
            kernel_size=(3,),
            strides=(1,),
            padding=(1,),
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, x: Array) -> Array:
        """
        Upsample the input tensor by a factor of 2.

        Parameters
        ----------
            x : Array
                Input tensor of shape (batch, length, channels).

        Returns
        -------
            Array
                Upsampled tensor.
        """
        # Assuming `x` is a 3D tensor with shape (batch, n, c)
        scale_factor = 2.0
        b, n, c = x.shape
        new_n = int(n * scale_factor)
        new_shape = (b, new_n, c)

        # Resize using nearest-neighbor interpolation
        x = jax.image.resize(x, new_shape, method="nearest")
        x = self.conv(x)
        return x


class Encoder1D(nnx.Module):
    """
    1D Encoder for autoencoder architectures.

    Parameters
    ----------
        resolution : int
            Input sequence length.
        in_channels : int
            Number of input channels.
        ch : int
            Base number of channels.
        ch_mult : list[int]
            Channel multipliers for each resolution.
        num_res_blocks : int
            Number of residual blocks per resolution.
        z_channels : int
            Number of latent channels.
        rngs : nnx.Rngs
            Random number generators for parameter initialization.
        param_dtype : DTypeLike
            Data type for parameters (default: jnp.bfloat16).
    """
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ) -> None:
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = nnx.Conv(
            in_features=in_channels,
            out_features=self.ch,
            kernel_size=(3,),
            strides=(1,),
            padding=(1,),
            rngs=rngs,
            param_dtype=param_dtype,
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nnx.Sequential()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nnx.Sequential()
            attn = nnx.Sequential()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.layers.append(
                    ResnetBlock1D(
                        in_channels=block_in,
                        out_channels=block_out,
                        rngs=rngs,
                        param_dtype=param_dtype,
                    )
                )
                block_in = block_out
            down = nnx.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.Downsample1D = Downsample1D(
                    in_channels=block_in,
                    rngs=rngs,
                    param_dtype=param_dtype,
                )
                curr_res = curr_res // 2
            self.down.layers.append(down)

        # middle
        self.mid = nnx.Module()
        self.mid.block_1 = ResnetBlock1D(
            in_channels=block_in,
            out_channels=block_in,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.mid.attn_1 = AttnBlock1D(
            in_channels=block_in,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.mid.block_2 = ResnetBlock1D(
            in_channels=block_in,
            out_channels=block_in,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        # end
        self.norm_out = nnx.GroupNorm(
            num_groups=32,
            num_features=block_in,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.conv_out = nnx.Conv(
            in_features=block_in,
            out_features=2 * z_channels,
            kernel_size=(3,),
            strides=(1,),
            padding=(1,),
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, x: Array) -> Array:
        """
        Forward pass for the encoder.

        Parameters
        ----------
            x : Array
                Input tensor of shape (batch, length, channels).

        Returns
        -------
            Array
                Encoded latent representation.
        """
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down.layers[i_level].block.layers[i_block](hs[-1])
                if len(self.down.layers[i_level].attn.layers) > 0:
                    h = self.down.layers[i_level].attn.layers[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down.layers[i_level].Downsample1D(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class Decoder1D(nnx.Module):
    """
    1D Decoder for autoencoder architectures.

    Parameters
    ----------
        ch : int
            Base number of channels.
        out_ch : int
            Number of output channels.
        ch_mult : list[int]
            Channel multipliers for each resolution.
        num_res_blocks : int
            Number of residual blocks per resolution.
        in_channels : int
            Number of input channels.
        resolution : int
            Output sequence length.
        z_channels : int
            Number of latent channels.
        rngs : nnx.Rngs
            Random number generators for parameter initialization.
        param_dtype : DTypeLike
            Data type for parameters (default: jnp.bfloat16).
    """
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ):
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, curr_res, z_channels) #(1, z_channels, curr_res)

        # z to block_in
        self.conv_in = nnx.Conv(
            in_features=z_channels,
            out_features=block_in,
            kernel_size=(3,),
            strides=(1,),
            padding=(1,),
            rngs=rngs,
            param_dtype=param_dtype,
        )

        # middle
        self.mid = nnx.Module()
        self.mid.block_1 = ResnetBlock1D(
            in_channels=block_in,
            out_channels=block_in,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.mid.attn_1 = AttnBlock1D(
            in_channels=block_in,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.mid.block_2 = ResnetBlock1D(
            in_channels=block_in,
            out_channels=block_in,
            rngs=rngs,
            param_dtype=param_dtype,
        )

        # upsampling
        self.up = nnx.Sequential()
        for i_level in reversed(range(self.num_resolutions)):
            block = nnx.Sequential()
            attn = nnx.Sequential()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.layers.append(
                    ResnetBlock1D(
                        in_channels=block_in,
                        out_channels=block_out,
                        rngs=rngs,
                        param_dtype=param_dtype,
                    )
                )
                block_in = block_out
            up = nnx.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.Upsample1D = Upsample1D(
                    in_channels=block_in,
                    rngs=rngs,
                    param_dtype=param_dtype,
                )
                curr_res = curr_res * 2
            self.up.layers.insert(0, up)

        # end
        self.norm_out = nnx.GroupNorm(
            num_groups=32,
            num_features=block_in,
            epsilon=1e-6,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.conv_out = nnx.Conv(
            in_features=block_in,
            out_features=out_ch,
            kernel_size=(3,),
            strides=(1,),
            padding=(1,),
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, z: Array) -> Array:
        """
        Forward pass for the decoder.

        Parameters
        ----------
            z : Array
                Latent tensor of shape (batch, latent_length, latent_channels).

        Returns
        -------
            Array
                Reconstructed output tensor.
        """
        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up.layers[i_level].block.layers[i_block](h)
                if len(self.up.layers[i_level].attn.layers) > 0:
                    h = self.up.layers[i_level].attn.layers[i_block](h)
            if i_level != 0:
                h = self.up.layers[i_level].Upsample1D(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h



class AutoEncoder1D(nnx.Module):
    """
    1D Autoencoder model with Gaussian latent space.

    Parameters
    ----------
        params : AutoEncoderParams
            Configuration parameters for the autoencoder.
    """
    def __init__(
        self,
        params: AutoEncoderParams,
    ):
        self.rngs = params.rngs
        self.Encoder1D = Encoder1D(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
            rngs=self.rngs,
            param_dtype=params.param_dtype,
        )
        self.Decoder1D = Decoder1D(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
            rngs=self.rngs,
            param_dtype=params.param_dtype,
        )
        self.reg = DiagonalGaussian()
        
        self.scale_factor = nnx.Param(jnp.array(params.scale_factor))
        self.shift_factor = nnx.Param(jnp.array(params.shift_factor))
        
        
        self.latent_shape = (1, params.resolution // (2 ** (len(params.ch_mult) - 1)), params.z_channels)

    def encode(self, x: Array, key=None) -> Array:
        """
        Encode input data into the latent space.

        Parameters
        ----------
            x : Array
                Input tensor.
            key : Array
                PRNG key for sampling the latent variable.

        Returns
        -------
            Array
                Latent representation.
        """
        if key is None:
            key = self.rngs.encode()
        z = self.reg(self.Encoder1D(x), key)
        z = self.scale_factor * (z - self.shift_factor)

        return z

    def decode(self, z: Array) -> Array:
        """
        Decode latent representation back to data space.

        Parameters
        ----------
            z : Array
                Latent tensor.

        Returns
        -------
            Array
                Reconstructed output.
        """

        z = z / self.scale_factor + self.shift_factor
        z = self.Decoder1D(z)

        return z

    def __call__(self, x: Array, key=None) -> Array:
        """
        Forward pass: encode and then decode the input.

        Parameters
        ----------
            x : Array
                Input tensor.

        Returns
        -------
            Array
                Reconstructed output.
        """
        return self.decode(self.encode(x, key))
