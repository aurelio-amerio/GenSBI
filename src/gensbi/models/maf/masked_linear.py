"""Dense layer with a fixed binary weight mask."""

import jax.numpy as jnp
from flax import nnx
from jax import Array
from jax.typing import DTypeLike

from gensbi.normalizing_flows.bijections.base import Mask


class MaskedLinear(nnx.Module):
    """``y = (kernel * mask).T @ x + bias`` with a non-trainable mask.

    Parameters
    ----------
    in_features, out_features : int
    mask : Array
        Boolean array of shape ``(in_features, out_features)``; stored as a
        :class:`Mask` buffer so it is excluded from ``nnx.Param``.
    rngs : nnx.Rngs
    param_dtype : DTypeLike, optional
        Dtype for the stored (master) kernel/bias parameters. Defaults to
        float32 (exact-likelihood model needs the precision).
    dtype : DTypeLike, optional
        Compute dtype: the kernel/bias/activations are cast to this dtype
        before the matmul. Defaults to float32, matching ``param_dtype``, so
        with default arguments this is a no-op cast (bit-identical).
    """

    def __init__(self, in_features, out_features, mask, rngs,
                 param_dtype: DTypeLike = jnp.float32,
                 dtype: DTypeLike = jnp.float32):
        self.dtype = dtype
        self.linear = nnx.Linear(
            in_features, out_features, use_bias=True,
            rngs=rngs, param_dtype=param_dtype, dtype=dtype,
        )
        self.mask = Mask(jnp.asarray(mask, dtype=param_dtype))

    def __call__(self, x: Array) -> Array:
        """Apply the masked linear transform ``y = (kernel * mask).T @ x + bias``.

        Parameters
        ----------
        x : Array
            Input of shape ``(in_features,)``.

        Returns
        -------
        Array
            Output of shape ``(out_features,)``.
        """
        masked_kernel = (self.linear.kernel[...] * self.mask[...]).astype(self.dtype)
        x = x.astype(self.dtype)
        bias = self.linear.bias[...].astype(self.dtype)
        return x @ masked_kernel + bias
