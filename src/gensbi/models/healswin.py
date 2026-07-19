"""Mirror of the standalone ``heal_swin_nnx`` package.

GenSBI depends on `heal-swin-nnx <https://pypi.org/p/heal-swin-nnx>`_ (the
HEALPix-native spherical Swin V2 U-Net in Flax NNX) and re-exports its public
API here, so spherical encoders are importable alongside the other gensbi
models. The SBI-relevant names — :class:`HealSwinEncoder` and
:class:`HealSwinParams` — are also exported from :mod:`gensbi.models`
directly; everything else (full U-Nets, decoders, HealConv, planar Swin) is
available from this module or from ``heal_swin_nnx`` itself.
"""

from heal_swin_nnx import (
    Buffer,
    HealConv,
    HealConvDecoder,
    HealConvEncoder,
    HealConvParams,
    HealSwin,
    HealSwinDecoder,
    HealSwinEncoder,
    HealSwinParams,
    SwinDecoder,
    SwinEncoder,
    SwinParams,
    SwinUnet,
)

__all__ = [
    "Buffer",
    "HealConv",
    "HealConvDecoder",
    "HealConvEncoder",
    "HealConvParams",
    "HealSwin",
    "HealSwinDecoder",
    "HealSwinEncoder",
    "HealSwinParams",
    "SwinDecoder",
    "SwinEncoder",
    "SwinParams",
    "SwinUnet",
]
