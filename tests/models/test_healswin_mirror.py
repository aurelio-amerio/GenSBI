import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx


def test_mirror_reexports_are_the_same_objects():
    import heal_swin_nnx
    from gensbi.models import HealSwinEncoder, HealSwinParams
    from gensbi.models import healswin

    assert HealSwinEncoder is heal_swin_nnx.HealSwinEncoder
    assert HealSwinParams is heal_swin_nnx.HealSwinParams
    assert set(healswin.__all__) == set(heal_swin_nnx.__all__)


def test_healswin_encoder_tiny_forward_via_gensbi():
    from gensbi.models import HealSwinEncoder, HealSwinParams

    # Known-good tiny config from HEAL-SWIN-nnx's own test suite
    # (tests/test_model.py::tiny_params): 8 faces at nside 16, 2 stages.
    p = HealSwinParams(
        nside=16, in_channels=3, out_channels=5, base_pixels=tuple(range(8)),
        embed_dim=16, depths=(2, 2), num_heads=(2, 4), drop_path_rate=0.0,
    )
    enc = HealSwinEncoder(p, rngs=nnx.Rngs(0))
    enc.eval()
    tokens, skips = enc(jnp.ones((1, p.npix, 3)))
    # N/(patch * 4^(L-1)) tokens, embed_dim * 2^(L-1) features
    assert tokens.shape == (1, p.npix // 4 // 4, 32)
