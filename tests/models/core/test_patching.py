import jax.numpy as jnp
from gensbi.models.core.patching import patchify_2d, depatchify_2d


def test_patchify_shape_and_roundtrip():
    x = jnp.arange(1 * 4 * 4 * 2).reshape(1, 4, 4, 2).astype(jnp.float32)
    p = patchify_2d(x, size=2)
    assert p.shape == (1, 4, 2 * 2 * 2)          # (B, (h w), C*ph*pw)
    xr = depatchify_2d(p, size=2)
    assert jnp.allclose(xr, x)


def test_depatchify_nonsquare_requires_grid():
    p = jnp.zeros((1, 6, 8))                       # 6 = 3*2 patches, not square
    xr = depatchify_2d(p, size=2, grid=(2, 3))
    assert xr.shape == (1, 4, 6, 2)
