import jax.numpy as jnp
from gensbi.models.maf.masks import make_mask


def test_make_mask_non_strict_is_ge():
    # in_ranks = [0,1,2], out_ranks = [0,1,2]; connect i->o if out_rank >= in_rank
    in_ranks = jnp.array([0, 1, 2])
    out_ranks = jnp.array([0, 1, 2])
    mask = make_mask(in_ranks, out_ranks, strict=False)  # shape (in=3, out=3)
    expected = jnp.array([
        [True,  True,  True],   # in-rank 0 -> out-ranks >= 0 : all
        [False, True,  True],   # in-rank 1 -> out-ranks >= 1
        [False, False, True],   # in-rank 2 -> out-ranks >= 2
    ])
    assert mask.shape == (3, 3)
    assert jnp.array_equal(mask, expected)


def test_make_mask_strict_is_gt():
    in_ranks = jnp.array([0, 1, 2])
    out_ranks = jnp.array([0, 1, 2])
    mask = make_mask(in_ranks, out_ranks, strict=True)  # connect if out_rank > in_rank
    expected = jnp.array([
        [False, True,  True],
        [False, False, True],
        [False, False, False],
    ])
    assert jnp.array_equal(mask, expected)


def test_make_mask_rectangular():
    in_ranks = jnp.array([0, 1])          # 2 inputs
    out_ranks = jnp.array([0, 0, 1, 1])   # 4 outputs
    mask = make_mask(in_ranks, out_ranks, strict=True)
    assert mask.shape == (2, 4)
    # input rank 1 connects to no output (no out_rank > 1)
    assert not mask[1].any()
    # input rank 0 connects to the two rank-1 outputs only (strict)
    assert jnp.array_equal(mask[0], jnp.array([False, False, True, True]))
