"""Golden-parity and property tests for the faithful RoPE port.

Golden values were generated once from the torch reference
(reference/ml-starflow/misc/pe.py, torch 2.5.1+cpu) and hard-coded here;
the test env has no torch. Generator: scratchpad/gen_rope_goldens.py
(session 2026-07-11).
"""

import jax.numpy as jnp
import numpy as np

from gensbi.models.tarflow.pe import (
    VisionRotaryEmbedding, apply_rope, get_positions, rotate_half,
)

# --- goldens from the torch reference (do not edit) -------------------------

# get_positions(h=2, w=3, txt_size=0, pt_seq_len=4, mode='2d')
GOLDEN_POS_2X3_PT4 = np.array(
    [[0.0, 0.0], [0.0, 1.6329931020736694], [0.0, 3.265986204147339],
     [1.6329931020736694, 0.0], [1.6329931020736694, 1.6329931020736694],
     [1.6329931020736694, 3.265986204147339]])

# get_positions(h=4, w=4, txt_size=0, pt_seq_len=4, mode='2d')
GOLDEN_POS_4X4_PT4 = np.array(
    [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0], [0.0, 3.0],
     [1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0],
     [2.0, 0.0], [2.0, 1.0], [2.0, 2.0], [2.0, 3.0],
     [3.0, 0.0], [3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])

# VisionRotaryEmbeddingFast(dim=4, pt_seq_len=4, no_buffer=True).freqs
GOLDEN_BASE_FREQS_DIM4 = np.array([1.0, 0.009999999776482582])

# rope(GOLDEN_POS_2X3_PT4)  -> (6, 8)
GOLDEN_FREQS_CIS_2X3_DIM4 = np.array(
    [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0, 1.6329931020736694, 1.6329931020736694,
      0.0163299310952425, 0.0163299310952425],
     [0.0, 0.0, 0.0, 0.0, 3.265986204147339, 3.265986204147339,
      0.032659862190485, 0.032659862190485],
     [1.6329931020736694, 1.6329931020736694, 0.0163299310952425,
      0.0163299310952425, 0.0, 0.0, 0.0, 0.0],
     [1.6329931020736694, 1.6329931020736694, 0.0163299310952425,
      0.0163299310952425, 1.6329931020736694, 1.6329931020736694,
      0.0163299310952425, 0.0163299310952425],
     [1.6329931020736694, 1.6329931020736694, 0.0163299310952425,
      0.0163299310952425, 3.265986204147339, 3.265986204147339,
      0.032659862190485, 0.032659862190485]])

# t = cos(0.1*arange(48)).reshape(6, 8); apply_rope(t, GOLDEN_FREQS_CIS_2X3_DIM4)
GOLDEN_APPLY_ROPE_OUT = np.array(
    [[1.0, 0.9950041770935059, 0.9800665974617004, 0.9553365111351013,
      0.9210609793663025, 0.8775825500488281, 0.8253356218338013,
      0.7648422122001648],
     [0.6967067122459412, 0.6216099262237549, 0.5403023362159729,
      0.4535961151123047, -0.2895044982433319, 0.3450302183628082,
      0.16878941655158997, 0.07350319623947144],
     [-0.029199546203017235, -0.12884454429149628, -0.2272021621465683,
      -0.32328954339027405, 0.3502935469150543, 0.5525779128074646,
      -0.5664307475090027, -0.6851376295089722],
     [0.845428466796875, -0.6861715912818909, -0.8420118093490601,
      -0.9179439544677734, -0.9422223567962646, -0.9709581732749939,
      -0.9899924993515015, -0.9991351366043091],
     [1.0476211309432983, -0.9349859952926636, -0.9513776898384094,
      -0.9521188735961914, 0.9021996855735779, -0.8423093557357788,
      -0.7790084481239319, -0.7387512922286987],
     [0.6143409013748169, -0.6166505813598633, -0.48365047574043274,
      -0.40875113010406494, 0.2788039743900299, 0.2472987025976181,
      -0.1116882860660553, -0.0160440094769001]])


def test_get_positions_nonsquare_matches_reference():
    pos = get_positions(h=2, w=3, pt_seq_len=4)
    assert pos.shape == (6, 2)
    assert np.allclose(np.asarray(pos), GOLDEN_POS_2X3_PT4, atol=1e-6)


def test_get_positions_square_is_integer_grid():
    """pt_seq_len == h == w -> plain integer raster coordinates."""
    pos = get_positions(h=4, w=4, pt_seq_len=4)
    assert np.allclose(np.asarray(pos), GOLDEN_POS_4X4_PT4, atol=1e-6)


def test_base_freqs_match_reference():
    rope = VisionRotaryEmbedding(dim=4, pt_seq_len=4)
    assert np.allclose(np.asarray(rope.freqs[...]), GOLDEN_BASE_FREQS_DIM4,
                       atol=1e-9)


def test_freqs_cis_match_reference():
    rope = VisionRotaryEmbedding(dim=4, pt_seq_len=4)
    freqs = rope(jnp.asarray(GOLDEN_POS_2X3_PT4, dtype=jnp.float32))
    assert freqs.shape == (6, 8)
    assert np.allclose(np.asarray(freqs), GOLDEN_FREQS_CIS_2X3_DIM4, atol=1e-5)


def test_apply_rope_matches_reference():
    t = jnp.cos(0.1 * jnp.arange(48, dtype=jnp.float32)).reshape(6, 8)
    freqs = jnp.asarray(GOLDEN_FREQS_CIS_2X3_DIM4, dtype=jnp.float32)
    out = apply_rope(t, freqs)
    assert np.allclose(np.asarray(out), GOLDEN_APPLY_ROPE_OUT, atol=1e-5)


def test_apply_rope_at_zero_position_is_identity():
    """Rotation by zero freqs is a no-op — the prefix-token property."""
    t = jnp.cos(0.1 * jnp.arange(48, dtype=jnp.float32)).reshape(6, 8)
    out = apply_rope(t, jnp.zeros((6, 8), dtype=jnp.float32))
    assert np.allclose(np.asarray(out), np.asarray(t), atol=1e-7)


def test_rotate_half_twice_negates():
    x = jnp.arange(16, dtype=jnp.float32).reshape(2, 8)
    assert np.allclose(np.asarray(rotate_half(rotate_half(x))),
                       -np.asarray(x), atol=1e-7)
