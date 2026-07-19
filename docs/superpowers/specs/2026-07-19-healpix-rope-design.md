# HEALPix RoPE: spherical rotary position embedding for Flux1

Status: approved design (brainstormed 2026-07-19)
Target: `gensbi` (Flux1 conditioning stream), first consumer HEAL-SWIN-nnx GRF example

## Goal

Give Flux1 a positional encoding for tokens that live on the HEALPix sphere, so
that attention sees true spherical geometry (geodesic relationships) instead of
a 1D NEST-order index. First application: HEAL-SWIN encoder bottleneck tokens
(nside 2–4, NEST order, optionally a `base_pixels` subset) feeding Flux1's
conditioning stream in the spherical GRF flow-matching example, replacing the
current `init_ids_1d` sinusoidal ids.

## Literature findings (2026-07-19)

- No published RoPE formulation on HEALPix exists. HEAL-SWIN (CVPR 2024) and
  HEAL-ViT use plain 1D learnable/sinusoidal embeddings over NEST order.
- STRATA (arXiv:2606.31248) reports naive 2D RoPE on HEALPix face coordinates
  produces discontinuous fields across polar faces; their fix (StereoRoPE) is
  per-tile stereographic projection on a cubed-sphere.
- SpheRoPE (arXiv:2606.32033) re-parameterizes RoPE channels with 3D Cartesian
  coordinates on the sphere for 360° panorama generation with pretrained FLUX —
  validating Cartesian-coordinates-as-RoPE-input. Its low/high frequency hybrid
  is a compromise for pretrained-model reuse we do not need (we train from
  scratch). ERP grid, not HEALPix.
- Exactly rotation-invariant rotary encodings cannot exist: RoPE requires a
  commuting family of rotations, and SO(3) is non-abelian. Published
  alternatives (geodesic attention biases, Wigner-D equivariant attention) are
  not rotary and require attention surgery. Rotation invariance is explicitly
  out of scope (HEAL-SWIN itself is not rotation-equivariant); the requirement
  is geodesic faithfulness: no projection artifacts.

## Formulation

**Coordinates.** Token i ↦ global HEALPix pixel p (via
`local_to_global(base_pixels, nside, i)` semantics for subsets) ↦ pixel-center
unit vector n_i = pix2vec(nside, p, nest=True) ∈ S² ⊂ ℝ³.

**Pixel-unit scaling.** ids_i = r(nside) · n_i with r(nside) ≈ 1/pixel_size
(pixel angular size ≈ 1.023/nside rad), so adjacent bottleneck tokens differ by
~1 in coordinate — the same convention standard RoPE assumes for integer token
indices. This keeps theta's semantics identical to 2D-image usage.

**Encoding.** The three Cartesian components are three continuous axes for the
existing `EmbedND`: `axes_dim = (d1, d2, d3)`, each even, summing to head dim
(e.g. `(22, 22, 20)` for 64). `rope()` already accepts float positions; no
change to `rope()`, `apply_rope()`, or attention.

**Theta from nside.** The project convention sets theta from resolution
(theta ≈ 10 × pixel count for 2D images). Spherical analogue:
`healpix_rope_theta(nside)` derives theta from nside so the frequency spectrum
spans pixel scale → sphere scale. nside (always known from the HEAL-SWIN
bottleneck) becomes the user-facing knob; theta remains a plain `Flux1Params`
passthrough supplied by the helper. Exact formula (total-pixel vs per-axis
extent generalization of the 10× rule) is pinned at implementation with a
phase-spectrum check.

**Properties.**
- Attention modulation depends on positions only through the chord vector
  Δ = n_q − n_k; |Δ| = 2 sin(γ/2) is strictly monotone in great-circle
  distance γ. Geodesic geometry enters exactly and smoothly — no projection.
- No face-seam discontinuities (adjacent pixels on different base faces have
  nearly equal n), no polar artifacts, exact longitude periodicity.
- `base_pixels` subsets and any nside work by construction (encoding depends
  only on directions). Full sky is the tested target; subsets must not break.
- Not rotation-invariant: encodes absolute directions plus displacement
  direction in a fixed global frame (accepted; same status as all published
  spherical RoPEs).
- Obs/θ-stream tokens use ids (0, 0, 0): the origin gives the identity
  rotation — a well-defined neutral position (replaces the 1D offset
  convention for spherical models; the "absolute" learned strategy remains
  available).

## Relation to prior work (docstring requirement)

This is NOT an adaptation of SpheRoPE. It is standard N-dimensional RoPE (the
RoFormer mechanism, as implemented by Flux1's `EmbedND`) applied uniformly —
all axes, all frequency bands — to 3D Cartesian coordinates of HEALPix pixel
centers. SpheRoPE's low/high frequency partition, harmonic quantization, and
horizontal-only Cartesian re-parameterization are compromises for zero-shot
reuse of a pretrained ERP-grid FLUX; trained from scratch, none apply. The
`init_ids_healpix` docstring must state the method in these terms and link:

- SpheRoPE, arXiv:2606.32033 — closest prior work: validates Cartesian
  coordinates as RoPE inputs for spherical topology (ERP grid, pretrained
  constraints; we drop those compromises).
- STRATA/StereoRoPE, arXiv:2606.31248 — documents the failure of index-based
  RoPE on HEALPix (discontinuities across polar faces) that motivates this
  design.
- RoFormer, arXiv:2104.09864 — the underlying rotary mechanism.

## Components

**`src/gensbi/recipes/utils.py`** (next to `init_ids_1d`):
- `init_ids_healpix(nside, base_pixels=None)` → `(1, N, 3)` float32 ids in
  NEST order over the selected base pixels (full sky when `base_pixels=None`,
  N = 12·nside²). Matches `init_ids_1d`'s return convention (verify tuple
  shape at implementation; if compatible, return the suggested theta in the
  second slot so units and theta cannot drift apart).
- `healpix_rope_theta(nside)` → suggested theta (exported; also usable as the
  default).

**Dependency: `healpy>=1.19` (regular dep, lazy import).** Directions are a
one-time host-side precompute per (nside, base_pixels): no jit, no gradients —
nothing for JAX to do. healpy computes NEST arithmetic in int64/double (CMB
community ground truth); final ids cast to float32 once. This avoids
jax-healpy's global `jax_enable_x64` recommendation entirely (a library must
not flip global precision flags). Dependency cost is only astropy given
matplotlib/scipy/numpy already required. HEAL-SWIN-nnx already depends on
healpy, so the first consumer gets it for free. Revisit jax-healpy only if
call-time traceable pixel functions are ever needed.

**Flux1: no attention changes.** 3-entry `axes_dim` in config;
`cond_ids`/`obs_ids` are existing passthroughs (verify the
`id_embedding_strategy` path accepts caller-supplied geometric ids as-is).

**HEAL-SWIN-nnx example `spherical_grf_flowmatch.py`:** config switch to
select spherical cond ids (`init_ids_healpix(2)`, `FLUX_AXES_DIM=(22,22,20)`,
theta from helper) with the old `pos1d` path kept selectable for A/B on the
GRF posterior (TARP + marginals). Encoder untouched (already emits NEST-order
tokens).

## Testing

1. Geometry: unscaled ids are unit-norm; NEST spot checks against healpy;
   `base_pixels` subset rows equal the corresponding full-sky rows.
2. Seam-freedom: for neighboring pixel pairs straddling different base faces
   (nside 2–4), coordinate distance equals chord distance — no
   face-discontinuity (the StereoRoPE-reported failure, as a test).
3. RoPE integration: 3-axis `EmbedND` → `apply_rope` shapes/finiteness; equal
   great-circle separation at different sky locations differs only through
   Δ direction (tolerance-based property test).
4. Flux1 e2e smoke: tiny model, spherical cond ids + origin obs ids, forward +
   backward, finite loss.
5. Acceptance evidence: GRF example A/B (spherical vs pos1d), full run on GPU
   by the user; CI keeps the `QUICK` smoke.

## Out of scope / future

- Rotation-invariant or zonal-spectral (Legendre/addition-theorem) attention
  bias — the mathematically exact route; requires attention surgery. Documented
  as a possible future extension.
- Lat/lon toroidal RoPE ablation, jax-healpy backend, TarFlow/pixelDiT
  consumers, fine-nside token grids.
