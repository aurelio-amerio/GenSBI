# Handoff: HEALPix RoPE landed — and the ConditionalPipeline ids gap it exposed

**Date:** 2026-07-19. **Purpose:** self-contained context for a new brainstorming session
on how GenSBI's recipe pipelines should support non-1D/2D (e.g. spherical) positional ids
properly, replacing the example-level workaround described below.

## 1. What was done (merged to GenSBI main)

Spherical rotary position embedding for HEALPix-grid tokens feeding Flux1's conditioning
stream: standard N-d RoPE applied to the 3D Cartesian coordinates of HEALPix pixel
centers, per the approved spec `docs/superpowers/specs/2026-07-19-healpix-rope-design.md`
and plan `docs/superpowers/plans/2026-07-19-healpix-rope.md`.

- **GenSBI** (merged, fast-forward `25b1d4a..dabd6c5`, branch deleted; main not yet
  pushed to origin at time of writing):
  - `healpix_rope_theta(nside) -> int` — `src/gensbi/recipes/utils.py:56`. Project
    convention `theta = 10 * token count` = `10 * 12 * nside**2`.
  - `init_ids_healpix(nside, base_pixels=None) -> (ids, num_tokens)` —
    `src/gensbi/recipes/utils.py:69`. Returns float32 `(1, N, 3)` NEST-order
    pixel-center unit vectors scaled to pixel units (radius `nside*sqrt(3/pi)`, so
    adjacent tokens differ by ~1 — keeps theta semantics identical to 2D-image usage).
    Lazy `import healpy` inside the function. Optional `base_pixels` subset for
    partial-sky grids.
  - `tests/recipes/test_healpix_ids.py` — 9 tests: theta convention; shape/dtype/scale;
    healpy `vec2pix` NEST round-trip oracle; nside/base_pixels validation; subset ids ==
    corresponding full-sky rows; face-seam-freedom property test (neighbor chord-distance
    max/min ratio measured 2.0 vs 4.0 bound — the StereoRoPE-documented index-RoPE
    failure mode does not occur); EmbedND consumption; full Flux1 forward.
  - **Zero changes to `src/gensbi/models/flux1/`** — the existing model consumes 3-axis
    float ids unmodified via `id_embedding_strategy=("absolute", "rope")` + 3-entry
    `axes_dim` (each even, summing to per-head dim, e.g. `(22, 22, 20)` for 64). Obs
    tokens get dummy zero rope ids (model.py ~427) = identity rotation = isotropic
    positional readout. Verified: int32 obs dummies × float32 cond ids concat promotes
    to float32 under JAX's lattice; zeros stay exact.
  - `pyproject.toml`: `healpy>=1.19.0` added as a regular dependency.

- **HEAL-SWIN-nnx** (`/lhome/ific/a/aamerio/data/github/HEAL-SWIN-nnx`, branch
  `healpix-rope` KEPT un-merged, single commit `0748eda`): first consumer.
  `examples/spherical_grf_flowmatch.py` gains a `COND_ID_KIND` switch (line 96):
  `"healpix"` (spherical RoPE, `ID_EMBEDDING=("absolute","rope")`, `FLUX_AXES_DIM=(22,22,20)`,
  `theta=healpix_rope_theta(NSIDE_BOTTLENECK)`) vs `"pos1d"` baseline (old config
  byte-identical, `theta=None` → old default 510) for the A/B. `pyproject.toml`
  temporarily sources gensbi from the local GenSBI working tree via `[tool.uv.sources]`.

Both QUICK CPU smokes green (healpix train 4.3665/val 5.2307; pos1d 4.3664/5.2230 —
losses differ, so the forward path really switches). All per-task reviews + final
whole-branch review clean, zero Critical/Important findings.

**Pending gates:** (1) user's full GPU A/B — `COND_ID_KIND="healpix"` vs `"pos1d"`, TARP
calibration + posterior marginals — gates promotion beyond the example; (2) re-sync the
HEAL-SWIN-nnx venv with CUDA extras first (the `uv sync` for the local gensbi source
uninstalled the CUDA jaxlib packages); (3) revert the temporary uv source once a gensbi
release ships the builder.

## 2. The ConditionalPipeline problem that emerged

**The plan had a blind spot.** Its example edits covered only the `SMOKE=1`-guarded
manual block (direct `Flux1` call with explicitly-built ids — the path Task 4's tests
also exercise). But the actual `QUICK=1`/full training path constructs a
`ConditionalPipeline`, and the pipeline **builds its own default ids internally**:

- `ConditionalPipeline.__init__` calls `_resolve_embedding_ids` for obs and cond
  (`src/gensbi/recipes/conditional_pipeline.py:148-152`), passing the per-stream
  strategy string plus only `dim` and `size`.
- `_resolve_embedding_ids` (`src/gensbi/recipes/utils.py:267`) dispatches on two closed
  string sets: `_EMBEDDINGS_1D = {"absolute", "pos1d", "rope1d"}` and
  `_EMBEDDINGS_2D = {"pos2d", "rope2d"}` (utils.py:263-264). Anything else raises
  `ValueError: Unknown id embedding strategy`.

So passing the example's `ID_EMBEDDING = ("absolute", "rope")` — which is what the
**model** (Flux1Params) wants — crashed pipeline construction: the model's strategy
namespace ("rope" = "apply RoPE to whatever ids you're given") and the pipeline's
id-builder namespace ("rope1d"/"rope2d" = "build 1D/2D grid ids") are different
vocabularies that happen to share words. Structurally, `_resolve_embedding_ids` cannot
build spherical ids even if taught the word: it receives only `dim_cond` (token count),
while HEALPix ids need `nside` (and optionally `base_pixels`) — the geometry, not just
the count. First smoke run failed with the ValueError from utils.py:300.

## 3. The fix (example-level workaround, reviewed sound)

In `examples/spherical_grf_flowmatch.py::make_pipeline` (lines 216-238): seed the
pipeline with a throwaway *valid* cond strategy so `__init__` doesn't raise, then
overwrite the ids attribute with the real spherical ids:

```python
pipeline = ConditionalPipeline(
    ...,
    id_embedding_strategy=(ID_EMBEDDING[0],
                           "absolute" if COND_ID_KIND == "healpix" else ID_EMBEDDING[1]),
    ...)
if COND_ID_KIND == "healpix":
    pipeline.cond_ids, _ = init_ids_healpix(NSIDE_BOTTLENECK)
```

`ID_EMBEDDING` as passed to `Flux1Params` (which actually controls the model's RoPE
dispatch) is untouched. Why this is safe — verified by two independent reviewers against
source:

- `id_embedding_strategy` is consumed **only** inside `ConditionalPipeline.__init__`
  for the two `_resolve_embedding_ids` calls; never read afterwards.
- Every downstream consumer reads `self.cond_ids` lazily, after the override:
  `get_loss_fn` (conditional_pipeline.py:187), `get_sampler` (:227),
  `get_log_prob_fn` (:303). Training, sampling, and log-prob all see spherical ids.
- `dim_cond` consistency is guarded by the example's module-level
  `assert COND_TOKENS == 12 * NSIDE_BOTTLENECK**2`.

It works, but it's the least elegant part of the branch: a consumer must know to seed a
fake strategy and reach into a pipeline attribute post-construction.

## 4. The brainstorming question for the new session

**How should GenSBI's recipe layer properly support positional ids it can't derive from
`dim` alone?** The final reviewer flagged this as the designed follow-up. Constraints
and observations to seed the discussion:

- The string-enum API has no channel for geometry (`nside`, `base_pixels`, or any future
  grid spec). Growing the enum with `"rope3d"`/`"healpix"` alone cannot work without
  also plumbing geometry parameters through `ConditionalPipeline.__init__`.
- Candidate directions (not yet evaluated): (a) accept prebuilt ids directly
  (`obs_ids=`/`cond_ids=` constructor params that bypass `_resolve_embedding_ids`);
  (b) accept a builder callable `(dim) -> (ids, dim)`; (c) a structured strategy object
  replacing/augmenting the string enum; (d) keep strings but add a parallel
  `id_geometry` config. Option (a) is the minimal change that would have made the
  example trivial.
- Whatever the design: the model-strategy vs pipeline-builder namespace collision
  ("rope" vs "rope1d"/"rope2d") is worth resolving or at least documenting — it is what
  actually bit here.
- Two sibling pipelines exist (`ConditionalPipeline` is the one used here); check
  whether joint/unconditional pipelines share `_resolve_embedding_ids` and need the same
  treatment.
- Prior art in-tree: `init_ids_1d`'s FIXME (utils.py, axis-order footgun vs
  `init_ids_2d`) already documents that the ids/axes conventions need unification —
  a redesign could fold that in.
- Backward compatibility: `_resolve_embedding_ids` is private but the
  `id_embedding_strategy` kwarg is public API used across examples/tests.

## 5. Minor loose ends (non-blocking, from final review)

- `init_ids_healpix(nside, base_pixels=[])` raises a raw
  `ValueError: need at least one array to concatenate` from `np.concatenate` — loud but
  unhelpful; a guard with a proper message would be nicer.
- Non-integer `base_pixels` (e.g. `[1.5]`) slip validation and reach `hp.pix2vec`.
- HEAL-SWIN-nnx pyproject comment says the gensbi source points at the "healpix-rope
  branch" — now stale (the work is on main); the path itself still resolves fine.
