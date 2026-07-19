# Design: first-class healpix-rope pipeline ids + HEAL-SWIN-nnx / examples restructuring

**Date:** 2026-07-19. **Status:** approved (brainstorm session, section-by-section).
**Follows:** `docs/superpowers/2026-07-19-healpix-rope-handoff.md` (the ConditionalPipeline
ids gap) and `docs/superpowers/specs/2026-07-19-healpix-rope-design.md` (the merged
spherical RoPE builders).

## Goal

Two coupled outcomes, one spec, phased:

- **A. GenSBI recipe layer:** spherical HEALPix RoPE ids become a first-class
  `ConditionalPipeline` id strategy ("healpix-rope"), replacing the example-level
  workaround (seed a fake strategy, overwrite `pipeline.cond_ids` post-construction).
- **B. Repo restructuring:** the spherical GRF flow-matching example moves from
  HEAL-SWIN-nnx to GenSBI-examples; HEAL-SWIN-nnx sheds its gensbi/sbibm-jax coupling
  and becomes a publishable standalone PyPI package; gensbi then depends on
  `heal-swin-nnx` and mirrors the encoder through `gensbi.models`, so HealSwin feels
  like part of gensbi. (Note: there is no dependency *cycle* today — HEAL-SWIN-nnx's
  gensbi coupling is an optional extra plus temporary `[tool.uv.sources]` overrides.
  The restructuring makes the direction clean and permanent.)

Scope decision: **minimal + document**. No renaming of existing strategy strings, no
fix of the `init_ids_1d` axis-order FIXME. The model-vocabulary vs pipeline-vocabulary
collision is documented, not resolved.

## Part A — strategy objects in the recipe layer

### API (approved shape)

`id_embedding_strategy` tuple slots accept `str | IdStrategy`. Strings behave
byte-identically to today. New module `src/gensbi/recipes/id_strategies.py`:

- **`IdStrategy`** — a duck-typed protocol (no ABC, no registry): an object with
  `name: str` and `build(dim) -> (ids, resolved_dim)`. Any user object satisfying it
  works, which covers arbitrary custom ids; therefore NO separate `obs_ids=`/`cond_ids=`
  passthrough kwargs are added (YAGNI).
- **`HealpixRope`** — frozen dataclass, `HealpixRope(nside, base_pixels=None)`,
  `name = "healpix-rope"`.
  - Construction validates: `nside` power of two; `base_pixels` (when given) unique
    integers in [0, 11], non-empty.
  - `build(dim)` cross-checks `dim == 12 * nside**2` (full sky) or
    `len(base_pixels) * nside**2`, raising a `ValueError` that names both numbers;
    then delegates to `init_ids_healpix(nside, base_pixels)` and returns
    `(ids, num_tokens)`. This replaces the example's module-level token-count assert.
  - `theta` property returns `healpix_rope_theta(nside)` so the model-side RoPE theta
    is derived from the same object that builds the ids.
- Both exported from `gensbi.recipes` (`from gensbi.recipes import HealpixRope`).

Example usage (the whole point — compare handoff §3):

```python
cond_strategy = HealpixRope(nside=NSIDE_BOTTLENECK)
pipeline = ConditionalPipeline(
    model=flux1, train_dataset=..., val_dataset=...,
    dim_obs=3, dim_cond=COND_TOKENS,
    method=FlowMatchingMethod(), ch_cond=512,
    id_embedding_strategy=("absolute", cond_strategy),
)
# Flux1Params side (unchanged vocabulary):
#   id_embedding_strategy=("absolute", "rope"), axes_dim=(22, 22, 20),
#   theta=cond_strategy.theta
```

### Dispatch change

`_resolve_embedding_ids` (`src/gensbi/recipes/utils.py:267`) gains one arm *before*
the string sets: `if hasattr(strategy, "build"): return strategy.build(dim)`.
`semantic_id`/`size` are not passed to objects — an `IdStrategy` owns its full geometry.
The unknown-strategy `ValueError` message is extended to state that strings or
`IdStrategy` objects are accepted, and to list the valid strings.

`ConditionalPipeline` needs zero structural change; objects flow through the existing
tuple slots. Grep confirms `ConditionalPipeline` is the only `_resolve_embedding_ids`
caller — sibling pipelines (joint/unconditional, Flux1Joint is `"absolute"`-only by
assertion) need no treatment. The config-driven model-specific pipelines
(`recipes/flux1.py` etc.) keep their string-only YAML path; strategy objects are
deliberately not YAML-serializable, which is acceptable because `ConditionalPipeline`
is the model-agnostic pipeline whose `init_pipeline_from_config` already raises
`NotImplementedError`.

### Folded-in fixes (handoff §5 loose ends)

Inside `init_ids_healpix` so both entry points benefit:

- `base_pixels=[]` → proper `ValueError` ("base_pixels must be non-empty"), replacing
  the raw numpy "need at least one array to concatenate".
- Non-integer `base_pixels` entries (e.g. `1.5`) rejected at validation instead of
  reaching `hp.pix2vec`.

### Namespace documentation (not resolution)

Docstrings in `ConditionalPipeline` and `_resolve_embedding_ids` gain an explicit note
distinguishing the two vocabularies that collided in the handoff incident:

- **Model-side** strategy strings (`Flux1Params.id_embedding_strategy`): "rope" means
  *apply RoPE to whatever ids arrive*.
- **Pipeline-side** builder strategies (`"rope1d"`/`"rope2d"`/`HealpixRope`): these
  *build* the ids.
- A `HealpixRope` pipeline strategy pairs with model-side `("absolute", "rope")` plus a
  3-entry even `axes_dim` summing to the per-head dim (e.g. `(22, 22, 20)` for 64).

No string renames, no deprecations.

## Part B — restructuring phases

Ordered so nothing ever depends on an unpublished package.

### Phase 1 — GenSBI: land Part A

On main (via short-lived branch), with tests (see Verification).

### Phase 2 — HEAL-SWIN-nnx: slim down, make publishable, STOP for manual publish

Repo: `/lhome/ific/a/aamerio/data/github/HEAL-SWIN-nnx` (currently checked out on
`healpix-rope`, one commit `0748eda` ahead of main; untracked `.github/` with drafted
`python-publish.yml` and `python-app.yml`).

1. **Branch resolution first:** fast-forward merge `healpix-rope` into main, then
   delete the branch. Rationale: `0748eda`'s example (the `COND_ID_KIND` A/B switch)
   is the source material for the Phase-4 port; merging puts it permanently in main's
   history instead of leaving the port source on an unmerged branch.
2. **Removal commit** (short-lived branch off main, merged when green):
   - Delete `examples/spherical_grf_flowmatch.py`,
     `examples/sub/spherical_grf_flowmatch.sub`,
     `examples/sub/run_spherical_grf_flowmatch.sh`,
     `examples/spherical_grf_fm_results.txt`,
     `examples/spherical_grf_fm_quick_results.txt`, and the 16 committed
     `examples/imgs/spherical_grf_*.png` files.
   - `pyproject.toml`: delete the `gensbi` optional extra and the **entire**
     `[tool.uv.sources]` table (both local-path entries — sbibm-jax and gensbi).
   - README: one-line pointer to the example's new home in GenSBI-examples.
   - Keep `docs/superpowers/` spec+plan (historical record); keep all MNIST examples
     and the `[examples]` extra (self-contained).
3. **Commit `.github/` workflows** after reviewing `python-publish.yml` (PyPI publish
   mechanism for the user's manual release) and `python-app.yml` (CI) for sanity.
4. **Publish-readiness metadata pass:** version stays `0.1.0`; add missing
   `[project.urls]`; check license/readme/`requires-python`. Basic classifiers only.
5. **Verification, then stop:** `uv sync` (side effect: restores the CUDA jaxlib
   packages the gensbi source-override uninstalled — closes handoff pending-gate 2);
   full `uv run pytest` green; `uv build` → sdist + wheel; install the wheel into a
   scratch venv and run the test suite against the installed package. Merge to main.
   **STOP — the user publishes `heal-swin-nnx 0.1.0` to PyPI manually.**

### Phase 3 — GenSBI: dependency + mirror

- `pyproject.toml`: add `heal-swin-nnx>=0.1.0` as a regular dependency (its runtime
  deps — jax, flax, einops, numpy, healpy — are a strict subset of gensbi's; zero
  added weight).
- New `src/gensbi/models/healswin.py` re-exporting the full `heal_swin_nnx` public
  surface (`heal_swin_nnx.__all__`).
- `gensbi.models.__init__` adds exactly two names to imports and `__all__`:
  `HealSwinEncoder`, `HealSwinParams` (the SBI-relevant surface; everything else is
  reachable via `gensbi.models.healswin` or `heal_swin_nnx` directly).
- Smoke test: import both names from `gensbi.models`, run a tiny CPU forward.

### Phase 4 — GenSBI-examples: the spherical_grf example

- New directory `examples/sbi-benchmarks/spherical_grf/` (underscore, matching
  `gravitational_waves` etc.) containing the ported script and its `sub/` files
  (submit file + runner script), source = HEAL-SWIN-nnx main history (`0748eda`).
- Two rewrites inside the script:
  1. `from gensbi.models import HealSwinEncoder, HealSwinParams` (dogfoods the
     Phase-3 mirror).
  2. `make_pipeline` uses `HealpixRope(nside=NSIDE_BOTTLENECK)`; the
     fake-strategy-then-overwrite workaround and the module-level token-count assert
     are deleted; Flux1's `theta` comes from `strategy.theta`. The `COND_ID_KIND`
     healpix-vs-pos1d A/B switch is **kept** — the GPU A/B gate runs from here.
- GenSBI-examples `pyproject.toml`, in passing: add `gensbi` as an explicit dependency
  (absent today); require `sbibm-jax[loader]>=0.1.3` (the version with the
  `spherical_grf` task and `TaskDataset`); update the stale `jax>=0.9,<0.10` pin to
  match gensbi's `>=0.10.2` floor.
- Until a gensbi release including Phases 1+3 exists, local runs use the existing env
  with the local gensbi tree (GenSBI-examples is `package = false`); no temporary
  uv-source plumbing is added.

## Verification

- **Phase 1 (GenSBI, mamba `gensbi` env per project convention):** new tests alongside
  `tests/recipes/test_healpix_ids.py`:
  - `HealpixRope(nside).build(dim)` ids identical to `init_ids_healpix(nside)`.
  - dim-mismatch `ValueError` names expected vs given token counts.
  - `base_pixels=[]` and non-integer `base_pixels` raise proper `ValueError`s.
  - `HealpixRope(nside).theta == healpix_rope_theta(nside)`.
  - `ConditionalPipeline` built with `HealpixRope` has `cond_ids` bit-identical to the
    handoff-§3 workaround's ids (regression against the A/B semantics).
  - Existing string-strategy tests pass untouched (byte-identical behavior).
- **Phase 2 (HEAL-SWIN-nnx):** pytest green; wheel-install test; `uv sync` CUDA
  restore observed.
- **Phase 3 (GenSBI):** mirror smoke test (imports + tiny CPU forward).
- **Phase 4 (GenSBI-examples):** `SMOKE=1 JAX_PLATFORMS=cpu` and
  `QUICK=1 JAX_PLATFORMS=cpu` runs of the moved example succeed against PyPI
  `sbibm-jax>=0.1.3`.

## Error handling

- `HealpixRope` fails at construction (geometry validation) or pipeline construction
  (token-count cross-check) with geometry-aware messages — never mid-training.
- `_resolve_embedding_ids` unknown-strategy error lists valid strings and mentions
  `IdStrategy` objects.

## Gates (user actions, in order)

1. After Phase 2 merges: user publishes `heal-swin-nnx 0.1.0` to PyPI. Work pauses here.
2. sbibm-jax `0.1.3` visible on PyPI (upload in flight at time of writing; local
   checkout already at 0.1.3 — verify `pip index`/resolution before executing Phase 4).
3. After Phase 4: full GPU A/B (`COND_ID_KIND` healpix vs pos1d, TARP calibration +
   posterior marginals) from GenSBI-examples — still the promotion gate for healpix
   RoPE itself.
4. A gensbi release including Phases 1+3, on the user's cadence (also allows
   HEAL-SWIN-nnx users to get gensbi integration purely from PyPI).

## Out of scope

- Renaming/deprecating existing id-strategy strings; the `init_ids_1d` axis-order
  FIXME (both documented, deferred).
- `obs_ids=`/`cond_ids=` passthrough kwargs (subsumed by custom `IdStrategy` objects).
- Joint/unconditional pipeline id-strategy changes.
- Converting the example to the `train-*.py` + notebook + YAML-config layout used by
  sibling benchmarks (the script keeps its headless single-file form).
