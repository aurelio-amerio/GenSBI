# Normalizing-Flow Model Restructure: `Params` + `Model` classes under `models/`

**Date:** 2026-06-26
**Status:** Design approved, ready for implementation plan
**Branch:** `maf` (un-merged, not pushed — a major release; breaking changes allowed)

## Context

The `maf` branch implemented two discrete normalizing-flow families — affine/spline
MAF and (S)TarFlow (transformer autoregressive). Both are currently built through
ad-hoc factory functions (`make_maf`, `make_tarflow`) that live next to generic
density containers (`Flow`, `TransformerFlow`) inside `normalizing_flows/`. The
flow-matching models (`Flux1`, `Simformer`, `Flux1Joint`) instead follow one uniform
convention: a `XxxParams` `@dataclass` plus a `Xxx(nnx.Module)` whose `__init__(params)`
builds everything, exported as `(Model, Params)` pairs from `gensbi.models`.

This restructure aligns the NF models with that convention and, in doing so, draws a
clean line between *pure normalizing-flow machinery* (the abstractions that define what
a flow is) and *model implementations* (the architecture-specific machinery of MAF and
TarFlow). It is the first step toward a uniform interface across both modelling tracks.

## Goals

- Introduce `MAFlowParams` + `MAFlow` and `TarFlowParams` + `TarFlow` as the canonical,
  uniform constructors (`Model(Params(...))`), mirroring `Flux1(Flux1Params(...))`.
- Move the concrete model implementations into `models/`, leaving `normalizing_flows/`
  as a pure, reusable abstraction layer.
- Establish `models/core/` as the home for shared, model-agnostic primitives.
- Keep all **general pipelines** (Flux1, Simformer, Flux1Joint, the unified pipelines)
  compiling and passing — that is the only user-facing contract.
- Behaviour-identical refactor: NF numerics unchanged; verified by the test suite.
  The one deliberate API change is the TarFlow head parameterization (decision 8),
  which preserves the default architecture.

## Non-goals (explicit follow-ups)

- **NF pipeline wiring.** `ConditionalFlowPipeline._make_model` / `get_default_params`
  stay as-is (raising `NotImplementedError`). Recipe-level `get_default_maf_params` /
  `get_default_tarflow_params` and pipeline construction-from-params come later.
- **Migrating `models/embedding/` into `models/core/`.** Documented as the intended next
  inhabitant of `core/`, but it is imported by every flow-matching model and is out of
  scope here.
- **Rewiring FieldDiT / glue to reuse the shared `ImageTokenizer`.** The tokenizer is
  *placed* in `core/` so this becomes cheap later; the rewire itself is deferred.
- **Promoting `MaskedAutoregressive` to a shared autoregressive-flow module.** It is
  MAF-only today; if an IAF appears, promote it then.

## Design decisions

1. **Self-contained model classes (Option B).** `MAFlow` / `TarFlow` are `nnx.Module`s
   that fold in the density methods (`log_prob`, `sample`, `set_standardization`). The
   generic `Flow` and `TransformerFlow` classes are removed; their logic is absorbed.
2. **Three-tier component taxonomy** (the organising principle — see below).
3. **`Params` dataclasses carry `rngs` as a field** (matching `Flux1Params`).
4. **Drop `make_maf` / `make_tarflow` entirely.** Canonical construction is
   `MAFlow(MAFlowParams(...))` / `TarFlow(TarFlowParams(...))`.
5. **Public API from `gensbi.models`.** `normalizing_flows/` exports Tier-1 only and
   never imports from `models/`.
6. **Breaking changes allowed** (major release); general pipelines must stay green.
7. **Names:** classes `MAFlow` / `TarFlow`; params `MAFlowParams` / `TarFlowParams`.
8. **TarFlow head parameterization: `(head_dim, num_heads)`, `channels` derived.**
   Expose the per-head dim and head count (Flux1-style), deriving total width
   `channels = head_dim * num_heads`; only two of the three are independent. This
   resolves the standing `blocks.py:17` TODO and normalizes `AttentionBlock` /
   `MetaBlock` to take `num_heads`. The default `(head_dim=16, num_heads=4)`
   reproduces the prior `channels=64` architecture exactly. There is no global
   head convention yet: Flux1 uses `(head_dim, num_heads)`, PixelDiT uses the dual
   `(channels, num_heads)`. We adopt Flux1's as the candidate house standard;
   reconciling PixelDiT is a separate follow-up (it does not touch the general
   pipelines' behaviour, but is its own change).

## Component taxonomy

The test for placement: *is a component reusable by a different flow, or does it define
one specific architecture?*

**Tier 1 — Pure NF abstractions** (define what a flow *is*) → stay in
`normalizing_flows/bijections/`:

| File | Exports | Depends on |
|------|---------|------------|
| `base.py` | `Bijection`, `Mask` | — |
| `chain.py` | `Chain` | `base` |
| `permutation.py` | `Permutation` | `base` |
| `standardize.py` | `Standardize` | `base` |
| `transformers.py` | `Affine`, `RQSpline` | — |

`Affine`/`RQSpline` are kept pure: they are generic elementwise bijections passed *as an
argument* to `MaskedAutoregressive` (which does not import them); a coupling flow or IAF
would reuse the same `Affine`. TarFlow does not use them (its `MetaBlock` has its own
inline affine).

**Tier 2 — Shared model primitives** (general, not flow-theory) → `models/core/`:

| File | Exports | Depends on |
|------|---------|------------|
| `patching.py` | `patchify_2d`, `depatchify_2d` | jax, einops |
| `tokenizers.py` | `VectorTokenizer`, `ImageTokenizer` | jax, `core.patching` |

Tokenizers are pure, stateless, volume-preserving reshapes (no convolution, no learned
parameters), which is why they are safe model-agnostic primitives and a reuse target for
FieldDiT/glue.

**Tier 3 — Model implementation** (defines one architecture) → co-located with its model,
exactly like `models/flux1/layers.py`:

| Model | Machinery files → location | Model file |
|-------|----------------------------|------------|
| MAF | `made.py` (`MADE`, `MaskedAutoregressive`), `masked_linear.py` (`MaskedLinear`), `masks.py` (`make_mask`) → `models/maf/` | `models/maf/model.py` (`MAFlowParams`, `MAFlow`) |
| TarFlow | `blocks.py` (`MetaBlock`, `AttentionBlock`), `conditioners.py` (`VectorConditioner`, `VectorPrefixConditioner`, `ImagePrefixConditioner`) → `models/tarflow/` | `models/tarflow/model.py` (`TarFlowParams`, `TarFlow`) |

MAF machinery depends only on Tier-1 `base` (plus its own siblings). TarFlow machinery
depends on Tier-1 `base` (for `Mask`) and `models/core` (for `patchify_2d`).

## Target structure

```
normalizing_flows/                       # PURE Tier-1 abstraction layer (leaf; deps: gensbi.core)
  __init__.py                            # re-exports Tier-1 only
  bijections/
    base.py          Bijection, Mask
    chain.py         Chain
    permutation.py   Permutation
    standardize.py   Standardize
    transformers.py  Affine, RQSpline
    __init__.py
  # DELETED: flow.py, transformer_flow/  (model.py, blocks, conditioners, tokenizers moved out)

models/
  core/
    __init__.py
    patching.py      patchify_2d, depatchify_2d        # moved from recipes/utils.py
    tokenizers.py    VectorTokenizer, ImageTokenizer   # moved from transformer_flow/
  maf/
    __init__.py      MAFlowParams, MAFlow
    made.py          MADE, MaskedAutoregressive        # moved from bijections/
    masked_linear.py MaskedLinear                       # moved from bijections/
    masks.py         make_mask                          # moved from bijections/
    model.py         MAFlowParams, MAFlow
  tarflow/
    __init__.py      TarFlowParams, TarFlow
    blocks.py        MetaBlock, AttentionBlock          # moved from transformer_flow/
    conditioners.py  *Conditioner family                # moved from transformer_flow/
    model.py         TarFlowParams, TarFlow             # absorbs TransformerFlow + LICENSE attribution
```

### Dependency direction (acyclic, verified)

```
models/{maf,tarflow}  →  normalizing_flows/bijections (Tier 1)
models/{maf,tarflow}  →  models/core (Tier 2, leaf)
models/core           →  (jax, einops)
normalizing_flows     →  (gensbi.core only)
```

There is **no** `normalizing_flows → models` edge (the prior
`normalizing_flows → recipes.utils → recipes/__init__ → models` back-edge is severed by
moving `patchify` out of `recipes`). `import gensbi.models` is therefore clean regardless
of sub-package import order.

## The `Params` + `Model` classes

`Params.__post_init__` validates configuration and fills scalar defaults only (it does not
build `nnx` modules — that is the model's `__init__`, which has `rngs`). This mirrors
`Flux1Params.__post_init__` computing derived scalars while `Flux1.__init__` builds modules.

### `MAFlowParams` / `MAFlow` (`models/maf/model.py`)

`MAFlowParams` (`@dataclass`) — fields from `make_maf`:

| field | type | default |
|-------|------|---------|
| `rngs` | `nnx.Rngs` | — |
| `dim` | `int` | — |
| `cond_dim` | `int` | `0` |
| `n_layers` | `int` | `5` |
| `transformer` | `Bijection \| None` | `None` → `Affine()` in `__post_init__` |
| `nn_width` | `int` | `64` |
| `nn_depth` | `int` | `2` |
| `permutation` | `str` | `"reverse"` |
| `standardize` | `bool` | `True` |
| `zero_init` | `bool` | `True` |

`__post_init__`: default `transformer` to `Affine()`; validate `permutation ∈ {"reverse","random"}`.

`MAFlow(nnx.Module).__init__(params)`: runs the current `make_maf` body — stack
`MaskedAutoregressive` + `Permutation` layers, append `Standardize` if requested, wrap in
`Chain`; store `chain`, `dim`, `cond_dim`. Folds in `Flow`'s `log_prob` / `sample`
(vmapped per-example over a `make_gaussian_prior((dim,))` base) and `set_standardization`.

### `TarFlowParams` / `TarFlow` (`models/tarflow/model.py`)

`TarFlowParams` (`@dataclass`) — fields from `make_tarflow`:

| field | type | default | notes |
|-------|------|---------|-------|
| `rngs` | `nnx.Rngs` | — | |
| `dim` | `int \| None` | `None` | required for `modeled="vector"` |
| `cond_dim` | `int` | `0` | |
| `modeled` | `str` | `"vector"` | `"vector"` \| `"image"` |
| `img_size` | `int \| None` | `None` | required for `modeled="image"` |
| `patch_size` | `int \| None` | `None` | required for `modeled="image"` |
| `img_channels` | `int` | `1` | |
| `cond` | `str` | `"add"` | `"add"` \| `"vector_prefix"` \| `"image_prefix"` |
| `cond_img_size` | `int \| None` | `None` | required for `cond="image_prefix"` |
| `cond_patch_size` | `int \| None` | `None` | required for `cond="image_prefix"` |
| `cond_channels` | `int` | `1` | |
| `prefix_tokens` | `int` | `1` | |
| `head_dim` | `int` | `16` | per-head attention dim (independent knob, Flux1-style) |
| `num_heads` | `int` | `4` | head count; width `channels = head_dim * num_heads` is derived; default (16, 4) ⇒ channels 64 |
| `num_blocks` | `int` | `8` | |
| `layers_per_block` | `int` | `2` | |
| `block_size` | `int` | `1` | |
| `permutation` | `str` | `"flip"` | `"flip"` \| `"random"` |
| `standardize` | `bool` | `True` | |
| `zero_init` | `bool` | `True` | |
| `use_softplus` | `bool` | `True` | |
| `soft_clip` | `float` | `4.0` | |

`__post_init__`: validate `modeled` / `cond` enums and the presence of the fields each
combination requires (the validation `make_tarflow` does inline today); compute the
derived width `channels = head_dim * num_heads`.

`TarFlow(nnx.Module).__init__(params)`: runs the current `make_tarflow` body — build the
tokenizer (`VectorTokenizer` / `ImageTokenizer` from `models/core`), the per-block
conditioner (`VectorConditioner` / `VectorPrefixConditioner` / `ImagePrefixConditioner`),
and the `MetaBlock` stack with per-block permutations. Folds in `TransformerFlow`'s
`log_prob` / `sample` / `set_standardization` and the `Mask` `mean`/`std` buffers.

## Migration plan

1. **`models/core/`**: new package. Move `patchify_2d` / `depatchify_2d` verbatim from
   `recipes/utils.py` to `models/core/patching.py` (no re-export left behind). Move
   `VectorTokenizer` / `ImageTokenizer` from `transformer_flow/tokenizers.py` to
   `models/core/tokenizers.py` (import `patchify` from `core.patching`).
2. **`models/maf/`**: move `made.py`, `masked_linear.py`, `masks.py` from
   `normalizing_flows/bijections/`. Add `model.py` with `MAFlowParams` + `MAFlow`
   (absorbing `flow.py`'s `Flow` density logic and the `make_maf` body). `__init__.py`
   re-exports the pair.
3. **`models/tarflow/`**: move `blocks.py`, `conditioners.py` from `transformer_flow/`
   (point `conditioners` at `models.core.patching`). Normalize `AttentionBlock` /
   `MetaBlock` to take `num_heads` instead of `head_dim` (derive `head_dim` / use the
   `channels = head_dim * num_heads` width), resolving the `blocks.py:17` TODO. Add
   `model.py` with `TarFlowParams` + `TarFlow` (absorbing `TransformerFlow` + the
   `make_tarflow` body + the Apple/STARFlow LICENSE attribution header). `__init__.py`
   re-exports the pair.
4. **`normalizing_flows/`**: delete `flow.py` and the `transformer_flow/` subpackage.
   Trim `bijections/__init__.py` and `normalizing_flows/__init__.py` to Tier-1 exports
   (`Bijection`, `Mask`, `Chain`, `Permutation`, `Standardize`, `Affine`, `RQSpline`).
5. **`gensbi.models`**: export `MAFlowParams`, `MAFlow`, `TarFlowParams`, `TarFlow`
   (import `.core` first in `models/__init__.py` for clarity, though order-independent).
6. **Update other importers** of the moved symbols:
   - `experimental/models/fielddit/codec.py`, `experimental/models/glue/embedder.py`:
     `patchify_2d` / `depatchify_2d` → `gensbi.models.core.patching`.
   - `Flux1Params` docstring reference `gensbi.recipes.utils.patchify_2d` →
     `gensbi.models.core.patching.patchify_2d`.
   - `recipes/flow_pipeline.py` docstrings / `NotImplementedError` messages mentioning
     `make_maf` → `MAFlow(MAFlowParams(...))` (wording only; behaviour unchanged).
7. **Tests & scripts**: rewrite NF call sites `make_maf(rngs, dim=…)` →
   `MAFlow(MAFlowParams(rngs=…, dim=…))` and `make_tarflow(...)` →
   `TarFlow(TarFlowParams(...))`; update imports to `gensbi.models`. TarFlow call sites
   that passed `channels`/`head_dim` adopt `(head_dim, num_heads)` with
   `num_heads = channels // head_dim` to preserve each test's architecture. Move machinery tests
   alongside their code (`tests/models/maf/`, `tests/models/tarflow/`); pure-bijection
   tests stay under `tests/normalizing_flows/bijections/`. Affected: the `test_flow*`,
   `test_nle`, and `transformer_flow/test_*` suites, and `scripts/{maf,tarflow}_*_recovery.py`.
   `transformer_flow/test_exports.py` is rewritten to assert the new `gensbi.models` surface.

## Testing & verification

- **Behaviour-identical:** the refactor only relocates code and renames the constructor
  surface. Numerics are unchanged. The full NF suite (the ~88+ fast tests these files
  cover) must pass after call-site rewrites.
- **General-pipeline contract:** the Flux1 / Simformer / Flux1Joint / unified-pipeline
  suites must remain green (they exercise the only user-facing surface).
- **No import cycle:** `python -c "import gensbi.models"` and
  `python -c "import gensbi.normalizing_flows"` both succeed; a smoke import of every new
  module path is added/checked.
- GPU recovery scripts remain smoke-only in CI (run full by the owner on GPU), unchanged
  in behaviour aside from the constructor swap.

## Risks & mitigations

- **Import-cycle regression** — mitigated by severing `normalizing_flows → recipes` (move
  `patchify`) and keeping `normalizing_flows` and `models/core` as leaves; verified by an
  explicit `import gensbi.models` check.
- **Silent EMA/buffer-seam breakage** — the `Mask` `mean`/`std` buffers and `optax.ema`
  exclusion behaviour must survive the move into `TarFlow`. Re-assert the existing EMA/
  buffer-seam tests after the move (carry them to `tests/models/tarflow/`).
- **Call-site churn introducing typos** — mechanical rewrites verified by the suite; the
  `Params` dataclass surfaces wrong/missing fields as immediate errors.

## Follow-ups (post-merge)

- Wire `ConditionalFlowPipeline` to build from `*Params` (`_make_model`,
  `get_default_{maf,tarflow}_params`) — the second uniformity step.
- Migrate `models/embedding/` into `models/core/`.
- Rewire FieldDiT / glue to reuse `models/core/tokenizers.ImageTokenizer`.
