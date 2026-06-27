# Safetensors Serialization: portable weight save/load for nnx models

**Date:** 2026-06-27
**Status:** Design approved, ready for implementation plan
**Branch:** `maf` (un-merged, not pushed — additive feature, no breaking changes)

## Context

GenSBI models are `flax.nnx` modules built from a `*Params` dataclass
(e.g. `MAFlowParams → MAFlow`, `TarFlowParams → TarFlow`, `Flux1Params → Flux1`).
The `*Params` object carries `rngs` plus architecture hyperparameters (and sometimes
objects like `transformer: Bijection`), so reconstructing a model abstractly requires
the caller to keep those params — it cannot be derived from weights alone.

Persistence today is **orbax-based and lives inside the training pipelines**
(`recipes/pipeline.py::save_model`/`restore_model`, mirrored in
`experimental/.../vae_pipeline.py`). It saves the primary model **and** an EMA model
into an orbax `CheckpointManager` directory keyed by `experiment_id`, via
`nnx.split(model) → (graphdef, state)` then `StandardSave(state)`; restore rebuilds an
abstract model, splits it, restores into that abstract state, and `nnx.merge`s. There
is **no** save/load on the standalone models, and the orbax format is a multi-file
directory tree — not a single portable artifact.

`safetensors[jax]>=0.8.0` is already a declared dependency (and installed in the
`gensbi` mamba env). Its flax API stores a **flat `{str: jax.Array}` dict plus an
optional `{str: str}` metadata blob** — it does *not* store pytree structure or the
model graph, so flatten-on-save / unflatten-on-load is on us.

### Goal (scope guard)

The one thing this feature enables is **portable weight sharing**: ship a single
framework-neutral, HF-Hub-friendly `.safetensors` file containing a model's weights.
To reload, the user **reconstructs the model from its `Params`** and then loads weights
*into* that model (matching the "update the current model with the weights" framing).

This is deliberately **not** a `from_pretrained`-style self-contained reload: we do not
embed enough architecture metadata to rebuild the model automatically. That is an
explicit non-goal (see below). The orbax checkpointing stays exactly as-is; safetensors
is an additive, parallel export path, not a replacement.

## Goals

- A small, **general-purpose** exporter/loader in `gensbi/utils/serialization.py` that
  works on **any** `nnx.Module` — not just the flow models.
- `save_safetensors(model, path, *, metadata=None, wrt=None)` → writes one file.
- `load_safetensors(model, path, *, strict=True)` → updates `model` **in place**, also
  returns it.
- Re-export both from `gensbi.utils`.
- A **thin** convenience pair on the base pipeline (`recipes/pipeline.py`):
  `export_safetensors(path, *, ema=True)` / `import_safetensors(path, *, ema=True)` —
  these only *select* the primary or EMA model and delegate to the standalone
  functions. EMA is the one piece of state a user cannot easily reach on their own, and
  EMA weights are usually the ones worth sharing for inference, so `ema=True` is the
  default.
- Fast CPU tests proving round-trip fidelity, strict validation, and the pipeline path.

## Non-goals

- **Auto-reconstruction from metadata** (HF `from_pretrained`-style). The caller rebuilds
  the model from `Params`; metadata is provenance only.
- **Per-model `save_safetensors`/`load_safetensors` methods.** They would add the same
  wrapper to every `nnx.Module` for no gain over `save_safetensors(model, path)` and
  would need maintenance as models come and go.
- **Replacing orbax.** The pipeline's existing orbax checkpointing is untouched.
- **Multi-host / sharded** save-load. Single-host arrays only.
- **Cross-framework (PyTorch) key remapping.** The file is round-trippable within GenSBI;
  porting keys to/from another framework's naming is out of scope for v1.

## Design

### Module layout

- New: `src/gensbi/utils/serialization.py` — the two standalone functions + private
  flatten/unflatten/validate helpers.
- Edit: `src/gensbi/utils/__init__.py` — re-export `save_safetensors`, `load_safetensors`
  (the file currently re-exports nothing; add an explicit `__all__`).
- Edit: `src/gensbi/recipes/pipeline.py` — add the two thin convenience methods to the
  base pipeline class (next to `save_model`/`restore_model`).
- New: `tests/utils/test_serialization.py`.

### The flatten/unflatten mechanism (verified against flax 0.12.7)

nnx state is a nested pytree, and `nnx.List` submodule containers use **integer** keys
for their indices. safetensors requires **string** keys. flax's own
`traverse_util.flatten_dict(pure, sep=".")` **crashes** on integer keys
(`sep.join(path)` → `TypeError: expected str instance, int found`). The resolution uses
flax's built-ins for all traversal and `nnx.restore_int_paths` for the inverse — we
hand-roll no per-layer naming.

**Save:**

```python
from flax import nnx
import flax.traverse_util as tu
import numpy as np
from safetensors.flax import save_file

state = nnx.state(model) if wrt is None else nnx.state(model, wrt)
pure = state.to_pure_dict()                       # nested; nnx.List indices are ints
flat = tu.flatten_dict(pure)                       # sep=None → {tuple: array}, ints preserved
tensors = {".".join(map(str, k)): np.asarray(v) for k, v in flat.items()}
save_file(tensors, path, metadata=full_metadata)
```

**Load** — `nnx.restore_int_paths` is flax's official inverse for "a flat checkpoint
stringified my integer paths", reconstructing the int-keyed pure dict *from the file
alone*. We then validate against the rebuilt model and update in place:

```python
from safetensors.flax import load_file

loaded = load_file(path)                           # {str: jax.Array}, flat
file_pure = nnx.restore_int_paths(tu.unflatten_dict(loaded, sep="."))  # → int keys restored

state = nnx.state(model)                           # the rebuilt model is the schema
ref = tu.flatten_dict(state.to_pure_dict())        # {tuple: array}
got = tu.flatten_dict(file_pure)                   # {tuple: array}
# ... validate (below), build `new` {tuple: array} with dtype cast ...
state.replace_by_pure_dict(tu.unflatten_dict(new))
nnx.update(model, state)                           # in-place
return model
```

Both directions were verified end-to-end (incl. a `BatchNorm`'s `BatchStat`,
int-dtype leaves, and `nnx.List` containers): every leaf matched after loading the
saved file into a differently-initialized model.

### What is saved (`wrt`)

Default saves the **full state** (`nnx.state(model)`, all variable collections — `Param`,
`BatchStat`, `Standardize` running stats, deterministic buffers like TarFlow `Mask` /
fielddit `RopeIds`, etc.). This matches what orbax saves here and gives a faithful
round-trip. `wrt` (e.g. `nnx.Param`) optionally narrows the saved collections for
advanced use; the loader naturally handles a subset under `strict=False`.

Deterministic buffers are config-derived and technically redundant (the rebuilt model
regenerates them), but saving them is harmless (int/bool dtypes round-trip cleanly) and
keeps "full state in, full state out" simple and obviously correct.

### Metadata

safetensors' `{str: str}` metadata blob carries **provenance only** — never used to
rebuild the model. Defaults, merged under any caller-supplied `metadata`:

- `format = "gensbi"`
- `version = "1"`
- `model_class = type(model).__name__`
- `framework = "flax-nnx"`

Readable without loading tensors via `safetensors.safe_open(path, framework="flax").metadata()`.

### Load validation & semantics

The rebuilt model is the schema; the file is a value store.

- **`strict=True` (default):** the file's key set must exactly equal the model's. Any
  missing or extra key → `ValueError`/`KeyError` with a readable diff (the offending
  keys). Any per-key shape mismatch → `ValueError` naming the key and both shapes.
- **`strict=False`:** load the **intersection** — keys present in both — leaving any
  model leaf absent from the file at its freshly-initialized value, and ignoring file
  keys absent from the model. Useful for partial / `wrt`-narrowed loads.
- **Dtype:** each loaded array is cast to the corresponding model leaf's dtype
  (`.astype(ref_dtype)`), so the reconstructed model keeps the dtype its `Params`
  dictate (mirrors orbax `StandardRestore` behaviour).
- **`model_class` mismatch** between metadata and the target model is a **warning**, not
  an error (a subclass or rename should not block a legitimate load).
- **In place:** load mutates the passed-in model via `nnx.update` and returns it, so
  `model = load_safetensors(MAFlow(params), path)` and bare
  `load_safetensors(model, path)` both work.

### Key-separator assumption

The `.` join/split assumes no nnx path component contains a literal `.`. This holds for
nnx attribute names (Python identifiers) and integer list indices. At **save** time we
guard it: if any non-leaf path component contains `.`, raise `ValueError` rather than
silently producing an unsplittable key. (`nnx.Dict` with dotted string keys is the only
way to violate this and is unused in the codebase.)

### Pipeline convenience (thin)

On the base pipeline class in `recipes/pipeline.py`, next to `save_model`:

```python
def export_safetensors(self, path, *, ema=True, metadata=None):
    """Export the trained model's weights to a single .safetensors file.
    ema=True (default) exports the EMA model — usually the inference weights."""
    model = self.ema_model if ema else self.model
    save_safetensors(model, path, metadata=metadata)

def import_safetensors(self, path, *, ema=True, strict=True):
    """Load weights from a .safetensors file into this pipeline's model in place."""
    model = self.ema_model if ema else self.model
    load_safetensors(model, path, strict=strict)
```

No reimplementation — pure selection + delegation. Symmetric `import_` makes round-trip
testing trivial and lets a user reload shared weights into an existing pipeline.

## API summary

```python
from gensbi.utils import save_safetensors, load_safetensors

# standalone (any nnx.Module)
save_safetensors(model, "model.safetensors")                 # full state + provenance metadata
save_safetensors(model, "weights.safetensors", wrt=nnx.Param)  # params only
model = load_safetensors(MAFlow(params), "model.safetensors")  # rebuild from Params, then load in place
load_safetensors(model, "weights.safetensors", strict=False)   # partial load

# pipeline convenience
pipeline.export_safetensors("ema.safetensors")               # ema=True by default
pipeline.import_safetensors("ema.safetensors")
```

## Testing

`tests/utils/test_serialization.py`, fast / CPU (`JAX_PLATFORMS=cpu`):

1. **Round-trip fidelity** — build a small `MAFlow`; save; build a second `MAFlow` with a
   *different* seed; `load_safetensors`; assert every state leaf is bit-identical and
   `log_prob(x, cond)` outputs match the source model.
2. **Generic module** — repeat the round-trip on a tiny hand-rolled `nnx.Module` that
   includes an `nnx.List` and a `BatchStat`-bearing layer, proving the functions are not
   flow-specific and that integer paths + non-`Param` collections round-trip.
3. **Keys & metadata** — assert saved keys are dot-joined strings and that
   `safe_open(...).metadata()` carries `format`/`version`/`model_class`/`framework`;
   caller metadata merges and overrides.
4. **Strict validation** — loading a file into a `MAFlow` built with a different `dim`
   raises with a readable key/shape error; `strict=False` loads the overlap without
   raising.
5. **Dtype cast** — a file saved from a float32 model loads into a model leaf and matches
   that leaf's dtype.
6. **Separator guard** — a module with a dotted `nnx.Dict` key raises at save time (or,
   if constructing such a module is awkward, a focused unit test on the key-join helper).
7. **Pipeline convenience** — via an existing concrete pipeline (e.g. the MAF/flow NLE
   pipeline), `export_safetensors(ema=True)` then `import_safetensors` into a fresh
   pipeline reproduces outputs; assert EMA vs primary selection picks the right model.

## Risks / mitigations

- **`replace_by_pure_dict` strictness varies by flax version.** Mitigation: we validate
  the key set and shapes ourselves *before* calling it, so errors are ours and readable
  regardless of flax's internal behaviour.
- **`.value` deprecation in flax 0.12.7.** Mitigation: the whole mechanism goes through
  `to_pure_dict` / `replace_by_pure_dict`, never touching `Variable.value`.
- **Non-array leaves in state.** Not expected for these models (lazily-built priors stay
  out of state), but a non-array leaf would surface as a clear `save_file` type error; if
  it ever bites, document filtering via `wrt`.
