# NumPy-style documentation pass for the `maf` branch modules

**Date:** 2026-06-28
**Branch:** `maf`
**Status:** Design — approved, pending spec review

## Goal

Bring every new/changed source module on the `maf` branch up to the repository's
documentation standard — the NumPy-style docstrings exemplified by
`src/gensbi/core/flow_matching.py` — so that the auto-generated API reference
(Sphinx `autoapi` + `napoleon` + `numpydoc`) renders complete, consistent pages
for the normalizing-flows / MAF / TarFlow / inference / serialization work that
has been added but whose documentation has lagged behind.

**This is a documentation-only pass.** No signature changes, no behavior
changes, no refactoring.

## The standard (anchor: `flow_matching.py`)

Each compliant docstring has:

- A one-line imperative summary, with an optional extended description.
- A `Parameters` section where every parameter carries a NumPy type descriptor
  (e.g. `x : Array`, `config : dict`, `event_shape : tuple of (int, int)`), with
  optional/default values noted (e.g. `step_size : float, optional`).
- A `Returns` section with the return type(s).
- A `Raises` section wherever the code raises.
- An `Examples` section only where it genuinely clarifies usage — not forced
  onto every object.
- Types are documented **in the docstring**. Signatures are left untouched.
  Because `conf.py` sets `autodoc_typehints = "description"`, docstring-described
  types and signature annotations render identically; we use the docstring form
  to match the existing convention and avoid any code change.

Cross-reference sibling classes/functions with `:class:` / `:func:` roles where
it aids navigation, as `flow_matching.py` does.

## Scope — files

The pass covers the new/changed source modules on `maf` flagged PARTIAL or POOR
in the docstring audit. Pre-existing, already-compliant modules and the
untracked `reference/ml-*` ports are out of scope.

**normalizing_flows/bijections/**
- `base.py` — class docstrings present; methods (`forward`, `inverse`, …) missing.
- `chain.py` — class docstring present; methods missing.
- `permutation.py` — class docstring present; methods missing.
- `standardize.py` — class docstring present; methods + types missing.
- `transformers.py` — class docstrings present; many methods missing.
- `__init__.py` — already GOOD (consistency check only).

**models/maf/**
- `made.py` — `MADE` has Parameters; methods partial.
- `masked_linear.py` — already GOOD (consistency check only).
- `masks.py` — already GOOD (consistency check only).
- `model.py` — class docstrings present; methods (`log_prob`, `sample`, …) mostly missing.
- `__init__.py` — POOR: add a module docstring and a curated `__all__`.

**models/tarflow/**
- `blocks.py` — classes documented; methods missing.
- `conditioners.py` — classes documented; some methods missing.
- `model.py` — classes documented; methods mostly missing.
- `__init__.py` — POOR: add a module docstring and a curated `__all__`.

**models/core/**
- `patching.py` — `depatchify_2d` documented; `patchify_2d` missing.
- `tokenizers.py` — class docstrings present; methods missing.
- `__init__.py` — already GOOD (consistency check only).

**inference/**
- `posterior.py` — `NLEPosterior` has Parameters; remainder partial.
- `samplers.py` — POOR: dataclasses (`MclmcInfo`, `SmcInfo`), functions, and
  methods (`run`, …) largely undocumented.
- `__init__.py` — already GOOD (consistency check only).

**utils/**
- `serialization.py` — functions mostly GOOD; fill remaining method/return gaps.

**recipes/**
- `flow_pipeline.py` — `ConditionalFlowPipeline` documented; many methods missing.

## Conventions / decisions

- **Depth: full public API surface.** Every public class, every public method
  (`forward`, `inverse`, `log_prob`, `sample`, `run`, `standardize`, …), every
  public function, and every public dataclass (`TarFlowParams`, `MAFlowParams`,
  `MclmcInfo`, `SmcInfo`) gets a full NumPy docstring. Private helpers (`_name`)
  and trivial dunders are skipped.
- **The two POOR `__init__.py` files** (`models/maf`, `models/tarflow`) get a
  module docstring and a curated `__all__`.
- **No code or behavior edits.** If a docstring claim and the code disagree, the
  code is the source of truth: document the actual behavior and flag the
  discrepancy to the owner rather than "fixing" the code.
- **Remove the dead `docs/requirements.txt`.** It lists only four packages while
  `conf.py` imports ~ten extensions, and nothing references it: the CI `docs`
  job installs via `uv sync --group docs` and builds with `make html`, the
  authoritative dependency list living in the `[docs]` group of `pyproject.toml`.
  There is no Read-the-Docs config. The file is removed as part of this pass.
- **No new narrative/guide pages.** `autoapi` generates the per-class/function
  API reference from the docstrings written here; conceptual prose guides are a
  separate, future effort.

## Verification

Builds run in the `gensbi` mamba environment, which has the full toolchain
(sphinx 8.1.3, `sphinx-autoapi`, `numpydoc`, `myst-nb`, `sphinx-design`,
`sphinx-copybutton`, `sphinxcontrib-mermaid`, `pydata-sphinx-theme`,
`sphinxext-rediraffe`, `sphinx-togglebutton`, `sphinx-favicon`) — the same set
CI installs from the `[docs]` group.

1. **Branch build.** Build the docs from `maf` (no branch swapping, no `main`
   baseline build). Capture the warning set and filter to the touched module
   paths. Any `autoapi` / `napoleon` / `numpydoc` warning referencing a touched
   file is a failure to fix. Because pre-existing modules are not edited, any
   long-standing unrelated warnings they emit are ignored — only warnings on the
   files this pass touches gate completion.
2. **Spot-check** the rendered HTML for the new subpackages
   (`normalizing_flows`, `models/maf`, `models/tarflow`, `models/core`,
   `inference`) to confirm Parameters/Returns sections render as expected.

Build command (matching CI), run with `mamba activate gensbi`:

```bash
cd docs && make html
```

## Execution shape

The ~19 files are independent of one another, so the work batches naturally into
per-subpackage units — bijections / maf / tarflow / core / inference /
serialization+recipes — each authored against the `flow_matching.py` exemplar
and validated by the build/warning diff. The detailed task breakdown is produced
by the implementation-plan step that follows this design.

## Out of scope (YAGNI)

- No signature type-hint additions, no refactoring, no behavior changes.
- No new narrative/tutorial pages and no toctree restructuring (autoapi already
  surfaces the new subpackages).
- Pre-existing already-compliant modules and the untracked `reference/ml-*`
  ports are untouched.
