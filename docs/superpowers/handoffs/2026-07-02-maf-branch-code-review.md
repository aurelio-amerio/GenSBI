# Code Review — `maf` branch → `main`

**Date:** 2026-07-02
**Scope:** `git diff main...HEAD` — ~21,800 insertions across 133 files, 50 commits (normalizing-flow track: MAF/TarFlow models, bijections, flow pipeline, NLE posterior sampling, safetensors serialization, docs pass, recovery scripts).
**Method:** medium-effort multi-agent review — 8 independent finder angles (line-by-line, removed-behavior, cross-file tracer, reuse, simplification, efficiency, altitude, conventions/docs), ~35 candidates deduped to 10, each verified by an independent verifier agent (one bug confirmed by executing the failing script).

**Verdict: NOT ready to merge yet.** The architecture and conventions are sound; four correctness items and the docs/repo-hygiene items below should land first.

---

## Answers to the review questions

### Is the NF implementation correctly integrated with the rest of the library?

**Mostly yes.** `ConditionalFlowPipeline` deliberately mirrors the flux1 flow-matching pipeline surface (`sample` / `sample_batched` / `get_sampler` / `get_log_prob_fn`, `x_o` as the conditioned variable), and `MAFlow`/`TarFlow` are drop-in for the pipeline and `NLEPosterior`. The two confirmed seams are enforcement gaps, not design flaws (Findings 2 and 3).

### Does it use the same channel and (x, θ) conventions as flux1 flow matching?

**Yes.** The `(B, dim, C)` channel convention is carried uniformly end-to-end — tokenizers, MAF, TarFlow, pipeline, and `NLEPosterior` all agree — and the conditioned-variable convention matches the FM pipelines. No place was found where the convention *itself* is violated; the problems are asymmetric *validation* of it (training path enforces the channel axis, inference path does not).

### Is the documentation up to par?

**Docstrings: yes.** The NumPy-style pass is thorough and accurate at HEAD (a stale-docstring candidate was checked and found already fixed by commit `f9370ab`).
**Docs site: no.** The entire NF/NLE track is invisible in curated docs (Finding 5), and the only end-to-end notebook is untracked and would be lost on merge.

### Is the new code well-written / non-convoluted?

**Broadly yes.** Residual cleanup that survived verification but was cut from the top-8 is listed under "Minor findings" below.

---

## Confirmed findings (ranked by severity)

### 1. `_rescale` mistranscribes the blackjax adjusted-MCLMC formula — CONFIRMED
**`src/gensbi/inference/samplers.py:10`**

The helper returns `mu / round(log2(2*mu - 1))`, but the blackjax reference (`adjusted_mclmc_dynamic.py`) is:

```python
k = jnp.floor(2 * mu - 1)
x = k * (mu - 0.5 * (k + 1)) / (k + 1 - mu)
return k + x   # ≈ 2*mu - 1, so ceil(U(0,1)*s) has expectation mu
```

**Impact:** with a typical tuned `L/step_size = 15`, the code yields `s = 3`, so integration-step draws average ~2 instead of ~15 — trajectories ~7× shorter than tuned, for **every** default `NLEPosterior.sample()` call, silently (no error; just autocorrelated chains and degraded posteriors). The docstring claims the formula is "from the blackjax adjusted-MCLMC tutorial" and neither the design doc nor the tests document a deliberate deviation — a transcription bug, not intent.

**Note:** this plausibly explains why recovery runs seemed to need `num_warmup ≈ 2000` instead of the default 500. Re-run the GPU recovery gates *after* this fix.

### 2. FM-pipeline regression: unbatched 1-D `x_o` is truncated to its first coordinate — CONFIRMED
**`src/gensbi/recipes/conditional_pipeline.py:63`** (`_single_cond_fm`, used by `get_sampler` and `get_log_prob_fn`)

On `main`, `get_sampler`/`get_log_prob_fn` passed `x_o` through `_expand_dims`, which correctly promotes a 1-D `(dim_cond,)` observation to `(1, dim_cond, 1)`. The new `_single_cond_fm` runs **before** `_expand_dims`, reads `x_o.shape[0]` as a batch size, warns, and slices to `x_o[0:1]` — shape `(1,)` — destroying the condition.

**Impact:** `pipeline.sample(key, jnp.array([0.3, -1.2]))` (previously correct) now conditions on a single scalar → downstream shape error at best, silently wrong posterior at worst, guided by a warning that misdiagnoses the input as a batch. Tests only cover 3-D inputs, so the regression is untested. (The related change to batched `get_log_prob_fn` — keep-first-with-warning — is deliberate and test-codified, so it is *not* flagged as a bug, but note it silently changes results for anyone relying on the old paired batched evaluation.)

### 3. Flow-pipeline inference path never enforces the documented channel contract — CONFIRMED
**`src/gensbi/recipes/flow_pipeline.py:270`** (also line 374 in `get_log_prob_fn`)

The training loss routes cond through `_prep_cond` → `_require_channel` (hard `ValueError` on a bare `(B, dim)` input, as the class docstring promises). The inference path routes `x_o` only through `_single_obs`, which merely checks `ndim >= 2`.

**Impact:**
- `x_o` of shape `(1, dim_cond)` (no channel axis): TarFlow `cond='vector'` fails with an opaque matmul error deep in `VectorConditioner` instead of the documented `ValueError`; MAFlow silently accepts it (coincidentally correct only for C=1). Same input accepted or rejected depending on the model.
- `x_o` of shape `(dim_cond, C)` (no batch axis): passes the ndim check, fires a bogus "batch dimension" warning, and conditions on `x_o[0]` — one coordinate's channels — a **silently wrong posterior**.

Tests only ever pass well-formed `(1, dim_cond, C)` inputs, so neither path is covered. Fix: route the inference path through the same `_require_channel` canonicalization as training.

### 4. `tarflow_image_npe_recovery.py` cannot run (branch's own landing gate) — CONFIRMED by execution
**`scripts/tarflow_image_npe_recovery.py:73`** and **:93**

- Line 73: `cond="image_prefix"` is stale after the conditioner rename (valid: `bias`/`vector`/`image`). Verified by running `--smoke`: `ValueError: unknown cond 'image_prefix'` at construction, before any training.
- Line 93: the script omits the `[..., 0]` channel squeeze its three sibling scripts apply. `pipe.sample()` returns `(nsamples, D, 1)`; `jnp.cov(s.T)` rejects 3-D input, and `mean_s` of shape `(D, 1)` is compared against an analytic `(D,)` mean.

**Impact:** the image-NPE GPU recovery gate for this branch can never produce a verdict in its current form.

### 5. NF/NLE track invisible in curated docs; the only example notebook is untracked — CONFIRMED
**`docs/examples.md:41`** (toctree) and repo root

- Grep across `docs/` (excluding auto-generated `docs/api/`) finds **zero** mentions of `MAFlow`, `TarFlow`, `ConditionalFlowPipeline`, `NLEPosterior`, `gensbi.inference`, or safetensors in `overview.md`, `inference.md`, `model_cards.md`, `training.md`, `examples.md`, or `index.md`. `training.md`'s "See Model Cards" pointer steers likelihood-dominated problems to Flux1Joint, never mentioning the new exact-density flows.
- `docs/notebooks/slcp_tarflow_nle.ipynb` is **untracked** (`git status` shows `??`) and referenced by no toctree — it will be silently lost on merge.

**Impact:** the branch's headline feature ships discoverable only via raw API reference, with zero runnable end-to-end examples.

### 6. `patchify_2d`/`depatchify_2d` moved with no back-compat re-export — CONFIRMED
**`src/gensbi/recipes/utils.py`**

The functions moved to `gensbi.models.core.patching`; all in-repo call sites were migrated, but `main`'s published docs (`docs/basics/data_and_embeddings.md`, four places) explicitly teach `from gensbi.recipes.utils import patchify_2d`, and gensbi is on PyPI (v0.3.5, live badge in README). External code following the published docs breaks with `ImportError` on upgrade.

**Fix:** one-line deprecating re-export in `recipes/utils.py` for at least one release cycle.

### 7. `reference/` gitlinks committed without `.gitmodules` — CONFIRMED by direct inspection
**`reference/flowjax`, `reference/ml-starflow`, `reference/ml-tarflow`**

`git ls-tree HEAD reference/` shows three mode-160000 entries; no `.gitmodules` exists.

**Impact:** fresh clones get permanently empty directories; `git submodule update --init` errors with "no submodule mapping found in .gitmodules"; CI with submodule checkout would break. Either add `.gitmodules` or drop the gitlinks before merging.

### 8. `sample_batched` runs B sequential AR sampling passes instead of one batched call — CONFIRMED
**`src/gensbi/recipes/flow_pipeline.py:344`**

The loop calls `get_sampler(x_o[i:i+1])(keys[i], nsamples)` per condition, yet both `MAFlow.sample` and `TarFlow.sample` natively accept a batched cond (indeed `get_sampler` itself already broadcasts one condition to a single batched call).

**Impact:** SBC/TARP diagnostics with B in the hundreds pay B sequential TarFlow autoregressive scans (the expensive direction) — near-B× avoidable wall-clock. Fix: `cond = jnp.repeat(x_o, nsamples, axis=0)`, one `flow.sample` call, reshape to `(nsamples, B, dim, C)`; chunk if `B*nsamples` memory is a concern.

---

## Minor findings (verified, below the reporting cut)

- **MCLMC runs chains sequentially in Python** (`samplers.py:169`) — `num_chains×` wall-clock; the in-code "MclmcInfo is not a pytree" blocker is not fundamental (return raw arrays, `jax.vmap`, build info after; the `float(mu)` host concretizations at lines 204–241 must also move out of the traced path). Mitigated by `num_chains=1` default and the module's stated "convenience sampler" scope. Related: `_inference_loop` (line 106) stacks the full state pytree per step when only positions and acceptance rates are consumed (~4× output memory).
- **`_fit_stat`/`set_standardization` copy-pasted** between `tarflow/model.py:288` and `maf/model.py:208`. Behavior verified identical for every stat shape the pipeline produces (`(dim,)`, `(dim, C)`, `(C,)`) — so maintenance risk only — but docstrings have already drifted (MAFlow's omits the `(C,)` case its code supports). Hoist into one shared helper (e.g. `models/core` or the `Standardize` bijection).
- **Warn-and-take-first helper exists in three near-copies** (`flow_pipeline.py:27` `_warn_if_batched`/`_single_obs`, `conditional_pipeline.py:52` `_single_cond_fm`) with identical warning text and subtly different semantics. Extract one shared helper in `recipes/utils.py` — and consider a hard error instead of warn+truncate, since silently discarding B−1 observations is the classic source of wrong-but-passing scientific results.
- **`MetaBlock` takes both `perm` and `inv_perm`** (`tarflow/blocks.py:128`) though `inv_perm` is derivable; a mismatched pair silently breaks invertibility. Compute `argsort(perm)` internally, as the sibling `Permutation` bijection already does.
- **`serialization.py` module docstring points to an internal spec path** (`docs/superpowers/specs/...`) that doesn't exist in an installed wheel; autoapi renders it on the public API page.
- **Recovery scripts are two near-identical copy-paste pairs** (~200 shared lines × 2 copies each); factor the common simulator/train/check harness before they drift further.

## Refuted / downgraded candidates (checked, not bugs)

- **TarFlow `log_prob`/`sample` cond docstrings "only describe the bias path"** — refuted: already fixed at HEAD (`f9370ab`) documents both `(B, cond_dim)` and `(B, cond_dim, C_cond)`. Residual nit: "(flattened internally by the conditioner)" is only true for `AdditiveBias`, and the `cond='image'` shape is still undescribed in those two methods.
- **`NLEPosterior.build_target` squeeze canonicalization breaks batched/multichannel `x_o`; scalar prior raises IndexError** — refuted: both edges are outside the documented contract (single observation; multichannel served by `structured_obs=True`; the class docstring requires `prior.sample(key, ()) → (dim,)`, and the project's `make_gaussian_prior` always builds a vector-event `Independent`). Earlier validation with clearer errors would still be nice.
- **Standardization behavior diverges between TarFlow and MAFlow** — refuted numerically: both `_fit_stat` bodies are logically identical and handle all pipeline-produced stat shapes the same way (survives only as the duplication cleanup above).
- **`requires-python` bumped 3.11→3.12 and CI dropped 3.11; `docs/requirements.txt` deleted** — flagged by finders but likely deliberate housekeeping (the commit message calls the requirements file dead and the docs build gate is on `uv sync --group docs`). Worth a one-line confirmation that no Read-the-Docs dashboard config still points at `docs/requirements.txt`, and that dropping 3.11 support is intentional for the next release.

---

## Recommended merge checklist

1. Fix `_rescale` (Finding 1) — then **re-run the GPU recovery gates**, which were already the branch's outstanding gate and may have been distorted by this bug.
2. Fix the two condition-validation seams (Findings 2, 3) + add tests for 1-D `x_o` and channel-less inference inputs.
3. Repair `tarflow_image_npe_recovery.py` (Finding 4).
4. `git add` the SLCP notebook; wire NF/NLE into `examples.md`, `overview.md`, `inference.md`, and a model card (Finding 5).
5. Add the `patchify_2d` deprecating re-export (Finding 6).
6. Resolve the `reference/` gitlinks (Finding 7).
7. Optional but cheap: batched `sample_batched` (Finding 8) and the minor-findings cleanups.
