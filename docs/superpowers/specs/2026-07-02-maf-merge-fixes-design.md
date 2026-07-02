# maf-branch merge fixes — design

**Date:** 2026-07-02
**Input:** code review handoff `docs/superpowers/handoffs/2026-07-02-maf-branch-code-review.md` (8 confirmed findings + verified minors). Verdict there: not ready to merge.
**Goal:** land every merge-blocking fix on `maf`, plus the cheap extras agreed below, so the branch is mergeable once the GPU recovery gates pass.

## Scope decisions (owner-approved)

- **In:** Findings 1–8, plus these minors: shared single-observation helper, MetaBlock `inv_perm` removal, `_fit_stat` hoist, `serialization.py` docstring path fix.
- **Out (post-merge backlog):** vmapped multi-chain MCLMC and `_inference_loop` memory reduction (default `num_chains=1`, nobody pays today); recovery-script harness dedup (run-once gate scripts, not library code).
- **Confirmed intentional, no action:** `requires-python` 3.11→3.12 bump; `docs/requirements.txt` deletion (docs build gate is `uv sync --group docs`). Recorded here so future reviews stop flagging them.
- **Execution:** single ordered pass on `maf`, one commit per work unit, TDD for the correctness fixes. No parallel subagents — Findings 2/3/8 and the shared helper all touch `flow_pipeline.py`.

## Work unit 1 — `_rescale` formula (Finding 1)

`src/gensbi/inference/samplers.py:10` mistranscribes the blackjax adjusted-MCLMC step-count rescaling. Purpose of the function: adjusted MCLMC randomizes integration steps per proposal as `ceil(U(0,1) · s)` to avoid resonances; `_rescale(mu)` must return the `s` for which `E[ceil(U·s)] = mu`, where `mu = L/step_size` comes from tuning. Correct formula (blackjax `adjusted_mclmc_dynamic.py`):

```python
k = jnp.floor(2 * mu - 1)
x = k * (mu - 0.5 * (k + 1)) / (k + 1 - mu)
return k + x
```

The buggy `mu / round(log2(2mu-1))` yields s=3 at mu=15 → proposals average ~2 steps instead of ~15, on the default `NLEPosterior.sample()` path, at both call sites (inside the tuning kernel and in the final sampler). Consequence is silent: exact but slow-mixing chains (plausibly the observed `num_warmup` 500-vs-2000 discrepancy).

- Fix the formula; keep the blackjax attribution in the docstring (now truthful).
- Rework `_check_rescale_domain` for the new formula's domain (`mu > 0.5`; no more `log`), keeping a host-side guard with the same actionable error message.
- Test: for `mu ∈ {1.5, 5.3, 15.0}`, assert `E[ceil(U·s)]` equals `mu` analytically (closed form over the floor/frac split of `s`), not by sampling.

## Work unit 2 — condition canonicalization (Findings 2+3, batch policy, shared helper)

One shared helper in `recipes/utils.py` replaces the three near-copies (`_warn_if_batched`/`_single_obs` in `flow_pipeline.py`, `_single_cond_fm` in `conditional_pipeline.py`):

```python
_single_obs(x_o, *, require_channel: bool, name="x_o") -> (1, dim, C) array
```

Semantics, in order:

1. **Promote first.** FM pipeline (`require_channel=False`): keep the historical `_expand_dims` leniency — 1-D `(dim,)` → `(1, dim, 1)`. Flow pipeline (`require_channel=True`): route through the same `_require_channel` used by the training path — hard `ValueError` on channel-less input, exactly as the class docstring documents.
2. **Then judge batch.** After canonicalization, `B > 1` raises `ValueError` naming the fix ("these methods take a single observation — use `sample_batched` or vectorize"). This **replaces** the warn-and-take-first behavior introduced in `4cc400b` (owner decision: silent discarding of B−1 observations is unacceptable in scientific code); the tests codifying warn+truncate are updated to expect the error.

This kills Finding 2 (FM: 1-D `x_o` read as a batch and truncated to one scalar coordinate) and Finding 3 (flow: `(dim_cond, C)` misdiagnosed as a batch → silently wrong posterior; `(1, dim_cond)` accepted or rejected depending on model) because misshaped input can no longer survive to the batch check.

Call sites: `conditional_pipeline.py` `get_sampler`/`get_log_prob_fn`; `flow_pipeline.py` `get_sampler`/`sample`/`get_log_prob_fn` (`sample_batched` keeps its own batched contract, see unit 4).

New tests: FM pipeline with 1-D `x_o` samples correctly (the regression case); flow pipeline `(1, dim)` and `(dim, C)` inputs raise the documented `ValueError` for **both** MAFlow and TarFlow; batched `(B>1, dim, C)` raises on both pipelines.

## Work unit 3 — repair `tarflow_image_npe_recovery.py` (Finding 4)

- Line 73: `cond="image_prefix"` → `cond="image"` (post-rename value).
- Line 93: add the `[..., 0]` channel squeeze the three sibling scripts apply before `jnp.cov`/mean comparison.
- Verify by running `--smoke` (the same execution that confirmed the bug).

## Work unit 4 — batched `sample_batched` (Finding 8)

`flow_pipeline.py:344`: replace the B-iteration Python loop (B sequential autoregressive sampling passes) with one batched call: repeat each condition `nsamples` times, single `flow.sample` over `B·nsamples`, reshape back. Output shape and which-samples-belong-to-which-condition must match the current loop exactly; test compares shape and per-condition statistics against the analytic posterior (exact sample equality not required — key splitting changes). No chunking (YAGNI until someone hits memory limits).

## Work unit 5 — small model cleanups

- **MetaBlock (`tarflow/blocks.py:128`):** constructor takes only `perm`; `inv_perm = jnp.argsort(perm)` computed internally (as the `Permutation` bijection already does). Removes the mismatched-pair invertibility footgun. Update call sites; existing invertibility tests cover it.
- **`_fit_stat` hoist:** one shared module-level implementation in `models/core` handling stat shapes `(dim,)`, `(dim, C)`, `(C,)`; both `MAFlow.set_standardization` and `TarFlow.set_standardization` delegate. Single docstring documents all three shapes (fixes MAFlow's drifted copy omitting `(C,)`).
- **`serialization.py`:** drop the internal `docs/superpowers/specs/...` path from the module docstring (keep the rationale sentence); autoapi renders this on the public API page.

## Work unit 6 — repo hygiene

- **Finding 6:** deprecating re-export of `patchify_2d`/`depatchify_2d` in `recipes/utils.py` via module-level `__getattr__` emitting `DeprecationWarning` pointing to `gensbi.models.core.patching`. Keep one release cycle. (Published docs on `main` teach the old import path; gensbi is live on PyPI.)
- **Finding 7:** `git rm --cached` the three `reference/` gitlinks (flowjax, ml-starflow, ml-tarflow); add `reference/` to `.gitignore`. Local checkouts untouched. Fresh clones currently get empty dirs and submodule-aware CI errors.

## Work unit 7 — docs (Finding 5)

- **New page `docs/advanced/normalizing_flows.md`** (added to `advanced/index.md` toctree) — the single prose home for the NF track. Content: experimental-support framing (APIs may change); what MAFlow and TarFlow are and how they differ (masked-MLP MAF vs transformer-autoregressive flow; exact one-pass log-density vs FM's ODE integration); when to prefer them over flow matching; training via `ConditionalFlowPipeline`; NLE posterior sampling via `NLEPosterior` with MCLMC/TemperedSMC; safetensors save/load; pointer to the SLCP notebook.
- `git add docs/notebooks/slcp_tarflow_nle.ipynb`; wire into the examples toctree.
- Short entries in `overview.md`, `inference.md`, and model cards for MAFlow/TarFlow, each **linking to the advanced page** rather than duplicating prose; update `training.md`'s "likelihood-dominated → Flux1Joint" pointer to mention the exact-density flows.
- Docs build must pass.

## Verification & merge gate

- Fast test suite green in the **mamba `gensbi` env** (not `.venv`) after every work unit.
- `--smoke` runs of all four recovery scripts at the end.
- **Final gate (owner):** full GPU recovery runs of all four scripts *after* the `_rescale` fix — prior runs may have been distorted by Finding 1; re-examine whether `num_warmup=500` default now suffices.

## Commit order

1. `_rescale` fix + domain-guard rework
2. canonicalization unit (shared `_single_obs`, hard-error batch policy, tests)
3. `tarflow_image_npe_recovery.py` repair
4. batched `sample_batched`
5. MetaBlock `inv_perm` removal
6. `_fit_stat` hoist
7. `patchify_2d` deprecating re-export
8. `reference/` gitlink removal + `.gitignore`
9. docs: advanced NF page + notebook + wiring
