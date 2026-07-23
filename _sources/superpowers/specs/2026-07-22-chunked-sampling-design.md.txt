# Chunked Sampling Design

**Date:** 2026-07-22
**Status:** Approved (brainstorming session)

## Problem

Deep models OOM when drawing many posterior samples in one device call
(e.g. 10k–100k samples for a single condition). The existing code has no
working mitigation:

1. `AbstractPipeline.sample_batched` (pipeline.py) loops over conditions
   one at a time, but draws **all** `nsamples` per condition in a single
   call. Its `chunk_size=50` and `show_progress_bars` parameters are
   dead — accepted and documented, never used.
2. `_get_batch_sampler` (pipeline.py:136) is unused legacy: it vmaps the
   sampler over one PRNG key *per sample*, duplicating (worse) what
   samplers already do natively with a batched `x_init`.
3. `FlowPipeline.sample_batched` (MAF/TarFlow) runs **one**
   `B * nsamples` autoregressive pass — memory scales with the product —
   and explicitly swallows `chunk_size` "for interface compatibility".

## Design

One shared helper, one nested code path. No split between
"single-condition" and "multi-condition" helpers: the outer loop over
conditions already exists, and inner chunking over `nsamples` covers
both failure modes (B = 1 degenerates to pure nsamples-chunking).

### `_chunked_draw` helper (pipeline.py)

Replaces `_get_batch_sampler` (deleted).

```python
def _chunked_draw(sampler, key, nsamples, chunk_size,
                  show_progress_bars=True, concat_axis=0,
                  sampler_kwargs=None):
```

- `chunk_size is None` **or** `chunk_size >= nsamples`: single call
  `sampler(key, nsamples)` with the **original** key — bit-identical to
  current behavior. This is the backward-compatibility guarantee.
- Otherwise: split `key` into `ceil(nsamples / chunk_size)` keys, call
  `sampler(key_i, chunk_size)` per chunk, `block_until_ready()` each
  chunk (accurate progress bar, paces the host), concatenate along
  `concat_axis`.
- The final chunk may be smaller than `chunk_size` → at most one extra
  JIT trace. Padding is deliberately not used (wasted compute,
  especially for AR flows); revisit only if the extra trace proves
  costly in practice.
- `concat_axis` handles `return_intermediates=True`: solver output is
  `(n_steps, nsamples, dim, C)` with a **static** step axis (steps come
  from `nsteps`/`time_grid`, not adaptive control), so chunks
  concatenate along axis 1. Callers pass
  `concat_axis=1 if return_intermediates else 0`.
- `sampler_kwargs` (e.g. `{"model_extras": ...}`) is forwarded to every
  sampler call, for `AbstractPipeline.sample_batched`'s per-condition
  extras swapping.
- Progress bar: `tqdm` over chunks, shown only when chunking is active
  and `show_progress_bars=True`.

### `sample()` gains chunking

`chunk_size: Optional[int] = None` and `show_progress_bars: bool = True`
are added to `sample()` in:

- `conditional_pipeline.py`
- `joint_pipeline.py`
- `flow_pipeline.py`
- `unconditional_pipeline.py`

Each body changes from `return sampler(key, nsamples)` to
`return _chunked_draw(sampler, key, nsamples, chunk_size, ...)`. No
sampling logic changes — the `(key, nsamples) -> samples` closure from
`get_sampler` is the seam.

The three Simformer overrides (`SimformerFlowPipeline`,
`SimformerSMPipeline`, `SimformerDiffusionPipeline` in simformer.py)
add `chunk_size=None, show_progress_bars=True` **explicitly** to their
signatures (house style there is explicit params, not `**kwargs`) and
forward them to `super().sample(...)`.

Pipelines supporting `return_intermediates` (joint/diffusion paths) set
`concat_axis=1` when the flag is on.

### `AbstractPipeline.sample_batched`

Keeps the outer per-condition loop and the build-sampler-once pattern.
The inner full-`nsamples` call becomes a `_chunked_draw` call with
`sampler_kwargs={"model_extras": extras_i}` — its `chunk_size` and
`show_progress_bars` parameters finally do something. Default
`chunk_size` changes from the dead `50` to `None` (no chunking — same
behavior as today, now honestly documented). Progress bar: a single
tqdm over `B × n_chunks` total chunks.

### `FlowPipeline.sample_batched`

Chunks the **flattened** `B * nsamples` batch: build
`cond = repeat(x_o, nsamples, axis=0)` as today, slice it (and the
split keys) into `chunk_size`-sized pieces along axis 0, one
`flow.sample(key_i, cond=cond_chunk)` per piece, concatenate, then the
existing reshape/moveaxis to `(nsamples, B, dim_obs, C)`. Chunk
boundaries may cross conditions — irrelevant, the pass is fully
batched per row. `chunk_size=None` keeps the current one-pass call
bit-identical. This path does not use `_chunked_draw` directly (it
chunks `cond` alongside keys) but follows the same contract.

### Semantics of `chunk_size`

One meaning everywhere: **maximum number of samples drawn per device
call**. Default `None` = no chunking (current behavior, zero surprise);
users opt in when memory requires it.

## Testing

Existing tests are untouched (default path is bit-identical by
construction). New tests:

1. `chunk_size >= nsamples` → bit-identical to unchunked (same key
   path).
2. Chunked draw (`nsamples` not a multiple of `chunk_size`) → correct
   output shape, finite values, sane statistics (chunked and unchunked
   use different key splits, so no bit-equality expected).
3. `return_intermediates=True` + `chunk_size` → shape
   `(n_steps, nsamples, dim, C)` with correct sample count.
4. `AbstractPipeline.sample_batched` with `chunk_size` set → shape
   `(nsamples, B, dim_obs, C)`, per-condition results consistent with
   unchunked run at the statistical level.
5. `FlowPipeline.sample_batched` chunked → same shape contract as the
   one-pass version.
6. Simformer `sample(..., chunk_size=...)` smoke test (passthrough
   works, no `TypeError`).

## Out of scope

- NLE/MCMC posterior sampling paths (`gensbi.inference`) — different
  memory profile, separate concern.
- Automatic chunk-size selection / OOM-retry heuristics.
- Chunking the condition dimension in groups > 1 (vmap over
  conditions) — the per-condition loop stays.
