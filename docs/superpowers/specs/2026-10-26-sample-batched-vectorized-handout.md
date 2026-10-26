# Handout: vectorized `sample_batched` for many conditions × few samples (2026-10-26)

Status: **problem statement only — not designed, not scheduled.** Surfaced by the
GenSBI-examples notebook re-check (`GenSBI-examples/docs/superpowers/specs/2026-10-26-notebook-recheck-handout.md`,
GenSBI issue 3). Branch `sampler-kwargs-fixes` fixed the sibling issues
(unknown sampler kwargs now raise, `sample_batched` forwards `model_extras`,
`JointPipeline.sample` batch warning); this one needs its own design pass.

## 1. The problem

`AbstractPipeline.sample_batched` (`src/gensbi/recipes/pipeline.py`) draws
samples for `B` conditions with a Python loop: one `sampler(...)` device call
per condition, plus `n_chunks` calls per condition when `chunk_size` is set.
It is the right shape for the common SBC/TARP workload (hundreds of conditions,
thousands of samples each, memory-bound per condition).

It is the wrong shape for the *other* common workload: **many conditions, one
(or a handful of) sample(s) each**. That is what L-C2ST needs
(`N_cal = 10 000` conditions × 1 sample) and what the SBC/TARP "one draw per
prior-predictive" variants need. Users used to get it in one device call with

```python
# old pattern (≤ 0.3.x): one sample per condition, vectorized
samples = pipeline.sample(key, x_o=xs, nsamples=xs.shape[0])   # xs: (N, dim_cond, C)
```

because the conditional models broadcast a cond batch of size `nsamples`
against the `nsamples` initial-noise rows. Since 0.4.0 (#51, `_single_obs` in
`recipes/utils.py`) `ConditionalPipeline.sample` **raises** on a leading batch
axis > 1 ("will not silently discard the rest. Use `sample_batched()`"), and
`sample_batched(key, xs, nsamples=1)[0]` is a 10 000-iteration Python loop.
Each iteration is a full solver run on a batch of one. The notebook re-check
rewrote every L-C2ST cell this way but never measured it at full size.
There is no isolated timing of the loop. The only numbers available are
whole-notebook CPU wall times with L-C2ST cut to 2 000 conditions:
`two_moons/flow_flux` 15.8 min end to end, `my_first_model` spot check
35 min (notebook re-check, 2026-10-26). Nobody has run 10 000 conditions.

`JointPipeline.sample` is inconsistent with this: it only *warns* on a batched
`x_o`, and its sampler still carries explicit support for the
"one condition per sample" case (padding `cond` alongside `x_init` when
`cond.shape[0] == nsamples > 1`, `joint_pipeline.py` ~L405). So the vectorized
path exists for joint models and is forbidden for conditional ones.

## 2. What a fix needs to decide

1. **Where the vectorized path lives.** The single-observation contract of
   `sample` was a deliberate choice in #51 (no silent discarding). The natural
   home is `sample_batched`, which already owns the "batch of conditions"
   semantics and already bypasses `_single_obs` (it calls `get_sampler` on
   `x_o[0:1]` for shape only and then overrides `model_extras` per call).

2. **The mechanism.** Two candidates:
   - *Tile-and-flatten:* build `cond_all = repeat(x_o, nsamples)` of shape
     `(B*nsamples, dim_cond, C)` and draw `B*nsamples` samples in one call
     (rows `i*nsamples:(i+1)*nsamples` belong to condition `i`), then reshape
     to `(nsamples, B, ...)`. This is exactly what
     `ConditionalFlowPipeline.sample_batched` (`recipes/flow_pipeline.py`)
     already does for MAF/TarFlow, chunking over flattened rows. It relies on
     the model accepting a cond batch equal to the obs batch, which Flux1 does
     and `MockConditionalModel` enforces via `broadcast_to`.
   - *`vmap` over conditions:* keeps per-condition semantics but recompiles
     per `B`, and interacts badly with the solver's internal `jit`/`scan`.
   Tile-and-flatten is the obvious pick; the design should say why not vmap.

3. **Chunking semantics.** Today `chunk_size` bounds *samples per device call
   for one condition*. With flattening it should bound *rows per device call*
   regardless of condition boundaries (as in the flow pipeline). Decide whether
   to keep the per-condition loop as the path for large `nsamples` (it has a
   bit-identical "historical behavior" guarantee in `test_chunked_sampling.py`)
   and add the flattened path only when `nsamples` is small, or to switch
   wholesale and accept a key-stream change. The former preserves the
   documented determinism; the latter is simpler.

4. **Sharding.** `_sharded_init` pads `x_init` to the data-parallel size and
   `_replicate`s the extras. A flattened `cond_all` must be padded and sharded
   like `x_init` (the joint pipeline's padding branch is the template); the
   conditional pipeline currently has no such branch because its `cond` is
   always a single row.

5. **SDE solvers.** `MockConditionalModel`'s comment notes SDE solvers expect
   `B = 1` for `cond`. Check whether `ZeroEndsSolver`/`NonSingularSolver` and
   the EDM/SM samplers are fine with `cond.shape[0] == x.shape[0]`, or whether
   the flattened path must be ODE-only.

6. **Key handling.** The loop splits `key` into `B` per-condition keys. The
   flattened path draws `B*nsamples` rows from one key. Document that results
   differ from the loop for the same key.

## 3. Acceptance

- `sample_batched(key, xs, nsamples=1)` with `xs` of 10 000 conditions runs
  in a handful of device calls, not 10 000.
- Output shape and dtype unchanged: `(nsamples, B, dim_obs, ch_obs)`.
- Works for FM (ODE), EDM and SM methods; SDE solvers either work or raise a
  clear error.
- Multi-device: a `mesh` test with `B*nsamples` not divisible by the
  data-parallel size.
- The L-C2ST cells in `my_first_model` and the six `two_moons` notebooks run
  at full size on a GPU in reasonable time (the gate that is currently
  untested).

## 4. Pointers

- `src/gensbi/recipes/pipeline.py`: `sample_batched`, `_chunked_draw`,
  `_sharded_init`, `_trim`, `_replicate`.
- `src/gensbi/recipes/conditional_pipeline.py`: `get_sampler` (per-call
  `model_extras` override), `sample`.
- `src/gensbi/recipes/joint_pipeline.py`: `get_sampler` padding branch for
  `cond.shape[0] == nsamples`.
- `src/gensbi/recipes/flow_pipeline.py`: `sample_batched` flattened-rows
  implementation (prior art).
- `src/gensbi/recipes/utils.py`: `_single_obs`.
- `tests/recipes/test_chunked_sampling.py`: current determinism guarantees.
