# Configurable training-time `t` sampling (logit-normal) + fixed-θ validation

Date: 2026-06-13
Status: Design approved (brainstorming), pending spec review
Repos: `GenSBI` (library feature), `GenSBI-examples` (validation experiment)
Branch: `FieldDiT`

## 1. Motivation

The first PixelDiT training on GRF-32 (`config_4b`, 50k steps) produced near-white-noise
samples with a flat radial power spectrum: the model learned the per-pixel **1-point**
variance (a normalization) but not the **2-point** spatial correlation. Forward parity
(3e-8 vs torch) and gate-2 (overfits a Nyquist-frequency checkerboard) rule out a code
bug; the failure is consistent with under-training plus a stripped training recipe.

One concrete, cheap, training-side lever the reference uses and the port dropped is the
**timestep distribution**: the paper samples `t` from a logit-normal (SD3 / Esser et al.),
GenSBI samples `t` uniformly (`flow_matching.py:129`). Logit-normal concentrates gradient
signal on the mid-noise regime where structure forms, changing the *learned velocity
field* (this is not a sampling-time speedup — it changes what the model learns).

This spec adds a **configurable training-time `t` distribution** to the flow-matching
method (default unchanged: uniform), exposes it via YAML for the experimental example
configs, and defines a **fixed-θ diagnostic run** to test whether, with enough training,
PixelDiT can learn a single GRF's 2-point structure.

Out of scope (explicitly deferred): inference-time "timeshift" / non-uniform ODE step
grid. For a well-trained velocity field any monotonic time reparametrization integrates
the same ODE to the same endpoint, so the sampling schedule is a discretization/step-count
speedup, not a correctness lever. Defer until the model learns.

## 2. Part 1 — Library feature: configurable `t` distribution

### 2.1 New pure helper

`src/gensbi/core/time_sampling.py` (new):

```python
def sample_time(key, n, *, dist="uniform", logitnorm_mean=0.0, logitnorm_std=1.0):
    """Sample n training timesteps in (0, 1).

    dist="uniform"     -> jax.random.uniform(key, (n,))  (bit-identical to the
                          previous inline call; preserves all existing runs)
    dist="logitnormal" -> sigmoid(logitnorm_mean + logitnorm_std * N(0,1))
                          (SD3 / Esser et al.; concentrates mass near
                          sigmoid(mean)). The reference's "lognorm_t" is a
                          misnomer for this logit-normal sampler.
    """
```

Pure, isolated, independently unit-testable; placed in its own module so
`score_matching` can reuse it later and `flow_matching.py` stays focused.

### 2.2 `FlowMatchingMethod` changes

`src/gensbi/core/flow_matching.py`:

- Constructor gains keyword-only params (defaults preserve current behavior):
  `FlowMatchingMethod(prior=None, *, time_dist="uniform", logitnorm_mean=0.0, logitnorm_std=1.0)`.
- Validate `time_dist in {"uniform", "logitnormal"}` in `__init__` (raise `ValueError`).
- `prepare_batch` replaces its `t = jax.random.uniform(rng_t, (n,))` line with
  `t = sample_time(rng_t, n, dist=self.time_dist, logitnorm_mean=..., logitnorm_std=...)`.

No changes to `FMLoss`, the path, the pipeline, the solver, or any model. Because
`prepare_batch` is the single training-time `t` source for every flow-matching recipe
(conditional, joint, unconditional), all of them gain the capability; with the uniform
default, none of them change behavior.

### 2.3 Convention (must be correct)

GenSBI's affine path is `x_t = (1-t)·x_0 + t·x_1` with **t=1 → data, t=0 → noise**, and
the CondOT loss is `‖v − (x_1 − x_0)‖²` with **no `1/t` weighting**. Therefore:
- `logitnorm_mean=0.0` centers mass at `t=0.5` (mid-noise) — the desired default.
- positive mean biases toward clean data, negative toward noise.
- `t ∈ {0, 1}` from sigmoid saturation is numerically safe (no singular weighting); no
  clamping is added.

### 2.4 Tests

`tests/core/test_time_sampling.py` (new):
- **Regression:** `sample_time(key, n, dist="uniform")` is bit-identical to
  `jax.random.uniform(key, (n,))` for a fixed key (guards existing runs).
- **logit-normal:** empirical mean and a few quantiles match the analytic logit-normal
  for a couple of `(mean, std)` settings (loose statistical tolerance, fixed key); all
  samples strictly in `(0, 1)`; deterministic for a fixed key; correct shape/dtype.
- **Validation:** unknown `dist` raises `ValueError`.
- **Method-level:** `FlowMatchingMethod()` (default) `prepare_batch` output is unchanged
  vs. the pre-change behavior; `FlowMatchingMethod(time_dist="logitnormal")` produces the
  logit-normal distribution.

## 3. Part 2 — Validation experiment: fixed-θ 100k diagnostic (GRF-32)

Goal: isolate "can PixelDiT learn a single GRF's 2-point structure with enough training"
from "can it generalize across θ". If yes → under-training/recipe was the issue and the
path forward is scale + logit-normal. If P(k) stays flat after 100k fixed-θ steps → the
problem is deeper (architecture for this ≤256² regime) and we reconsider PixelDiT vs
FieldDiT.

**Honest caveat (record in run notes):** with θ fixed the cond input is constant, so this
run does **not** test conditioning — the model can ignore `cond` and learn the marginal at
that θ. That is intentional; it tests generative/2-point capacity only.

### 3.1 New training script

`examples/sbi-benchmarks/gaussian_random_field/train-grf-fixedtheta.py` (new):
- Reuses the existing example's helpers (`radial_power_spectrum`, `plot_losses`,
  `plot_power_spectra`, `swap_obs_cond`, `resolve_model_section`, `build_model`). Prefer
  importing them by factoring the shared helpers into a small `grf_common.py` module in
  the same directory (the current `train-grf.py` cannot be imported directly — its name
  contains a dash); duplicating the few helpers is an acceptable fallback if factoring
  proves noisy. Resolve at implementation time.
- **Fixed-θ data:** draw one θ (configurable — see config) once, hold it constant, and
  generate fresh GRF realizations each batch via the task's simulator. Implement as a thin
  loader wrapper in the script over the existing `OnlineTaskDataset` machinery
  (`task.get_prior` → replace with the constant θ broadcast to `batch_size`; `simulator`
  unchanged). Do **not** modify `sbibm-jax`. Validation loader serves the same fixed-θ
  stream.
- Trains for **100,000 steps** (from config). Uses the new `time_sampling` knob (logit-
  normal) wired through `FlowMatchingMethod` exactly as in §4 (Wiring).
- After training: sample N fields at the fixed θ; write the loss curve and a P(k) overlay
  (truth at that θ vs. sample mean ± σ) plus a small truth-vs-samples field grid, to
  `imgs/` with a distinct experiment suffix. The P(k) overlay is the decisive plot.
- Checkpoints route to `checkpoints_pixeldit/` (already wired) under the new
  `experiment_id`, so it never collides with the conditional runs.

### 3.2 New config

`examples/sbi-benchmarks/gaussian_random_field/config/config_4f.yaml` (new; `4f` =
"4b fixed-θ"):
- `pixeldit:` section copied from `config_4b.yaml` (same architecture).
- `time_sampling:` block: `dist: logitnormal`, `logitnorm_mean: 0.0`, `logitnorm_std: 1.0`.
- `training:` `nsteps: 100000`, `experiment_id: <distinct, e.g. 41>`, `train_model: true`,
  `restore_model: false`, otherwise mirror `config_4b`.
- `fixed_theta:` the held-constant θ. Choose a θ with clear spatial structure so the 2-point
  signal is visually obvious (e.g. a moderately smooth case, `alpha ≈ 2.5–4`). Specify as
  raw `[log_std, alpha]` (the script normalizes it for the model and for the loader's
  simulator call). Configurable.
- `sampling:` reuse `nsamples`, `step_size`; `num_thetas` is effectively 1 (the fixed θ).

### 3.3 YAML schema for `time_sampling` (shared with Part 4)

```yaml
time_sampling:           # optional; omit -> uniform (backward compatible)
  dist: logitnormal      # uniform (default) | logitnormal
  logitnorm_mean: 0.0    # median t = sigmoid(mean); 0.0 -> 0.5
  logitnorm_std: 1.0
```

### 3.4 Condor submission (final step)

`GenSBI-examples/sub/train_model_grf_PixelDiT_fixedtheta.sub` (new), modeled on
`train_model_grf_PixelDiT.sub`:
- `version = 4f`, `script_path = train-grf-fixedtheta.py`, same `workdir`, resources
  (32 GB, 8 CPUs, 1 A100), `getenv = True`, logs keyed by experiment/version.
- **Final action:** submit it (`condor_submit train_model_grf_PixelDiT_fixedtheta.sub`)
  and report the job id. This is a real cluster submission — confirm with the user before
  firing, and report the outcome faithfully.

### 3.5 Optional A/B (note, not a required deliverable)

For clean attribution of any improvement specifically to logit-normal, an otherwise-
identical `dist: uniform` fixed-θ run can be submitted alongside. Mention to the user;
do not run unless requested.

## 4. Wiring `time_sampling` into the existing scripts

Both `train-grf.py` (32² and 256²) and the new fixed-θ script read
`ts = cfg.get("time_sampling", {})` and construct:

```python
method = FlowMatchingMethod(
    time_dist=ts.get("dist", "uniform"),
    logitnorm_mean=ts.get("logitnorm_mean", 0.0),
    logitnorm_std=ts.get("logitnorm_std", 1.0),
)
```

Configs without the block are unchanged (uniform). This is the only edit to the existing
`train-grf.py` scripts for Part 1.

## 5. Files touched

GenSBI (library):
- New: `src/gensbi/core/time_sampling.py`, `tests/core/test_time_sampling.py`
- Edit: `src/gensbi/core/flow_matching.py` (ctor + `prepare_batch`)

GenSBI-examples (experiment):
- New: `examples/sbi-benchmarks/gaussian_random_field/train-grf-fixedtheta.py`
- New: `examples/sbi-benchmarks/gaussian_random_field/config/config_4f.yaml`
- New: `sub/train_model_grf_PixelDiT_fixedtheta.sub`
- Edit: both `train-grf.py` scripts (read `time_sampling` block → `FlowMatchingMethod`)
- Edit (opt-in): add `time_sampling:` to PixelDiT conditional configs (`config_4b`,
  `config_6b`) if/when we want logit-normal on the full conditional runs too.
- Possibly new: `grf_common.py` (shared helpers) if factoring is chosen over duplication.

## 6. Success criteria

- Part 1: uniform default provably bit-identical (regression test green); logit-normal
  produces the correct distribution; full suite stays green.
- Part 2: the fixed-θ 100k run completes; the decisive read is its P(k) overlay —
  **success** = sample P(k) develops low-k power matching the truth at the fixed θ
  (2-point structure learned); **failure** = P(k) still flat (escalate: architecture/
  regime, reconsider PixelDiT vs FieldDiT at ≤256²).

## 7. Decisions deferred to implementation (planning-prototype-restraint)

- Whether to factor `grf_common.py` vs. duplicate the handful of plotting helpers.
- Exact `experiment_id` value for the fixed-θ run.
- Exact fixed-θ value (a structurally-clear case) and whether it is drawn from the prior
  or hard-set in the config.
- Precise form of the fixed-θ loader wrapper (inline generator vs. small class).
