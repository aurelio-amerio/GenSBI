# Handout: three numerics tests fail on JAX 0.12.1 (2026-11-23)

Status: **resolved — see §6 (root causes, fix, evidence).** Original status: problem statement
and reproduction — cause not identified, nothing fixed. Found
on PR #55 (`fix/diagnostics-calibration`), whose `test` job went red for reasons unrelated to
that PR. This worktree (`worktree-jax-0.12.1-numerics`) is branched from `main` at `0dea64bc`
and contains no code changes.

## 1. The problem

CI's `test` job installs the newest dependencies that `pyproject.toml` allows
(`uv sync --group test`; `uv.lock` is gitignored, so nothing is pinned). `jax` is constrained
to `>=0.11.1, <0.13.0`, so the job picked up **jax / jaxlib 0.12.1** when it was released.
With that version three tests of the fast tier fail:

| test | failure |
|---|---|
| `tests/models/maf/test_maf_precision.py::test_refactor_is_bit_identical` | log-probs differ from the stored values in the last float32 digit |
| `tests/models/tarflow/test_tarflow_precision.py::test_refactor_is_bit_identical` | same |
| `tests/experimental/models/pixeldit/test_training.py::test_tiny_overfit_and_cond_sensitivity` | loss drops 52.6x after 3000 steps, the test requires 100x |

`main` is affected in the same way: its last CI run (2026-10-27, `0dea64bc`) passed only
because it resolved jax 0.12.0.1. Any PR opened now gets a red `test` job.

The `test-jax-floor` job, which pins `jax==0.11.1`, still passes.

## 2. Evidence

**CI, PR #55, run 45221344336, job `test`** (ubuntu-26.04 runner): `3 failed, 1292 passed,
2 skipped`. Versions installed: jax 0.12.1, jaxlib 0.12.1, numpy 2.5.5, optax 0.2.9,
flax 0.12.12, equinox 0.13.8, diffrax 0.7.2, scipy 1.18.1.

**CI, `main`, run 40486100994, job `test`** (ubuntu-24.04 runner): all passed. Versions:
jax 0.12.0.1, jaxlib 0.12.0.1, numpy 2.5.4, optax 0.2.8, the rest identical.

**Local reproduction** (Intel Xeon Gold 6130, the repo's `.venv` with jax 0.12.0.1,
numpy 2.5.4, optax 0.2.8, flax 0.12.10), changing only jax and jaxlib through a throwaway
overlay so that `.venv` is not touched:

```bash
for v in 0.12.0.1 0.12.1; do
  uv run --no-sync --with "jax==$v" --with "jaxlib==$v" python -m pytest \
    tests/models/maf/test_maf_precision.py::test_refactor_is_bit_identical \
    tests/models/tarflow/test_tarflow_precision.py::test_refactor_is_bit_identical \
    tests/experimental/models/pixeldit/test_training.py::test_tiny_overfit_and_cond_sensitivity \
    -q -n 3 -p no:cacheprovider
done
```

| jax / jaxlib | result |
|---|---|
| 0.12.0.1 | 3 passed |
| 0.12.1 | 3 failed, with the same numbers as CI |

So the jax version alone is enough to cause it. numpy 2.5.5, optax 0.2.9 and the newer
runner image are not needed to reproduce.

The numbers under 0.12.1 (stored value in brackets where it differs):

- MAF log-probs: `-2.781574, -3.5820527, -5.5947804 [-5.5947814], -3.1017501 [-3.1017504], -3.853031`
- TarFlow log-probs: `-2.7815738, -3.5820527, -5.59478 [-5.594781], -3.1017501, -3.8530312 [-3.853031]`
- PixelDiT overfit: loss `3.7532 -> 0.0713`, ratio 52.6x. The test's docstring records
  118-136x at jax 0.11.1 and 0.12.0.1.

**The existing workaround does not help.** `pyproject.toml` sets
`XLA_FLAGS=--xla_cpu_enable_platform_dependent_math=false` for the test suite, added for an
earlier rounding change (XNNPACK's f32 dot kernel on AMD Zen 4/5, google/XNNPACK#11736,
which drove the same PixelDiT ratio to about 57x). Evaluating the MAF log-probs directly,
with and without that flag:

| jax / jaxlib | flag set | bit-identical to the stored values |
|---|---|---|
| 0.12.0.1 | yes | yes |
| 0.12.0.1 | no | yes |
| 0.12.1 | yes | no |
| 0.12.1 | no | no |

On this Intel CPU the flag changes nothing in either version, and 0.12.1 differs in both
cases.

## 3. What is not known

- **Which change in 0.12.1 moved the rounding.** Not looked up. The 52.6x ratio is close to
  the 57x of the XNNPACK episode, which suggests the same kind of kernel change now applies
  on more CPUs or is no longer switched off by the flag. That is a guess, not a finding.
- **Whether another flag or setting restores the old arithmetic in 0.12.1.**
- **Whether the PixelDiT result is only slower convergence or a worse optimum.** The run was
  not extended beyond 3000 steps, and the loss curve was not compared between versions.
- **Whether the slow and experimental tiers are affected.** CI stops after the fast tier
  fails, so they did not run. They were not run locally under 0.12.1 either.
- **Whether GPU results change.** Everything above is CPU.

## 4. Options

These are not exclusive.

1. **Cap `jax<0.12.1`** in `pyproject.toml` (all four entries: the base dependency and the
   `cuda12`, `cuda13`, `tpu` extras). Turns CI green at once. It also stops users from
   installing 0.12.1 with GenSBI, although nothing here shows the library misbehaving on it:
   the failures are last-digit differences and one convergence threshold. A CI-only pin
   (as `test-jax-floor` does with `uv pip install`) avoids restricting users, but then the
   `test` job no longer tests the newest release.
2. **Compare the MAF and TarFlow log-probs with a tolerance** instead of `jnp.array_equal`.
   The stored values were captured to prove that one refactor (the mixed-precision work) was
   a no-op. Exact equality across JAX releases is a stronger promise than that, and this is
   the second release to change CPU rounding. The observed differences are about 2e-7
   relative; a tolerance near 1e-6 would pass 0.12.1 and still catch a real change.
3. **Investigate the PixelDiT test before touching its threshold.** Its assertion message
   says not to weaken it without investigating. Things to establish: what changed in
   0.12.1, whether a flag restores the old path, and whether more steps reach 100x.

## 5. Related

- `tests/experimental/models/pixeldit/test_training.py` is under `tests/experimental/` but
  `test_tiny_overfit_and_cond_sensitivity` carries no `experimental` marker, so it runs in
  the fast tier. Whether that is intended was not checked.
- `uv.lock` being gitignored is what lets a dependency release break CI without any commit.
- PR #55 is blocked on a green `test` job by this issue, not by its own changes.

## 6. Resolution (2026-11-23, same day)

Environment: worktree `.venv` built with `uv sync --group dev --upgrade` (jax/jaxlib 0.12.1, the
newest release; numpy 2.5.5, optax 0.2.9, flax 0.12.12, grain 0.2.19). No dependency bump was
needed for the fix. Intel Xeon Gold 6130.

### 6.1 MAF / TarFlow goldens: the RNG moved, not the model

`JAX_ACCURATE_ERF_INV=False` alone restores both sets of log-probs bit for bit
(`JAX_ACCURATE_DEFAULT_ERFC`, `JAX_SOFTMAX_CUSTOM_JVP`, `JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS` do
nothing). jax 0.12.1 made the accurate `erf_inv` the default, and `jax.random.normal` is built on
it: the first float of `jax.random.normal(PRNGKey(1), (5, 3))` changes in its last bit. The tests
draw both the input `x` and the initial weights from `jax.random`, so they moved by ~1 ULP and
the log-probs by <=1.8e-7 relative. The model's arithmetic is unchanged.

**Follow-up: the goldens tested nothing about the models, so they were removed.** They were
captured from GenSBI's own JAX code (not PyTorch) on 2026-07-20 (mixed-precision plan, Tasks 7/8).
Both models default to `zero_init=True`, so at init each flow is the identity and its log-prob is
the base density: MAF's goldens equal `N(0, I).logpdf(x)` exactly and TarFlow's to 1 ULP, which is
why the two "independent" golden sets are the same numbers. They pinned `jax.random.normal` bits,
which is why an `erf_inv` change broke them, and they duplicated the existing in-run
`test_zero_init_*_is_standard_normal` tests. Replaced in both `*_precision.py` files by the bf16
compute-knob tests every other model already had, on a non-trivial flow (`zero_init=False`):
fp32 master weights + fp32 finite log-prob under bf16 compute, and bf16 log-prob differs from fp32
(knob reaches the compute) by < 2e-2 relative (measured 1.2e-3 MAF, 4.5e-3 TarFlow). No stored
numbers, so nothing to drift across JAX releases.

### 6.2 PixelDiT overfit: an unstable test recipe, reshuffled by rounding

Evidence, in the order it was gathered (ratio after 3000 steps; test threshold 100x):

1. Seed 2 loss curves: 0.12.0.1 -> 119x; 0.12.1 -> 52.6x; 0.12.1 with `erf_inv` off -> **1.0x**
   (no learning at all).
2. Crossover with pinned init weights (erf_inv off): 0.12.1 arithmetic + 0.12.0.1 init -> 1.0x;
   0.12.0.1 arithmetic + 0.12.1 init -> 118.9x. Init is irrelevant; the training-step arithmetic
   decides.
3. Flag toggles on that pinned setup: `JAX_SOFTMAX_CUSTOM_JVP=False` -> 129x; the constants and
   erfc flags leave the trajectory bit-identical (they do not touch this model).
4. But it is **not** a softmax defect. Both softmax gradient rules are exact (new:
   `y*(xdot - sum(y*xdot))`; old: autodiff through exp/sum/div with stop-gradient max). 8-seed
   sweeps (model seeds 0-7):

   | config | failures (<100x) |
   |---|---|
   | 0.12.0.1 default | 0/8 (114-140x) |
   | 0.12.1 default | 2/8 (67x, 53x) |
   | 0.12.0.1 + `JAX_SOFTMAX_CUSTOM_JVP=True` | 2/8 (67x, 11x) |
   | 0.12.1 + `JAX_SOFTMAX_CUSTOM_JVP=False` | 1/8 (11x) |

   Every rounding perturbation, including turning the *old* rule back on, just moves which seeds
   fail; pooled ~1 in 5 fail, and 0.12.0.1's 0/8 is consistent with luck (P ~ 0.17).
5. Mechanism (traced every 20 steps): the failing run tracks the healthy one until warmup ends
   (peak lr 5e-3 at step 50), takes a gradient spike (global norm 9.5 at step 80), loss jumps
   1.07 -> 3.49, recovers briefly, then by step 160 collapses for good to ~3.6 (the
   constant-prediction loss) with a runaway parameter norm. Plain Adam at 5e-3, 50 warmup steps,
   no clipping, 4 samples. The library's own pipelines use max_lr ~1e-4 with gradient clipping.

Fix: `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(schedule))`. Schedule, step count
and the 100x threshold are unchanged. Clipped recipe, 8 seeds x 4 setups:

| setup | ratios (seeds 0-7) |
|---|---|
| jax 0.11.1 | 156 183 143 183 181 178 158 178 |
| jax 0.12.0.1 | 155 173 169 170 166 171 161 158 |
| jax 0.12.1 | 175 161 151 180 158 156 147 173 |
| jax 0.12.1, old softmax rule | 188 152 155 180 180 153 154 164 |

32/32 pass, worst 143x. The XNNPACK/AMD Zen 4/5 episode (~57x) is the same instability; not
re-measured on AMD (no AMD host available), but clipping removes the failure mode rather than
avoiding one rounding variant.

### 6.3 Still open

- AMD Zen 4/5 runners: not measured with the clipped recipe.
- Done in the same branch: floor raised to `grain>=0.2.19` (which catches the absl error
  upstream) and `gensbi/_compat.py` deleted; `tests/test_grain_compat.py` kept as the floor guard
  (fails on grain 0.2.18 without the shim, passes on 0.2.19).
- `test_tiny_overfit_and_cond_sensitivity` still has no `experimental` marker (unchanged).
