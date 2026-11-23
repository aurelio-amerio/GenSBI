# Changelog

All notable changes to GenSBI are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) with a pre-1.0 caveat: minor
releases may contain breaking changes, and they are listed under
**Breaking** when they do.

The changelog starts at 0.5.0. Breaking changes from earlier releases that
still bite code written against the first tutorials are collected in
[Migration notes](#migration-notes).

## [Unreleased]

### Added

- `plot_marginal_coverage` takes `parameter_labels` for the panel titles, as
  `sbc_rank_plot` does. The default stays "Dim 1", "Dim 2", ...
- `plot_marginal_coverage` takes `num_posterior_samples`. If given, the curve
  is dashed above the confidence level that this many samples can resolve (a
  tail mass of `1 / (N + 1)`, z = 3.3 for 1000 samples), where it is an
  extrapolation.
- `TARPResult` carries `coverage_values`, the TARP coverage value of every
  (theta, posterior) pair, and `num_posterior_samples`.

### Changed

- `check_tarp`'s second return value is now the p-value of a KS test of the
  per-pair coverage values against Uniform(0, 1). It used to compare the
  points of the binned ECP curve with the alpha grid, as `sbi` does, which
  cannot detect miscalibration: it returned p > 0.6 for posteriors 30% too
  narrow or 50% too wide. **Low p-values are now meaningful**, and a
  calibrated posterior no longer returns p close to 1 every time. `check_tarp`
  takes a `seed` for the jitter that makes the discrete values continuous.
- The confidence (z-score) panel of `plot_tarp` is computed from the per-pair
  coverage values instead of the binned ECP curve. With the default 30 bins
  the old panel had no grid point between z = 2.1 and an endpoint clipped at
  z = 4.9, so everything above 2 sigma, including the 3 sigma label, was an
  interpolation that understated miscalibration. **The panel changes above
  z = 2.** The values are jittered to remove their discreteness (`seed`
  argument), the band is the 95% Jeffreys interval also with
  `bootstrap=True`, and the curve is dashed above the level the posterior
  samples can resolve. The credibility panel is unchanged.

- `gensbi.diagnostics.marginal_coverage.compute_marginal_coverage` uses a new
  estimator for both `method="histogram"` and `method="KDE"`. The old ones
  reported calibrated posteriors as overconfident (about 7 points low at the
  68% level for a Gaussian marginal with 1000 samples and the histogram
  method, far more for flat or bounded marginals), so **coverage values and
  plots change**. The true value is now pooled with the posterior samples
  before the density is estimated, and the credibility is the rank of its
  density among the samples', with ties split at random. This is exactly
  uniform for a calibrated posterior. Consequences:
  - results depend on a new `seed` argument (default `0`);
  - credibilities are strictly inside `(0, 1)`: a true value far outside the
    samples no longer returns exactly `1.0`;
  - non-finite samples or true values give `NaN` with one summary warning
    (the histogram method used to raise), and `estimate_hat_z` /
    `plot_marginal_coverage` exclude `NaN` values instead of counting them as
    misses;
  - a single progress bar replaces the per-dimension prints, and
    `show_progress_bar=False` silences it;
  - mismatched input shapes raise `ValueError`.
- Diagnostics synced with fixes made upstream in `sbi`. These change numbers:
  - `check_tarp` returns the area-to-curve as an area. It was a plain sum
    over the grid, so it grew with `num_bins`; values are now smaller by a
    factor of `num_bins`.
  - `LC2ST.p_value` counts the observed statistic among the null ones,
    `(1 + #{T_null >= T_obs}) / (1 + num_trials_null)`, so it is never 0.
  - `LC2ST` builds its null samples by swapping the two labels within each
    pair that shares an observation, instead of shuffling all rows, and the
    null trials now depend on `seed`.
  - `check_sbc`: the KS test jitters the integer ranks and the C2ST compares
    them with uniform integers. Both used a continuous uniform before, which
    is wrong for discrete ranks and rejected calibrated posteriors when the
    number of posterior samples was small.
  - `sbc_rank_plot` bins the ranks over `[0, num_posterior_samples]` and
    labels the CDF axis in rank units. It used the range of the observed
    ranks, which hid ranks that never reach the extremes.
  - `run_sbc` validates its inputs again: it raises if `thetas` and `xs`
    differ in length (the missing rows used to get rank 0) and warns below
    100 SBC or posterior samples.

### Fixed

- `LC2ST(z_score=True)` normalised the null-hypothesis training data twice
  and re-normalised the observation on every null trial, so the null
  statistics were evaluated at a wrong observation and the test could not
  reject. Data is now normalised exactly once. Constant features no longer
  produce `NaN`.
- `run_tarp(num_bins=None)` raised `TypeError`; it now uses
  `num_tarp_samples // 10` bins as documented.
- `sbc_rank_plot` raised `AttributeError` on JAX arrays.
- `c2st` raised `UnboundLocalError` for a classifier class or instance and
  `TypeError` when `classifier_kwargs` contained `random_state`; an unknown
  classifier name now raises `ValueError`.
- `lc2st.EnsembleClassifier` works with estimators that have no
  `random_state`.

### Docs

- `docs/basics/validation.md`: TARP snippet fixed (`run_tarp` returns a
  `TARPResult`), marginal coverage section added.

## [0.5.2] - 2026-10-27

### Changed

- The sampler and log-probability builders of `FlowMatchingMethod`,
  `DiffusionEDMMethod` and `ScoreMatchingMethod` no longer accept arbitrary
  keyword arguments. An argument that the method does not support (for
  example `time_grid=` on a diffusion pipeline, or a model keyword such as
  `encoder_key=` passed bare to `sample`/`sample_batched`) now raises
  `TypeError` instead of being silently ignored. This is a bug fix, but it
  can break code that passed such arguments: they never had an effect, so
  remove them, or pass model keyword arguments through
  `model_extras={...}`.

### Fixed

- `AbstractPipeline.sample_batched` forwards the caller's `model_extras` to
  the model for every condition. Previously they were validated and then
  dropped, so models that need an extra at inference (e.g. a VAE-embedded
  condition needing an `encoder_key`) fell back to their defaults.
- `JointPipeline.sample` no longer warns "x_o has batch dimension N > 1" for a
  single 1-D observation of shape `(dim_cond,)`.

### Docs

- `CHANGELOG.md` added.
- Batched-sampling snippets in `docs/basics/inference.md` and
  `docs/basics/validation.md` fixed (`x_o`/`nsamples` argument names, output
  shape).
- Example notebooks refreshed: small bugs and typos fixed, and the
  gravitational-wave and lensing examples pass the VAE encoder key via
  `model_extras`.

## [0.5.1] - 2026-10-25

### Added

- Multi-device training and sampling via `gensbi.sharding` (#54).

## [0.5.0] - 2026-10-22

### Breaking

- JAX floor raised to 0.11.1 (JAX 0.12 supported); dependency floors
  refreshed (#52).

### Added

- Chunked sampling: `chunk_size` on `sample` and `sample_batched` for every
  pipeline bounds the number of samples per device call.
- Periodic intermediate checkpoints via `training.save_every`.
- Mixed precision: fp32 master weights with a per-model bf16 compute knob
  (Flux1, Flux1Joint, Simformer, MAF, TarFlow, PixelDiT, FieldDiT,
  autoencoders).
- HEALPix RoPE ids: `init_ids_healpix` and the `HealpixRope` `IdStrategy`;
  `ConditionalPipeline` accepts `IdStrategy` objects.
- `gensbi.models.healswin` mirror of `heal-swin-nnx`.

### Fixed

- `training.save_every` exposed in `parse_training_config`.
- Stale work-in-progress checkpoint clearing and restore path.

## Migration notes

Code written against the early tutorials (≤ 0.3) fails on 0.5 in three
places. None of these has a compatibility alias; update the call sites.

| Was | Now | Since |
|---|---|---|
| `UnconditionalFlowPipeline(...)` | `UnconditionalPipeline(..., method=FlowMatchingMethod())` | 0.2.1 (#43) |
| `ecp, alpha = run_tarp(...)` | `result = run_tarp(...)`; `plot_tarp(result)`; fields `result.ecp`, `result.alpha`, … (`TARPResult`) | 0.1.0 |
| `pipeline.sample(key, x_o=xs, nsamples=len(xs))` for a batch of conditions | `pipeline.sample_batched(key, xs, nsamples=1)[0]`; `sample` conditions on a single observation and raises on a batch | 0.4.0 (#51) |

The last replacement is a per-condition Python loop; a vectorized path for
many conditions × few samples is an open design item
(`docs/superpowers/specs/2026-10-26-sample-batched-vectorized-handout.md`).
