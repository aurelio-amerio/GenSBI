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

### Breaking

- The sampler and log-probability builders of `FlowMatchingMethod`,
  `DiffusionEDMMethod` and `ScoreMatchingMethod` no longer accept arbitrary
  keyword arguments. An argument that the method does not support (for
  example `time_grid=` on a diffusion pipeline, or a model keyword such as
  `encoder_key=` passed bare to `sample`/`sample_batched`) now raises
  `TypeError` instead of being silently ignored. Pass model keyword arguments
  through `model_extras={...}`.

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
