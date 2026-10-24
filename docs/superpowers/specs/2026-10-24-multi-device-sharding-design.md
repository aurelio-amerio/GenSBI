# Multi-device sharding for GenSBI and HEAL-SWIN-nnx

Date: 2026-10-24
Status: reviewed, awaiting approval
Repos: `GenSBI` (this repo) and `HEAL-SWIN-nnx` (`/home/aure/github/HEAL-SWIN-nnx`).
A second pass on `sbibm-jax` is out of scope here and listed under follow-ups.

## 1. Intent

Let a user train and sample GenSBI models, and HEAL-SWIN models, on all the
GPUs or TPUs in one machine with one extra line of code, and with no change
to the model code they already have. Today neither repo contains any
sharding, `pmap`, `device_put` or mesh code; everything runs on the default
device.

The parallelism wanted now is **data parallel**: every device holds the
full model, each device processes a slice of the batch, gradients are
averaged. The models all fit on one device today. The design must leave
the door open to **FSDP** (shard parameters and optimizer state across the
same devices to save memory) and to **tensor parallel** later, without a
second rewrite. FSDP is in scope as a tested opt-in, **marked experimental**
in docs and in the warning it emits, because its memory benefit cannot be
measured on simulated CPU devices (see 6.4). A `tensor` rule also ships,
**experimental and untuned**: it is one row in the rule table plus a
correctness smoke test, and exists mainly to validate the `heads`/`mlp`
annotations early. Making it fast (activation sharding constraints in
attention and MLP blocks, GPU tuning) is a follow-up.

### Why

* Larger effective batch or faster steps when a multi-GPU box is available.
* Posterior sampling (10k samples by default) is embarrassingly parallel.
* HEAL-SWIN at high HEALPix resolution is the first model likely to need
  the memory relief of FSDP.

## 2. Goals, non-goals, anti-goals

**Goals**

1. One mesh, passed once to the pipeline, then forgotten. Same user code on
   one device and on eight.
2. All models under `src/gensbi/models` (Flux1, Flux1Joint, Simformer,
   TarFlow, MAF, the embedding layers, wrappers) and the HEAL-SWIN models
   carry sharding annotations on every parameter.
3. Three shipped sharding rules: `replicated` (plain data parallel, the
   default when a mesh is given), `fsdp` (experimental) and `tensor`
   (experimental, correctness only).
4. Batches, the per-step random key, samplers, the EMA model, the optimizer
   state and checkpoints all behave correctly under both rules.
5. Everything is exercised in CI on 8 simulated CPU devices, in both repos.
6. A sharded loss curve matches the unsharded one for the same seed, within
   floating-point tolerance.

**Non-goals (this pass)**

* Multi-host / multi-process training. One process, many local devices.
* A *tuned* tensor-parallel rule (activation constraints, benchmarks),
  sequence parallelism, pipeline parallelism.
* Sharding anything inside `sbibm-jax` (its loaders, simulators). Second pass.
* Ensemble parallelism (K independent models, one per device). Separate design.
* Annotating the experimental models (PixelDiT, FieldDiT, autoencoders, the
  VAE pipeline). They still work under the `replicated` rule through the
  fallback described below; the `fsdp` rule will replicate them too.
* Removing the per-step host syncs in the training loop (tqdm postfix,
  validation comparisons). Performance work, not correctness.

**Anti-goals (would make this a failure even if it works)**

* Any change in behaviour, speed or numerics when no mesh is given.
* A user having to annotate their own layers to get data parallel.
* A cryptic XLA error where a one-line validation message would do.
* Code in the models, losses or solvers that knows which rule is active.

## 3. Constraints

* Flax `>= 0.12.7` NNX, JAX `>= 0.11.1` (GenSBI pins `>=0.11.1,<0.13`;
  HEAL-SWIN's pyproject is being bumped to the same floor). Verified on both
  installed environments (GenSBI jax 0.12.0.1 / flax 0.12.10, HEAL-SWIN jax
  0.10.2 / flax 0.12.7): `nnx.with_partitioning`, `nnx.logical_axis_rules`,
  `nnx.get_partition_spec`, `nnx.get_named_sharding`, `*_metadata` kwargs on
  `Linear`, `Conv`, `LayerNorm`, `RMSNorm`, `Embed`, `MultiHeadAttention`;
  `jax.set_mesh` exists (`jax.sharding.use_mesh` does not); `device_put` of
  typed PRNG key arrays onto a `NamedSharding` works; `nnx.clone` preserves
  `out_sharding` and `eager_sharding` metadata; an `nnx.Optimizer` built on
  placed parameters inherits their sharding because optax initialises its
  state with `zeros_like` on the placed arrays (verified for adam, adamw and
  the pipeline's `MultiSteps(chain(adaptive_grad_clip, adamw))`, with no
  mesh context); the moments keep their `data` sharding across `nnx.jit`
  steps, so no sharding constraint inside the train step is needed.
* **Flax eager sharding (FLIP 4844).** In Flax 0.12 a variable created with
  `out_sharding` metadata is placed on the mesh at creation, and creation
  **requires** an active mesh context (`jax.set_mesh`) plus, for logical
  names, an active `nnx.logical_axis_rules` context. Without them it raises.
  Unannotated variables are unaffected, which is why stock Flax models never
  meet this. Since every parameter in both repos will be annotated, and
  models must keep building with no mesh anywhere, every library annotation
  sets the per-variable opt-out `eager_sharding=False` (the FLIP's documented
  per-variable escape hatch). Models therefore never touch a mesh; placement
  happens once, explicitly, in `shard_model` (5.3). We do **not** flip the
  global `flax_always_shard_variable` flag or use `nnx.use_eager_sharding`,
  which would silently change Flax semantics for the user's own variables.
  Verified on both installed Flax versions: with the opt-out, construction,
  `nnx.Optimizer`, `ModelEMA`, `nnx.eval_shape` and `nnx.get_abstract_model`
  all work with no mesh; the flag survives `clone`, `split`/`merge`,
  `OptVariable` creation and orbax restore. The only thing given up is
  parameters being born sharded, which is irrelevant while models fit on one
  device (and the future large-model path, `nnx.get_abstract_model` plus a
  `jax.jit(init, out_shardings=...)`, is verified to work with the opt-out).
* `jax_threefry_partitionable` is on by default in the installed JAX. Random
  draws inside a jitted step therefore give the same values regardless of
  how the output is sharded. The design relies on this for the
  sharded-equals-unsharded test.
* GenSBI depends on `heal-swin-nnx`, never the reverse. Shared vocabulary
  between the two must be a documented convention, not an import.
* Models are constructed eagerly with a live `nnx.Rngs`. HEAL-SWIN
  construction runs tens of seconds of host-side numpy geometry. Abstract
  initialisation is therefore not used; we build eagerly and place state
  afterwards.
* Batch sizes are powers of two in practice. We validate divisibility and
  error clearly; we do not pad or reshape training batches.
* Both test suites use `pytest-xdist` (`-n 2`) and `pytest-env`, and GenSBI
  hard-sets `XLA_FLAGS` in `pyproject.toml`.

## 4. User-facing behaviour (decided)

```python
from gensbi.sharding import make_mesh

mesh = make_mesh()                      # all local devices on the 'data' axis
pipeline = Flux1FlowPipeline(train_ds, val_ds, dim_obs, dim_cond,
                             mesh=mesh)             # rule defaults to "replicated"
pipeline.train(rngs)
samples = pipeline.sample(key, x_o, nsamples=100_000)
```

* `mesh=None` (the default) is exactly today's behaviour. No code path
  differs.
* `sharding_rule="fsdp"` on the pipeline switches to FSDP. Nothing else
  changes for the user.
* Both are also accepted through `training_config` / the YAML recipes as
  `mesh: auto` and `sharding_rule: fsdp`, so `init_pipeline_from_config`
  users get them without Python changes.
* A user who passes their own `nnx.Module` with no annotations gets plain
  data parallel. Under `fsdp` their unannotated parameters are replicated
  and a single warning lists them.
* Batch size must be divisible by the mesh `data` size, for both train and
  validation loaders. The pipeline checks the first batch of each and
  raises a `ValueError` naming the batch size, the device count and the
  loader.
* `make_mesh(devices=None, model_axis=1)` returns a 2-D
  `jax.sharding.Mesh` with axes `("data", "model")`, `data = n_devices //
  model_axis`, and **Auto** axis types. On the installed JAX,
  `jax.make_mesh` defaults to Explicit axis types, and Flax's eager
  sharding is documented to work only in auto mode, so the helper sets
  `axis_types` explicitly; a user-supplied mesh with Explicit axes gets a
  warning (construction never depends on the mesh, only `shard_model` does). `model_axis` stays 1 in this pass; it
  exists so a tensor rule can be added later without changing meshes or
  checkpoints.
* A user model with **no** annotations works unchanged: verified by
  constructing a plain `nnx.Module` and its optimizer inside a 2x4 mesh on 8
  simulated devices and running a jitted train step with the batch split
  over `data`. Parameters and optimizer state come out replicated, which is
  plain data parallel.

## 5. Architecture

### 5.1 Logical axis names (the contract shared by both repos)

Every parameter is annotated with **logical** axis names, never with mesh
axis names. A rule maps logical names to mesh axes at placement time. One
annotation serves every rule, now and later.

| Logical name | Meaning | `replicated` | `fsdp` | `tensor` |
|---|---|---|---|---|
| `embed` | model/residual hidden width | – | `data` | – |
| `mlp` | MLP hidden width | – | – | `model` |
| `heads` | attention qkv / head-merged width | – | – | `model` |
| `vocab` | rows of an embedding table | – | – | `model` |
| `in` | raw input feature dim (data dims, channels) | – | – | – |
| `out` | raw output feature dim | – | – | – |
| `None` | any other axis (window, 1-sized broadcast dims, kernel taps) | – | – | – |

The `fsdp` rule shards only `embed`. That is the standard choice: every
large kernel has one `embed` axis, so every large kernel and its optimizer
state gets split, while small vectors stay cheap to gather. The `tensor`
rule shards `mlp`, `heads` and `vocab` over `model` and leaves `embed`
replicated (Megatron-style column/row split); the same divisibility
fallback applies to the sharded dims (head count and hidden widths must
divide by the `model` axis size). XLA guarantees correct results for any
placement, so this rule is safe to ship untuned; it only affects speed.

**Batch axes are derived from the mesh**, not hard-coded: a batch is
sharded over every mesh axis except `model`. With today's 2-D mesh that is
`P("data")`; a future 3-D mesh `("data","fsdp","model")`, where the FSDP
rule maps `embed` to `fsdp` instead of `data`, needs only a change to
`make_mesh` and the rule table, with models, checkpoints and the pipeline
untouched.

The vocabulary and the rule tables above must be identical in
`gensbi/sharding.py` and `heal_swin_nnx/sharding.py`. Each file imports
only JAX and Flax; neither imports the other (GenSBI depends on
`heal-swin-nnx` for the model re-export only, and that stays the only
coupling). GenSBI's `shard_model` reads plain Flax metadata off each
variable, so it places a HEAL-SWIN model without knowing where it came
from. A test in GenSBI pins the two tables against each other (skipped when
the installed `heal_swin_nnx` has no `sharding` module). For robustness
against drift, `shard_model` treats a logical name missing from the active
rule as replicated and warns, instead of failing inside Flax with
"resource axis not found".

### 5.2 How annotations are written

Through the standard layers' metadata kwargs where they exist, otherwise
through `nnx.with_partitioning` on the initializer. Two tiny helpers in each
repo's `sharding.py` keep the sites short and identical:

```python
from gensbi.sharding import axes, partitioned   # heal_swin_nnx.sharding has the same two

nnx.Linear(d_in, d_hidden, rngs=rngs,
           kernel_metadata=axes("embed", "mlp"),
           bias_metadata=axes("mlp"))

self.condition_embedding = nnx.Param(
    partitioned(nnx.initializers.zeros, (None, None, "embed"))(key, (1, 1, hidden)))
```

`axes(*names)` returns `{"out_sharding": names, "eager_sharding": False}`;
`partitioned(init, names)` is `nnx.with_partitioning(init, names,
eager_sharding=False)`. **All library annotations go through these two
helpers**; a raw `out_sharding` written by hand would raise at construction
without a mesh, so the annotation-coverage test (6.2) also asserts
`eager_sharding is False` on every annotated variable.

Consequences: models and building blocks never import mesh code and build
on the default device exactly as today, whether called directly
(`Flux1(params)` in a notebook), composed into a user's own module, or
built by the pipeline. `nnx.Optimizer`, `ModelEMA` and `nnx.eval_shape` on
them work with no mesh. A user with an unrelated active mesh is unaffected
(placement only happens in `shard_model`, with the pipeline's mesh).

Users who do **not** annotate their own modules are unaffected everywhere:
construction, optimizer, placement (all parameters resolve to replicated,
which is data parallel), FSDP (replicated plus one warning). Users who want
FSDP on their own layers use the same two helpers.

Cosmetic: `nnx.display` shows `eager_sharding=False` on every library
parameter because it differs from the global default. Documented, accepted.

Rank of the annotation must equal rank of the array. Non-parameter state
(`Mask`, `PEMatrix`, `Buffer`, `RopeIds`, `RngKey`, `RngCount`, `BatchStat`)
is left unannotated and resolves to replicated through Flax's existing
fallback (`get_partition_spec` returns an empty `PartitionSpec` for any
variable without `out_sharding`).

### 5.3 `gensbi.sharding` module

Small, pure, no pipeline imports. Public surface:

* `LOGICAL_AXES`, `RULES: dict[str, tuple[tuple[str, str | None], ...]]`.
  Every rule maps **every** logical name explicitly (to `None` when
  unsharded); `get_named_sharding` fails on a logical name the mesh does
  not know, so the `replicated` rule cannot be an empty tuple.
* `axes(*names)`, `partitioned(init, names)` (see 5.2).
* `make_mesh(devices=None, model_axis=1) -> Mesh`.
* `shard_model(module, mesh, rule="replicated") -> None`. Resolves every
  variable's `PartitionSpec` under the rule, turns it into a
  `NamedSharding`, and `device_put`s the state in place. Idempotent: calling
  it again re-places anything that drifted (used after `fit_standardization`
  and after safetensors import). For `fsdp`, any array whose `embed` axis is
  not divisible by the `data` size, or which has no annotation, is placed
  replicated; one warning lists them.
* `shard_batch(batch, mesh) -> batch`. `tree_map` over arrays (numpy or
  `jax.Array`), placing axis 0 on the batch axes (`batch_axes(mesh)`: every
  mesh axis except `model`) and the rest replicated. Rank-agnostic. Leaves
  non-array leaves alone.
* `batch_axes(mesh) -> tuple[str, ...]` and `batch_size_divisor(mesh)`
  (product of the batch axes' sizes), used by `check_batch_divisible`.
* `check_batch_divisible(batch, mesh, what: str)`. Raises the `ValueError`
  described above.
* `sharding_of(module) -> dict[path, NamedSharding]` for tests and debugging.
* `shard_sample_init(x_init, mesh)` and `pad_for_mesh(n, mesh) -> (n_padded, n)`
  used by samplers.

### 5.4 Pipeline integration (`recipes/pipeline.py` and subclasses)

* `AbstractPipeline.__init__` gains `mesh: Mesh | None = None` and
  `sharding_rule: str = "replicated"`. `training_config` / recipe parsing
  accept the same two keys.
* The pipeline builds the model (`_make_model`, or takes the user's) as
  today, on one device, then calls `shard_model(model, mesh, rule)` once.
  `ema_model = nnx.clone(model)` keeps that placement (verified; the clone
  happens after placement). `shard_model` is idempotent and is also the
  "fix-up" after buffer writes and imports.
* `train()`: `optimizer = self._get_optimizer()` runs as today and
  inherits the parameters' placement through optax's `zeros_like` init;
  the pipeline then calls `shard_model(optimizer, mesh, rule)` once as an
  idempotent safety net for transforms that allocate state with
  `jnp.zeros(shape)` instead, and a test asserts equivalence with the
  parameters' sharding. The same applies to `ModelEMA`. The training loop
  body runs inside `jax.set_mesh(mesh)` for `with_sharding_constraint`
  compatibility; correctness does not depend on it (verified with and
  without). Every `next(iter)` result passes through `shard_batch` before
  `train_step`/`val_step`. The first train and first validation batch pass
  through `check_batch_divisible`. The per-step key is left as is (a scalar
  key is replicated automatically).
* `train_step`, `val_step`, `ema_step`, loss functions, `prepare_batch`:
  **unchanged**. Sharding propagates from the placed inputs and parameters.
* `best_state = nnx.state(model)`: unchanged; the copies keep their sharding.
* `fit_standardization` (flow pipeline): call `shard_model` again after it
  writes the standardisation buffers.
* `import_safetensors` / `load_safetensors`: call `shard_model` again after
  `nnx.update`. Orbax `restore_model` and `restore_intermediate` already
  restore onto the concrete current state, so sharding is preserved; a test
  pins this.
* `get_sampler` in the conditional, joint and unconditional pipelines
  already draws `x_init` outside the jitted solver. When a mesh is set, the
  returned sampler pads `nsamples` up to a multiple of the `data` size, draws
  `x_init` and places it with `shard_sample_init`, runs the existing jitted
  solver unchanged, and trims back to `nsamples`. The SDE solver splits its
  per-sample keys inside its own jit, so nothing is needed for keys.
* `ConditionalFlowPipeline.get_sampler` (TarFlow, MAF) calls
  `flow.sample` eagerly today, unlike the other pipelines. It is wrapped in
  `nnx.jit` (verified jit-compatible on both models; on one CPU device 4096
  samples drop from ~650 ms to ~4 ms for MAF and ~730 ms to ~31 ms for
  TarFlow), and when a mesh is set the broadcast condition is placed on the
  batch axes with the same `nsamples` padding; a condition sharded over
  `data` propagates through the models' internal `vmap`/`scan` and the
  output lands on `data` with identical values (verified).
* The concrete pipelines (`Flux1*Pipeline`, `Simformer*Pipeline`,
  `Flux1Joint*Pipeline`, …) build the model before calling
  `AbstractPipeline.__init__`, so `mesh` and `sharding_rule` are threaded
  through each of their constructors. Model extras (broadcast `cond`, ids) stay replicated.
  `_chunked_draw` and `sample_batched` sit on top unchanged.

### 5.5 HEAL-SWIN-nnx

* Annotate every parameter in `models/healswin.py`, `models/swin.py`,
  `models/healconv.py`, `layers.py` with the shared vocabulary:
  qkv `("embed","heads")`, proj `("heads","embed")` (bias `("embed",)`),
  Mlp fc1 `("embed","mlp")`, fc2 `("mlp","embed")`, PatchEmbed conv
  `(None,"in","embed")`, PatchMerging reduction `(None,"embed")` (its input
  axis is 4×C concatenated tokens, not the residual width, so it stays
  `None`; its output is the next stage's width);
  PatchExpand / FinalPatchExpand `("embed", None)`; output conv
  `(None,"embed","out")`; LayerNorm scale/bias `("embed",)`;
  `logit_scale` `(None,None,None)`; `relative_position_bias_table`
  `(None,"heads")`; `rope_freqs` `(None,"heads",None)`. The flat `Swin`
  model's conv kernels are rank 4 (`(None,None,"in","embed")`).
  `Buffer` variables (shift indices, attention masks, RoPE tables, validity)
  stay unannotated and replicated. The pixel axis is never sharded.
* `heal_swin_nnx/sharding.py`: self-contained copy of `LOGICAL_AXES`,
  `RULES`, `make_mesh`, `shard_model`, `shard_batch`,
  `check_batch_divisible`. Same semantics as GenSBI's; GenSBI does not
  import it.
* New example `examples/mnist_healpix_classify_sharded.py`: a copy of the
  classifier script with a mesh, `shard_model`, `shard_batch` per step,
  `drop_remainder=True` on the grain batch, and `JAX_PLATFORMS` left to the
  user. The original script is untouched.
* Dropout and DropPath RNG state lives on the modules and is replicated;
  DropPath's per-sample Bernoulli draw over the batch-sharded shape is
  correct under partitionable threefry. A test pins that a sharded forward
  in training mode matches an unsharded one for the same RNG state.

### 5.6 Random keys

The pipeline draws one key per step on the host and passes it in. Inside
the step, `prepare_batch` draws noise and times of global batch shape. With
partitionable threefry those draws are identical whether the output lands
on one device or eight, so no per-device key folding is needed anywhere.
This is a property to test, not to assume: the end-to-end test below checks
it.

## 6. Testing

### 6.1 Environment

Both repos add `--xla_force_host_platform_device_count=8` to the
`XLA_FLAGS` entry set by `pytest-env` in `pyproject.toml`, so the whole
suite runs in processes that see 8 CPU devices. This doubles as a check
that single-device behaviour is unchanged by the mere presence of extra
devices. Sharding tests carry a `sharding` marker and skip with a clear
reason when `jax.device_count() < 8`. CI needs no new job. (My call; if the
suite slows down measurably, the fallback is a separate pytest invocation
for `tests/sharding/` with its own flags.)

### 6.2 GenSBI tests (`tests/sharding/`)

1. `make_mesh` shape, axis names and Auto axis types; `model_axis`
   splitting. No-mesh behaviour: every exported model builds with no mesh,
   every variable on `SingleDeviceSharding`; `nnx.Optimizer`, `ModelEMA`,
   `nnx.eval_shape` and `nnx.get_abstract_model` work on it with no mesh; a
   building block composed into a user module builds; an unannotated user
   module passed to a pipeline with a mesh trains and samples.
2. Annotation coverage: for every model class exported from
   `gensbi.models` (built small), every `nnx.Param` has `out_sharding`
   metadata of matching rank, using only names from `LOGICAL_AXES`, and
   `eager_sharding is False`.
3. `shard_model` with `replicated`: every variable leaf, including `Mask`,
   `PEMatrix`, `RngKey`, has a fully replicated `NamedSharding` on the mesh.
   With `fsdp`: every annotated `embed` axis divisible by 8 is sharded over
   `data`; indivisible or unannotated arrays are replicated and warned about.
4. `shard_batch`: tuple and single-array batches, numpy and jax inputs,
   ranks 3 and 4; `check_batch_divisible` error text.
5. End to end, for a small Flux1, Simformer and Flux1Joint pipeline each,
   under both rules: train 20 steps with `mesh` and without, same seed; the
   loss sequences agree to fp32 tolerance; parameters agree afterwards;
   optimizer and EMA state carry the same sharding as the parameters before
   and after steps. Sharding comparisons use
   `NamedSharding.is_equivalent_to(other, ndim)`, never spec equality
   (`P('data')` and `P('data', None)` are the same sharding).
6. Sampling: ODE and EDM samplers, and the flow pipeline's jitted MAF and
   TarFlow samplers, give the same samples with and without a mesh for the
   same key (tolerance); `nsamples` not divisible by 8 is padded and
   trimmed; SDE sampler runs and has the right shape.
7. Checkpointing: orbax save/restore and safetensors export/import each
   preserve sharding.
8. Recipe config: `mesh: auto` and `sharding_rule: fsdp` parsed and applied.
9. Tensor smoke test: `make_mesh(model_axis=2)` (4x2), a small Flux1 and
   Simformer under the `tensor` rule, 20 training steps and one sampling
   call match the unsharded run to tolerance; qkv and MLP kernels are
   placed over `model`; an indivisible head count falls back with a
   warning. Correctness only, no performance assertion.

### 6.3 Cross-repo sequencing

GenSBI installs `heal-swin-nnx` from PyPI, and `gensbi.models` re-exports
`HealSwinEncoder`. The two repos are implemented **in parallel, in separate
sessions**, each from its own plan; the shared contract (5.1) is fully
written in both plans, so neither needs the other's working tree. GenSBI
develops against the *released* HEAL-SWIN; its HEAL-SWIN-specific checks
(annotation coverage of `HealSwinEncoder`, vocabulary pin against
`heal_swin_nnx.sharding`) skip with a clear reason while the installed
`heal_swin_nnx` has no `sharding` module, so GenSBI CI is never blocked on
the other repo. GenSBI's final task is a release gate: it asks whether the
annotated HEAL-SWIN has been uploaded to PyPI and which version, bumps the
`heal-swin-nnx>=` floor to it, and only then runs the final full
verification in which those two checks must run rather than skip.

### 6.4 FSDP acceptance

On 8 CPU devices we can verify placement (parameters, optimizer state and
EMA state actually split along `data`) and numerics (loss and parameters
match the unsharded run). We cannot measure peak device memory, and XLA's
partitioner may keep gathered weights live through a step. The accepted
criterion for this pass is **placement plus numerics**. FSDP is documented
and warned as experimental; a manual multi-GPU memory measurement is a
follow-up before it is advertised as a memory feature.

### 6.5 HEAL-SWIN tests (`tests/test_sharding.py`)

Annotation coverage for `HealSwin`, `HealConv` and the flat `Swin`; buffers
replicated; vocabulary identical to GenSBI's table; sharded vs unsharded
forward (eval and train mode) and one gradient step agree; `fsdp` places
qkv and MLP kernels over `data`; the sharded example script's train step
runs for two steps on a synthetic batch.

## 7. Calls I made (not discussed; flag if wrong)

* Logical axis names resolved through rules, rather than mesh names written
  directly into layers. Costs nothing now, avoids a second sweep later.
* Per-variable `eager_sharding=False` on every library annotation, via
  the two helpers, instead of keeping eager sharding on with a mesh guard
  in constructors or flipping Flax's global flag. Reason: models stay
  declarative and mesh-free; no guard on ~69 module classes; users building
  their own optimizer or holding an unrelated mesh are unaffected; the only
  loss (born-sharded parameters) is irrelevant while models fit on one
  device, and the large-model path is verified to work with the opt-out.
  An independent review reached the same conclusion.
* `fsdp` shards the `embed` axis only; `tensor` ships as a correctness-only
  experimental rule with a 4x2 smoke test, so mislabelled `heads`/`mlp`
  axes are caught now rather than when someone needs them.
* Batch axes are derived from the mesh (everything except `model`) so a
  3-D mesh later is a helper change, not a redesign.
* Indivisible or unannotated arrays under `fsdp` fall back to replication
  with one warning, instead of raising.
* Sampling pads `nsamples` up to a multiple of the device count and trims,
  instead of refusing or falling back to one device.
* Eager construction then in-place placement, instead of abstract init in a
  sharded `nnx.jit`.
* The 8-device flag goes into the whole test suite's environment.
* HEAL-SWIN gets its own small `sharding.py` instead of importing GenSBI's.
* Experimental models and the VAE pipeline are not annotated in this pass.

## 8. Left to the builder

* Exact placement of the `jax.set_mesh` context in the training loop and
  the samplers.
* Tolerances in the sharded-vs-unsharded tests.
* How `mesh: auto` is spelled in the recipe parser and whether a device
  count can be given.
* Order of annotation sweep and module layout, provided the public surface
  in 5.3 is kept.

## 9. Follow-ups (not this pass)

* `sbibm-jax`: the online loader converts to host numpy and back to device 0
  (`data/dataset.py:304,311`, `process.py:115`); a sharded placement or a
  `shard_options`-aware loader belongs there. Offline `TaskDataset` needs
  nothing: its batches are host numpy with a leading batch axis and are
  placed by `shard_batch`.
* Tuning the `tensor` rule: `with_sharding_constraint` on activations in
  attention and MLP blocks (written with logical names, resolved through
  the rules, so models still do not know which rule is active) and GPU
  benchmarks.
* 3-D mesh `("data","fsdp","model")`: `make_mesh` grows an axis and the
  `fsdp` rule maps `embed` to `fsdp`; batch sharding follows automatically
  from `batch_axes(mesh)`.
* Ensemble parallelism.
* Multi-host.
* Annotations for experimental models.
* Multi-GPU peak-memory measurement for FSDP (6.4).
* Born-sharded / abstract init for models that do not fit on one device:
  `nnx.get_abstract_model(init_fn, mesh)` under the rule's logical axis
  rules, then `jax.jit(init_fn, out_shardings=nnx.get_named_sharding(...))`.
  Verified to work with the existing annotations and the opt-out.
