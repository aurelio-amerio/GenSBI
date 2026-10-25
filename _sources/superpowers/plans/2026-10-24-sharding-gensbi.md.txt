# GenSBI Multi-Device Sharding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A user passes one `mesh` to any GenSBI pipeline and trains, checkpoints and samples on every local device, with no change to model code; `mesh=None` is byte-for-byte today's behaviour; everything is verified on 8 simulated CPU devices.

**Architecture:** A pure `gensbi.sharding` module (mesh helper, logical-axis rule tables, `shard_model`, `shard_batch`, sampler helpers). Every parameter in `gensbi.models` carries logical `out_sharding` metadata with the per-variable `eager_sharding=False` opt-out, written through `axes()`/`partitioned()`. `AbstractPipeline` takes `mesh`/`sharding_rule`, places model, EMA and optimizer once, shards every batch, and pads/places sampler inputs. Train step, losses and solvers are unchanged.

**Tech Stack:** JAX 0.11.1–0.12 (`jax.set_mesh`, `NamedSharding`, Auto mesh axes), Flax NNX ≥ 0.12.10, optax, orbax-checkpoint, grain, pytest + pytest-env + pytest-xdist.

**Spec:** `docs/superpowers/specs/2026-10-24-multi-device-sharding-design.md`. Shares the `LOGICAL_AXES`/`RULES` contract with the HEAL-SWIN plan (`HEAL-SWIN-nnx/docs/superpowers/plans/2026-10-24-sharding-healswin.md`). **The two plans are executed in parallel in separate sessions.** Tasks 1-10 here run against the *released* `heal-swin-nnx` (no path override), so HEAL-SWIN-specific tests skip until Task 11; only Task 11 needs the HEAL-SWIN work finished and published.

## Global Constraints

- `src/gensbi/sharding.py` imports only `jax`, `flax.nnx`, `numpy`, `warnings`, `math`, `contextlib`. It never imports `heal_swin_nnx` or anything under `gensbi.recipes`/`gensbi.models`.
- `LOGICAL_AXES = ("embed", "mlp", "heads", "vocab", "in", "out")` and `RULES` (`replicated`, `fsdp`, `tensor`) are byte-identical to `heal_swin_nnx/sharding.py` (HEAL-SWIN plan Task 2). Mesh axes `("data", "model")`, Auto types; batch axes = every mesh axis except `"model"`.
- Every library annotation goes through `axes(...)` / `partitioned(...)` (both set `eager_sharding=False`). Rank of the tuple equals rank of the array; no logical name appears twice in one tuple.
- `mesh=None` changes nothing: no `device_put`, no mesh context, same arrays, same values. Experimental models (`gensbi.experimental.*`) and the VAE pipeline are **not** annotated.
- Models, losses, solvers and `train_step`/`val_step`/`ema_step` must not learn which rule is active (anti-goal 4). All placement lives in `gensbi.sharding` and `AbstractPipeline`.
- Sharding comparisons in tests use `NamedSharding.is_equivalent_to(other, ndim)`, never `.spec ==`.
- The sandbox cannot run `uv` (read-only cache). Run tests with `.venv/bin/python -m pytest`; ask the user to run `uv sync`/`uv lock` outside the sandbox when `pyproject.toml` dependencies change.
- Test env: `pyproject.toml` `[tool.pytest.ini_options].env` already sets `XLA_FLAGS=--xla_cpu_enable_platform_dependent_math=false`; the device-count flag is appended to that same string (Task 1).
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Review Focus

1. **Validation loader with a different batch size than train** (notebooks use 512 for validation and 128 for training): both are checked separately and the error names the loader. Pinned in Task 6 (`test_validation_batch_not_divisible_names_loader`).
2. **Batches that already live on device 0 as `jax.Array`** (the notebook `split_data` pattern builds `jnp` arrays inside a grain `.map`): `shard_batch` must re-place them, not fail or copy twice. Pinned in Task 2 (`test_shard_batch_jax_input_already_on_device`).
3. **A one-device mesh** (`make_mesh(devices=jax.devices()[:1])`) must train to the same losses as `mesh=None`. Pinned in Task 7 (`test_one_device_mesh_matches_no_mesh`).
4. **Sample counts smaller than the device count and chunked sampling with a chunk size that is not a multiple of 8** (`nsamples=3`, `chunk_size=5`): every chunk is padded and trimmed, final shape exact. Pinned in Task 9 (`test_sampling_pads_small_and_chunked`).
5. **Checkpoint restore across placements**: a checkpoint saved by a pipeline without a mesh restored into one with a mesh (and the reverse) must land on the restoring pipeline's placement. Pinned in Task 8 (`test_restore_follows_current_placement`).

---

### Task 1: Branch and 8-device test environment

**Files:**
- Modify: `pyproject.toml` (`[tool.pytest.ini_options].env`)
- Create: `tests/sharding/__init__.py`, `tests/sharding/helpers.py`

**Interfaces:**
- Produces: `tests/sharding/helpers.py` with
  - `requires_8_devices = pytest.mark.skipif(jax.device_count() < 8, reason="needs XLA_FLAGS=--xla_force_host_platform_device_count=8")`
  - `has_healswin_sharding() -> bool` (`importlib.util.find_spec("heal_swin_nnx.sharding") is not None`)
  - `small_flux1_params() -> Flux1Params`, `small_simformer_params() -> SimformerParams`, `small_flux1joint_params() -> Flux1JointParams`: copies of the configs in `tests/recipes/test_pipeline_flux1.py:24`, `tests/recipes/test_chunked_sampling.py:384`, `tests/recipes/test_pipeline_flux1joint.py:24`, adjusted so the residual width is **16** (Flux1: `axes_dim=[8], num_heads=2`; Simformer: `val_emb_dim=8, id_emb_dim=4, cond_emb_dim=4, qkv_features=8, fourier_features=16`; Flux1Joint: `val_emb_dim=8, cond_emb_dim=4, id_emb_dim=4`), `depth=1`/`depth_single_blocks=1`, `mlp_ratio=1`, fresh `nnx.Rngs(0)` on every call. Each factory asserts the resulting hidden size is divisible by 8 (so FSDP actually shards).
  - `train_dataset_cond`, `val_dataset_cond`, `train_dataset_joint`, `val_dataset_joint`, `dim_obs=2`, `dim_cond=7`: re-exported from `tests/recipes/model_test_helpers.py` (batch 32, channels 2).
  - `test_training_config(tmp_path, nsteps=20) -> dict`: `AbstractPipeline.get_default_training_config()` with `nsteps`, `val_every=5`, `warmup_steps=2`, `early_stopping=False`, `checkpoint_dir=str(tmp_path / "ckpt")`.

- [ ] **Step 1: Branch and environment edits**

```bash
git checkout -b sharding
```
In `pyproject.toml`, replace the `XLA_FLAGS` entry of `[tool.pytest.ini_options].env` with:
```toml
    "XLA_FLAGS=--xla_cpu_enable_platform_dependent_math=false --xla_force_host_platform_device_count=8",
```
Do **not** add a path override for `heal-swin-nnx`: the HEAL-SWIN work proceeds in parallel in another session and its working tree may be half-annotated. The two HEAL-SWIN-specific tests (`test_vocabulary_pinned_to_heal_swin`, `test_healswin_encoder_annotated_via_gensbi`) skip while the installed package has no `sharding` module.

- [ ] **Step 2: Nothing to install.** The dependency set is unchanged.

- [ ] **Step 3: Write the helpers module and a device-count test**

```python
# tests/sharding/test_env.py
import jax
def test_eight_cpu_devices_present():
    assert jax.device_count() == 8 and all(d.platform == "cpu" for d in jax.devices())
```

- [ ] **Step 4: Run** `.venv/bin/python -m pytest tests/sharding -q -p no:xdist` → PASS; then the full suite `.venv/bin/python -m pytest -q -m "not slow and not extraslow and not experimental"` → PASS (no behaviour change from extra devices).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/sharding/__init__.py tests/sharding/helpers.py tests/sharding/test_env.py
git commit -m "test: run the suite on 8 simulated CPU devices"
```

---

### Task 2: `gensbi.sharding` module

**Files:**
- Create: `src/gensbi/sharding.py`
- Test: `tests/sharding/test_module.py`

**Interfaces:**
- Produces, identical in name and semantics to HEAL-SWIN plan Task 2: `LOGICAL_AXES`, `RULES`, `MODEL_AXIS`, `axes`, `partitioned`, `make_mesh`, `batch_axes`, `batch_size_divisor`, `shard_model`, `shard_batch`, `check_batch_divisible`, `sharding_of`. Plus:

```python
def resolve_mesh(spec) -> jax.sharding.Mesh | None
    # None -> None; Mesh -> itself; "auto" -> make_mesh(); dict -> make_mesh(**spec); else ValueError
def pad_for_mesh(n: int, mesh) -> tuple[int, int]      # (ceil(n / batch_size_divisor(mesh)) * divisor, n)
def shard_sample_init(x_init, mesh)                      # = shard_batch(x_init, mesh) for a single array or pytree
def mesh_context(mesh) -> contextlib.AbstractContextManager   # jax.set_mesh(mesh) if mesh is not None else nullcontext()
```

- `shard_model` semantics exactly as in HEAL-SWIN plan Task 2 (per-variable spec from metadata; unknown name or indivisible dim → `P()` + one collected warning; `rule != "replicated"` emits `UserWarning("sharding rule '<rule>' is experimental ...")` on **every** call — Python's default filter already shows it once per call site, and tests can rely on it deterministically; Explicit-axis mesh warns; `ValueError` on unknown rule; writes back via `nnx.update`).

- [ ] **Step 1: Write the failing tests** — port the `Toy`-module tests from HEAL-SWIN plan Task 2 verbatim (same names, `from gensbi import sharding as sh`, `Mask` from `gensbi.normalizing_flows.bijections.base` in place of `Buffer`), and add:

```python
def test_vocabulary_pinned_to_heal_swin():
    if not has_healswin_sharding():
        pytest.skip("installed heal_swin_nnx predates its sharding module")
    from heal_swin_nnx import sharding as hs
    assert hs.LOGICAL_AXES == sh.LOGICAL_AXES and hs.RULES == sh.RULES and hs.MODEL_AXIS == sh.MODEL_AXIS

@requires_8_devices
def test_resolve_mesh_forms():
    assert sh.resolve_mesh(None) is None
    assert dict(sh.resolve_mesh("auto").shape) == {"data": 8, "model": 1}
    assert dict(sh.resolve_mesh({"model_axis": 2}).shape) == {"data": 4, "model": 2}
    m = sh.make_mesh(); assert sh.resolve_mesh(m) is m
    with pytest.raises(ValueError): sh.resolve_mesh("eight")

@requires_8_devices
def test_pad_for_mesh():
    mesh = sh.make_mesh()
    assert sh.pad_for_mesh(16, mesh) == (16, 16) and sh.pad_for_mesh(3, mesh) == (8, 3) and sh.pad_for_mesh(9, mesh) == (16, 9)

@requires_8_devices
def test_shard_batch_jax_input_already_on_device():
    mesh = sh.make_mesh()
    x = jnp.ones((16, 4, 1))                        # committed to device 0 as in notebook .map(split_data)
    y = sh.shard_batch(x, mesh)
    assert y.sharding.is_equivalent_to(NamedSharding(mesh, P("data")), 3)
    np.testing.assert_array_equal(np.asarray(y), np.asarray(x))

def test_mesh_context_is_nullcontext_without_mesh():
    with sh.mesh_context(None):
        assert jax.sharding.get_mesh().empty
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/sharding/test_module.py -q -p no:xdist` → FAIL `ModuleNotFoundError: gensbi.sharding`.
- [ ] **Step 3: Implement `src/gensbi/sharding.py`.** Copy the HEAL-SWIN module's body for the shared names (keep the docstring note that the two files are a contract), add the four extra functions. Export nothing from `gensbi/__init__.py` (import path is `gensbi.sharding`).
- [ ] **Step 4: Run** the same command → PASS.
- [ ] **Step 5: Commit**

```bash
git add src/gensbi/sharding.py tests/sharding/test_module.py
git commit -m "feat: gensbi.sharding — logical axes, rules, mesh helpers, shard_model/shard_batch, sampler helpers"
```

---

### Task 3: Annotate embeddings, Flux1 and Flux1Joint

**Files:**
- Modify: `src/gensbi/models/embedding/embedding.py:42-60, 189, 405`
- Modify: `src/gensbi/models/flux1/layers.py:87-100, 119-123, 149-165, 195-200, 263-280, 314-331, 412-427, 485-502`
- Modify: `src/gensbi/models/flux1/model.py:270-310`
- Modify: `src/gensbi/models/flux1joint/model.py:165-202`
- Create: `tests/sharding/test_annotations.py`

**Interfaces:**
- Consumes: `axes`, `partitioned` from Task 2.
- Produces: `assert_every_param_annotated(model)` in `tests/sharding/helpers.py` (same body as HEAL-SWIN plan Task 3, with the non-Param check over `Mask` and `PEMatrix` variables).

Annotation table (kernel / bias). Flux1 `LayerNorm`s have `use_scale=False, use_bias=False` → no parameters → nothing to annotate.

| Site | kernel | bias |
|---|---|---|
| `MLPEmbedder.in_layer` (in→hidden) | `("in","embed")` | `("embed",)` |
| `MLPEmbedder.out_layer` (hidden→hidden) | `("embed",None)` | `(None,)` |
| `MLPEmbedder.p_skip` `(1,1,hidden)` | `nnx.Param(value, **axes(None,None,"embed"))` | |
| `GaussianFourierEmbedding.B` `(half_dim,1)` | `**axes(None,None)` | |
| `Embed` wrapper → `nnx.Embed(..., embedding_metadata=axes("vocab","embed"))` unless the caller passed one | | |
| `QKNorm.query_norm/key_norm` RMSNorm scale `(head_dim,)` | `scale_metadata=axes(None)` | |
| `SelfAttention.qkv` (dim→3·qkv) | `("embed","heads")` | `("heads",)` |
| `SelfAttention.proj` (qkv→dim) | `("heads","embed")` | `("embed",)` |
| `Modulation.lin` (dim→k·dim) | `("embed",None)` | `(None,)` |
| `DoubleStreamBlock.obs_mlp[0]`, `cond_mlp[0]` (hidden→mlp) | `("embed","mlp")` | `("mlp",)` |
| `DoubleStreamBlock.obs_mlp[2]`, `cond_mlp[2]` (mlp→hidden) | `("mlp","embed")` | `("embed",)` |
| `SingleStreamBlock.linear1` (hidden→3·qkv+mlp) | `("embed","mlp")` | `("mlp",)` |
| `SingleStreamBlock.linear2` (qkv+mlp→hidden) | `("mlp","embed")` | `("embed",)` |
| `LastLayer.linear` (hidden→p²·out) | `("embed","out")` | `("out",)` |
| `LastLayer.adaLN_modulation[1]` (hidden→2·hidden) | `("embed",None)` | `(None,)` |
| `Flux1.obs_in`, `Flux1.cond_in`, `Flux1Joint.obs_in` (channels→hidden/token) | `("in","embed")` | `("embed",)` |
| `Flux1Joint.condition_embedding` `(1,1,H)` (both branches) | `**axes(None,None,"embed")` | |

- [ ] **Step 1: Write the failing test**

```python
# tests/sharding/test_annotations.py
from gensbi.models import Flux1, Flux1Joint
from .helpers import assert_every_param_annotated, small_flux1_params, small_flux1joint_params

def test_flux1_every_param_annotated():
    m = Flux1(small_flux1_params()); assert_every_param_annotated(m)
    k = m.obs_in.kernel
    assert isinstance(k[...].sharding, jax.sharding.SingleDeviceSharding)   # no mesh anywhere

def test_flux1joint_every_param_annotated():
    assert_every_param_annotated(Flux1Joint(small_flux1joint_params()))

def test_no_mesh_optimizer_ema_eval_shape_abstract_model():
    import optax
    from gensbi.recipes.pipeline import ModelEMA
    m = Flux1(small_flux1_params())
    nnx.Optimizer(m, optax.adamw(1e-3), wrt=nnx.Param)            # OptVariables copy eager_sharding=False
    ModelEMA(nnx.clone(m), optax.ema(0.999))
    nnx.eval_shape(lambda: Flux1(small_flux1_params()))
    assert jax.sharding.get_mesh().empty

def test_building_block_composes_into_user_module_without_mesh():
    from gensbi.models.embedding import MLPEmbedder
    class Mine(nnx.Module):
        def __init__(self, rngs):
            self.emb = MLPEmbedder(in_dim=4, hidden_dim=16, rngs=rngs); self.head = nnx.Linear(16, 1, rngs=rngs)
    m = Mine(nnx.Rngs(0))
    assert isinstance(m.emb.in_layer.kernel[...].sharding, jax.sharding.SingleDeviceSharding)

@requires_8_devices
def test_shard_model_replicated_on_flux1_places_every_leaf():
    m = Flux1(small_flux1_params()); mesh = sh.make_mesh(); sh.shard_model(m, mesh, "replicated")
    for path, s in sh.sharding_of(m).items():
        assert isinstance(s, NamedSharding) and s.is_equivalent_to(NamedSharding(mesh, P()), 1), path
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/sharding/test_annotations.py -q -p no:xdist` → FAIL `has no annotation`.
- [ ] **Step 3: Annotate every site in the table.** Import `from gensbi.sharding import axes` in each file.
- [ ] **Step 4: Run** `tests/sharding` and `tests/models/flux1 tests/models/flux1joint tests/recipes/test_pipeline_flux1.py tests/recipes/test_pipeline_flux1joint.py` → PASS (existing precision/pipeline tests unchanged).
- [ ] **Step 5: Commit**

```bash
git add src/gensbi/models/embedding src/gensbi/models/flux1 src/gensbi/models/flux1joint tests/sharding
git commit -m "feat: logical sharding annotations on embeddings, Flux1, Flux1Joint"
```

---

### Task 4: Annotate Simformer

**Files:**
- Modify: `src/gensbi/models/simformer/transformer.py:60-70, 117-153, 219-221`
- Modify: `src/gensbi/models/simformer/model.py:131-165`
- Test: `tests/sharding/test_annotations.py`

| Site | annotation |
|---|---|
| `AttentionBlock.layer_norm`, `Transformer.layer_norm`, `DenseBlock.layer_norm` (over din) | scale/bias `("embed",)` |
| `AttentionBlock.attn` `nnx.MultiHeadAttention` | `kernel_metadata=axes("embed","heads",None)` (q/k/v kernels `(din,H,hd)`), `bias_metadata=axes("heads",None)`, `out_kernel_metadata=axes("heads",None,"embed")`, `out_bias_metadata=axes("embed")` |
| `DenseBlock.hidden_blocks[0]` (n→w·n) | kernel `("embed","mlp")`, bias `("mlp",)` |
| `DenseBlock.hidden_blocks[1:-1]` (w·n→w·n) | kernel `("mlp",None)`, bias `(None,)` |
| `DenseBlock.hidden_blocks[-1]` (w·n→din) | kernel `("mlp","embed")`, bias `("embed",)` |
| `DenseBlock.context_block` (dcontext→din) | kernel `("in","embed")`, bias `("embed",)` |
| `Simformer.embedding_net_id` `nnx.Embed` | `embedding_metadata=axes("vocab","embed")` |
| `Simformer.condition_embedding` `(1,1,cond_emb)` | `**axes(None,None,"embed")` |
| `Simformer.output_fn` (total_tokens→in_channels) | kernel `("embed","out")`, bias `("out",)` |

- [ ] **Step 1: Tests** `test_simformer_every_param_annotated()` with `Simformer(small_simformer_params())`; the model stores `transformer.rngs`, so also assert that `RngKey`/`RngCount` variables carry no `out_sharding`. And `test_shard_model_replicated_on_simformer_places_rng_state()`: after `shard_model(m, mesh, "replicated")`, every leaf of `sharding_of(m)` — including the paths containing `rngs` — is a `NamedSharding` equivalent to `P()` on `mesh`.
- [ ] **Step 2: Run** → FAIL. **Step 3:** annotate. **Step 4:** `tests/sharding tests/models/simformer tests/recipes/test_pipeline_simformer*.py` → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: logical sharding annotations on Simformer"`.

---

### Task 5: Annotate TarFlow, MAF; HEAL-SWIN coverage through GenSBI

**Files:**
- Modify: `src/gensbi/models/tarflow/blocks.py:59-69, 243-271`, `src/gensbi/models/tarflow/conditioners.py:59-61, 133-136, 207-210`
- Modify: `src/gensbi/models/maf/masked_linear.py:30-38` (new kwargs `kernel_axes: tuple | None = None, bias_axes: tuple | None = None`, forwarded as `kernel_metadata=axes(*kernel_axes)` when given), `src/gensbi/models/maf/made.py:95-108` (pass axes per layer)
- Test: `tests/sharding/test_annotations.py`

| Site | kernel | bias |
|---|---|---|
| `tarflow.AttentionBlock.norm1/norm2` | scale/bias `("embed",)` | |
| `tarflow.AttentionBlock.qkv` | `("embed","heads")` | `("heads",)` |
| `tarflow.AttentionBlock.proj` | `("heads","embed")` | `("embed",)` |
| `tarflow.AttentionBlock.mlp_in` / `mlp_out` | `("embed","mlp")` / `("mlp","embed")` | `("mlp",)` / `("embed",)` |
| `MetaBlock.proj_in` (F→channels) | `("in","embed")` | `("embed",)` |
| `MetaBlock.sos_embed` `(1,1,F)` | `**axes(None,None,"in")` | |
| `MetaBlock.pos_embed` `(T,channels)` | `**axes(None,"embed")` | |
| `MetaBlock.proj_out` (channels→2F) | `("embed","out")` | `("out",)` |
| `AdditiveBiasConditioner.l1` / `l2` | `("in","embed")` / `("embed",None)` | `("embed",)` / `(None,)` |
| `VectorConditioner.proj`, `ImageConditioner.proj` | `("in","embed")` | `("embed",)` |
| `VectorConditioner.pos`, `ImageConditioner.pos` `(M,channels)` | `**axes(None,"embed")` | |
| `MADE.input_layer` (dim+cond→width) | `("in","embed")` | `("embed",)` |
| `MADE.hidden_layers[i]` (width→width) | `("embed",None)` | `(None,)` |
| `MADE.output_layer` (width→dim·k) | `("embed","out")` | `("out",)` |

`Mask` buffers (`freqs_cis`, `perm`, `inv_perm`, `mask`, `mean`, `std`) stay unannotated.

- [ ] **Step 1: Tests**

```python
from gensbi.models import TarFlow, TarFlowParams, MAFlow, MAFlowParams
def test_tarflow_every_param_annotated():
    assert_every_param_annotated(TarFlow(TarFlowParams(rngs=nnx.Rngs(0), dim=4, cond_dim=2, num_blocks=2, num_heads=2, head_dim=8)))
def test_maf_every_param_annotated():
    assert_every_param_annotated(MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=3, cond_dim=2, n_layers=2, nn_width=16)))
def test_healswin_encoder_annotated_via_gensbi():
    if not has_healswin_sharding(): pytest.skip("installed heal_swin_nnx predates its sharding module")
    from gensbi.models import HealSwinEncoder, HealSwinParams
    p = HealSwinParams(nside=16, in_channels=1, out_channels=1, base_pixels=tuple(range(8)), embed_dim=16, depths=(2, 2), num_heads=(2, 4), drop_path_rate=0.0)
    assert_every_param_annotated(HealSwinEncoder(p, rngs=nnx.Rngs(0)))
```

- [ ] **Step 2: Run** → FAIL for tarflow/maf. **Step 3:** annotate. **Step 4:** `tests/sharding tests/models/tarflow tests/models/maf tests/normalizing_flows` → PASS.
- [ ] **Step 5: Commit** `git commit -m "feat: logical sharding annotations on TarFlow and MAF"`.

---

### Task 6: Pipeline core: `mesh`/`sharding_rule`, placement, batches, config

**Files:**
- Modify: `src/gensbi/recipes/pipeline.py` — `AbstractPipeline.__init__` (276-330), `train()` (885-1059), new helpers
- Modify: `src/gensbi/recipes/conditional_pipeline.py:117`, `joint_pipeline.py:213`, `unconditional_pipeline.py:84`, `flow_pipeline.py:68` (constructor signatures forward the two kwargs)
- Modify: `src/gensbi/recipes/flux1.py:133,273,415`, `flux1joint.py:124,262,385`, `simformer.py:120,279,430` (constructor signatures forward the two kwargs to `super().__init__`)
- Modify: `src/gensbi/recipes/utils.py:parse_training_config` (411-473)
- Test: `tests/sharding/test_pipeline.py`

**Interfaces:**
- Produces on `AbstractPipeline`:
  - `__init__(..., training_config=None, mesh=None, sharding_rule=None)`; resolution: `self.mesh = resolve_mesh(mesh if mesh is not None else self.training_config.get("mesh"))`; `self.sharding_rule = sharding_rule or self.training_config.get("sharding_rule") or "replicated"`. If `self.mesh is not None`: `shard_model(self.model, self.mesh, self.sharding_rule)` **before** `self.ema_model = nnx.clone(model)`.
  - `_reshard(self) -> None`: `shard_model` on `self.model` and `self.ema_model` when a mesh is set (idempotent fix-up).
  - `_prepare_batch(self, batch, what: str)`: no mesh → `batch` unchanged; with mesh → `check_batch_divisible(batch, mesh, what)` the first time per `what` (track in `self._checked_loaders: set[str]`), then `shard_batch(batch, mesh)`.
  - `train()`: after `optimizer`/`ema_optimizer` creation, `shard_model(optimizer, ...)` and `shard_model(ema_optimizer, ...)` when a mesh is set; `batch_val = self._prepare_batch(next(self.val_dataset_iter), "validation")`; each `batch = self._prepare_batch(next(self.train_dataset_iter), "train")`; the `for j in pbar:` loop runs inside `with mesh_context(self.mesh):`.
- `parse_training_config` adds `"mesh": train_params.get("mesh")` and `"sharding_rule": train_params.get("sharding_rule")` (both `None` when absent). `get_default_training_config` is unchanged (keys absent → no mesh).
- Every concrete pipeline constructor gains `mesh=None, sharding_rule=None` and forwards them; `init_pipeline_from_config(**kwargs)` already forwards.

- [ ] **Step 1: Write the failing tests**

```python
# tests/sharding/test_pipeline.py
import numpy as np, jax, jax.numpy as jnp, pytest
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P
from gensbi import sharding as sh
from gensbi.recipes import ConditionalPipeline, Flux1FlowPipeline
from gensbi.core import FlowMatchingMethod
from .helpers import *
import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parents[1] / "recipes"))
from mock_models import MockConditionalModel

def _train(pipeline_cls, params, mesh, rule, tmp_path, nsteps=20, **kw):
    p = pipeline_cls(train_dataset_cond, val_dataset_cond, dim_obs, dim_cond, ch_obs=2, ch_cond=2,
                     params=params, training_config=test_training_config(tmp_path, nsteps),
                     mesh=mesh, sharding_rule=rule, **kw)
    losses, val_losses = p.train(nnx.Rngs(0), save_model=False)
    return p, np.asarray([float(l) for l in losses]), np.asarray([float(l) for l in val_losses])

def _assert_params_close(a, b, rtol=1e-3, atol=1e-5):
    fa = nnx.to_flat_state(nnx.state(a, nnx.Param)); fb = nnx.to_flat_state(nnx.state(b, nnx.Param))
    for (pa, va), (pb, vb) in zip(fa, fb):
        assert pa == pb
        np.testing.assert_allclose(np.asarray(vb[...]), np.asarray(va[...]), rtol=rtol, atol=atol, err_msg=str(pa))

def test_no_mesh_is_unchanged(tmp_path):
    p = Flux1FlowPipeline(train_dataset_cond, val_dataset_cond, dim_obs, dim_cond, ch_obs=2, ch_cond=2,
                          params=small_flux1_params(), training_config=test_training_config(tmp_path))
    assert p.mesh is None and p.sharding_rule == "replicated"
    assert isinstance(p.model.obs_in.kernel[...].sharding, jax.sharding.SingleDeviceSharding)

@requires_8_devices
def test_flux1_replicated_matches_no_mesh(tmp_path):
    ref, l_ref, v_ref = _train(Flux1FlowPipeline, small_flux1_params(), None, None, tmp_path / "a")
    p, l, v = _train(Flux1FlowPipeline, small_flux1_params(), sh.make_mesh(), "replicated", tmp_path / "b")
    np.testing.assert_allclose(l, l_ref, rtol=1e-3, atol=1e-5); np.testing.assert_allclose(v, v_ref, rtol=1e-3, atol=1e-5)
    _assert_params_close(ref.model, p.model); _assert_params_close(ref.ema_model, p.ema_model)
    mesh = p.mesh
    for path, s in sh.sharding_of(p.model).items():
        assert s.is_equivalent_to(NamedSharding(mesh, P()), 1), path

@requires_8_devices
def test_unannotated_user_model_trains_with_mesh(tmp_path):
    p = ConditionalPipeline(MockConditionalModel(), train_dataset_cond, val_dataset_cond, dim_obs, dim_cond,
                            method=FlowMatchingMethod(), ch_obs=2, ch_cond=2,
                            training_config=test_training_config(tmp_path, 5), mesh=sh.make_mesh())
    p.train(nnx.Rngs(0), save_model=False)
    assert sh.sharding_of(p.model)["dummy"].is_equivalent_to(NamedSharding(p.mesh, P()), 1)

@requires_8_devices
def test_validation_batch_not_divisible_names_loader(tmp_path):
    import grain
    bad_val = (grain.MapDataset.source(np.zeros((100, dim_obs + dim_cond, 2), np.float32)).repeat()
               .to_iter_dataset().batch(12).map(lambda d: (d[:, :dim_obs], d[:, dim_obs:])))
    p = Flux1FlowPipeline(train_dataset_cond, bad_val, dim_obs, dim_cond, ch_obs=2, ch_cond=2,
                          params=small_flux1_params(), training_config=test_training_config(tmp_path, 2), mesh=sh.make_mesh())
    with pytest.raises(ValueError, match=r"validation batch size 12 is not divisible by the data-parallel device count 8"):
        p.train(nnx.Rngs(0), save_model=False)

@requires_8_devices
def test_training_config_keys_select_mesh_and_rule(tmp_path):
    tc = test_training_config(tmp_path); tc["mesh"] = "auto"; tc["sharding_rule"] = "fsdp"
    with pytest.warns(UserWarning, match="experimental"):
        p = Flux1FlowPipeline(train_dataset_cond, val_dataset_cond, dim_obs, dim_cond, ch_obs=2, ch_cond=2,
                              params=small_flux1_params(), training_config=tc)
    assert dict(p.mesh.shape) == {"data": 8, "model": 1} and p.sharding_rule == "fsdp"

def test_parse_training_config_reads_mesh_keys(tmp_path):
    from gensbi.recipes.utils import parse_training_config
    cfg = tmp_path / "c.yaml"; cfg.write_text("training:\n  nsteps: 10\n  mesh: auto\n  sharding_rule: fsdp\n")
    tc = parse_training_config(str(cfg)); assert tc["mesh"] == "auto" and tc["sharding_rule"] == "fsdp"
    cfg.write_text("training:\n  nsteps: 10\n"); tc = parse_training_config(str(cfg))
    assert tc["mesh"] is None and tc["sharding_rule"] is None
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/sharding/test_pipeline.py -q -p no:xdist` → FAIL (`unexpected keyword argument 'mesh'`).
- [ ] **Step 3: Implement** per the Interfaces block. Keep `train_step`/`val_step`/`ema_step` untouched. Thread the two kwargs through all 13 constructors (4 intermediate + 9 model pipelines) with docstring lines `mesh : jax.sharding.Mesh or None, optional` / `sharding_rule : {"replicated", "fsdp", "tensor"} or None, optional` (FSDP/tensor marked experimental).
- [ ] **Step 4: Run** `tests/sharding` then `tests/recipes` → PASS.
- [ ] **Step 5: Commit**

```bash
git add src/gensbi/recipes tests/sharding/test_pipeline.py
git commit -m "feat(recipes): mesh and sharding_rule on every pipeline; batch placement; config keys"
```

---

### Task 7: End-to-end matrix: Simformer, Flux1Joint, FSDP, optimizer/EMA placement, one-device mesh

**Files:**
- Test: `tests/sharding/test_pipeline.py`

**Interfaces:** consumes Task 6 helpers; `SimformerFlowPipeline(train_dataset_joint, val_dataset_joint, dim_obs, dim_cond, ch_obs=2, params=..., ...)` and `Flux1JointFlowPipeline(...)` take no `ch_cond`.

- [ ] **Step 1: Write the tests**

```python
from gensbi.recipes import SimformerFlowPipeline, Flux1JointFlowPipeline

def _train_joint(cls, params, mesh, rule, tmp_path, nsteps=20):
    p = cls(train_dataset_joint, val_dataset_joint, dim_obs, dim_cond, ch_obs=2, params=params,
            training_config=test_training_config(tmp_path, nsteps), mesh=mesh, sharding_rule=rule)
    losses, _ = p.train(nnx.Rngs(0), save_model=False)
    return p, np.asarray([float(l) for l in losses])

@requires_8_devices
@pytest.mark.parametrize("rule", ["replicated", "fsdp"])
@pytest.mark.parametrize("cls,params_fn", [(SimformerFlowPipeline, small_simformer_params), (Flux1JointFlowPipeline, small_flux1joint_params)])
def test_joint_pipelines_match_no_mesh(cls, params_fn, rule, tmp_path):
    ref, l_ref = _train_joint(cls, params_fn(), None, None, tmp_path / "a")
    p, l = _train_joint(cls, params_fn(), sh.make_mesh(), rule, tmp_path / "b")
    np.testing.assert_allclose(l, l_ref, rtol=1e-3, atol=1e-5)
    _assert_params_close(ref.model, p.model)

@requires_8_devices
def test_flux1_fsdp_matches_and_shards_params_opt_and_ema(tmp_path):
    ref, l_ref, _ = _train(Flux1FlowPipeline, small_flux1_params(), None, None, tmp_path / "a")
    mesh = sh.make_mesh()
    p = Flux1FlowPipeline(train_dataset_cond, val_dataset_cond, dim_obs, dim_cond, ch_obs=2, ch_cond=2,
                          params=small_flux1_params(), training_config=test_training_config(tmp_path / "b"), mesh=mesh, sharding_rule="fsdp")
    s = sh.sharding_of(p.model)
    sharded = [k for k, v in s.items() if not v.is_equivalent_to(NamedSharding(mesh, P()), 1)]
    assert sharded, "FSDP sharded nothing — hidden size must be divisible by 8"
    optimizer = p._get_optimizer(); sh.shard_model(optimizer, mesh, "fsdp")
    mu = {"/".join(map(str, k)): v for k, v in nnx.to_flat_state(nnx.state(optimizer)) if "mu" in map(str, k)}
    for name, var in s.items():
        twin = next((v for k, v in mu.items() if k.endswith(name)), None)
        if twin is not None:
            assert twin[...].sharding.is_equivalent_to(var, twin[...].ndim), name
    losses, _ = p.train(nnx.Rngs(0), save_model=False)
    np.testing.assert_allclose(np.asarray([float(l) for l in losses]), l_ref, rtol=1e-3, atol=1e-5)
    _assert_params_close(ref.model, p.model)
    for name, var in sh.sharding_of(p.ema_model).items():
        assert var.is_equivalent_to(s[name], 1), name           # EMA mirrors the model's placement after training

@requires_8_devices
def test_one_device_mesh_matches_no_mesh(tmp_path):
    ref, l_ref, _ = _train(Flux1FlowPipeline, small_flux1_params(), None, None, tmp_path / "a")
    one = sh.make_mesh(devices=jax.devices()[:1])
    p, l, _ = _train(Flux1FlowPipeline, small_flux1_params(), one, "replicated", tmp_path / "b")
    np.testing.assert_allclose(l, l_ref, rtol=1e-5, atol=1e-6)
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest tests/sharding/test_pipeline.py -q -p no:xdist` → PASS. A failure is a defect in Tasks 3-6; fix there. Loosen tolerances only with printed evidence of the max deviation.
- [ ] **Step 3: Commit** `git commit -m "test: sharded pipelines match unsharded for Simformer/Flux1Joint; FSDP placement of params, optimizer, EMA"`.

---

### Task 8: Checkpoints, safetensors, standardization keep placement

**Files:**
- Modify: `src/gensbi/recipes/pipeline.py` — `restore_model` (658-708), `restore_intermediate` (710-778), `import_safetensors` (790-798): call `self._reshard()` at the end
- Modify: `src/gensbi/recipes/flow_pipeline.py:fit_standardization` (154-182): call `self._reshard()` after `set_standardization`
- Test: `tests/sharding/test_checkpoint.py`

- [ ] **Step 1: Write the failing tests**

```python
@requires_8_devices
def test_restore_follows_current_placement(tmp_path):
    # saved WITHOUT a mesh, restored INTO a pipeline WITH fsdp -> fsdp placement; and the reverse
    tc = test_training_config(tmp_path, 3)
    a = Flux1FlowPipeline(train_dataset_cond, val_dataset_cond, dim_obs, dim_cond, ch_obs=2, ch_cond=2, params=small_flux1_params(), training_config=tc)
    a.train(nnx.Rngs(0), save_model=True)
    mesh = sh.make_mesh()
    b = Flux1FlowPipeline(train_dataset_cond, val_dataset_cond, dim_obs, dim_cond, ch_obs=2, ch_cond=2, params=small_flux1_params(), training_config=dict(tc), mesh=mesh, sharding_rule="fsdp")
    before = sh.sharding_of(b.model); b.restore_model()
    after = sh.sharding_of(b.model)
    assert all(after[k].is_equivalent_to(before[k], 1) for k in before)
    _assert_params_close(a.model, b.model, rtol=0, atol=0)
    # reverse direction: save from the fsdp pipeline under a NEW step id (orbax silently skips an existing step)
    b.save_model(experiment_id=2)
    c = Flux1FlowPipeline(train_dataset_cond, val_dataset_cond, dim_obs, dim_cond, ch_obs=2, ch_cond=2, params=small_flux1_params(), training_config=dict(tc))
    c.restore_model(experiment_id=2)
    assert isinstance(c.model.obs_in.kernel[...].sharding, jax.sharding.SingleDeviceSharding)
    _assert_params_close(b.model, c.model, rtol=0, atol=0)

@requires_8_devices
def test_safetensors_roundtrip_keeps_placement(tmp_path):
    mesh = sh.make_mesh()
    p = Flux1FlowPipeline(train_dataset_cond, val_dataset_cond, dim_obs, dim_cond, ch_obs=2, ch_cond=2, params=small_flux1_params(), training_config=test_training_config(tmp_path), mesh=mesh, sharding_rule="fsdp")
    before = sh.sharding_of(p.ema_model)
    p.export_safetensors(tmp_path / "w.safetensors"); p.import_safetensors(tmp_path / "w.safetensors")
    after = sh.sharding_of(p.ema_model)
    assert all(after[k].is_equivalent_to(before[k], 1) for k in before)

@requires_8_devices
def test_fit_standardization_keeps_placement(tmp_path):
    from gensbi.recipes import ConditionalFlowPipeline
    from gensbi.models import MAFlow, MAFlowParams
    mesh = sh.make_mesh()
    flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=dim_obs, cond_dim=dim_cond, n_layers=2, nn_width=16, channels=2, cond_channels=2))
    p = ConditionalFlowPipeline(flow, train_dataset_cond, val_dataset_cond, dim_obs, dim_cond, ch_obs=2, ch_cond=2, training_config=test_training_config(tmp_path), mesh=mesh)
    p.fit_standardization(jnp.ones((64, dim_obs, 2)), axis=(0, 1))
    for name, s in sh.sharding_of(p.model).items():
        assert isinstance(s, NamedSharding) and s.mesh == mesh, name
```

- [ ] **Step 2: Run** → FAIL on the safetensors and standardization tests (restored arrays on device 0). **Step 3:** add the `_reshard()` calls. **Step 4:** run `tests/sharding tests/recipes/test_checkpoint*.py tests/normalizing_flows` → PASS.
- [ ] **Step 5: Commit** `git commit -m "fix(recipes): keep mesh placement across restore, safetensors import, standardization"`.

---

### Task 9: Samplers: padding and placement; jitted flow sampling

**Files:**
- Modify: `src/gensbi/recipes/pipeline.py` — new `_sharded_init(self, draw, key, nsamples) -> (x_init, n_pad)` and `_trim(self, out, nsamples, axis)`; `_chunked_draw` unchanged
- Modify: `src/gensbi/recipes/conditional_pipeline.py:244-254`, `joint_pipeline.py:376-388`, `unconditional_pipeline.py:181-187` (inner `sampler` closures)
- Modify: `src/gensbi/recipes/flow_pipeline.py:217-247` (`get_sampler`) and `:316-364` (`sample_batched`)
- Test: `tests/sharding/test_sampling.py`

**Interfaces:**
- `_sharded_init(draw, key, nsamples)`: no mesh → `(draw(key, nsamples), nsamples)`; with mesh → `n_pad, _ = pad_for_mesh(nsamples, mesh)`; `(shard_sample_init(draw(key, n_pad), mesh), n_pad)`.
- `_trim(out, nsamples, axis)`: `out` if `out.shape[axis] == nsamples` else `jax.lax.slice_in_dim(out, 0, nsamples, axis=axis)`; `axis = _sample_concat_axis(sampler_kwargs)` (0, or 1 with intermediates).
- Conditional/unconditional `sampler`: `x_init, _ = self._sharded_init(lambda k, n: self.method.sample_init(k, n), key_init, nsamples)`; `return self._trim(sampler_fn(key, x_init, _extras), nsamples, axis)`. Joint: the `draw` lambda also slices `[:, :self.dim_obs, :]`.
- Flow pipeline: module-level `_jit_flow_sample = nnx.jit(lambda flow, key, cond: flow.sample(key, cond=cond))`. `get_sampler`'s closure: `n_pad, _ = pad_for_mesh(...)` when a mesh is set (else `nsamples`); `cond_b = jnp.broadcast_to(cond, (n_pad,) + cond.shape)`; placed with `shard_sample_init` when a mesh is set; `return self._trim(_jit_flow_sample(flow, key, cond_b), nsamples, 0)`. `sample_batched`: replace both `flow.sample(...)` calls with `_jit_flow_sample(flow, ...)`; place `cond` (or each chunk) on the mesh only when its leading size is divisible by `batch_size_divisor(mesh)`.

- [ ] **Step 1: Write the failing tests**

```python
from gensbi.recipes import Flux1FlowPipeline, Flux1DiffusionPipeline, Flux1SMPipeline, ConditionalFlowPipeline
from gensbi.models import MAFlow, MAFlowParams, TarFlow, TarFlowParams

def _pipe(cls, mesh, tmp_path, params=None):
    p = cls(train_dataset_cond, val_dataset_cond, dim_obs, dim_cond, ch_obs=2, ch_cond=2,
            params=params or small_flux1_params(), training_config=test_training_config(tmp_path), mesh=mesh)
    p.ema_model = p.model; p._wrap_model(); return p

x_o = jax.random.normal(jax.random.key(2), (1, dim_cond, 2))

@requires_8_devices
@pytest.mark.parametrize("cls", [Flux1FlowPipeline, Flux1DiffusionPipeline])
def test_ode_and_edm_sampling_matches_no_mesh(cls, tmp_path):
    a = _pipe(cls, None, tmp_path / "a"); b = _pipe(cls, sh.make_mesh(), tmp_path / "b")
    sa = a.sample(jax.random.key(1), x_o, nsamples=16, use_ema=False, show_progress_bars=False)
    sb = b.sample(jax.random.key(1), x_o, nsamples=16, use_ema=False, show_progress_bars=False)
    np.testing.assert_allclose(np.asarray(sb), np.asarray(sa), rtol=1e-4, atol=1e-5)

@requires_8_devices
def test_sampling_pads_small_and_chunked(tmp_path):
    b = _pipe(Flux1FlowPipeline, sh.make_mesh(), tmp_path)
    assert b.sample(jax.random.key(1), x_o, nsamples=3, use_ema=False, show_progress_bars=False).shape == (3, dim_obs, 2)
    assert b.sample(jax.random.key(1), x_o, nsamples=13, chunk_size=5, use_ema=False, show_progress_bars=False).shape == (13, dim_obs, 2)
    out = b.sample(jax.random.key(1), x_o, nsamples=5, use_ema=False, show_progress_bars=False, time_grid=jnp.linspace(0, 1, 3))
    assert out.shape[1] == 5                       # intermediates: trim on axis 1

@requires_8_devices
def test_sde_sampler_runs_with_mesh(tmp_path):
    b = _pipe(Flux1SMPipeline, sh.make_mesh(), tmp_path)
    assert b.sample(jax.random.key(1), x_o, nsamples=16, use_ema=False, show_progress_bars=False).shape == (16, dim_obs, 2)

def _flow_pipe(flow_fn, mesh, tmp_path):
    p = ConditionalFlowPipeline(flow_fn(), train_dataset_cond, val_dataset_cond, dim_obs, dim_cond, ch_obs=2, ch_cond=2,
                                training_config=test_training_config(tmp_path), mesh=mesh)
    p.ema_model = p.model; p._wrap_model(); return p

_maf = lambda: MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=dim_obs, cond_dim=dim_cond, n_layers=2, nn_width=16, channels=2, cond_channels=2))
_tar = lambda: TarFlow(TarFlowParams(rngs=nnx.Rngs(0), dim=dim_obs, cond_dim=dim_cond, num_blocks=2, num_heads=2, head_dim=8, vec_channels=2, cond_channels=2))

@requires_8_devices
@pytest.mark.parametrize("flow_fn", [_maf, _tar], ids=["maf", "tarflow"])
def test_flow_sampling_jitted_and_matches_no_mesh(flow_fn, tmp_path):
    a = _flow_pipe(flow_fn, None, tmp_path / "a"); b = _flow_pipe(flow_fn, sh.make_mesh(), tmp_path / "b")
    sa = a.sample(jax.random.key(1), x_o, nsamples=16, use_ema=False, show_progress_bars=False)
    sb = b.sample(jax.random.key(1), x_o, nsamples=16, use_ema=False, show_progress_bars=False)
    np.testing.assert_allclose(np.asarray(sb), np.asarray(sa), rtol=1e-4, atol=1e-5)
    assert b.sample(jax.random.key(1), x_o, nsamples=3, use_ema=False, show_progress_bars=False).shape == (3, dim_obs, 2)
    xb = jax.random.normal(jax.random.key(3), (4, dim_cond, 2))
    assert b.sample_batched(jax.random.key(1), xb, nsamples=4, use_ema=False, show_progress_bars=False).shape == (4, 4, dim_obs, 2)
```

(Check `TarFlowParams`/`MAFlowParams` channel kwargs against the dataclasses at `tarflow/model.py:137-161` and `maf/model.py:52-65`; the datasets carry 2 channels.)

- [ ] **Step 2: Run** → `test_sampling_pads_small_and_chunked` and the flow tests FAIL (shape/placement). **Step 3:** implement per Interfaces. **Step 4:** `tests/sharding tests/recipes/test_chunked_sampling.py tests/normalizing_flows tests/models/tarflow` → PASS; equality tests must pass with no tolerance change.
- [ ] **Step 5: Commit** `git commit -m "feat(recipes): sharded sampler inputs with padding; jitted flow-pipeline sampling"`.

---

### Task 10: Tensor-rule smoke test (4x2 mesh)

**Files:**
- Test: `tests/sharding/test_tensor_rule.py`

- [ ] **Step 1: Write the tests**

```python
@requires_8_devices
@pytest.mark.parametrize("cls,params_fn,joint", [(Flux1FlowPipeline, small_flux1_params, False), (SimformerFlowPipeline, small_simformer_params, True)])
def test_tensor_rule_matches_no_mesh(cls, params_fn, joint, tmp_path):
    mesh = sh.make_mesh(model_axis=2)
    run = _train_joint if joint else (lambda c, p, m, r, t: _train(c, p, m, r, t)[:2])
    ref, l_ref = run(cls, params_fn(), None, None, tmp_path / "a")
    with pytest.warns(UserWarning, match="experimental"):
        p, l = run(cls, params_fn(), mesh, "tensor", tmp_path / "b")
    np.testing.assert_allclose(l, l_ref, rtol=1e-3, atol=1e-5)
    s = sh.sharding_of(p.model)
    over_model = [k for k, v in s.items() if "model" in str(v.spec)]
    assert any(k.endswith("qkv/kernel") or "linear1/kernel" in k or "hidden_blocks/0/kernel" in k or "attn/query/kernel" in k for k in over_model), over_model

@requires_8_devices
def test_tensor_rule_indivisible_heads_fall_back(tmp_path):
    params = small_simformer_params(); params.num_heads = 1          # heads axis of size 1 on a model axis of 2
    with pytest.warns(UserWarning) as rec:
        p = SimformerFlowPipeline(train_dataset_joint, val_dataset_joint, dim_obs, dim_cond, ch_obs=2, params=params,
                                  training_config=test_training_config(tmp_path), mesh=sh.make_mesh(model_axis=2), sharding_rule="tensor")
    assert any("not divisible" in str(w.message) or "replicated" in str(w.message) for w in rec)
```

- [ ] **Step 2: Run** → PASS (XLA guarantees correctness; a failure means a mislabelled `heads`/`mlp` axis or a divisibility fallback bug — fix in Tasks 3-5 or 2).
- [ ] **Step 3: Commit** `git commit -m "test: experimental tensor rule smoke test on a 4x2 mesh"`.

---

### Task 11: Docs, HEAL-SWIN release gate, final verification

**Files:**
- Modify: `docs/basics/training.md` (new section "Multi-device training"), `README.md` (one bullet), `CHANGELOG.md` if present
- Modify: `pyproject.toml` (`heal-swin-nnx>=<released version>`), `uv.lock`

- [ ] **Step 1: Docs.** In `docs/basics/training.md` add ≤ 40 lines: the four-line usage from spec §4, `mesh: auto` / `sharding_rule:` YAML keys, the batch-size rule, the `replicated` default, FSDP and tensor marked experimental (FSDP memory benefit unmeasured; tensor untuned), "your own `nnx.Module` needs no annotations for data parallel; use `gensbi.sharding.axes` for FSDP", and the one-liner for the test environment. Run `.venv/bin/python -m pytest tests/sharding -q -p no:xdist` → PASS. Commit the docs on their own:

```bash
git add docs/basics/training.md README.md
git commit -m "docs: multi-device training with gensbi.sharding"
```

- [ ] **Step 2: Release gate — ask the user, and stop until they answer.** Use `AskUserQuestion`: "Has the HEAL-SWIN-nnx release with the sharding annotations been uploaded to PyPI, and which version is it?" Options: *Yes, version …* / *Not yet*. On *Not yet*, stop here and report that Tasks 1-10 are complete and only the final verification is pending; do not proceed to Step 3. On *Yes*, record the version.
- [ ] **Step 3: Pin the release.** Set `heal-swin-nnx>=<that version>` in `pyproject.toml`; ask the user to run `uv lock && uv sync --group test` outside the sandbox (suggest `! uv lock && uv sync --group test`). Verify `.venv/bin/python -c "import heal_swin_nnx.sharding"` succeeds.
- [ ] **Step 4: Full suite as CI runs it**: `.venv/bin/python -m pytest -m "not slow and not experimental and not extraslow" -n auto -q`, then `-m slow`, then `-m experimental` → PASS. Confirm with `-rs` that `test_vocabulary_pinned_to_heal_swin` and `test_healswin_encoder_annotated_via_gensbi` **ran** (not skipped) against the released package.
- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "build: depend on released heal-swin-nnx with sharding annotations"
```

Then superpowers:finishing-a-development-branch.
