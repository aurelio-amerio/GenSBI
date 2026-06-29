# Uniform `(B, dim, C)` channel convention + conditioner rename/redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the discrete-flow track (`MAFlow`/`TarFlow` + `ConditionalFlowPipeline`) carry a mandatory channel axis `(B, dim, C)` end-to-end for every `C ≥ 1` (no `C=1`-vs-`C>1` branching), and bring TarFlow's conditioners onto one uniform channel fold with clearer names.

**Architecture:** A normalizing flow is a bijection on ℝ^d; a size-1 channel axis is pure bookkeeping (numerically free). We (1) rename conditioner classes and de-conflate the `channels` config word **first** to avoid double-names, then (2) stop collapsing the channel in the tokenizer / models / pipeline, then (3) redesign the vector conditioner to tokenize the condition per-coordinate, then (4) wire and smoke-train the examples.

**Tech Stack:** Python, JAX (0.10.2), Flax NNX, pytest (9.0.2). Two repos: `GenSBI` (library) and `GenSBI-examples` (benchmark scripts).

## Global Constraints

- **Run environment:** ALL python/pytest/scripts run in the mamba env `gensbi` and on **CPU**. Prefix every command with `JAX_PLATFORMS=cpu mamba run -n gensbi …`. (Do **not** use any `.venv`.)
- **Branch:** `maf` (GenSBI). Commit after every task.
- **Spec:** `docs/superpowers/specs/2026-06-29-uniform-channel-convention-and-conditioner-redesign-design.md` — the binding contract.
- **Universal shape contract:** `log_prob(x:(B,dim,C_obs), cond:(B,cond_dim,C_cond)) -> (B,)`; `sample(...) -> (B,dim,C_obs)`; `example_shape=(dim,C_obs)` with `C=1 → (dim,1)`, never collapsed. A bare 2-D `(B,dim)` is rejected at the pipeline boundary.
- **`log_prob` returns `(B,)`**, never `(B,1)` — the channel is summed away by change-of-variables.
- **Dev branch:** breaking changes acceptable when mathematically correct; no checkpoint/numerical-identity gate. `C=1` data bookkeeping is byte-identical internally; the per-coordinate `VectorConditioner` is a genuine architecture change (zero blast radius — no example/test trains a prefix-conditioned model).
- **Repo paths:** GenSBI = `/lhome/ific/a/aamerio/data/github/GenSBI`; GenSBI-examples = `/lhome/ific/a/aamerio/data/github/GenSBI-examples`.

---

## File-structure map

| File | Responsibility | Tasks |
|---|---|---|
| `src/gensbi/models/tarflow/conditioners.py` | the 3 conditioner classes | 1, 5, 6 |
| `src/gensbi/models/tarflow/model.py` | TarFlowParams, make_cond dispatch, example_shape/mean/std, cond_channels wiring | 1, 3, 5, 6 |
| `src/gensbi/models/tarflow/blocks.py` | MetaBlock docstring (conditioner union) | 1 |
| `src/gensbi/models/core/tokenizers.py` | VectorTokenizer example_shape/detokenize | 3 |
| `src/gensbi/models/maf/model.py` | MAFlow.sample / set_standardization channel-carry | 4 |
| `src/gensbi/recipes/flow_pipeline.py` | strict channel, drop squeeze/expand, unify cond-prep, fit_standardization | 4 |
| `tests/models/{core,tarflow,maf}/…`, `tests/normalizing_flows/test_flow_pipeline.py` | updated to channel-carrying shapes + renames | 1,3,4,5,6 |
| GenSBI-examples `…/{two_moons/tarflow, slcp/tarflow_NLE, slcp/tarflow_NPE}` + 3 MAF benchmarks | configs + scripts | 2, 7 |

---

## Task 1: Rename conditioners + `cond` strings (library, behaviour-preserving)

**Files:**
- Modify: `src/gensbi/models/tarflow/conditioners.py`
- Modify: `src/gensbi/models/tarflow/model.py`
- Modify: `src/gensbi/models/tarflow/blocks.py` (docstring only)
- Test: `tests/models/tarflow/test_conditioners.py`, `test_blocks_meta.py`, `test_model.py`, `test_tarflow.py`, `test_structured_integration.py`

**Interfaces:**
- Produces (renamed, signatures UNCHANGED in this task):
  - `AdditiveBiasConditioner(cond_dim, channels, rngs)` — was `VectorConditioner`
  - `VectorConditioner(cond_dim, channels, num_tokens, rngs)` — was `VectorPrefixConditioner` (still dense prefix logic here)
  - `ImageConditioner(cond_channels, patch_size, channels, num_tokens, rngs)` — was `ImagePrefixConditioner`
  - `cond=` strings: `"bias"` (default), `"vector"`, `"image"`

- [ ] **Step 1: Update the conditioner classes (SWAP — order matters)**

In `conditioners.py`, apply the renames in this exact order using word-boundary edits:
1. `VectorConditioner` → `AdditiveBiasConditioner` (the additive-bias class, lines ~21).
2. `VectorPrefixConditioner` → `VectorConditioner` (lines ~76).
3. `ImagePrefixConditioner` → `ImageConditioner` (lines ~131).

> ⚠️ Do #1 BEFORE #2. Reverse order creates two `VectorConditioner`s and corrupts the swap. `VectorPrefixConditioner` does not contain the token `VectorConditioner`, so word-boundary replacement of #1 is safe.

Update the class docstrings' opening sentence to match the new role names (e.g. `AdditiveBiasConditioner` "Embed a vector condition as a per-token additive bias.").

- [ ] **Step 2: Update `model.py` imports, dispatch, validation, docstrings**

In `model.py`:
- Imports (lines ~21-23): `from gensbi.models.tarflow.conditioners import (AdditiveBiasConditioner, VectorConditioner, ImageConditioner,)`.
- `make_cond()` (lines ~177-186): map `cond == "bias"` → `AdditiveBiasConditioner(...)`; `cond == "vector"` → `VectorConditioner(...)`; `cond == "image"` → `ImageConditioner(...)` (keep the same constructor args as today).
- `__post_init__` (line ~142): validation set becomes `("bias", "vector", "image")`; the guard at line ~144 becomes `if self.cond == "image" and (...)`.
- `TarFlowParams.cond` default (line ~119): `cond: str = "bias"`.
- Update the `cond` docstring block (lines ~60-67) to the new strings/class names.

- [ ] **Step 3: Update the tests to the new names/strings**

- `test_conditioners.py`: `from …conditioners import AdditiveBiasConditioner` for the bias tests (lines 5-25); `from …conditioners import (VectorConditioner, ImageConditioner)` (lines 28-30); `VectorConditioner(cond_dim, channels, num_tokens=…)` in `test_vector_prefix_shapes` (line 34); `ImageConditioner(...)` (line 44).
- `test_blocks_meta.py`: line 5 `import AdditiveBiasConditioner`; line 6 `import VectorConditioner`; line 14 `cond = AdditiveBiasConditioner(cond_dim, channels, rngs=rngs)`; line 84 `cond = VectorConditioner(cond_dim, channels, num_tokens, rngs=rngs)`.
- `test_model.py:158`, `test_tarflow.py:15`, `test_structured_integration.py:55`: `cond="image_prefix"` → `cond="image"`.

- [ ] **Step 4: Run the full tarflow + model suites to verify green (behaviour unchanged)**

Run:
```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest \
  /lhome/ific/a/aamerio/data/github/GenSBI/tests/models/tarflow -q
```
Expected: PASS (same count as before the rename; pure refactor).

- [ ] **Step 5: Confirm no stale references remain**

Run:
```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI && \
  grep -rn "VectorPrefixConditioner\|ImagePrefixConditioner\|\"add\"\|'add'\|vector_prefix\|image_prefix" src/ tests/
```
Expected: no output. (The old `VectorConditioner` name now refers to the new class; that is intended.)

- [ ] **Step 6: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI && git add -A && \
git commit -m "refactor(tarflow): rename conditioners (AdditiveBias/Vector/Image) + cond strings

Behaviour-preserving swap rename: VectorConditioner->AdditiveBiasConditioner,
VectorPrefixConditioner->VectorConditioner, ImagePrefixConditioner->ImageConditioner;
cond strings add/vector_prefix/image_prefix -> bias/vector/image (default bias).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Examples — drop the `channels:` width key (3 tarflow benchmarks)

**Files (GenSBI-examples):**
- Modify: `examples/sbi-benchmarks/two_moons/tarflow/train_tarflow_npe.py` (`build_flow`)
- Modify: `examples/sbi-benchmarks/two_moons/tarflow/config/*.yaml` (all variants)
- Modify: `examples/sbi-benchmarks/slcp/tarflow_NLE/train_tarflow_nle.py` + `config/*.yaml`
- Modify: `examples/sbi-benchmarks/slcp/tarflow_NPE/train_tarflow_npe.py` + `config/*.yaml`

**Interfaces:**
- Consumes: `TarFlowParams(head_dim=…, num_heads=…)` (unchanged library API).
- Produces: configs that specify `num_heads:` directly; `build_flow` reads it. Width `head_dim·num_heads` unchanged.

- [ ] **Step 1: Edit each config — replace `channels:` with `num_heads:`**

For every tarflow `config/*.yaml`, delete the `channels:` line and add `num_heads:` with the value `old_channels // head_dim`. Example for `two_moons/tarflow/config/config_tarflow_npe.yaml` (`channels: 80`, `head_dim: 20` → `num_heads: 4`):
```yaml
model:
  num_blocks: 8
  num_heads: 4               # attention heads; total width = head_dim * num_heads
  layers_per_block: 2
  head_dim: 20               # attention head dim
  block_size: 1
  permutation: flip
  standardize: true
  zero_init: true
```
Apply the same transform to each variant (`config_tarflow_npe_v2..v5.yaml`, the slcp NLE/NPE configs): compute `num_heads = channels // head_dim` from that file's own values and drop `channels:`.

- [ ] **Step 2: Edit each `build_flow` to read `num_heads` directly**

Replace the `channels`/`num_heads` block in each `build_flow` (e.g. `two_moons/tarflow/train_tarflow_npe.py:42-54`) with:
```python
    head_dim = int(model_cfg.get("head_dim", 16))
    num_heads = int(model_cfg.get("num_heads", 4))
    return TarFlow(TarFlowParams(
        rngs=rngs,
        dim=dim_obs,
        cond_dim=dim_cond,
        num_blocks=int(model_cfg.get("num_blocks", 8)),
        layers_per_block=int(model_cfg.get("layers_per_block", 2)),
        head_dim=head_dim,
        num_heads=num_heads,
        block_size=int(model_cfg.get("block_size", 1)),
        permutation=str(model_cfg.get("permutation", "flip")),
        standardize=bool(model_cfg.get("standardize", True)),
        zero_init=bool(model_cfg.get("zero_init", True)),
    ))
```
Also update the `build_flow` docstring to drop the "config keeps specifying channels" sentence.

- [ ] **Step 3: Verify each config builds a model (CPU, gensbi env)**

For each benchmark, run a one-liner that loads the YAML and calls `build_flow` (adjust import path per script). Example:
```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -c "
import yaml, sys; sys.path.insert(0,'/lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/two_moons/tarflow')
from flax import nnx
import train_tarflow_npe as m
cfg = yaml.safe_load(open('/lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/two_moons/tarflow/config/config_tarflow_npe.yaml'))
f = m.build_flow(nnx.Rngs(0), 2, 3, cfg['model']); print('built', type(f).__name__)
"
```
Expected: `built TarFlow`.

- [ ] **Step 4: Confirm no `channels:` width key remains**

```bash
grep -rn "channels" /lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/{two_moons/tarflow,slcp/tarflow_NLE,slcp/tarflow_NPE}/config/
```
Expected: no `channels:` lines (only `num_heads`/`head_dim`).

- [ ] **Step 5: Commit (in the examples repo)**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples && git add -A && \
git commit -m "refactor(tarflow examples): replace channels: width key with num_heads

De-conflates 'channels' (now means data channels C). Width head_dim*num_heads
unchanged. Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `VectorTokenizer` + TarFlow obs channel-carry

**Files:**
- Modify: `src/gensbi/models/core/tokenizers.py` (`VectorTokenizer`)
- Modify: `src/gensbi/models/tarflow/model.py` (no logic change — `example_shape` flows from the tokenizer; verify mean/std)
- Test: `tests/models/core/test_tokenizers.py`, `tests/models/tarflow/test_blocks_meta.py`, and any TarFlow C=1 assertions in `test_model.py`/`test_tarflow.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `VectorTokenizer.example_shape == (dim, channels)` always; `detokenize(...) -> (B, dim, channels)` always. TarFlow `example_shape == (dim, C_obs)`; `log_prob((B,dim,C_obs)) -> (B,)`; `sample(...) -> (B,dim,C_obs)`.

- [ ] **Step 1: Pre-check — confirm no test wires TarFlow through the flow pipeline**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI && \
  grep -rln "ConditionalFlowPipeline" tests/ | xargs grep -ln "TarFlow" 2>/dev/null
```
Expected: no output. (If any appears, those tests must move to `(B,dim,1)` here too.)

- [ ] **Step 2: Write the failing tokenizer tests (C=1 carries the channel)**

In `tests/models/core/test_tokenizers.py`, update:
```python
def test_shapes_scalar_per_token():
    tok = VectorTokenizer(dim=6, block_size=1)
    assert (tok.T, tok.F) == (6, 1)
    x = jnp.arange(12.0).reshape(2, 6, 1)          # channel-carrying input
    t = tok.tokenize(x)
    assert t.shape == (2, 6, 1)

def test_vector_tokenizer_example_shape():
    tok = VectorTokenizer(dim=6, block_size=1)
    assert tok.example_shape == (6, 1)             # was (6,)

def test_vector_tokenizer_channels_one_unchanged():
    tok = VectorTokenizer(dim=6, block_size=2)
    assert tok.F == 2 and tok.T == 3 and tok.example_shape == (6, 1)   # was (6,)
    x = jnp.arange(2 * 6).reshape(2, 6, 1).astype(jnp.float32)
    assert jnp.array_equal(tok.detokenize(tok.tokenize(x)), x)         # (2,6,1) roundtrip
```
(Leave the `block_size`, multichannel, and Image tests as-is — they already use channel-carrying shapes.)

- [ ] **Step 3: Run to verify failure**

Run:
```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest \
  /lhome/ific/a/aamerio/data/github/GenSBI/tests/models/core/test_tokenizers.py -q
```
Expected: FAIL on `example_shape == (6, 1)` (currently `(6,)`).

- [ ] **Step 4: Implement the tokenizer change**

In `tokenizers.py` `VectorTokenizer`:
```python
        self.example_shape = (dim, channels)           # always carry the channel

    # ... in detokenize, replace the channels==1 branch with:
    def detokenize(self, tokens):
        B = tokens.shape[0]
        return tokens.reshape(B, self.dim, self.channels)
```
Update the class/`detokenize` docstrings to state the input is `(B, dim, C)` (with `C=1` as `(dim,1)`) and `detokenize` returns `(B, dim, channels)`.

- [ ] **Step 5: Update TarFlow C=1 model tests**

In `tests/models/tarflow/test_blocks_meta.py`, `test_tarflow_vector_channels_one_unchanged`:
```python
def test_tarflow_vector_channels_one_unchanged():
    m = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), modeled="vector", dim=4,
                              num_blocks=2, head_dim=8, num_heads=2))
    x = jnp.zeros((3, 4, 1))                          # channel-carrying
    assert m.log_prob(x).shape == (3,)
    assert m.sample(jax.random.PRNGKey(0), nsamples=3).shape == (3, 4, 1)
```
Grep `test_model.py`/`test_tarflow.py` for any other bare `(B, dim)` TarFlow vector inputs / `(B, dim)` sample assertions and add the trailing `1`.

- [ ] **Step 6: Run the tokenizer + tarflow suites green**

Run:
```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest \
  /lhome/ific/a/aamerio/data/github/GenSBI/tests/models/core \
  /lhome/ific/a/aamerio/data/github/GenSBI/tests/models/tarflow -q
```
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI && git add -A && \
git commit -m "feat(tokenizer,tarflow): carry channel for C=1 (example_shape always (dim,C))

VectorTokenizer.example_shape=(dim,channels) and detokenize->(B,dim,channels)
always; TarFlow C=1 now (B,dim,1) end-to-end (byte-identical internals).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: MAF channel-carry + flow-pipeline strictness (end-to-end)

> The MAF output shape and the pipeline's squeeze/expand are a **matched pair** (the flow-pipeline e2e tests assert end-to-end shapes), so they change together to stay green.

**Files:**
- Modify: `src/gensbi/models/maf/model.py` (`sample`, `set_standardization`)
- Modify: `src/gensbi/recipes/flow_pipeline.py`
- Test: `tests/models/maf/test_maflow.py`, `tests/normalizing_flows/test_flow_pipeline.py`

**Interfaces:**
- Consumes: `MAFlow.log_prob((B,dim,C)) -> (B,)` (already flattens internally).
- Produces:
  - `MAFlow.sample(...) -> (B, dim, channels)` always (C=1 → `(B,dim,1)`).
  - `MAFlow.set_standardization(mean, std)` accepts `(dim, channels)`-broadcastable stats.
  - Pipeline: `_prep_obs`/`_prep_cond` reject bare `(B,dim)`; `_single_obs(x_o) -> (cond_dim,C_cond)` (strip batch, keep channel); `get_sampler`/`sample` return `(nsamples,dim,C_obs)`; `log_prob_fn` takes `(B,dim,C_obs)` → `(B,)`; `fit_standardization` computes on native `(N,dim,C)`.

- [ ] **Step 1: Write the failing MAF tests**

In `tests/models/maf/test_maflow.py`, update `test_maflow_sample_shape_and_standardize`:
```python
def test_maflow_sample_shape_and_standardize():
    flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=4, cond_dim=2, n_layers=3))
    c = jax.random.normal(jax.random.key(0), (7, 2, 1))     # channel-carrying cond
    s = flow.sample(jax.random.key(1), cond=c)
    assert s.shape == (7, 4, 1)                              # was (7, 4)
    assert flow.log_prob(jnp.zeros((5, 4, 1)), jnp.zeros((5, 2, 1))).shape == (5,)
    flow.set_standardization(jnp.ones((4, 1)), 2.0 * jnp.ones((4, 1)))   # (dim,1)
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest \
  /lhome/ific/a/aamerio/data/github/GenSBI/tests/models/maf/test_maflow.py -q
```
Expected: FAIL (`sample` returns `(7,4)`; `set_standardization((4,1))` raises on broadcast).

- [ ] **Step 3: Implement the MAF change**

In `maf/model.py` `sample`, replace the trailing reshape:
```python
        x = x.reshape(x.shape[0], self.dim, self.channels)   # always carry the channel
        return x
```
And `set_standardization`:
```python
    def set_standardization(self, mean, std) -> None:
        target = (self.dim, self.channels)
        mean = jnp.broadcast_to(jnp.asarray(mean), target).reshape(-1)
        std = jnp.broadcast_to(jnp.asarray(std), target).reshape(-1)
        for b in self.chain.bijections:
            if isinstance(b, Standardize):
                b.set_stats(mean, std)
                return
        raise ValueError(
            "MAFlow has no Standardize bijection (built with standardize=False).")
```
Update the `sample` docstring "Returns" to `(nsamples, dim, channels)` for all `C`.

- [ ] **Step 4: Run MAF tests green**

Run:
```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest \
  /lhome/ific/a/aamerio/data/github/GenSBI/tests/models/maf -q
```
Expected: PASS. (If `test_maflow_density.py`/others assert `(B,dim)` sample shapes, add the trailing `1`.)

- [ ] **Step 5: Write the failing pipeline tests**

In `tests/normalizing_flows/test_flow_pipeline.py`:
- Replace the import `_squeeze_ch, _single_cond, _structured_cond` with `_require_channel, _single_obs`.
- Replace `test_squeeze_ch` with:
```python
def test_require_channel_rejects_bare_2d():
    assert _require_channel(jnp.zeros((4, DIM_OBS, 1)), "obs").shape == (4, DIM_OBS, 1)
    with pytest.raises(ValueError):
        _require_channel(jnp.zeros((4, DIM_OBS)), "obs")        # bare (B,dim) rejected
```
- Replace `test_single_cond` / `test_single_cond_batched_warns_and_takes_first` / `test_structured_cond_strips_only_batch_axis` with:
```python
def test_single_obs_keeps_channel_strips_batch():
    assert _single_obs(jnp.zeros((1, DIM_COND, 1))).shape == (DIM_COND, 1)
    img = jnp.arange(1*1*4*2).reshape(1, 1, 4, 2)               # (B=1,H=1,W=4,C=2)
    assert _single_obs(img).shape == (1, 4, 2)                  # H==1 preserved

def test_single_obs_batched_warns_and_takes_first():
    x_o = jnp.arange(3 * DIM_COND).reshape(3, DIM_COND, 1)
    with pytest.warns(UserWarning, match="batch dimension"):
        out = _single_obs(x_o)
    assert out.shape == (DIM_COND, 1) and jnp.array_equal(out, x_o[0])
```
- The unknown-kwarg tests pass a bare `(1, DIM_COND)` cond — change each to `(1, DIM_COND, 1)` (lines ~227, 233, 239, 245).

- [ ] **Step 6: Run to verify failure**

Run:
```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest \
  /lhome/ific/a/aamerio/data/github/GenSBI/tests/normalizing_flows/test_flow_pipeline.py -q
```
Expected: FAIL on import of `_require_channel`/`_single_obs` (not defined yet).

- [ ] **Step 7: Implement the pipeline change**

In `flow_pipeline.py`:
- Remove the `from gensbi.utils.math import _expand_dims` import and the `_squeeze_ch`, `_single_cond`, `_structured_cond` helpers.
- Add:
```python
def _require_channel(x, name="input"):
    """Enforce a tabular channel axis (B, dim, C); reject a bare (B, dim)."""
    x = jnp.asarray(x)
    if x.ndim < 3:
        raise ValueError(
            f"{name} must carry a channel axis (B, dim, C); got shape "
            f"{tuple(x.shape)}. A bare (B, dim) is not accepted — add a trailing "
            f"channel axis (e.g. x[..., None] for C=1).")
    return x


def _single_obs(x_o):
    """Strip the leading batch axis from ONE observation, keeping the channel
    (and any structured) axes. Warn + take-first on a batch axis > 1."""
    x_o = jnp.asarray(x_o)
    if x_o.ndim < 2:
        raise ValueError(
            "x_o must carry a leading batch axis (e.g. (1, dim_cond, C)); got "
            f"shape {tuple(x_o.shape)}.")
    _warn_if_batched(x_o.shape[0])
    return x_o[0]
```
- In `__init__`, drop `_obs_passthrough`/`_cond_passthrough`; keep `structured_obs`/`structured_cond`, `ch_obs`/`ch_cond`.
- `_prep_obs`/`_prep_cond`:
```python
    def _prep_obs(self, x):
        x = jnp.asarray(x)
        return x if self.structured_obs else _require_channel(x, "obs")

    def _prep_cond(self, x):
        x = jnp.asarray(x)
        return x if self.structured_cond else _require_channel(x, "cond")
```
- `fit_standardization`: drop the squeeze branch — compute on native shape:
```python
        obs = jnp.asarray(obs_data)
        mean = jnp.mean(obs, axis=axis)
        std = jnp.std(obs, axis=axis)
        std = jnp.where(std < 1e-6, 1.0, std)
        self.model.set_standardization(mean, std)
        self.ema_model.set_standardization(mean, std)
        self._standardized = True
```
- `get_sampler` (replace the passthrough branching):
```python
        _warn_unused_kwargs(kwargs)
        flow = self.ema_model if use_ema else self.model
        cond = _single_obs(x_o)                         # (cond_dim, C_cond) or (H,W,C)

        def sampler(key, nsamples):
            cond_b = jnp.broadcast_to(cond, (nsamples,) + cond.shape)
            return flow.sample(key, cond=cond_b)        # model owns (nsamples, dim, C)
        return sampler
```
- `get_log_prob_fn` (replace the passthrough branching):
```python
        _warn_unused_kwargs(kwargs)
        flow = self.ema_model if use_ema else self.model
        cond = _single_obs(x_o)

        def log_prob_fn(x_1):
            obs = self._prep_obs(x_1)
            cond_b = jnp.broadcast_to(cond, (obs.shape[0],) + cond.shape)
            return flow.log_prob(obs, cond_b)           # (B,)
        return log_prob_fn
```
- Update docstrings that mention `(nsamples, dim_obs, 1)` via `_expand_dims` to "the model's native `(nsamples, dim, C)`", and the `_squeeze_ch`/`_single_cond` references.

- [ ] **Step 8: Run the flow-pipeline suite green**

Run:
```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest \
  /lhome/ific/a/aamerio/data/github/GenSBI/tests/normalizing_flows/test_flow_pipeline.py -q
```
Expected: PASS (sample → `(64,DIM_OBS,1)`; multichannel → `(7,DIM_OBS,2)`; `log_prob` → `(B,)`).

- [ ] **Step 9: Run the broader suite to catch fallout (inference/diagnostics)**

Run:
```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest \
  /lhome/ific/a/aamerio/data/github/GenSBI/tests/normalizing_flows \
  /lhome/ific/a/aamerio/data/github/GenSBI/tests/inference -q
```
Expected: PASS. (If `inference/posterior` tests squeeze `x_o`, confirm they still pass `(B,)`/scalar from `log_prob` — unchanged.)

- [ ] **Step 10: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI && git add -A && \
git commit -m "feat(maf,pipeline): channel-carrying I/O end-to-end; reject bare (B,dim)

MAF.sample->(B,dim,C) always; set_standardization accepts (dim,C). Pipeline drops
_squeeze_ch/_expand_dims, requires the channel axis, unifies cond-prep to keep it,
and standardizes on native (N,dim,C).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: `AdditiveBiasConditioner` channel fold

**Files:**
- Modify: `src/gensbi/models/tarflow/conditioners.py` (`AdditiveBiasConditioner`)
- Modify: `src/gensbi/models/tarflow/model.py` (`make_cond` passes `cond_channels`)
- Test: `tests/models/tarflow/test_conditioners.py`

**Interfaces:**
- Produces: `AdditiveBiasConditioner(cond_dim, channels, rngs, cond_channels=1)`; `embed((B,cond_dim,C_cond)) -> ((B,channels), None)`; accepts flat `(B,cond_dim)` too (it flattens).

- [ ] **Step 1: Write the failing test**

In `test_conditioners.py`:
```python
def test_additive_bias_channel_carrying_cond():
    cond_dim, cond_channels, channels, B = 3, 2, 8, 4
    c = AdditiveBiasConditioner(cond_dim, channels, rngs=nnx.Rngs(0),
                                cond_channels=cond_channels)
    cond = jax.random.normal(jax.random.PRNGKey(1), (B, cond_dim, cond_channels))
    bias, prefix = c.embed(cond)
    assert bias.shape == (B, channels) and prefix is None
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest \
  "/lhome/ific/a/aamerio/data/github/GenSBI/tests/models/tarflow/test_conditioners.py::test_additive_bias_channel_carrying_cond" -q
```
Expected: FAIL (`cond_channels` kwarg unknown / Linear shape mismatch on the 3-D input).

- [ ] **Step 3: Implement the channel fold**

In `conditioners.py` `AdditiveBiasConditioner`:
```python
    def __init__(self, cond_dim, channels, rngs, cond_channels: int = 1):
        self.cond_dim = cond_dim
        self.cond_channels = cond_channels
        if cond_dim > 0:
            self.l1 = nnx.Linear(cond_dim * cond_channels, channels, rngs=rngs)
            self.l2 = nnx.Linear(channels, channels, rngs=rngs)

    def embed(self, cond):
        if self.cond_dim == 0:
            return (None, None)
        if cond is None:
            raise ValueError(
                "cond is required: this conditioner was built with cond_dim > 0")
        cond = jnp.asarray(cond).reshape(cond.shape[0], -1)    # (B, cond_dim*C_cond)
        bias = self.l2(jax.nn.silu(self.l1(cond)))
        return (bias, None)
```
Add `import jax.numpy as jnp` to `conditioners.py` if not present.

In `model.py` `make_cond`, pass `cond_channels`:
```python
            if params.cond == "bias":
                return AdditiveBiasConditioner(params.cond_dim, channels, rngs=rngs,
                                               cond_channels=params.cond_channels)
```

- [ ] **Step 4: Run the conditioner + tarflow suites green**

Run:
```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest \
  /lhome/ific/a/aamerio/data/github/GenSBI/tests/models/tarflow -q
```
Expected: PASS (the existing bias test passing flat `(B,cond_dim)` still works — `reshape(B,-1)` is a no-op there).

- [ ] **Step 5: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI && git add -A && \
git commit -m "feat(tarflow): AdditiveBiasConditioner folds cond channel (cond_channels)

Flattens (B,cond_dim,C_cond)->(B,cond_dim*C_cond); Linear sized cond_dim*cond_channels.
C_cond=1 numerically identical. Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: `VectorConditioner` per-coordinate redesign (+ ImageConditioner verify)

**Files:**
- Modify: `src/gensbi/models/tarflow/conditioners.py` (`VectorConditioner`)
- Modify: `src/gensbi/models/tarflow/model.py` (`make_cond` for `"vector"`; remove `prefix_tokens`)
- Test: `tests/models/tarflow/test_conditioners.py`, `test_blocks_meta.py`

**Interfaces:**
- Produces: `VectorConditioner(cond_dim, cond_channels, channels, rngs)`; `embed((B,cond_dim,C_cond)) -> (None, (B,cond_dim,channels))` (M = cond_dim prefix tokens; no flatten). `prefix_tokens` removed from `TarFlowParams`.

- [ ] **Step 1: Write the failing tests**

In `test_conditioners.py`, replace `test_vector_prefix_shapes` and `test_prefix_depends_on_condition`:
```python
def test_vector_conditioner_per_coordinate_tokens():
    cond_dim, cond_channels, channels, B = 3, 2, 8, 4
    c = VectorConditioner(cond_dim, cond_channels, channels, rngs=nnx.Rngs(0))
    assert c.M == cond_dim                                   # one token per coordinate
    cond = jax.random.normal(jax.random.PRNGKey(1), (B, cond_dim, cond_channels))
    bias, prefix = c.embed(cond)
    assert bias is None and prefix.shape == (B, cond_dim, channels)

def test_vector_conditioner_depends_on_condition():
    c = VectorConditioner(3, 1, 8, rngs=nnx.Rngs(0))
    _, p1 = c.embed(jnp.zeros((2, 3, 1)))
    _, p2 = c.embed(jnp.ones((2, 3, 1)))
    assert not jnp.allclose(p1, p2)
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest \
  /lhome/ific/a/aamerio/data/github/GenSBI/tests/models/tarflow/test_conditioners.py -q -k vector_conditioner
```
Expected: FAIL (old signature `(cond_dim, channels, num_tokens, rngs)`).

- [ ] **Step 3: Implement the per-coordinate redesign**

In `conditioners.py`, rewrite `VectorConditioner`:
```python
class VectorConditioner(nnx.Module):
    """Embed a vector condition as one prefix token per coordinate.

    Each of the ``cond_dim`` coordinates is a token carrying ``C_cond``
    channels; a shared ``Linear(cond_channels, channels)`` projects each to the
    transformer width, plus per-coordinate positional embeddings. Produces
    ``M = cond_dim`` prefix tokens (no flatten). Mirrors the modeled-variable
    tokenizer and the image conditioner.
    """

    def __init__(self, cond_dim: int, cond_channels: int, channels: int,
                 rngs: nnx.Rngs):
        self.cond_dim = cond_dim
        self.cond_channels = cond_channels
        self.channels = channels
        self.M = cond_dim
        self.proj = nnx.Linear(cond_channels, channels, rngs=rngs)
        self.pos = nnx.Param(
            jax.random.normal(rngs.params(), (cond_dim, channels)) * 1e-2)

    def embed(self, cond):
        if cond is None:
            raise ValueError("cond is required for VectorConditioner")
        cond = jnp.asarray(cond)                       # (B, cond_dim, C_cond)
        h = self.proj(cond)                            # (B, cond_dim, channels)
        return (None, h + self.pos[...][None])
```

In `model.py`:
- `make_cond` `"vector"` branch:
```python
            if params.cond == "vector":
                return VectorConditioner(params.cond_dim, params.cond_channels,
                                         channels, rngs=rngs)
```
- Remove the `prefix_tokens` field from `TarFlowParams` and its docstring entry (no longer used: `M = cond_dim`).

- [ ] **Step 4: Update the block-level prefix tests to channel-carrying cond**

In `test_blocks_meta.py` `_make_prefix` and the `test_prefix_*` tests:
```python
def _make_prefix(T=4, F=1, channels=8, cond_dim=2, cond_channels=1, zero_init=False,
                 rngs=None):
    rngs = rngs or nnx.Rngs(0)
    perm = jnp.arange(T)
    cond = VectorConditioner(cond_dim, cond_channels, channels, rngs=rngs)
    return MetaBlock(F=F, channels=channels, T=T, perm=perm,
                     inv_perm=jnp.argsort(perm), conditioner=cond, num_layers=2,
                     num_heads=2, expansion=2, rngs=rngs, zero_init=zero_init)
```
In each `test_prefix_*`, give the cond a channel axis: replace `(2,)`/`jnp.array([0.3,-0.4])` conds with `(..., 1)` shapes, e.g. `cond = jnp.array([0.3, -0.4])[:, None]` (→ `(2,1)`) for the unbatched jacrev cases, and `jax.random.normal(key,(3,2,1))` for batched roundtrips. The `MetaBlock`/`AttentionBlock` are unchanged; only the conditioner input rank grows by one.

- [ ] **Step 5: Run the tarflow suite green; confirm ImageConditioner already folds**

Run:
```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest \
  /lhome/ific/a/aamerio/data/github/GenSBI/tests/models/tarflow -q
```
Expected: PASS. `ImageConditioner` already folds the channel via `patchify_2d` (`in_f = cond_channels·patch²`) and needs no change beyond the Task-1 rename — `test_image_prefix_shapes` (now `ImageConditioner`) confirms it.

- [ ] **Step 6: Confirm `prefix_tokens` is fully removed**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI && grep -rn "prefix_tokens" src/ tests/
```
Expected: no output.

- [ ] **Step 7: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI && git add -A && \
git commit -m "feat(tarflow): per-coordinate VectorConditioner (no flatten, M=cond_dim)

Linear(cond_channels,channels) shared across coordinates -> (B,cond_dim,channels)
prefix tokens; removes prefix_tokens. ImageConditioner already folds via patchify.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Examples — channel-carrying wiring + smoke-train all 6

**Files (GenSBI-examples):**
- Modify (TarFlow): the 3 tarflow benchmark scripts/configs (add `vec_channels`/`cond_channels` if `>1`; channel-carrying inference `x_o`).
- Modify (MAF): `two_moons/maf_NPE`, `two_moons/maf_NLE`, `slcp/maf_NLE` scripts (confirm channel-carrying inference `x_o`).

**Interfaces:**
- Consumes: the finished library (Tasks 1–6).
- Produces: 6 scripts that train a few steps end-to-end on CPU.

- [ ] **Step 1: Audit each script's inference `x_o` shape**

```bash
grep -rn "sample\|log_prob\|x_o\|build_target\|\[None\]" \
  /lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/{two_moons/tarflow,slcp/tarflow_NLE,slcp/tarflow_NPE,two_moons/maf_NPE,two_moons/maf_NLE,slcp/maf_NLE}/*.py
```
For each call into `pipeline.sample(x_o, …)` / `pipeline.log_prob(x_1, x_o, …)`, ensure `x_o` carries a leading batch axis **and** a channel axis: `(1, dim_cond, 1)`. The dataset already emits `(B, dim, 1)` (sbibm-jax collate), so `x_o = data_x[idx][None]` already yields `(1, dim_cond, 1)`; fix any site that squeezes it to `(1, dim_cond)` or `(dim_cond,)`.

- [ ] **Step 2: For any `C>1` TarFlow benchmark, set the data-channel params**

If a benchmark genuinely has `C_obs>1`/`C_cond>1` (the 3 named SBI benchmarks are `C=1`, so this is usually a no-op), pass `vec_channels=C_obs`/`cond_channels=C_cond` in `build_flow`'s `TarFlowParams(...)` and matching `ch_obs`/`ch_cond` to `ConditionalFlowPipeline(...)`.

- [ ] **Step 3: Smoke-train each of the 6 scripts a few steps (CPU)**

For each benchmark, run its trainer with a tiny step budget (use the script's CLI/override for `nsteps`; if none, temporarily set `nsteps≈3` via the config). Example pattern:
```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python \
  /lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/two_moons/tarflow/train_tarflow_npe.py \
  --config <path-or-flag> --nsteps 3        # adapt to each script's arg parser
```
Expected for each: training loop runs, loss is finite, sampling returns `(nsamples, dim, 1)`, no shape errors. Capture the final line of each run.

- [ ] **Step 4: Full library regression (CPU, gensbi env)**

Run:
```bash
JAX_PLATFORMS=cpu mamba run -n gensbi python -m pytest \
  /lhome/ific/a/aamerio/data/github/GenSBI/tests -q
```
Expected: PASS (whole suite).

- [ ] **Step 5: Commit (examples repo)**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples && git add -A && \
git commit -m "feat(examples): channel-carrying inference for the 6 SBI flow benchmarks

x_o passed as (1,dim_cond,1); vec_channels/cond_channels set where C>1. Smoke-trained
3 MAF + 3 TarFlow on CPU. Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Universal contract (`log_prob->(B,)`, `sample->(B,dim,C)`, reject bare 2-D) → Tasks 3, 4. ✓
- Part 1a rename + cond strings → Task 1. ✓
- Part 1b examples `channels:` key → Task 2. ✓
- Part 2.1 VectorTokenizer → Task 3. ✓
- Part 2.2 TarFlow → Task 3. ✓
- Part 2.3 MAF → Task 4. ✓
- Part 3 AdditiveBias fold → Task 5; per-coordinate VectorConditioner + `prefix_tokens` removal → Task 6; ImageConditioner (already folds) → verified Task 6. ✓
- Part 4 pipeline strictness/unified cond-prep/fit_standardization → Task 4. ✓
- Part 5 tests → folded into each task (TDD). ✓
- Part 6 examples + 6-script smoke-train → Task 7. ✓

**Placeholder scan:** No "TBD"/"add error handling"-style placeholders; every code step shows code; Task 7 Step 3 notes per-script arg-parser adaptation (unavoidable — scripts differ) but gives the exact command shape and the pass criterion.

**Type consistency:** `_require_channel`/`_single_obs` (Task 4) used consistently; `VectorConditioner(cond_dim, cond_channels, channels, rngs)` defined in Task 6 and called identically in `make_cond` and `_make_prefix`; `AdditiveBiasConditioner(cond_dim, channels, rngs, cond_channels=1)` consistent in Task 5; `cond` strings `bias/vector/image` consistent across Tasks 1, 5, 6.

## Execution Handoff

Choose how to execute (see end of message).
