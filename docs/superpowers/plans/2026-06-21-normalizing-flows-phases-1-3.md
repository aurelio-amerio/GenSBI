# Normalizing Flows — Phases 1–3 (NPE / NLE / RQ-NSF) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wrap the Phase-0 affine-MAF density core into first-class SBI inference — a trainable **NPE** pipeline (`ConditionalFlowPipeline`), an **NLE** posterior over NumPyro NUTS (`NLEPosterior`), and an **RQ-NSF** spline transformer (`RQSpline`) — reusing the existing training loop / EMA / checkpointing / diagnostics unchanged.

**Architecture:** A *parallel track* to the flow-matching/diffusion methods. The flow **is** the density model (not the CNF-shaped `GenerativeMethod`). Phase 1 subclasses `AbstractPipeline` and overrides only the method-specific hooks (no `ConditionalWrapper`; the flow is the model). Phase 2 builds a NumPyro **potential function** `-(log q(x_o|θ) + log p(θ))` and hands it to NUTS — no `Distribution` wrapper. Phase 3 adds one new elementwise transformer behind the existing `(value, params)` interface; MADE already generalizes via `transformer.num_params`.

**Tech Stack:** JAX, Flax NNX (0.12.x), Optax, NumPyro (0.21), Grain (datasets), pytest (+ pytest-xdist `-n 2`, CPU-forced via `JAX_PLATFORMS=cpu`).

---

## Context the implementer must not rediscover

**Phase-0 core is COMPLETE on branch `maf`** (28 tests pass). These signatures are LOCKED — call them, don't change them:

```python
# gensbi/normalizing_flows/flow.py
make_maf(rngs, dim, cond_dim=0, n_layers=5, transformer=None,
         nn_width=64, nn_depth=2, permutation="reverse",
         standardize=True, zero_init=True) -> Flow
Flow.log_prob(x, cond=None) -> (batch,)          # base.log_prob(u) + logdet
Flow.sample(key, cond=None, nsamples=None) -> (n, dim)   # n defaults to cond.shape[0]
Flow.chain                                       # the Chain (nnx.List of bijections)

# gensbi/normalizing_flows/bijections/transformers.py
Affine().num_params == 2                          # [shift mu, log-scale a]
Affine.forward(u, params)/inverse(x, params) -> (val, logdet)
Affine.forward_dim(u_i, params_i) -> x_i          # scalar, used by the sampling scan

# gensbi/normalizing_flows/bijections/standardize.py
Standardize(dim).set_stats(mean, std)             # in-place; default identity
Standardize.inverse(x) -> ((x-mean)/std, -sum(log std))   # data->noise (applied FIRST in chain.inverse)
```

**Direction convention (LOCKED):** `inverse` = data→noise = fast one-pass = used by `log_prob`. `forward` = noise→data = slow `lax.scan` = used by `sample`. Each method returns `(output, log_det)` where `log_det` is the log|det| of *that* method's Jacobian.

**Float32 everywhere** (exact-likelihood model; bf16 wrecks the Jacobian/log-det precision).

**`AbstractPipeline` facts** (`src/gensbi/recipes/pipeline.py`) — verified by reading the source:

- `__init__(self, model, train_dataset, val_dataset, dim_obs, dim_cond, ch_obs=1, ch_cond=None, params=None, training_config=None)`. It stores the model, sets `self.ema_model = nnx.clone(model)`, sets `self.model_wrapped = self.ema_model_wrapped = None`, fills `training_config` from `get_default_training_config()` when `None`, and `os.makedirs(checkpoint_dir)`. It does **not** call any abstract method during construction.
- **Abstract methods a subclass MUST define** (else instantiation raises `TypeError`): `init_pipeline_from_config` (classmethod), `_make_model`, `get_default_params` (classmethod), `get_loss_fn`, `_wrap_model`, `get_sampler`, `sample`, `get_log_prob_fn`, `log_prob`.
- **Training loop** (`train(self, rngs, nsteps=None, save_model=True)`): each step `loss = train_step(self.model, optimizer, batch, rngs.train_step())` where the jitted `train_step` does `loss, grads = nnx.value_and_grad(loss_fn)(model, batch, key); optimizer.update(model, grads, value=loss)`. EMA every `multistep` via `ema_step(self.ema_model, self.model, ema_optimizer)`. **The loss function signature is `loss_fn(model, batch, key) -> scalar`.** Batches come from `next(self.train_dataset_iter)`.
- The optimizer is `nnx.Optimizer(self.model, opt, wrt=nnx.Param)` and EMA averages only `nnx.Param`/`nnx.BatchStat`. **Phase-0 masks/buffers are `Mask(nnx.Variable)` — non-Param — so the optimizer and EMA never touch them, and checkpointing (`nnx.split(model)`, full state) still saves/restores them.** This is why `Standardize` works.
- At the end of `train()` and `restore_model()`, the base calls `_wrap_model()`.

**`ConditionalPipeline` conventions to mirror** (`src/gensbi/recipes/conditional_pipeline.py`): the training batch is a tuple `(obs, cond)`; for **NPE** `obs = θ`, `cond = x`. Tabular SBI data carries a trailing channel axis of size 1 → shape `(B, dim, ch=1)`. `.sample(key, x_o, nsamples)` returns `(nsamples, dim_obs, ch_obs)`; `.log_prob(x_1, x_o)` returns `(B,)`. `_expand_dims` lives at `gensbi.utils.math` and maps `(n, dim) -> (n, dim, 1)` and `(dim,) -> (1, dim, 1)`.

**Test harness** (`tests/recipes/test_unified_conditional_pipeline.py`): datasets are Grain pipelines `grain.MapDataset.source(np_array).shuffle(s).repeat().to_iter_dataset().batch(bs).map(split_obs_cond)`. Tests force CPU with `os.environ["JAX_PLATFORMS"]="cpu"` at the top, train 2 steps, use `tempfile.TemporaryDirectory(dir=home)` for `checkpoint_dir`, and `save_model=False`. `pyproject.toml` registers markers `slow` and `experimental` and sets `addopts="-n 2"`, `env=["JAX_PLATFORMS=cpu"]`.

**Run the new tests** (GPUs are usually busy — keep the CPU prefix):

```bash
JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/ -v
```

The fast battery excludes the exploratory end-to-end tests: add `-m "not slow"`. Run only the slow ones with `-m slow`.

**Each of the three phases is an independently shippable milestone.** Do them in order (Phase 2 and the slow Phase-1 test reuse the Phase-1 pipeline; Phase 3 is independent of 1–2 but shares the core).

---

# PHASE 1 — NPE: `ConditionalFlowPipeline`

Trains `q(θ | x)` by max-likelihood and exposes `.sample(x_o)` / `.log_prob(θ, x_o)` with the exact `(obs, cond)` convention as `ConditionalPipeline`, so SBC/TARP/C²ST in `diagnostics/` run unchanged → a clean FM-NPE vs NF-NPE comparison.

---

### Task 1: `Flow.set_standardization` — reach the data-end Standardize buffer

The pipeline must set the `Standardize` mean/std from training stats. Add a method to `Flow` that finds the standardize bijection in its chain and sets it.

**Files:**
- Modify: `src/gensbi/normalizing_flows/flow.py` (add a method to `Flow`; `Standardize` is already imported at the top)
- Test: `tests/normalizing_flows/test_flow.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/normalizing_flows/test_flow.py`:

```python
def test_set_standardization_sets_buffers():
    import jax.numpy as jnp
    from flax import nnx
    from gensbi.normalizing_flows import make_maf
    from gensbi.normalizing_flows.bijections.standardize import Standardize

    flow = make_maf(nnx.Rngs(0), dim=3, cond_dim=2, n_layers=2, standardize=True)
    mean = jnp.array([1.0, -2.0, 0.5])
    std = jnp.array([2.0, 0.5, 3.0])
    flow.set_standardization(mean, std)

    std_bij = [b for b in flow.chain.bijections if isinstance(b, Standardize)][0]
    assert jnp.allclose(std_bij.mean.value, mean)
    assert jnp.allclose(std_bij.std.value, std)


def test_set_standardization_raises_without_bijection():
    import pytest
    from flax import nnx
    from gensbi.normalizing_flows import make_maf

    flow = make_maf(nnx.Rngs(0), dim=2, cond_dim=1, n_layers=2, standardize=False)
    with pytest.raises(ValueError):
        flow.set_standardization([0.0, 0.0], [1.0, 1.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_flow.py::test_set_standardization_sets_buffers -v`
Expected: FAIL — `AttributeError: 'Flow' object has no attribute 'set_standardization'`.

- [ ] **Step 3: Add the method**

In `src/gensbi/normalizing_flows/flow.py`, inside `class Flow`, after `sample(...)`:

```python
    def set_standardization(self, mean, std) -> None:
        """Set the data-end Standardize bijection's mean/std buffers in place.

        Raises ValueError if the flow was built with ``standardize=False``.
        """
        mean = jnp.asarray(mean)
        std = jnp.asarray(std)
        for b in self.chain.bijections:
            if isinstance(b, Standardize):
                b.set_stats(mean, std)
                return
        raise ValueError(
            "Flow has no Standardize bijection (built with standardize=False).")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_flow.py -k standardization -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/flow.py tests/normalizing_flows/test_flow.py
git commit -m "feat(nflows): Flow.set_standardization to set data-end buffers"
```

---

### Task 2: `ConditionalFlowPipeline` skeleton — subclass, ch adapters, stubs, `_wrap_model`

Create the pipeline file with the `(B, dim, ch)`↔`(B, dim)` adapters, the constructor, the three not-implemented stubs (mirroring `ConditionalPipeline`), and the identity `_wrap_model`.

**Files:**
- Create: `src/gensbi/recipes/flow_pipeline.py`
- Test: `tests/normalizing_flows/test_flow_pipeline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/normalizing_flows/test_flow_pipeline.py`:

```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import grain
import pytest

from gensbi.normalizing_flows import make_maf
from gensbi.recipes.flow_pipeline import (
    ConditionalFlowPipeline, _squeeze_ch, _single_cond,
)

DIM_OBS = 2
DIM_COND = 3
N = 1024

_key = jax.random.PRNGKey(0)
_kth, _kx = jax.random.split(_key)
_theta = jax.random.normal(_kth, (N, DIM_OBS))
_W = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])          # (DIM_COND, DIM_OBS)
_x = _theta @ _W.T + 0.1 * jax.random.normal(_kx, (N, DIM_COND))
DATA = jnp.concatenate([_theta[..., None], _x[..., None]], axis=1)  # (N, 5, 1)


def split_obs_cond(d):
    return d[:, :DIM_OBS], d[:, DIM_OBS:]


def _make_ds(arr, bs=128):
    return (grain.MapDataset.source(np.array(arr)).shuffle(0).repeat()
            .to_iter_dataset().batch(bs).map(split_obs_cond))


def build_pipeline(**cfg):
    flow = make_maf(nnx.Rngs(0), dim=DIM_OBS, cond_dim=DIM_COND,
                    n_layers=4, nn_width=32, nn_depth=2, standardize=True)
    train_ds = _make_ds(DATA[:800])
    val_ds = _make_ds(DATA[800:])
    training_config = ConditionalFlowPipeline.get_default_training_config()
    training_config["val_every"] = 1
    training_config.update(cfg)
    return ConditionalFlowPipeline(
        flow, train_ds, val_ds, DIM_OBS, DIM_COND,
        ch_obs=1, ch_cond=1, training_config=training_config)


def test_squeeze_ch():
    x = jnp.zeros((4, DIM_OBS, 1))
    assert _squeeze_ch(x).shape == (4, DIM_OBS)
    assert _squeeze_ch(jnp.zeros((4, DIM_OBS))).shape == (4, DIM_OBS)
    with pytest.raises(ValueError):
        _squeeze_ch(jnp.zeros((4, DIM_OBS, 2)))


def test_single_cond():
    assert _single_cond(jnp.zeros((1, DIM_COND, 1))).shape == (DIM_COND,)
    assert _single_cond(jnp.zeros((DIM_COND,))).shape == (DIM_COND,)


def test_init_and_wrap():
    pipe = build_pipeline()
    assert isinstance(pipe, ConditionalFlowPipeline)
    assert pipe.ema_model is not None
    assert pipe.model_wrapped is None            # not wrapped yet
    pipe._wrap_model()
    assert pipe.model_wrapped is pipe.model      # identity, no ConditionalWrapper
    assert pipe.ema_model_wrapped is pipe.ema_model


def test_stubs_raise():
    with pytest.raises(NotImplementedError):
        ConditionalFlowPipeline.get_default_params(2, 3, 1, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_flow_pipeline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gensbi.recipes.flow_pipeline'`.

- [ ] **Step 3: Create the pipeline skeleton**

Create `src/gensbi/recipes/flow_pipeline.py`:

```python
"""NPE pipeline for discrete normalizing flows (parallel track).

The flow IS the density model: no ``ConditionalWrapper``, no ``GenerativeMethod``.
Trains ``q(obs | cond)`` by max-likelihood. NPE convention: ``obs = theta``,
``cond = x`` (mirrors ``ConditionalPipeline`` so the diagnostics run unchanged).
"""

import warnings

import jax.numpy as jnp

from gensbi.recipes.pipeline import AbstractPipeline
from gensbi.utils.math import _expand_dims


def _squeeze_ch(x):
    """``(B, dim, 1) -> (B, dim)``; pass ``(B, dim)`` through. Asserts ch == 1."""
    x = jnp.asarray(x)
    if x.ndim == 3:
        if x.shape[-1] != 1:
            raise ValueError(
                f"flow pipeline requires a singleton channel axis (ch == 1), "
                f"got shape {x.shape}")
        return x[..., 0]
    if x.ndim == 2:
        return x
    raise ValueError(f"expected (B, dim) or (B, dim, 1), got shape {tuple(x.shape)}")


def _single_cond(x_o):
    """Reduce a single conditioning observation to a 1-D ``(dim_cond,)`` vector."""
    x_o = jnp.squeeze(jnp.asarray(x_o))
    if x_o.ndim == 0:
        x_o = x_o[None]
    if x_o.ndim != 1:
        raise ValueError(
            f"x_o must reduce to a single (dim_cond,) vector; got shape "
            f"{tuple(jnp.asarray(x_o).shape)}. sample()/log_prob() take ONE "
            f"observation at a time.")
    return x_o


class ConditionalFlowPipeline(AbstractPipeline):
    """Max-likelihood NPE pipeline wrapping a Phase-0 ``Flow``.

    Parameters
    ----------
    model : Flow
        A pre-built flow (e.g. ``make_maf(rngs, dim=dim_obs, cond_dim=dim_cond)``).
    train_dataset, val_dataset : iterable
        Yield ``(obs, cond)`` batches of shape ``(B, dim, 1)`` each.
    dim_obs, dim_cond : int
    ch_obs, ch_cond : int
        Must both be 1 (tabular SBI). Default 1.
    """

    def __init__(self, model, train_dataset, val_dataset, dim_obs, dim_cond,
                 ch_obs=1, ch_cond=1, params=None, training_config=None):
        super().__init__(
            model, train_dataset, val_dataset, dim_obs, dim_cond,
            ch_obs=ch_obs, ch_cond=ch_cond, params=params,
            training_config=training_config)
        self._standardized = False

    # --- abstract methods the flow pipeline does not use (mirror ConditionalPipeline) ---
    @classmethod
    def init_pipeline_from_config(cls, *args, **kwargs):
        raise NotImplementedError(
            "ConditionalFlowPipeline takes a pre-built Flow; build it with "
            "make_maf and pass it as model=.")

    def _make_model(self, params):
        raise NotImplementedError(
            "Pass a pre-built Flow as model=; the flow pipeline does not build "
            "models from params.")

    @classmethod
    def get_default_params(cls, *args, **kwargs):
        raise NotImplementedError(
            "ConditionalFlowPipeline takes a pre-built Flow; there are no model "
            "params to default.")

    # --- the flow IS the model: no wrapper ---
    def _wrap_model(self):
        self.model_wrapped = self.model
        self.ema_model_wrapped = self.ema_model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_flow_pipeline.py -v`
Expected: PASS — `test_squeeze_ch`, `test_single_cond`, `test_init_and_wrap`, `test_stubs_raise` all pass.

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/recipes/flow_pipeline.py tests/normalizing_flows/test_flow_pipeline.py
git commit -m "feat(nflows): ConditionalFlowPipeline skeleton + ch adapters"
```

---

### Task 3: `get_loss_fn` — max-likelihood loss with ch squeeze

**Files:**
- Modify: `src/gensbi/recipes/flow_pipeline.py` (add `get_loss_fn`)
- Test: `tests/normalizing_flows/test_flow_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/normalizing_flows/test_flow_pipeline.py`:

```python
def test_loss_fn_scalar_and_finite():
    pipe = build_pipeline()
    loss_fn = pipe.get_loss_fn()
    obs = jnp.asarray(DATA[:32, :DIM_OBS])      # (32, DIM_OBS, 1)
    cond = jnp.asarray(DATA[:32, DIM_OBS:])     # (32, DIM_COND, 1)
    loss = loss_fn(pipe.model, (obs, cond), key=jax.random.PRNGKey(0))
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_loss_fn_has_param_gradients():
    # The flow's MaskedLinear kernels are nnx.Param; grads must flow to them.
    pipe = build_pipeline()
    loss_fn = pipe.get_loss_fn()
    obs = jnp.asarray(DATA[:32, :DIM_OBS])
    cond = jnp.asarray(DATA[:32, DIM_OBS:])
    # zero_init=True zeroes the MADE OUTPUT layer, so at init the OUTPUT-layer
    # weights carry the gradient (hidden/input grads are 0 until the output
    # moves off zero). It suffices that SOME Param leaf has a non-zero gradient.
    grads = nnx.grad(loss_fn)(pipe.model, (obs, cond), jax.random.PRNGKey(0))
    leaves = jax.tree_util.tree_leaves(grads)
    assert len(leaves) > 0
    assert any(jnp.any(jnp.abs(g) > 0) for g in leaves)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_flow_pipeline.py -k loss_fn -v`
Expected: FAIL — `AttributeError`/`TypeError` (no concrete `get_loss_fn`; class is still abstract for `get_loss_fn`).

- [ ] **Step 3: Add `get_loss_fn`**

In `src/gensbi/recipes/flow_pipeline.py`, inside `ConditionalFlowPipeline`, after `_wrap_model`:

```python
    def get_loss_fn(self):
        """Return ``loss_fn(model, batch, key) -> scalar`` (key unused).

        ``batch = (obs, cond)`` each ``(B, dim, 1)``. NPE: obs=theta, cond=x.
        Loss is the mean negative log-likelihood ``-mean(log q(obs | cond))``.
        """
        def loss_fn(model, batch, key):
            obs, cond = batch
            obs = _squeeze_ch(obs)        # (B, dim_obs)
            cond = _squeeze_ch(cond)      # (B, dim_cond)
            return -jnp.mean(model.log_prob(obs, cond))

        return loss_fn
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_flow_pipeline.py -k loss_fn -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/recipes/flow_pipeline.py tests/normalizing_flows/test_flow_pipeline.py
git commit -m "feat(nflows): ConditionalFlowPipeline.get_loss_fn (max-likelihood)"
```

---

### Task 4: `fit_standardization` + `train` guard

Set the `Standardize` buffers on **both** `model` and `ema_model` (EMA is a `nnx.clone` made at construction and only averages `nnx.Param`, so its non-Param standardize buffer must be set explicitly). Add a `train()` guard that warns if standardization was never fit.

**Files:**
- Modify: `src/gensbi/recipes/flow_pipeline.py`
- Test: `tests/normalizing_flows/test_flow_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append:

```python
from gensbi.normalizing_flows.bijections.standardize import Standardize


def _get_std(flow):
    return [b for b in flow.chain.bijections if isinstance(b, Standardize)][0]


def test_fit_standardization_sets_both_models():
    pipe = build_pipeline()
    theta = DATA[:800, :DIM_OBS]                 # (800, DIM_OBS, 1)
    pipe.fit_standardization(theta)

    expected_mean = jnp.mean(theta[..., 0], axis=0)
    expected_std = jnp.std(theta[..., 0], axis=0)
    for flow in (pipe.model, pipe.ema_model):
        sb = _get_std(flow)
        assert jnp.allclose(sb.mean.value, expected_mean, atol=1e-4)
        assert jnp.allclose(sb.std.value, expected_std, atol=1e-4)
    assert pipe._standardized is True


def test_train_warns_without_standardization(tmp_path):
    pipe = build_pipeline(checkpoint_dir=str(tmp_path))
    with pytest.warns(UserWarning, match="fit_standardization"):
        pipe.train(nnx.Rngs(0), nsteps=1, save_model=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_flow_pipeline.py -k standardiz -v`
Expected: FAIL — `AttributeError: 'ConditionalFlowPipeline' object has no attribute 'fit_standardization'`.

- [ ] **Step 3: Add `fit_standardization` and override `train`**

In `ConditionalFlowPipeline`, after `get_loss_fn`:

```python
    def fit_standardization(self, obs_data):
        """Set the Standardize buffers from training-obs stats (call BEFORE train).

        ``obs_data`` is ``(N, dim_obs)`` or ``(N, dim_obs, 1)`` (the autoregressive
        target, i.e. theta for NPE). Sets the buffer on both ``model`` and
        ``ema_model`` (EMA only averages Params, so its non-Param buffer must be
        set here too).
        """
        obs = jnp.asarray(obs_data)
        if obs.ndim == 3:
            obs = _squeeze_ch(obs)
        mean = jnp.mean(obs, axis=0)
        std = jnp.std(obs, axis=0)
        std = jnp.where(std < 1e-6, 1.0, std)     # guard zero-variance dims
        self.model.set_standardization(mean, std)
        self.ema_model.set_standardization(mean, std)
        self._standardized = True

    def train(self, rngs, nsteps=None, save_model=True):
        if not self._standardized:
            warnings.warn(
                "fit_standardization() was not called before train(); the "
                "Standardize bijection stays at identity. Call "
                "pipeline.fit_standardization(theta_train) first if you want "
                "input standardization.",
                UserWarning, stacklevel=2)
        return super().train(rngs, nsteps=nsteps, save_model=save_model)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_flow_pipeline.py -k standardiz -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/recipes/flow_pipeline.py tests/normalizing_flows/test_flow_pipeline.py
git commit -m "feat(nflows): fit_standardization (both models) + train guard"
```

---

### Task 5: `get_sampler` / `sample` — single-condition sampling with ch re-expand

**Files:**
- Modify: `src/gensbi/recipes/flow_pipeline.py`
- Test: `tests/normalizing_flows/test_flow_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_sample_shape(tmp_path):
    pipe = build_pipeline(checkpoint_dir=str(tmp_path))
    pipe.fit_standardization(DATA[:800, :DIM_OBS])
    pipe.train(nnx.Rngs(0), nsteps=2, save_model=False)

    x_o = jnp.zeros((1, DIM_COND, 1))
    s = pipe.sample(jax.random.PRNGKey(1), x_o, nsamples=64, use_ema=False)
    assert s.shape == (64, DIM_OBS, 1)
    assert jnp.all(jnp.isfinite(s))

    s_ema = pipe.sample(jax.random.PRNGKey(1), x_o, nsamples=64, use_ema=True)
    assert s_ema.shape == (64, DIM_OBS, 1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_flow_pipeline.py -k test_sample_shape -v`
Expected: FAIL — class is still abstract for `get_sampler`/`sample` (`TypeError: Can't instantiate ...`) or `AttributeError`.

- [ ] **Step 3: Add `get_sampler` and `sample`**

In `ConditionalFlowPipeline`, after `train`:

```python
    def get_sampler(self, x_o, use_ema=True):
        """Return ``sampler(key, nsamples) -> (nsamples, dim_obs, 1)`` for one x_o."""
        flow = self.ema_model if use_ema else self.model
        cond = _single_cond(x_o)                  # (dim_cond,)

        def sampler(key, nsamples):
            cond_b = jnp.broadcast_to(cond, (nsamples, cond.shape[0]))
            samples = flow.sample(key, cond=cond_b)    # (nsamples, dim_obs)
            return _expand_dims(samples)               # (nsamples, dim_obs, 1)

        return sampler

    def sample(self, key, x_o, nsamples=10_000, use_ema=True):
        return self.get_sampler(x_o, use_ema=use_ema)(key, nsamples)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_flow_pipeline.py -k test_sample_shape -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/recipes/flow_pipeline.py tests/normalizing_flows/test_flow_pipeline.py
git commit -m "feat(nflows): ConditionalFlowPipeline get_sampler/sample"
```

---

### Task 6: `get_log_prob_fn` / `log_prob`

**Files:**
- Modify: `src/gensbi/recipes/flow_pipeline.py`
- Test: `tests/normalizing_flows/test_flow_pipeline.py` (append)

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_log_prob_shape(tmp_path):
    pipe = build_pipeline(checkpoint_dir=str(tmp_path))
    pipe.fit_standardization(DATA[:800, :DIM_OBS])
    pipe.train(nnx.Rngs(0), nsteps=2, save_model=False)

    x_1 = jnp.zeros((5, DIM_OBS, 1))
    x_o = jnp.zeros((1, DIM_COND, 1))
    lp = pipe.log_prob(x_1, x_o, use_ema=False)
    assert lp.shape == (5,)
    assert jnp.all(jnp.isfinite(lp))


def test_log_prob_depends_on_condition(tmp_path):
    # Test the property on a LIVE flow (zero_init=False) so cond-dependence is
    # present immediately and does not rely on training dynamics. Phase-0
    # conditioning is concat-at-rank −1, so every output dim depends on cond.
    flow = make_maf(nnx.Rngs(0), dim=DIM_OBS, cond_dim=DIM_COND, n_layers=4,
                    nn_width=32, nn_depth=2, standardize=True, zero_init=False)
    cfg = ConditionalFlowPipeline.get_default_training_config()
    cfg["checkpoint_dir"] = str(tmp_path)
    pipe = ConditionalFlowPipeline(
        flow, _make_ds(DATA[:800]), _make_ds(DATA[800:]),
        DIM_OBS, DIM_COND, ch_obs=1, ch_cond=1, training_config=cfg)

    x_1 = jnp.zeros((5, DIM_OBS, 1))
    lp_a = pipe.log_prob(x_1, jnp.zeros((1, DIM_COND, 1)), use_ema=False)
    lp_b = pipe.log_prob(x_1, jnp.ones((1, DIM_COND, 1)), use_ema=False)
    assert not jnp.allclose(lp_a, lp_b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_flow_pipeline.py -k test_log_prob -v`
Expected: FAIL — abstract `get_log_prob_fn`/`log_prob` not implemented.

- [ ] **Step 3: Add `get_log_prob_fn` and `log_prob`**

In `ConditionalFlowPipeline`, after `sample`:

```python
    def get_log_prob_fn(self, x_o, use_ema=True):
        """Return ``log_prob_fn(x_1) -> (B,)`` for one conditioning x_o."""
        flow = self.ema_model if use_ema else self.model
        cond = _single_cond(x_o)                  # (dim_cond,)

        def log_prob_fn(x_1):
            obs = _squeeze_ch(x_1)                 # (B, dim_obs)
            cond_b = jnp.broadcast_to(cond, (obs.shape[0], cond.shape[0]))
            return flow.log_prob(obs, cond_b)      # (B,)

        return log_prob_fn

    def log_prob(self, x_1, x_o, use_ema=True):
        return self.get_log_prob_fn(x_o, use_ema=use_ema)(x_1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_flow_pipeline.py -v`
Expected: PASS — the whole file is green (class is now fully concrete).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/recipes/flow_pipeline.py tests/normalizing_flows/test_flow_pipeline.py
git commit -m "feat(nflows): ConditionalFlowPipeline get_log_prob_fn/log_prob"
```

---

### Task 7: Export `ConditionalFlowPipeline`

**Files:**
- Modify: `src/gensbi/recipes/__init__.py`
- Test: `tests/normalizing_flows/test_flow_pipeline.py` (append a 1-line import test)

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_exported_from_recipes():
    from gensbi.recipes import ConditionalFlowPipeline as CFP
    assert CFP is ConditionalFlowPipeline
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_flow_pipeline.py -k exported -v`
Expected: FAIL — `ImportError: cannot import name 'ConditionalFlowPipeline' from 'gensbi.recipes'`.

- [ ] **Step 3: Add the export**

In `src/gensbi/recipes/__init__.py`, add after the `from .unconditional_pipeline import UnconditionalPipeline` line:

```python
from .flow_pipeline import ConditionalFlowPipeline
```

And add `"ConditionalFlowPipeline",` to the `__all__` list (insert after `"UnconditionalPipeline",`).

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_flow_pipeline.py -k exported -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/recipes/__init__.py tests/normalizing_flows/test_flow_pipeline.py
git commit -m "feat(nflows): export ConditionalFlowPipeline from recipes"
```

---

### Task 8 (slow / exploratory): end-to-end linear-Gaussian NPE recovery

Per the design, the end-to-end recovery test is **exploratory** — marked `slow` so the fast battery skips it. It trains a real NF-NPE on a linear-Gaussian task whose posterior is analytic, then checks the recovered posterior mean/cov at a held-out `x_o`.

**Task setup (analytic posterior).** Prior `θ ~ N(0, I_d)`. Simulator `x = G θ + ε`, `ε ~ N(0, σ² I_m)`. Given `x_o`, the Gaussian posterior is `Σ_post = (I + Gᵀ G / σ²)⁻¹`, `μ_post = Σ_post Gᵀ x_o / σ²`.

**Files:**
- Test: `tests/normalizing_flows/test_flow_pipeline_e2e.py`

- [ ] **Step 1: Write the test (it is the deliverable)**

Create `tests/normalizing_flows/test_flow_pipeline_e2e.py`:

```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import grain
import pytest

from gensbi.normalizing_flows import make_maf
from gensbi.recipes.flow_pipeline import ConditionalFlowPipeline

D = 2          # dim theta
M = 3          # dim x
SIGMA = 0.5
G = jnp.array([[1.0, 0.5], [0.0, 1.0], [0.5, -1.0]])   # (M, D)


def _simulate(key, n):
    kth, ke = jax.random.split(key)
    theta = jax.random.normal(kth, (n, D))
    x = theta @ G.T + SIGMA * jax.random.normal(ke, (n, M))
    return theta, x


def _analytic_posterior(x_o):
    prec = jnp.eye(D) + (G.T @ G) / SIGMA**2
    cov = jnp.linalg.inv(prec)
    mean = cov @ (G.T @ x_o) / SIGMA**2
    return mean, cov


def split_obs_cond(d):
    return d[:, :D], d[:, D:]


@pytest.mark.slow
def test_npe_recovers_linear_gaussian(tmp_path):
    key = jax.random.PRNGKey(0)
    theta, x = _simulate(key, 20_000)
    data = jnp.concatenate([theta[..., None], x[..., None]], axis=1)  # (N, D+M, 1)

    def make_ds(arr):
        return (grain.MapDataset.source(np.array(arr)).shuffle(0).repeat()
                .to_iter_dataset().batch(256).map(split_obs_cond))

    flow = make_maf(nnx.Rngs(0), dim=D, cond_dim=M,
                    n_layers=6, nn_width=64, nn_depth=2, standardize=True)
    cfg = ConditionalFlowPipeline.get_default_training_config()
    cfg.update(dict(nsteps=4000, val_every=200, max_lr=3e-4,
                    checkpoint_dir=str(tmp_path), early_stopping=False))
    pipe = ConditionalFlowPipeline(flow, make_ds(data[:18_000]),
                                   make_ds(data[18_000:]), D, M,
                                   ch_obs=1, ch_cond=1, training_config=cfg)
    pipe.fit_standardization(data[:18_000, :D])
    pipe.train(nnx.Rngs(0), nsteps=4000, save_model=False)

    x_o = jnp.array([1.0, -0.5, 0.3])
    mean_a, cov_a = _analytic_posterior(x_o)

    s = pipe.sample(jax.random.PRNGKey(7), x_o[None, :, None],
                    nsamples=20_000, use_ema=True)[..., 0]   # (n, D)
    mean_s = jnp.mean(s, axis=0)
    cov_s = jnp.cov(s.T)

    assert jnp.allclose(mean_s, mean_a, atol=0.1), (mean_s, mean_a)
    assert jnp.allclose(cov_s, cov_a, atol=0.1), (cov_s, cov_a)
```

- [ ] **Step 2: Run the slow test**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_flow_pipeline_e2e.py -m slow -v`
Expected: PASS. (If mean/cov are off, increase `nsteps` to 8000 or `n_layers` to 8 — this is a fit-quality knob, not a correctness bug. Do not loosen `atol` below 0.15.)

- [ ] **Step 3: Confirm the fast battery skips it**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/ -m "not slow" -q`
Expected: PASS, and the e2e test is deselected.

- [ ] **Step 4: Commit**

```bash
git add tests/normalizing_flows/test_flow_pipeline_e2e.py
git commit -m "test(nflows): slow end-to-end linear-Gaussian NPE recovery"
```

**Phase 1 milestone:** NF-NPE trains and is a drop-in for `diagnostics/` (SBC/TARP/C²ST consume `pipeline.sample(...)` arrays — assemble `(num_posterior_samples, num_obs, dim_theta)` by calling `.sample` per observation and squeezing the ch axis). Ship.

---

# PHASE 2 — NLE: `NLEPosterior` (NumPyro NUTS)

Takes an NLE-trained flow (`obs = x`, `cond = θ`), a NumPyro prior over θ, and an observation `x_o`. Builds `potential(θ) = -[log q(x_o | θ) + log p(θ)]` and hands it to NUTS — the **potential-function** route, **not** a `Distribution` wrapper (NNX modules and NumPyro transforms fight each other). `∇_θ log q` is free by autodiff. Exposes `.sample(key, x_o, n) -> (n, dim_θ, 1)` so the same diagnostics run for NLE.

---

### Task 9: `inference` package + `NLEPosterior.__init__` + `potential`

**Files:**
- Create: `src/gensbi/inference/__init__.py`
- Create: `src/gensbi/inference/nle.py`
- Test: `tests/normalizing_flows/test_nle.py`

- [ ] **Step 1: Write the failing test**

Create `tests/normalizing_flows/test_nle.py`:

```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
import pytest

from gensbi.core.prior import make_gaussian_prior
from gensbi.normalizing_flows import make_maf
from gensbi.inference import NLEPosterior


class GaussianMock:
    """log q(x | theta) = sum_i N(x_i; theta_i, 1) (batched over rows)."""
    def log_prob(self, x, cond):
        return -0.5 * jnp.sum((x - cond) ** 2, axis=-1)   # (B,)


def test_potential_value_and_grad_real_flow():
    dim = 2
    # zero_init=False so the flow actually depends on theta -> non-trivial grad.
    flow = make_maf(nnx.Rngs(0), dim=dim, cond_dim=dim,
                    n_layers=3, nn_width=16, zero_init=False)
    prior = make_gaussian_prior((dim,))
    post = NLEPosterior(flow, prior)

    U = post.potential(jnp.array([0.5, -0.5]))
    theta = jnp.array([0.1, 0.2])
    val = U(theta)
    grad = jax.grad(U)(theta)
    assert val.shape == ()
    assert jnp.isfinite(val)
    assert grad.shape == (dim,)
    assert jnp.all(jnp.isfinite(grad))


def test_potential_equals_neg_loglike_plus_logprior():
    dim = 2
    prior = make_gaussian_prior((dim,))
    post = NLEPosterior(GaussianMock(), prior)
    x_o = jnp.array([1.0, -1.0])
    theta = jnp.array([0.3, 0.4])
    expected = -(GaussianMock().log_prob(x_o[None], theta[None])[0]
                 + prior.log_prob(theta))
    assert jnp.allclose(post.potential(x_o)(theta), expected)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_nle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gensbi.inference'`.

- [ ] **Step 3: Create the package and `NLEPosterior` (init + potential)**

Create `src/gensbi/inference/__init__.py`:

```python
"""Inference wrappers built on the normalizing-flows density core."""

from gensbi.inference.nle import NLEPosterior

__all__ = ["NLEPosterior"]
```

Create `src/gensbi/inference/nle.py`:

```python
"""NLE posterior: a trained likelihood flow + a prior -> NumPyro NUTS.

The flow is NLE-trained (``obs = x``, ``cond = theta``), so ``flow.log_prob(x, theta)``
is ``log q(x | theta)``. We form the (unnormalized) posterior potential
``U(theta) = -(log q(x_o | theta) + log p(theta))`` and run NUTS on it directly
(potential-function route; NOT a numpyro Distribution wrapper). The flow's params
are frozen constants inside the potential; only ``theta`` is traced/differentiated.
"""

import jax
import jax.numpy as jnp

from gensbi.utils.math import _expand_dims


class NLEPosterior:
    """Amortized NLE posterior over a trained likelihood flow.

    Parameters
    ----------
    flow : object
        Anything exposing ``log_prob(x, cond) -> (B,)`` with x the observation
        and cond the parameter (an NLE-trained ``Flow``).
    prior : numpyro.distributions.Distribution
        Prior over theta; ``prior.log_prob(theta)`` returns a scalar and
        ``prior.sample(key, ())`` returns ``(dim_theta,)``.
    num_warmup, num_samples, num_chains : int
        NUTS defaults (overridable per ``sample`` call via ``nsamples``).
    """

    def __init__(self, flow, prior, *, num_warmup=500, num_samples=1000,
                 num_chains=1):
        self.flow = flow
        self.prior = prior
        self.num_warmup = num_warmup
        self.num_samples = num_samples
        self.num_chains = num_chains

    def potential(self, x_o):
        """Return ``U(theta) = -(log q(x_o|theta) + log p(theta))`` for one x_o."""
        x_o = jnp.atleast_1d(jnp.squeeze(jnp.asarray(x_o)))   # (dim_x,)
        flow = self.flow
        prior = self.prior

        def U(theta):
            theta = jnp.asarray(theta)
            log_like = flow.log_prob(x_o[None, :], theta[None, :])[0]
            log_prior = prior.log_prob(theta)
            return -(log_like + log_prior)

        return U
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_nle.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/inference/__init__.py src/gensbi/inference/nle.py tests/normalizing_flows/test_nle.py
git commit -m "feat(nflows): NLEPosterior potential function (inference package)"
```

---

### Task 10: `NLEPosterior.sample` — NUTS over the potential

**Files:**
- Modify: `src/gensbi/inference/nle.py`
- Test: `tests/normalizing_flows/test_nle.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/normalizing_flows/test_nle.py`:

```python
def test_sample_shape_and_prior_recovery():
    # zero_init=True (default): q(x|theta) is theta-independent (identity flow),
    # so the posterior collapses to the prior. Exercises NUTS + the real flow.
    dim = 2
    flow = make_maf(nnx.Rngs(0), dim=dim, cond_dim=dim,
                    n_layers=3, nn_width=16, zero_init=True)
    prior = make_gaussian_prior((dim,))
    post = NLEPosterior(flow, prior, num_warmup=300, num_samples=800)

    s = post.sample(jax.random.PRNGKey(0), jnp.array([1.0, -1.0]))
    assert s.shape == (800, dim, 1)
    assert jnp.all(jnp.isfinite(s))
    # posterior ~ prior N(0, I)
    assert jnp.allclose(jnp.mean(s[..., 0], axis=0), 0.0, atol=0.25)


def test_gaussian_mock_matches_analytic_posterior():
    # likelihood N(x; theta, I), prior N(0, I)  =>  posterior N(x_o/2, 0.5 I)
    dim = 2
    prior = make_gaussian_prior((dim,))
    post = NLEPosterior(GaussianMock(), prior, num_warmup=500, num_samples=3000)
    x_o = jnp.array([1.0, -1.0])

    s = post.sample(jax.random.PRNGKey(1), x_o)[..., 0]    # (3000, dim)
    assert jnp.allclose(jnp.mean(s, axis=0), x_o / 2, atol=0.1)
    assert jnp.allclose(jnp.var(s, axis=0), 0.5 * jnp.ones(dim), atol=0.15)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_nle.py -k "shape or analytic" -v`
Expected: FAIL — `AttributeError: 'NLEPosterior' object has no attribute 'sample'`.

- [ ] **Step 3: Add `sample`**

In `src/gensbi/inference/nle.py`, inside `NLEPosterior`, after `potential`:

```python
    def sample(self, key, x_o, nsamples=None):
        """Draw posterior samples via NUTS. Returns ``(n, dim_theta, 1)``.

        Uses the potential-function route: ``init_params`` is a single draw from
        the prior, so ``mcmc.get_samples()`` returns a raw ``(n, dim_theta)`` array.
        """
        from numpyro.infer import MCMC, NUTS

        n = self.num_samples if nsamples is None else nsamples
        potential = self.potential(x_o)
        kernel = NUTS(potential_fn=potential)
        mcmc = MCMC(kernel, num_warmup=self.num_warmup, num_samples=n,
                    num_chains=self.num_chains, progress_bar=False)
        key, key_init = jax.random.split(key)
        init_params = self.prior.sample(key_init, ())        # (dim_theta,)
        mcmc.run(key, init_params=init_params)
        samples = jnp.asarray(mcmc.get_samples())            # (n, dim_theta)
        return _expand_dims(samples)                         # (n, dim_theta, 1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_nle.py -v`
Expected: PASS (4 tests). NUTS on these tiny problems runs in a few seconds on CPU.

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/inference/nle.py tests/normalizing_flows/test_nle.py
git commit -m "feat(nflows): NLEPosterior.sample via NumPyro NUTS"
```

---

### Task 11 (slow / exploratory): end-to-end linear-Gaussian NLE recovery

Train a real NLE flow `q(x | θ)` on the same linear-Gaussian task, run NUTS, and compare to the analytic posterior. Marked `slow`.

**Files:**
- Test: `tests/normalizing_flows/test_nle_e2e.py`

- [ ] **Step 1: Write the test (it is the deliverable)**

Create `tests/normalizing_flows/test_nle_e2e.py`:

```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import grain
import pytest

from gensbi.core.prior import make_gaussian_prior
from gensbi.normalizing_flows import make_maf
from gensbi.recipes.flow_pipeline import ConditionalFlowPipeline
from gensbi.inference import NLEPosterior

D = 2
M = 3
SIGMA = 0.5
G = jnp.array([[1.0, 0.5], [0.0, 1.0], [0.5, -1.0]])   # (M, D)


def _simulate(key, n):
    kth, ke = jax.random.split(key)
    theta = jax.random.normal(kth, (n, D))
    x = theta @ G.T + SIGMA * jax.random.normal(ke, (n, M))
    return theta, x


def _analytic_posterior(x_o):
    prec = jnp.eye(D) + (G.T @ G) / SIGMA**2
    cov = jnp.linalg.inv(prec)
    mean = cov @ (G.T @ x_o) / SIGMA**2
    return mean, cov


@pytest.mark.slow
def test_nle_recovers_linear_gaussian(tmp_path):
    key = jax.random.PRNGKey(0)
    theta, x = _simulate(key, 20_000)
    # NLE: obs = x, cond = theta. Build the joint with x FIRST.
    data = jnp.concatenate([x[..., None], theta[..., None]], axis=1)  # (N, M+D, 1)

    def split(d):
        return d[:, :M], d[:, M:]            # (obs=x, cond=theta)

    def make_ds(arr):
        return (grain.MapDataset.source(np.array(arr)).shuffle(0).repeat()
                .to_iter_dataset().batch(256).map(split))

    # the likelihood flow: dim = M (x), cond_dim = D (theta)
    flow = make_maf(nnx.Rngs(0), dim=M, cond_dim=D,
                    n_layers=6, nn_width=64, nn_depth=2, standardize=True)
    cfg = ConditionalFlowPipeline.get_default_training_config()
    cfg.update(dict(nsteps=4000, val_every=200, max_lr=3e-4,
                    checkpoint_dir=str(tmp_path), early_stopping=False))
    pipe = ConditionalFlowPipeline(flow, make_ds(data[:18_000]),
                                   make_ds(data[18_000:]), M, D,
                                   ch_obs=1, ch_cond=1, training_config=cfg)
    pipe.fit_standardization(data[:18_000, :M])     # standardize x
    pipe.train(nnx.Rngs(0), nsteps=4000, save_model=False)

    x_o = jnp.array([1.0, -0.5, 0.3])
    mean_a, cov_a = _analytic_posterior(x_o)

    prior = make_gaussian_prior((D,))
    post = NLEPosterior(pipe.ema_model, prior, num_warmup=500, num_samples=4000)
    s = post.sample(jax.random.PRNGKey(7), x_o)[..., 0]   # (n, D)

    assert jnp.allclose(jnp.mean(s, axis=0), mean_a, atol=0.15), (jnp.mean(s, 0), mean_a)
    assert jnp.allclose(jnp.cov(s.T), cov_a, atol=0.15)
```

- [ ] **Step 2: Run the slow test**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_nle_e2e.py -m slow -v`
Expected: PASS. (Fit-quality knobs: `nsteps`, `n_layers`. Keep `atol` ≥ 0.15.)

- [ ] **Step 3: Commit**

```bash
git add tests/normalizing_flows/test_nle_e2e.py
git commit -m "test(nflows): slow end-to-end linear-Gaussian NLE recovery"
```

**Phase 2 milestone:** NF-NLE produces a posterior via NUTS that runs through the same diagnostics (assemble per-observation `.sample(...)` arrays). Ship.

---

# PHASE 3 — RQ-NSF: `RQSpline` transformer

Add a monotonic rational-quadratic spline transformer alongside `Affine`, behind the **same** `(value, params)` interface. MADE already sizes its output to `transformer.num_params`, so wider output is automatic — **no conditioner changes**. Reference: Durkan et al. 2019; mirrors `reference/flowjax/flowjax/bijections/rational_quadratic_spline.py`.

**Design notes the implementer must honor:**
- `num_params = 3K - 1` for `K` bins: `[widths(K), heights(K), inner_derivatives(K-1)]`.
- Widths/heights via `softmax` (positive, floored by `min_bin_*`, summing to the `2B` span); inner derivatives via `softplus` (+`min_derivative`); boundary derivatives fixed to 1 (linear tails on `[-B, B]`).
- **Identity warm-start**: the derivative `softplus` input is offset by `inv_softplus(1 - min_derivative)` so that *zero* params (what `zero_init=True` produces) give uniform bins (slope `s=1`) **and** unit derivatives → the spline is the exact identity. This keeps the Phase-0 `zero_init` contract (flow starts as a standard normal) for splines too.
- Direction mapping (consistent with `Affine`): `inverse` (data→noise) applies the spline map `g`, returns `logdet = +Σ log g'(x)`; `forward` (noise→data) applies `g⁻¹`, returns `logdet = -Σ log g'(x)`; `forward_dim` returns the scalar `g⁻¹(u_i)` for the sampling scan.

---

### Task 12: `RQSpline` parameterization (knots + derivatives from raw params)

**Files:**
- Modify: `src/gensbi/normalizing_flows/bijections/transformers.py` (add `_inv_softplus`, `RQSpline`, its `_knots` helper)
- Test: `tests/normalizing_flows/bijections/test_rqspline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/normalizing_flows/bijections/test_rqspline.py`:

```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest

from gensbi.normalizing_flows.bijections.transformers import RQSpline


def test_num_params():
    assert RQSpline(num_bins=8).num_params == 3 * 8 - 1
    assert RQSpline(num_bins=4).num_params == 3 * 4 - 1


def test_knots_constraints():
    K, B = 8, 5.0
    spline = RQSpline(num_bins=K, range_bound=B)
    params = jax.random.normal(jax.random.PRNGKey(0), (spline.num_params,))
    x_knots, y_knots, d = spline._knots(params)

    assert x_knots.shape == (K + 1,)
    assert y_knots.shape == (K + 1,)
    assert d.shape == (K + 1,)
    # span exactly [-B, B], strictly increasing, positive derivatives
    assert jnp.allclose(x_knots[0], -B) and jnp.allclose(x_knots[-1], B)
    assert jnp.allclose(y_knots[0], -B) and jnp.allclose(y_knots[-1], B)
    assert jnp.all(jnp.diff(x_knots) > 0)
    assert jnp.all(jnp.diff(y_knots) > 0)
    assert jnp.all(d > 0)
    # linear tails: boundary derivatives are 1
    assert jnp.allclose(d[0], 1.0) and jnp.allclose(d[-1], 1.0)


def test_zero_params_give_identity_knots():
    K, B = 8, 5.0
    spline = RQSpline(num_bins=K, range_bound=B)
    x_knots, y_knots, d = spline._knots(jnp.zeros(spline.num_params))
    # uniform bins => x_knots == y_knots, and all derivatives == 1
    assert jnp.allclose(x_knots, y_knots, atol=1e-5)
    assert jnp.allclose(d, jnp.ones(K + 1), atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/bijections/test_rqspline.py -v`
Expected: FAIL — `ImportError: cannot import name 'RQSpline'`.

- [ ] **Step 3: Add `_inv_softplus`, `RQSpline.__init__`, `RQSpline._knots`**

In `src/gensbi/normalizing_flows/bijections/transformers.py`, append (the existing `jax`, `jnp`, `Array` imports are already present at the top):

```python
def _inv_softplus(y: Array) -> Array:
    """Inverse of softplus: ``x`` such that ``softplus(x) == y`` (y > 0)."""
    return jnp.log(jnp.expm1(y))


class RQSpline:
    """Elementwise monotonic rational-quadratic spline on ``[-B, B]``.

    Linear tails outside the interval. Same ``(value, params)`` interface as
    :class:`Affine`. params layout per dim (length ``3K-1``):
    ``[widths(K), heights(K), inner_derivatives(K-1)]``.

    With zero params (the ``zero_init`` MADE output) the spline is the identity,
    so the flow warm-starts as a standard normal (same contract as Affine).
    Reference: Durkan et al. 2019 (https://arxiv.org/abs/1906.04032).
    """

    def __init__(self, num_bins: int = 8, range_bound: float = 5.0,
                 min_bin_width: float = 1e-3, min_bin_height: float = 1e-3,
                 min_derivative: float = 1e-3):
        self.num_bins = num_bins
        self.B = range_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.num_params = 3 * num_bins - 1

    def _knots(self, params: Array):
        """Raw params -> (x_knots, y_knots, derivatives), each over K+1 knots."""
        K, B = self.num_bins, self.B
        raw_w = params[:K]
        raw_h = params[K:2 * K]
        raw_d = params[2 * K:3 * K - 1]                  # (K-1,)

        w = jax.nn.softmax(raw_w)
        w = self.min_bin_width + (1.0 - self.min_bin_width * K) * w
        h = jax.nn.softmax(raw_h)
        h = self.min_bin_height + (1.0 - self.min_bin_height * K) * h

        x_knots = -B + 2.0 * B * jnp.concatenate([jnp.zeros(1), jnp.cumsum(w)])
        y_knots = -B + 2.0 * B * jnp.concatenate([jnp.zeros(1), jnp.cumsum(h)])

        # offset so raw_d == 0 -> derivative == 1 (identity warm-start)
        d_inner = self.min_derivative + jax.nn.softplus(
            raw_d + _inv_softplus(1.0 - self.min_derivative))
        derivatives = jnp.concatenate([jnp.ones(1), d_inner, jnp.ones(1)])
        return x_knots, y_knots, derivatives
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/bijections/test_rqspline.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/bijections/transformers.py tests/normalizing_flows/bijections/test_rqspline.py
git commit -m "feat(nflows): RQSpline parameterization (knots from MADE params)"
```

---

### Task 13: `RQSpline` spline math — `inverse` / `forward` / `forward_dim`

**Files:**
- Modify: `src/gensbi/normalizing_flows/bijections/transformers.py` (add `_rqs_apply`, the three methods + `_fwd_scalar`/`_inv_scalar`)
- Test: `tests/normalizing_flows/bijections/test_rqspline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/normalizing_flows/bijections/test_rqspline.py`:

```python
def _rand_params(key, spline):
    return jax.random.normal(key, (spline.num_params,))


def test_scalar_roundtrip_inside_interval():
    spline = RQSpline(num_bins=8, range_bound=5.0)
    params = _rand_params(jax.random.PRNGKey(1), spline)
    xs = jnp.linspace(-4.5, 4.5, 50)
    for x in xs:
        u, _ = spline._fwd_scalar(x, params)
        x_rec, _ = spline._inv_scalar(u, params)
        assert jnp.allclose(x_rec, x, atol=1e-4), (x, x_rec)


def test_scalar_logdet_matches_autodiff():
    spline = RQSpline(num_bins=8, range_bound=5.0)
    params = _rand_params(jax.random.PRNGKey(2), spline)
    for x in jnp.linspace(-4.0, 4.0, 25):
        _, logderiv = spline._fwd_scalar(x, params)
        g = jax.grad(lambda z: spline._fwd_scalar(z, params)[0])(x)
        assert jnp.allclose(logderiv, jnp.log(jnp.abs(g)), atol=1e-4), (x, logderiv, g)


def test_tails_are_identity():
    spline = RQSpline(num_bins=8, range_bound=5.0)
    params = _rand_params(jax.random.PRNGKey(3), spline)
    for x in [-8.0, 7.5]:
        u, logderiv = spline._fwd_scalar(jnp.array(x), params)
        assert jnp.allclose(u, x)              # identity outside [-B, B]
        assert jnp.allclose(logderiv, 0.0)


def test_vector_inverse_forward_roundtrip():
    spline = RQSpline(num_bins=6, range_bound=4.0)
    dim = 4
    key = jax.random.PRNGKey(4)
    kp, kx = jax.random.split(key)
    params = jax.random.normal(kp, (dim, spline.num_params))
    x = jax.random.uniform(kx, (dim,), minval=-3.5, maxval=3.5)
    u, ld_inv = spline.inverse(x, params)
    x_rec, ld_fwd = spline.forward(u, params)
    assert jnp.allclose(x_rec, x, atol=1e-4)
    assert jnp.allclose(ld_inv + ld_fwd, 0.0, atol=1e-4)   # logdets cancel
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/bijections/test_rqspline.py -k "roundtrip or logdet or tails" -v`
Expected: FAIL — `AttributeError: 'RQSpline' object has no attribute '_fwd_scalar'`.

- [ ] **Step 3: Add the spline math**

In `src/gensbi/normalizing_flows/bijections/transformers.py`, add the module-level helper and the methods on `RQSpline`:

```python
def _rqs_apply(z: Array, x_knots: Array, y_knots: Array, derivatives: Array,
               inverse: bool):
    """Apply the RQ spline (or its inverse) to a scalar ``z``.

    Returns ``(out, logderiv)`` where ``logderiv = log(dy/dx)`` evaluated at the
    relevant x (the same forward derivative is used for both directions; the
    caller flips its sign for ``forward``). Outside ``[-B, B]`` the map is the
    identity (logderiv 0).
    """
    lo, hi = x_knots[0], x_knots[-1]
    in_bounds = (z >= lo) & (z <= hi)
    n_bins = x_knots.shape[0] - 1

    if not inverse:                                  # z is x; bins on x_knots
        k = jnp.clip(jnp.searchsorted(x_knots, z) - 1, 0, n_bins - 1)
    else:                                            # z is y; bins on y_knots
        k = jnp.clip(jnp.searchsorted(y_knots, z) - 1, 0, n_bins - 1)

    xk, xk1 = x_knots[k], x_knots[k + 1]
    yk, yk1 = y_knots[k], y_knots[k + 1]
    dk, dk1 = derivatives[k], derivatives[k + 1]
    w = xk1 - xk
    s = (yk1 - yk) / w                               # bin slope

    if not inverse:
        xi = jnp.clip((z - xk) / w, 0.0, 1.0)
        num = (yk1 - yk) * (s * xi ** 2 + dk * xi * (1 - xi))
        den = s + (dk1 + dk - 2 * s) * xi * (1 - xi)
        out_in = yk + num / den
    else:
        dy = z - yk
        c2 = dk1 + dk - 2 * s
        a = (yk1 - yk) * (s - dk) + dy * c2
        b = (yk1 - yk) * dk - dy * c2
        c = -s * dy
        disc = jnp.clip(b ** 2 - 4 * a * c, a_min=0.0)
        xi = jnp.clip((2 * c) / (-b - jnp.sqrt(disc)), 0.0, 1.0)
        out_in = xk + xi * w

    out = jnp.where(in_bounds, out_in, z)
    num_d = s ** 2 * (dk1 * xi ** 2 + 2 * s * xi * (1 - xi) + dk * (1 - xi) ** 2)
    den_d = (s + (dk1 + dk - 2 * s) * xi * (1 - xi)) ** 2
    deriv = jnp.where(in_bounds, num_d / den_d, 1.0)
    return out, jnp.log(deriv)
```

Then add these methods inside `class RQSpline` (after `_knots`):

```python
    def _fwd_scalar(self, x: Array, params: Array):
        x_knots, y_knots, d = self._knots(params)
        return _rqs_apply(x, x_knots, y_knots, d, inverse=False)

    def _inv_scalar(self, u: Array, params: Array):
        x_knots, y_knots, d = self._knots(params)
        return _rqs_apply(u, x_knots, y_knots, d, inverse=True)

    def inverse(self, x: Array, params: Array):
        """data -> noise (fast). logdet = +sum log g'(x)."""
        u, logderiv = jax.vmap(self._fwd_scalar)(x, params)
        return u, jnp.sum(logderiv)

    def forward(self, u: Array, params: Array):
        """noise -> data. logdet = -sum log g'(x)."""
        x, logderiv = jax.vmap(self._inv_scalar)(u, params)
        return x, -jnp.sum(logderiv)

    def forward_dim(self, u_i: Array, params_i: Array) -> Array:
        """Scalar noise->data for one dim (used by the sequential sampling scan)."""
        x_i, _ = self._inv_scalar(u_i, params_i)
        return x_i
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/bijections/test_rqspline.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/bijections/transformers.py tests/normalizing_flows/bijections/test_rqspline.py
git commit -m "feat(nflows): RQSpline inverse/forward/forward_dim (RQ-NSF math)"
```

---

### Task 14: Identity warm-start at the Flow level (zero-init spline ≈ standard normal)

**Files:**
- Test: `tests/normalizing_flows/test_flow.py` (append)

- [ ] **Step 1: Write the failing test**

> This test should pass immediately once Tasks 12–13 land (the spline already supports `make_maf` via `transformer=`). Write it now to lock the contract; if it fails, the bug is in Task 12/13, not here.

Append to `tests/normalizing_flows/test_flow.py`:

```python
def test_zero_init_spline_flow_is_standard_normal():
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from gensbi.normalizing_flows import make_maf
    from gensbi.normalizing_flows.bijections.transformers import RQSpline
    from gensbi.core.prior import make_gaussian_prior

    dim, cond_dim = 3, 2
    flow = make_maf(nnx.Rngs(0), dim=dim, cond_dim=cond_dim, n_layers=4,
                    transformer=RQSpline(num_bins=8, range_bound=5.0),
                    standardize=True, zero_init=True)
    base = make_gaussian_prior((dim,))

    x = jax.random.normal(jax.random.PRNGKey(1), (16, dim))
    cond = jax.random.normal(jax.random.PRNGKey(2), (16, cond_dim))
    lp = flow.log_prob(x, cond)
    lp_base = jax.vmap(base.log_prob)(x)
    # zero-init spline is the identity => flow density == base density
    assert jnp.allclose(lp, lp_base, atol=1e-4)
```

- [ ] **Step 2: Run test to verify behavior**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_flow.py::test_zero_init_spline_flow_is_standard_normal -v`
Expected: PASS (confirms the identity warm-start). If it FAILS, fix the derivative-offset / softmax-floor math in Task 12 (`_knots`) — do not weaken this test.

- [ ] **Step 3: Commit**

```bash
git add tests/normalizing_flows/test_flow.py
git commit -m "test(nflows): zero-init RQSpline flow is the standard normal"
```

---

### Task 15: Full official battery (§11) with the spline transformer

Re-run the Phase-0 battery — invertibility, log-det vs autodiff, 1D-integrates-to-1, MADE autoregression — but with `transformer=RQSpline(...)` to confirm RQ-NSF is a correct density model.

**Files:**
- Test: `tests/normalizing_flows/test_flow_spline_battery.py`

- [ ] **Step 1: Write the test**

Create `tests/normalizing_flows/test_flow_spline_battery.py`:

```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
from scipy.integrate import trapezoid

from gensbi.normalizing_flows import make_maf
from gensbi.normalizing_flows.bijections.transformers import RQSpline


def _spline_flow(dim, cond_dim, **kw):
    return make_maf(nnx.Rngs(0), dim=dim, cond_dim=cond_dim, n_layers=4,
                    nn_width=32, nn_depth=2,
                    transformer=RQSpline(num_bins=8, range_bound=6.0),
                    standardize=True, zero_init=False, **kw)


def test_spline_flow_invertibility():
    dim, cond_dim = 3, 2
    flow = _spline_flow(dim, cond_dim)
    cond = jax.random.normal(jax.random.PRNGKey(1), (cond_dim,))
    u = jax.random.normal(jax.random.PRNGKey(2), (dim,))
    x, _ = flow.chain.forward(u, cond)
    u_rec, _ = flow.chain.inverse(x, cond)
    assert jnp.allclose(u_rec, u, atol=1e-4)


def test_spline_flow_logdet_matches_autodiff():
    dim, cond_dim = 4, 2
    flow = _spline_flow(dim, cond_dim)
    cond = jax.random.normal(jax.random.PRNGKey(1), (cond_dim,))
    x = jax.random.normal(jax.random.PRNGKey(3), (dim,)) * 0.5

    _, logdet = flow.chain.inverse(x, cond)
    jac = jax.jacobian(lambda z: flow.chain.inverse(z, cond)[0])(x)
    sign, logabsdet = jnp.linalg.slogdet(jac)
    assert jnp.allclose(logdet, logabsdet, atol=1e-4), (logdet, logabsdet)


def test_spline_flow_1d_density_integrates_to_one():
    flow = make_maf(nnx.Rngs(0), dim=1, cond_dim=0, n_layers=4, nn_width=32,
                    transformer=RQSpline(num_bins=8, range_bound=6.0),
                    standardize=True, zero_init=False)
    grid = jnp.linspace(-8.0, 8.0, 4001)[:, None]       # (G, 1)
    dens = jnp.exp(flow.log_prob(grid))                 # (G,)
    integral = trapezoid(dens, grid[:, 0])
    assert jnp.allclose(integral, 1.0, atol=1e-2), integral


def test_spline_made_autoregression_preserved():
    # output dim d depends on x_d (through the transformer) and x_{<d} (through
    # the MADE params), but MUST have zero Jacobian w.r.t. x_{>d}. The strict
    # MADE mask is unchanged by the wider spline output, so this must still hold.
    dim, cond_dim = 4, 2
    flow = _spline_flow(dim, cond_dim)
    cond = jax.random.normal(jax.random.PRNGKey(1), (cond_dim,))
    x = jax.random.normal(jax.random.PRNGKey(5), (dim,)) * 0.5
    jac = jax.jacobian(lambda z: flow.chain.bijections[0].inverse(z, cond)[0])(x)
    # bijections[0] is the first MaskedAutoregressive layer (no permutation yet)
    for d in range(dim):
        for j in range(d + 1, dim):                     # strictly greater than d
            assert jnp.allclose(jac[d, j], 0.0, atol=1e-6), (d, j, jac[d, j])
```

The Phase-0 1D test uses `from scipy.integrate import trapezoid` (matched above), not `jnp.trapz`.

- [ ] **Step 2: Run the tests**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/test_flow_spline_battery.py -v`
Expected: PASS (4 tests). The autoregression test reads `flow.chain.bijections[0]`, which is the first `MaskedAutoregressive` layer (the permutation only appears between layers).

- [ ] **Step 3: Commit**

```bash
git add tests/normalizing_flows/test_flow_spline_battery.py
git commit -m "test(nflows): official battery with RQSpline (RQ-NSF)"
```

---

### Task 16: Export `RQSpline`

**Files:**
- Modify: `src/gensbi/normalizing_flows/bijections/__init__.py`
- Modify: `src/gensbi/normalizing_flows/__init__.py`
- Test: `tests/normalizing_flows/bijections/test_rqspline.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/normalizing_flows/bijections/test_rqspline.py`:

```python
def test_rqspline_exported():
    from gensbi.normalizing_flows.bijections import RQSpline as A
    from gensbi.normalizing_flows import RQSpline as B
    assert A is RQSpline and B is RQSpline
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/bijections/test_rqspline.py -k exported -v`
Expected: FAIL — `ImportError: cannot import name 'RQSpline' from 'gensbi.normalizing_flows.bijections'`.

- [ ] **Step 3: Add the exports**

In `src/gensbi/normalizing_flows/bijections/__init__.py`, change the transformers import and `__all__`:

```python
from gensbi.normalizing_flows.bijections.transformers import Affine, RQSpline
```

and add `"RQSpline",` to `__all__` (after `"Affine",`).

In `src/gensbi/normalizing_flows/__init__.py`, add the re-export:

```python
from gensbi.normalizing_flows.bijections.transformers import Affine, RQSpline

__all__ = ["Flow", "make_maf", "Affine", "RQSpline"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/bijections/test_rqspline.py -k exported -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/gensbi/normalizing_flows/__init__.py src/gensbi/normalizing_flows/bijections/__init__.py tests/normalizing_flows/bijections/test_rqspline.py
git commit -m "feat(nflows): export RQSpline transformer"
```

**Phase 3 milestone:** RQ-NSF available via `make_maf(..., transformer=RQSpline(num_bins=K, range_bound=B))` for both NPE and NLE (drop-in for `Affine`). Ship.

---

## Final verification (run before declaring done)

- [ ] Fast battery green: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/ -m "not slow" -v`
- [ ] Slow/exploratory recovery tests green: `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/normalizing_flows/ -m slow -v`
- [ ] No regression in the Phase-0 battery (same command above covers it).
- [ ] Optional smoke that diagnostics consume the NPE pipeline: build `(num_posterior_samples, num_obs, dim_theta)` from per-observation `pipeline.sample(...)[..., 0]` and run `run_sbc` / `run_tarp` from `gensbi.diagnostics` (no code change required — they take arrays, not the pipeline object).

---

## Known watch-items carried from Phase 0 (do not regress)

- **Standardization ordering:** `fit_standardization` mutates non-Param buffers in place; it MUST be called *before* `train()` so the buffers are correct when `nnx.jit` first traces `train_step`. The `train()` guard warns if skipped. Buffers are saved/restored by checkpointing (full `nnx.split` state) and are untouched by the optimizer/EMA (non-Param). (Handout §5.)
- **Affine log-scale clamp** uses a straight-through gradient; raw pre-clamp `a` can drift while `exp(a)` is pinned. If Phase-1 training is unstable, add a diagnostic histogram of `params[..., 1]` (not a code change).
- **`.value` accessor deprecation:** the subpackage uses `var.value` (flax `DeprecationWarning`). New code here follows the same convention for consistency; a repo-wide sweep to `.get_value()`/`var[...]` is a separate cleanup (handout §3d).
- **`reference/flowjax/`** is an algorithm reference clone; it should be `.gitignore`d, not committed.
