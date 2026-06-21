# Logit-normal training-time `t` sampling + fixed-θ validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the flow-matching training-time `t` distribution a configurable knob (default uniform, opt-in logit-normal), expose it in YAML, and add a fixed-θ 100k diagnostic run for PixelDiT on GRF-32 plus its Condor job.

**Architecture:** A pure `sample_time` helper feeds `FlowMatchingMethod.prepare_batch` (the single training-time `t` source for every FM recipe); default `"uniform"` is bit-identical to today. The example scripts read a `time_sampling:` YAML block. A new self-contained `train-grf-fixedtheta.py` holds θ constant by monkeypatching the task prior and reuses the full online-simulation pipeline.

**Tech Stack:** JAX, Flax NNX, optax, grain (via `sbibm_jax`), pytest, HTCondor.

**Spec:** `docs/superpowers/specs/2026-06-13-logitnormal-time-sampling-design.md`

---

## Repos, branches, environments

- **GenSBI (library)** — `/lhome/ific/a/aamerio/data/github/GenSBI`, branch `FieldDiT`. Tasks 1–2.
- **GenSBI-examples** — `/lhome/ific/a/aamerio/data/github/GenSBI-examples`, branch `gaussian_random_field`. Tasks 3–7.
- **Tests / smoke:** run with `JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/data/github/GenSBI/.venv/bin/python -m pytest ...` (pytest 9.0.3, `gensbi` importable). The cluster job (Task 7) uses the `gensbi` conda env via `train_model.sh`.

## Implementer model selection

SDD dispatches the default `superpowers-sdd-implementer` (Sonnet). For tasks flagged **Implementer model: Opus 4.8**, the orchestrator MUST dispatch that task's subagent with `model: opus` (the work is subtle: a bit-identical-default invariant, and novel glue against the simulator API). Unflagged tasks use the Sonnet default.

## File structure

GenSBI (library):
- `src/gensbi/core/time_sampling.py` (new) — pure `sample_time(key, n, *, dist, logitnorm_mean, logitnorm_std)`.
- `src/gensbi/core/flow_matching.py` (modify) — ctor params + `prepare_batch` call the helper.
- `tests/core/test_time_sampling.py` (new) — helper + method-level tests.

GenSBI-examples:
- `examples/sbi-benchmarks/gaussian_random_field/train-grf.py` (modify) — read `time_sampling`.
- `examples/sbi-benchmarks/gaussian_random_field_256/train-grf.py` (modify) — same.
- `examples/sbi-benchmarks/gaussian_random_field/train-grf-fixedtheta.py` (new) — fixed-θ diagnostic.
- `examples/sbi-benchmarks/gaussian_random_field/config/config_4f.yaml` (new).
- `sub/train_model_grf_PixelDiT_fixedtheta.sub` (new).

---

## Task 1: `sample_time` pure helper (library)

**Implementer model:** Sonnet (default)

**Files:**
- Create: `src/gensbi/core/time_sampling.py`
- Test: `tests/core/test_time_sampling.py`

- [ ] **Step 1: Write the failing test**

Create `tests/core/test_time_sampling.py`:

```python
import jax
import jax.numpy as jnp
import pytest

from gensbi.core.time_sampling import sample_time


def test_uniform_is_bit_identical_to_jax_uniform():
    key = jax.random.PRNGKey(0)
    n = 257
    got = sample_time(key, n, dist="uniform")
    expected = jax.random.uniform(key, (n,))
    assert got.shape == (n,)
    assert jnp.array_equal(got, expected)


def test_logitnormal_in_unit_interval_and_deterministic():
    key = jax.random.PRNGKey(1)
    n = 100_000
    t = sample_time(key, n, dist="logitnormal", logitnorm_mean=0.0, logitnorm_std=1.0)
    assert t.shape == (n,)
    assert bool(jnp.all(t > 0.0)) and bool(jnp.all(t < 1.0))
    t2 = sample_time(key, n, dist="logitnormal", logitnorm_mean=0.0, logitnorm_std=1.0)
    assert jnp.array_equal(t, t2)  # deterministic for a fixed key


def test_logitnormal_logit_is_normal_m_s():
    # logit(t) = log(t) - log(1-t) must be Normal(mean, std) by construction.
    key = jax.random.PRNGKey(2)
    n = 200_000
    m, s = 0.5, 1.3
    t = sample_time(key, n, dist="logitnormal", logitnorm_mean=m, logitnorm_std=s)
    z = jnp.log(t) - jnp.log1p(-t)
    assert abs(float(jnp.mean(z)) - m) < 0.02
    assert abs(float(jnp.std(z)) - s) < 0.02


def test_unknown_dist_raises():
    with pytest.raises(ValueError):
        sample_time(jax.random.PRNGKey(0), 8, dist="cosmap")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/data/github/GenSBI/.venv/bin/python -m pytest tests/core/test_time_sampling.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gensbi.core.time_sampling'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/gensbi/core/time_sampling.py`:

```python
"""Training-time timestep samplers for flow matching.

A small, pure helper so the timestep distribution is a configurable knob on
:class:`~gensbi.core.flow_matching.FlowMatchingMethod` without touching the
loss, path, or models.
"""

import jax
import jax.numpy as jnp


def sample_time(key, n, *, dist="uniform", logitnorm_mean=0.0, logitnorm_std=1.0):
    """Sample ``n`` flow-matching timesteps in ``(0, 1)``.

    Parameters
    ----------
    key : jax.random.PRNGKey
    n : int
        Number of timesteps (batch size).
    dist : str
        ``"uniform"`` (default) -> ``jax.random.uniform(key, (n,))``, bit-identical
        to the previous inline sampling so existing runs are unchanged.
        ``"logitnormal"`` -> ``sigmoid(logitnorm_mean + logitnorm_std * N(0, 1))``
        (SD3 / Esser et al.); concentrates mass near ``sigmoid(logitnorm_mean)``.
        The reference's ``lognorm_t`` flag is a misnomer for this logit-normal sampler.
    logitnorm_mean, logitnorm_std : float
        Mean/std of the underlying normal (used only for ``"logitnormal"``).

    Returns
    -------
    jax.Array
        Shape ``(n,)`` timesteps.
    """
    if dist == "uniform":
        return jax.random.uniform(key, (n,))
    if dist == "logitnormal":
        eps = jax.random.normal(key, (n,))
        return jax.nn.sigmoid(logitnorm_mean + logitnorm_std * eps)
    raise ValueError(
        f"unknown time dist {dist!r}; expected 'uniform' or 'logitnormal'"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/data/github/GenSBI/.venv/bin/python -m pytest tests/core/test_time_sampling.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI
git add src/gensbi/core/time_sampling.py tests/core/test_time_sampling.py
git commit -m "feat(core): add configurable sample_time helper (uniform | logitnormal)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Wire `sample_time` into `FlowMatchingMethod` (library)

**Implementer model: Opus 4.8** — must preserve the bit-identical uniform default (same split order) and the t=1→data convention.

**Files:**
- Modify: `src/gensbi/core/flow_matching.py` (`__init__` lines 54-56; `prepare_batch` lines 127-130)
- Test: `tests/core/test_time_sampling.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/core/test_time_sampling.py`:

```python
def _build_method(**kw):
    from gensbi.core import FlowMatchingMethod
    method = FlowMatchingMethod(**kw)
    method.build_path(config={}, event_shape=(4, 1))  # sets method.prior
    return method


def test_method_default_prepare_batch_bit_identical():
    method = _build_method()
    key = jax.random.PRNGKey(7)
    x1 = jnp.zeros((16, 4, 1))
    _, _, t = method.prepare_batch(key, x1, path=None)
    # reproduce the exact historical computation (split, then uniform on 2nd sub-key)
    _, rng_t = jax.random.split(key)
    expected_t = jax.random.uniform(rng_t, (16,))
    assert jnp.array_equal(t, expected_t)


def test_method_logitnormal_prepare_batch():
    method = _build_method(time_dist="logitnormal", logitnorm_mean=0.0, logitnorm_std=1.0)
    key = jax.random.PRNGKey(7)
    x1 = jnp.zeros((4096, 4, 1))
    _, _, t = method.prepare_batch(key, x1, path=None)
    assert bool(jnp.all(t > 0)) and bool(jnp.all(t < 1))
    _, rng_t = jax.random.split(key)
    assert jnp.array_equal(t, sample_time(rng_t, 4096, dist="logitnormal"))


def test_method_invalid_time_dist_raises():
    from gensbi.core import FlowMatchingMethod
    with pytest.raises(ValueError):
        FlowMatchingMethod(time_dist="bogus")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/data/github/GenSBI/.venv/bin/python -m pytest tests/core/test_time_sampling.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'time_dist'`.

- [ ] **Step 3: Modify `FlowMatchingMethod`**

In `src/gensbi/core/flow_matching.py`, add the import near the other core imports (after line 20, `from gensbi.flow_matching.loss import FMLoss`):

```python
from gensbi.core.time_sampling import sample_time
```

Replace the constructor (currently lines 54-56):

```python
    def __init__(self, prior=None):
        self._user_prior = prior
        self.prior = None
```

with:

```python
    def __init__(self, prior=None, *, time_dist="uniform",
                 logitnorm_mean=0.0, logitnorm_std=1.0):
        if time_dist not in ("uniform", "logitnormal"):
            raise ValueError(
                f"time_dist must be 'uniform' or 'logitnormal', got {time_dist!r}"
            )
        self._user_prior = prior
        self.prior = None
        self.time_dist = time_dist
        self.logitnorm_mean = float(logitnorm_mean)
        self.logitnorm_std = float(logitnorm_std)
```

Replace the body of `prepare_batch` (currently lines 127-130):

```python
        rng_x0, rng_t = jax.random.split(key)
        x_0 = self.prior.sample(rng_x0, (x_1.shape[0],))
        t = jax.random.uniform(rng_t, (x_1.shape[0],))
        return (x_0, x_1, t)
```

with (keep the split order so the uniform default stays bit-identical):

```python
        rng_x0, rng_t = jax.random.split(key)
        x_0 = self.prior.sample(rng_x0, (x_1.shape[0],))
        t = sample_time(
            rng_t, x_1.shape[0],
            dist=self.time_dist,
            logitnorm_mean=self.logitnorm_mean,
            logitnorm_std=self.logitnorm_std,
        )
        return (x_0, x_1, t)
```

Also update the `prepare_batch` docstring line that says ``t`` is uniform in ``[0, 1)`` to note it follows `time_dist` (uniform default).

- [ ] **Step 4: Run tests to verify they pass**

Run: `JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/data/github/GenSBI/.venv/bin/python -m pytest tests/core/test_time_sampling.py tests/core/test_generative_method.py -q`
Expected: PASS (7 in test_time_sampling + existing generative-method tests green).

- [ ] **Step 5: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI
git add src/gensbi/core/flow_matching.py tests/core/test_time_sampling.py
git commit -m "feat(core): configurable t distribution on FlowMatchingMethod (default uniform unchanged)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Read `time_sampling` in the existing `train-grf.py` scripts (examples)

**Implementer model:** Sonnet (default)

**Files:**
- Modify: `examples/sbi-benchmarks/gaussian_random_field/train-grf.py`
- Modify: `examples/sbi-benchmarks/gaussian_random_field_256/train-grf.py`

Both files are identical in this region (the pipeline construction block).

- [ ] **Step 1: Edit both scripts**

In each file, replace this block (it currently constructs the pipeline with an inline method):

```python
    pipeline = FieldConditionalPipeline(
        model,
        train_loader,
        val_loader,
        field_shape=tuple(model_cfg["field_shape"]),
        dim_cond=model_cfg["cond_dim"],
        method=FlowMatchingMethod(),
        ch_obs=1,
        ch_cond=1,
        training_config=training_config,
    )
```

with:

```python
    ts_cfg = cfg.get("time_sampling", {})
    method = FlowMatchingMethod(
        time_dist=ts_cfg.get("dist", "uniform"),
        logitnorm_mean=ts_cfg.get("logitnorm_mean", 0.0),
        logitnorm_std=ts_cfg.get("logitnorm_std", 1.0),
    )

    pipeline = FieldConditionalPipeline(
        model,
        train_loader,
        val_loader,
        field_shape=tuple(model_cfg["field_shape"]),
        dim_cond=model_cfg["cond_dim"],
        method=method,
        ch_obs=1,
        ch_cond=1,
        training_config=training_config,
    )
```

- [ ] **Step 2: Verify both scripts still compile**

Run:
```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
python -m py_compile \
  examples/sbi-benchmarks/gaussian_random_field/train-grf.py \
  examples/sbi-benchmarks/gaussian_random_field_256/train-grf.py && echo "compile ok"
```
Expected: `compile ok` (no output from py_compile, then the echo).

- [ ] **Step 3: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
git add examples/sbi-benchmarks/gaussian_random_field/train-grf.py \
        examples/sbi-benchmarks/gaussian_random_field_256/train-grf.py
git commit -m "feat(grf): read optional time_sampling block -> FlowMatchingMethod

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Fixed-θ diagnostic training script (examples)

**Implementer model: Opus 4.8** — novel glue against the `sbibm_jax` simulator/normalization API (monkeypatched prior, simulate-truth, normalize/unnormalize). Verify shapes carefully.

**Files:**
- Create: `examples/sbi-benchmarks/gaussian_random_field/train-grf-fixedtheta.py`

Mechanism: construct `OnlineTaskDataset` normally, then replace `online_task.task.get_prior` with a function returning the constant θ. The simulator, reshape, normalization, and tokenization run unchanged, so batches are format-identical to the real loader (`num_workers=0` default → no pickling). Truth fields for the P(k) overlay are simulated directly via `task.simulator(key, theta_batch)`. Helpers are duplicated (not factored) to avoid touching the working conditional script.

- [ ] **Step 1: Create the script**

Create `examples/sbi-benchmarks/gaussian_random_field/train-grf-fixedtheta.py`:

```python
"""Fixed-theta diagnostic: can PixelDiT learn ONE GRF's 2-point structure?

Trains PixelDiT on a single held-constant theta (no conditioning generalization)
with logit-normal t-sampling, then overlays the radial power spectrum of samples
vs. simulated truth at that theta. Decisive plot: the P(k) overlay — success =
samples develop low-k power matching truth; failure = flat P(k) (white noise).

NOTE: with theta fixed the cond input is constant, so this does NOT test
conditioning — it isolates generative/2-point capacity. Intentional.

Usage (conda env `gensbi`):
    python train-grf-fixedtheta.py --config config/config_4f.yaml
"""

import os

if __name__ != "__main__":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
    os.environ.setdefault("JAX_PLATFORMS", "cuda")

import argparse

import jax
from jax import numpy as jnp
import numpy as np
from flax import nnx
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gensbi.core import FlowMatchingMethod
from gensbi.experimental.recipes import FieldConditionalPipeline

from sbibm_jax.data import OnlineTaskDataset

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
TASK_NAME = "gaussian_random_field"

_PIPELINE_KEYS = (
    "nsteps", "max_lr", "min_lr", "warmup_steps", "ema_decay",
    "decay_transition", "val_every", "early_stopping", "multistep",
    "experiment_id", "val_error_ratio",
)


def swap_obs_cond(batch):
    theta, x = batch
    return x, theta


def build_pixeldit(model_cfg, seed):
    from gensbi.experimental.models import PixelDiT, PixelDiTParams

    kw = dict(model_cfg)
    kw["field_shape"] = tuple(kw["field_shape"])
    kw["param_dtype"] = getattr(jnp, kw.pop("param_dtype", "bfloat16"))
    return PixelDiT(PixelDiTParams(rngs=nnx.Rngs(seed), **kw))


def radial_power_spectrum(field, nbins=40):
    field = np.asarray(field, dtype=np.float64)
    H, W = field.shape
    pk2d = np.abs(np.fft.fft2(field)) ** 2 / (H * W)
    kx, ky = np.meshgrid(np.fft.fftfreq(H), np.fft.fftfreq(W), indexing="ij")
    knorm = np.sqrt(kx**2 + ky**2).ravel()
    kbins = np.geomspace(knorm[knorm > 0].min(), 0.5, nbins + 1)
    counts, _ = np.histogram(knorm, kbins)
    power, _ = np.histogram(knorm, kbins, weights=pk2d.ravel())
    kcent = np.sqrt(kbins[1:] * kbins[:-1])
    good = counts > 0
    return kcent[good], power[good] / counts[good]


def plot_losses(loss_array, val_loss_array, val_every, path):
    loss = np.asarray(loss_array, dtype=np.float32)
    val = np.asarray(val_loss_array, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(1, len(loss) + 1) * val_every, loss, label="train (smoothed)", alpha=0.5)
    ax.plot(np.arange(1, len(val) + 1) * val_every, val, label="val")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_pk_overlay(truths, samples, theta_raw, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    k = radial_power_spectrum(truths[0])[0]
    pkt = np.stack([radial_power_spectrum(t)[1] for t in truths])
    pks = np.stack([radial_power_spectrum(s)[1] for s in samples])
    for arr, color, lab in ((pkt, "k", "truth"), (pks, "C0", "samples")):
        mean, std = arr.mean(0), arr.std(0)
        ax.loglog(k, mean, color=color, label=f"{lab} (mean)")
        ax.fill_between(k, np.maximum(mean - std, 1e-20), mean + std, color=color, alpha=0.2)
    ax.set_xlabel("k [cycles/pixel]")
    ax.set_ylabel("P(k)")
    ax.set_title(f"fixed theta: log_std={theta_raw[0]:.2f}, alpha={theta_raw[1]:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_field_grid(truths, samples, theta_raw, n_show, path):
    n_show = min(n_show, len(samples))
    fig, axes = plt.subplots(1, n_show + 1, figsize=(3 * (n_show + 1), 3.2), squeeze=False)
    vmax = float(np.percentile(np.abs(truths[0]), 99.5))
    axes[0][0].imshow(truths[0], vmin=-vmax, vmax=vmax, cmap="coolwarm")
    axes[0][0].set_title(f"truth | log_std={theta_raw[0]:.2f}, alpha={theta_raw[1]:.2f}", fontsize=9)
    for j in range(n_show):
        axes[0][j + 1].imshow(samples[j], vmin=-vmax, vmax=vmax, cmap="coolwarm")
        axes[0][j + 1].set_title(f"sample {j + 1}", fontsize=9)
    for ax in axes[0]:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def make_fixed_theta_loader(online_task, batch_size, theta_raw):
    """Monkeypatch the task prior to a constant theta; reuse the full online pipeline."""
    dim_theta = online_task.dim_theta
    theta_const = jnp.asarray(theta_raw, dtype=jnp.float32).reshape(dim_theta)
    orig_get_prior = online_task.task.get_prior

    def fixed_get_prior(key, n):
        template = orig_get_prior(key, n)  # (n, dim_theta), correct dtype/shape
        return jnp.broadcast_to(theta_const.astype(template.dtype), template.shape)

    online_task.task.get_prior = fixed_get_prior
    return online_task.get_online_train_loader(batch_size).map(swap_obs_cond)


def main(config_path):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    tcfg = cfg["training"]
    scfg = cfg["sampling"]
    experiment = tcfg["experiment_id"]
    model_cfg = cfg["pixeldit"]
    field_shape = tuple(model_cfg["field_shape"])
    theta_raw = np.asarray(cfg["fixed_theta"], dtype=np.float32)  # (dim_theta,)
    H, W = field_shape

    imgs_dir = os.path.join(EXAMPLE_DIR, "imgs")
    os.makedirs(imgs_dir, exist_ok=True)

    # --- data (fixed theta) ---
    task = OnlineTaskDataset(TASK_NAME, normalize=True, dtype=jnp.bfloat16)
    train_loader = make_fixed_theta_loader(task, tcfg["batch_size"], theta_raw)
    val_task = OnlineTaskDataset(TASK_NAME, normalize=True, dtype=jnp.bfloat16, seed=123)
    val_loader = make_fixed_theta_loader(val_task, tcfg["val_batch_size"], theta_raw)

    # --- model + method + pipeline ---
    model = build_pixeldit(model_cfg, seed=tcfg.get("seed", 0))
    n_params = sum(leaf.size for leaf in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
    print(f"pixeldit parameters: {n_params / 1e6:.1f}M (fixed-theta diagnostic)")

    ts_cfg = cfg.get("time_sampling", {})
    method = FlowMatchingMethod(
        time_dist=ts_cfg.get("dist", "uniform"),
        logitnorm_mean=ts_cfg.get("logitnorm_mean", 0.0),
        logitnorm_std=ts_cfg.get("logitnorm_std", 1.0),
    )

    training_config = FieldConditionalPipeline.get_default_training_config()
    training_config.update({k: tcfg[k] for k in _PIPELINE_KEYS if k in tcfg})
    training_config["checkpoint_dir"] = os.path.join(EXAMPLE_DIR, "checkpoints_pixeldit")

    pipeline = FieldConditionalPipeline(
        model, train_loader, val_loader,
        field_shape=field_shape, dim_cond=model_cfg["cond_dim"],
        method=method, ch_obs=1, ch_cond=1, training_config=training_config,
    )

    # --- train / restore ---
    if tcfg["train_model"]:
        loss_array, val_loss_array = pipeline.train(nnx.Rngs(0), save_model=True)
        plot_losses(loss_array, val_loss_array, training_config["val_every"],
                    os.path.join(imgs_dir, f"grf_fixedtheta_loss_conf{experiment}.png"))
    if tcfg["restore_model"]:
        pipeline.restore_model()
    pipeline._wrap_model()

    # --- truth fields at the fixed theta (simulate) ---
    n_eval = scfg["nsamples"]
    key = jax.random.PRNGKey(tcfg.get("seed", 0))
    key, kt = jax.random.split(key)
    theta_batch = jnp.broadcast_to(jnp.asarray(theta_raw), (n_eval, theta_raw.shape[0]))
    truth_flat = task.simulator(kt, theta_batch)                       # (n_eval, H*W) raw
    truths = np.asarray(truth_flat, dtype=np.float32).reshape(n_eval, H, W)

    # --- posterior samples at the fixed theta ---
    theta_norm = np.asarray(task.normalize_theta(theta_raw[None, :, None]))  # (1, dim_theta, 1)
    key, sub = jax.random.split(key)
    s = pipeline.sample(sub, jnp.asarray(theta_norm), nsamples=n_eval, step_size=scfg["step_size"])
    s = np.asarray(task.unnormalize_x(s), dtype=np.float32)[..., 0]          # (n_eval, H, W) raw
    print(f"sampled {s.shape}, finite={np.isfinite(s).all()}")

    plot_pk_overlay(truths, s, theta_raw,
                    os.path.join(imgs_dir, f"grf_fixedtheta_pk_conf{experiment}.png"))
    plot_field_grid(truths, s, theta_raw, scfg["nsamples_grid"],
                    os.path.join(imgs_dir, f"grf_fixedtheta_fields_conf{experiment}.png"))
    print(f"Plots written to {imgs_dir} (experiment {experiment})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config_4f.yaml")
    main(parser.parse_args().config)
```

- [ ] **Step 2: Verify it compiles**

Run:
```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
python -m py_compile examples/sbi-benchmarks/gaussian_random_field/train-grf-fixedtheta.py && echo "compile ok"
```
Expected: `compile ok`.

- [ ] **Step 3: Best-effort CPU smoke (depends on `gensbi` env + cached task metadata)**

Create a throwaway smoke config `examples/sbi-benchmarks/gaussian_random_field/config/config_4f_smoke.yaml` identical to `config_4f.yaml` (Task 5) but with `nsteps: 2`, `val_every: 1`, `nsamples: 4`, `experiment_id: 411`. Then run:
```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/gaussian_random_field
JAX_PLATFORMS=cpu /lhome/ific/a/aamerio/miniforge3/envs/gensbi/bin/python \
  train-grf-fixedtheta.py --config config/config_4f_smoke.yaml
```
Expected: prints param count, `sampled (4, 32, 32) finite=True`, and writes `imgs/grf_fixedtheta_*_conf411.png`. If it fails because `OnlineTaskDataset` needs network/HF metadata not cached locally, record that and rely on the Condor run (Task 7) as the integration test — do NOT weaken the script to make a local smoke pass. Delete `config_4f_smoke.yaml` and the `*conf411*` artifacts after.

- [ ] **Step 4: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
git add examples/sbi-benchmarks/gaussian_random_field/train-grf-fixedtheta.py
git commit -m "feat(grf): fixed-theta PixelDiT diagnostic script (logit-normal t, P(k) overlay)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Fixed-θ config `config_4f.yaml` (examples)

**Implementer model:** Sonnet (default)

**Files:**
- Create: `examples/sbi-benchmarks/gaussian_random_field/config/config_4f.yaml`

- [ ] **Step 1: Create the config**

Create `examples/sbi-benchmarks/gaussian_random_field/config/config_4f.yaml`:

```yaml
# GRF-32 PixelDiT — fixed-theta diagnostic (experiment 41 / version 4f)
# Single held-constant theta + logit-normal t-sampling, 100k steps. Tests whether
# PixelDiT can learn ONE GRF's 2-point structure with enough training.
# NOTE: cond is constant here -> does NOT test conditioning (intentional).
pixeldit:
  in_channels: 1
  field_shape: [32, 32]
  patch_size: 4
  hidden_size: 384
  pixel_hidden_size: 16
  patch_depth: 6
  pixel_depth: 2
  num_heads: 6
  cond_dim: 2
  cond_in_channels: 1
  cond_id_embedding: absolute
  use_cond_rope: false
  param_dtype: bfloat16

time_sampling:
  dist: logitnormal
  logitnorm_mean: 0.0
  logitnorm_std: 1.0

fixed_theta: [0.1, 3.0]        # [log_std, alpha]; alpha=3 = intermediate smoothness

training:
  batch_size: 128
  val_batch_size: 128
  nsteps: 100000
  max_lr: 1.0e-4
  val_every: 100
  early_stopping: false
  multistep: 1
  experiment_id: 41
  train_model: true
  restore_model: false
  seed: 0

sampling:
  nsamples: 64                 # truth + posterior samples for P(k) statistics
  nsamples_grid: 4
  step_size: 0.01              # Euler ODE step (100 steps)
```

- [ ] **Step 2: Verify it parses**

Run:
```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/gaussian_random_field
python -c "import yaml; c=yaml.safe_load(open('config/config_4f.yaml')); assert c['fixed_theta']==[0.1,3.0]; assert c['time_sampling']['dist']=='logitnormal'; assert c['training']['nsteps']==100000; print('config ok')"
```
Expected: `config ok`.

- [ ] **Step 3: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
git add examples/sbi-benchmarks/gaussian_random_field/config/config_4f.yaml
git commit -m "feat(grf): config_4f — fixed-theta logit-normal 100k diagnostic

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Condor submit file (examples)

**Implementer model:** Sonnet (default)

**Files:**
- Create: `sub/train_model_grf_PixelDiT_fixedtheta.sub`

- [ ] **Step 1: Create the submit file**

Create `/lhome/ific/a/aamerio/data/github/GenSBI-examples/sub/train_model_grf_PixelDiT_fixedtheta.sub`:

```
experiment_name = grf
version = 4f
workdir = /lhome/ific/a/aamerio/data/github/GenSBI-examples/examples/sbi-benchmarks/gaussian_random_field
script_path = train-grf-fixedtheta.py

universe = vanilla

request_memory = 32 GB
request_cpus = 8

executable = train_model.sh
arguments = "$(workdir) $(script_path) --config config/config_$(version).yaml"
getenv = True
request_gpus = 1
# bf16 matmuls (param_dtype: bfloat16) need Ampere+ -> A100/H100 nodes only
+UseNvidiaA100 = True

log                     = condor_logs/logs_$(experiment_name)_fixedtheta_$(version).log
output                  = condor_logs/outfile_$(experiment_name)_fixedtheta_$(version).out
error                   = condor_logs/errors_$(experiment_name)_fixedtheta_$(version).err

#########

queue
```

- [ ] **Step 2: Sanity-check it references the new script + config**

Run:
```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples/sub
grep -E "script_path|version|config_" train_model_grf_PixelDiT_fixedtheta.sub
test -f ../examples/sbi-benchmarks/gaussian_random_field/train-grf-fixedtheta.py && \
test -f ../examples/sbi-benchmarks/gaussian_random_field/config/config_4f.yaml && echo "targets exist"
```
Expected: shows `version = 4f`, `script_path = train-grf-fixedtheta.py`, the `config_$(version)` line, then `targets exist`.

- [ ] **Step 3: Commit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples
git add sub/train_model_grf_PixelDiT_fixedtheta.sub
git commit -m "feat(grf): Condor submit for fixed-theta PixelDiT diagnostic (v4f)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Submit the Condor job (final step — human-confirmed)

**Implementer model:** orchestrator (do not delegate); requires user confirmation before submitting a real cluster job.

**Files:** none (cluster action).

- [ ] **Step 1: Confirm with the user**

Ask the user to confirm submission (this consumes A100 cluster time). Do not proceed without an explicit yes.

- [ ] **Step 2: Submit**

```bash
cd /lhome/ific/a/aamerio/data/github/GenSBI-examples/sub
condor_submit train_model_grf_PixelDiT_fixedtheta.sub
```
Expected: `1 job(s) submitted to cluster <ID>.`

- [ ] **Step 3: Report and watch**

Record the cluster ID. Point the user at the decisive output once it runs:
`examples/sbi-benchmarks/gaussian_random_field/imgs/grf_fixedtheta_pk_conf41.png` (P(k) overlay — success = sample band tracks truth's low-k rise; failure = flat) and `..._loss_conf41.png`. Monitor `sub/condor_logs/errors_grf_fixedtheta_4f.err` for failures.

---

## Self-review

**Spec coverage:**
- §2.1 `sample_time` helper → Task 1. ✓
- §2.2 `FlowMatchingMethod` params + `prepare_batch` → Task 2. ✓
- §2.3 convention (bit-identical default, no clamp) → Task 2 (bit-identical test + split order preserved). ✓
- §2.4 tests (regression, logit-normal, invalid, method default) → Tasks 1 & 2. ✓
- §3.1 fixed-θ script (monkeypatched prior, simulate-truth, P(k) overlay, checkpoints_pixeldit) → Task 4. ✓
- §3.2 config_4f (100k, logit-normal, fixed θ, distinct experiment_id) → Task 5. ✓ (θ=[0.1,3.0], alpha=3 per user)
- §3.3 YAML schema → Tasks 3 & 5 read the same block. ✓
- §3.4 Condor submit + final submission → Tasks 6 & 7. ✓
- §3.5 uniform A/B → intentionally omitted per user ("don't write the uniform case yet"). ✓
- §4 wiring into existing scripts → Task 3. ✓

**Placeholder scan:** no TBD/TODO; every code/edit step shows full content; the smoke step (Task 4 Step 3) is explicitly best-effort with a defined fallback, not a placeholder.

**Type/name consistency:** `sample_time(key, n, *, dist, logitnorm_mean, logitnorm_std)` used identically in Tasks 1, 2, 3, 4, 5. `FlowMatchingMethod(time_dist=, logitnorm_mean=, logitnorm_std=)` consistent across Tasks 2–4. `make_fixed_theta_loader`, `swap_obs_cond`, config keys (`time_sampling.dist`, `fixed_theta`, `experiment_id: 41`, `version 4f`) consistent across Tasks 4–6.

**Decisions locked from spec §7:** helpers duplicated (not factored) to avoid touching the working script; `experiment_id: 41` / version `4f`; fixed θ `[0.1, 3.0]`; loader = monkeypatched `get_prior` reusing the online pipeline.
