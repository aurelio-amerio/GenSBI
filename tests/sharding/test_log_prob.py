# tests/sharding/test_log_prob.py
"""log_prob with a mesh: committed inputs are placed, values match the no-mesh pipeline."""
import dataclasses
import numpy as np, jax, jax.numpy as jnp, pytest
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P
from gensbi import sharding as sh
from gensbi.core import FlowMatchingMethod
from gensbi.models import MAFlow, MAFlowParams, Simformer
from gensbi.recipes import (ConditionalFlowPipeline, Flux1FlowPipeline, SimformerFlowPipeline,
                            UnconditionalPipeline)
from .helpers import *

TOL = dict(rtol=1e-4, atol=1e-5)

# Value comparisons use a fixed-step solver. Dopri5's step-size controller
# takes a norm over the whole batch, so any change in floating-point
# reduction order (sharding x_1, or even splitting the batch in two without a
# mesh) can change the accepted steps and move log-probs by ~1e-2 on values of
# ~20. Fixed steps isolate placement from that solver sensitivity.
FIXED = dict(method="Euler", step_size=0.05)

x_o = jax.random.normal(jax.random.key(2), (1, dim_cond, 2))


def _committed(x):
    return jax.device_put(x, jax.devices()[0])


def _x1(n, dim=dim_obs, ch=2):
    return jax.random.normal(jax.random.key(5), (n, dim, ch))


def _spec(n):
    """Expected placement of ``n`` evaluated points on the default 8-device mesh."""
    return P("data") if n % 8 == 0 else P()


def _is(x, mesh, spec):
    return x.sharding.is_equivalent_to(NamedSharding(mesh, spec), x.ndim)


@pytest.fixture
def lp_calls(monkeypatch):
    """Record ``(x_1, model_extras)`` passed to every method-built log-prob closure."""
    calls = []
    def install(pipe):
        build = pipe.method.build_log_prob_fn
        def spy_build(*args, **kw):
            fn = build(*args, **kw)
            def log_prob_fn(x_1, model_extras, *, key=None):
                calls.append((x_1, model_extras)); return fn(x_1, model_extras, key=key)
            return log_prob_fn
        monkeypatch.setattr(pipe.method, "build_log_prob_fn", spy_build)
    return install, calls


def _dev(got, want, label):
    dev = np.max(np.abs(np.asarray(got) - np.asarray(want)))
    print(f"{label}: max |mesh - no-mesh| = {dev:.3e}")


def _close(got, want, label):
    _dev(got, want, label)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), **TOL)


def _adaptive_runs(got, want, n, label):
    """Default Dopri5 with a mesh: runs and is finite; deviation reported, not asserted."""
    assert got.shape == (n,) and np.all(np.isfinite(np.asarray(got)))
    _dev(got, want, label + " (Dopri5, informational)")


def _cond_pipe(mesh, tmp_path):
    p = Flux1FlowPipeline(train_dataset_cond, val_dataset_cond, dim_obs, dim_cond, ch_obs=2, ch_cond=2,
                          params=small_flux1_params(), training_config=test_training_config(tmp_path), mesh=mesh)
    p.ema_model = p.model; p._wrap_model(); return p


def _joint_pipe(mesh, tmp_path):
    p = SimformerFlowPipeline(train_dataset_joint, val_dataset_joint, dim_obs, dim_cond, ch_obs=2,
                              params=small_simformer_params(), training_config=test_training_config(tmp_path), mesh=mesh)
    p.ema_model = p.model; p._wrap_model(); return p


def _flow_pipe(mesh, tmp_path):
    flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=dim_obs, cond_dim=dim_cond, n_layers=2, nn_width=16,
                               channels=2, cond_channels=2, zero_init=False))
    p = ConditionalFlowPipeline(flow, train_dataset_cond, val_dataset_cond, dim_obs, dim_cond, ch_obs=2, ch_cond=2,
                                training_config=test_training_config(tmp_path), mesh=mesh)
    p.ema_model = p.model; p._wrap_model(); return p


def _uncond_pipe(mesh, tmp_path):
    # UnconditionalWrapper builds a per-channel condition mask, so the joint
    # models need a single channel here.
    params = dataclasses.replace(small_simformer_params(), in_channels=1, rngs=nnx.Rngs(0))
    p = UnconditionalPipeline(Simformer(params), train_dataset_joint, val_dataset_joint,
                              dim_joint, method=FlowMatchingMethod(), ch_obs=1,
                              training_config=test_training_config(tmp_path), mesh=mesh)
    p.ema_model = p.model; p._wrap_model(); return p


@requires_8_devices
@pytest.mark.parametrize("n", [16, 3], ids=["divisible", "non_divisible"])
@pytest.mark.parametrize("make", [_cond_pipe, _joint_pipe], ids=["conditional", "joint"])
def test_log_prob_committed_inputs_match_no_mesh(make, n, tmp_path, lp_calls):
    a = make(None, tmp_path / "a"); b = make(sh.make_mesh(), tmp_path / "b")
    install, calls = lp_calls; install(b)
    x_1, name = _x1(n), make.__name__
    la = a.log_prob(x_1, x_o, use_ema=False, **FIXED)
    lb = b.log_prob(_committed(x_1), _committed(x_o), use_ema=False, **FIXED)
    assert lb.shape == (n,)
    _close(lb, la, f"{name} n={n} exact")
    # Hutchinson: the same key gives the same probe noise with and without a mesh.
    key = jax.random.key(42)
    ha = a.log_prob(x_1, x_o, use_ema=False, key=key, exact_divergence=False, **FIXED)
    hb = b.log_prob(_committed(x_1), _committed(x_o), use_ema=False, key=_committed(key),
                    exact_divergence=False, **FIXED)
    _close(hb, ha, f"{name} n={n} hutchinson")
    # get_log_prob_fn: the closure places x_1 on each call.
    fb = b.get_log_prob_fn(_committed(x_o), use_ema=False, **FIXED)
    _close(fb(_committed(x_1)), la, f"{name} n={n} get_log_prob_fn")
    # Extras passed at call time are placed too.
    extras = {"cond": _committed(x_o), "obs_ids": _committed(b.obs_ids), "cond_ids": _committed(b.cond_ids)}
    _close(fb(_committed(x_1), extras), la, f"{name} n={n} call-time extras")
    _adaptive_runs(b.log_prob(_committed(x_1), _committed(x_o), use_ema=False),
                   a.log_prob(x_1, x_o, use_ema=False), n, f"{name} n={n} exact")
    # x_1 sharded over the batch axes when divisible, else replicated; extras replicated.
    assert len(calls) == 5
    for x, ex in calls:
        assert _is(x, b.mesh, _spec(n))
        for k in ("cond", "obs_ids", "cond_ids"):
            assert _is(ex[k], b.mesh, P()), k


@requires_8_devices
@pytest.mark.parametrize("n", [16, 3], ids=["divisible", "non_divisible"])
def test_flow_log_prob_committed_inputs_match_no_mesh(n, tmp_path, monkeypatch):
    a = _flow_pipe(None, tmp_path / "a"); b = _flow_pipe(sh.make_mesh(), tmp_path / "b")
    calls, orig = [], MAFlow.log_prob
    def spy(self, x, cond):
        calls.append((x, cond)); return orig(self, x, cond)
    x_1 = _x1(n)
    la = a.log_prob(x_1, x_o, use_ema=False)
    monkeypatch.setattr(MAFlow, "log_prob", spy)
    lb = b.log_prob(_committed(x_1), _committed(x_o), use_ema=False)
    assert lb.shape == (n,)
    _close(lb, la, f"flow n={n}")
    fb = b.get_log_prob_fn(_committed(x_o), use_ema=False)
    _close(fb(_committed(x_1)), la, f"flow n={n} get_log_prob_fn")
    # The broadcast condition has the same leading size as x_1: both follow
    # the batch policy (sharded when divisible, replicated otherwise).
    assert len(calls) == 2
    for x, c in calls:
        assert _is(x, b.mesh, _spec(n)) and _is(c, b.mesh, _spec(n))


@requires_8_devices
@pytest.mark.parametrize("n", [16, 3], ids=["divisible", "non_divisible"])
def test_unconditional_log_prob_committed_inputs_match_no_mesh(n, tmp_path, lp_calls):
    a = _uncond_pipe(None, tmp_path / "a"); b = _uncond_pipe(sh.make_mesh(), tmp_path / "b")
    install, calls = lp_calls; install(b)
    x_1 = _x1(n, dim_joint, 1)
    la = a.log_prob(x_1, use_ema=False, **FIXED)
    lb = b.log_prob(_committed(x_1), use_ema=False, **FIXED)
    assert lb.shape == (n,)
    _close(lb, la, f"unconditional n={n}")
    key = jax.random.key(42)
    ha = a.log_prob(x_1, use_ema=False, key=key, exact_divergence=False, **FIXED)
    hb = b.log_prob(_committed(x_1), use_ema=False, key=_committed(key), exact_divergence=False, **FIXED)
    _close(hb, ha, f"unconditional n={n} hutchinson")
    _adaptive_runs(b.log_prob(_committed(x_1), use_ema=False), a.log_prob(x_1, use_ema=False),
                   n, f"unconditional n={n}")
    assert len(calls) == 3
    for x, ex in calls:
        assert _is(x, b.mesh, _spec(n)) and _is(ex["obs_ids"], b.mesh, P())


@requires_8_devices
@pytest.mark.parametrize("n, spec", [(16, P("data")), (3, P())], ids=["divisible", "non_divisible"])
def test_place_batch_or_replicate(n, spec, tmp_path):
    b = _cond_pipe(sh.make_mesh(), tmp_path)
    x = b._place_batch_or_replicate(_committed(_x1(n)))
    assert x.sharding.is_equivalent_to(NamedSharding(b.mesh, spec), x.ndim)
    a = _cond_pipe(None, tmp_path / "a")
    y = _x1(n)
    assert a._place_batch_or_replicate(y) is y


@requires_8_devices
@pytest.mark.parametrize("make", [_cond_pipe, _joint_pipe, _flow_pipe], ids=["conditional", "joint", "flow"])
def test_log_prob_without_mesh_does_no_placement(make, tmp_path, monkeypatch):
    a = make(None, tmp_path)
    def boom(*args, **kw):
        raise AssertionError("device_put called without a mesh")
    monkeypatch.setattr(jax, "device_put", boom)
    assert a.log_prob(_x1(3), x_o, use_ema=False).shape == (3,)
