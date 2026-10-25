# tests/sharding/test_sampling.py
import numpy as np, jax, jax.numpy as jnp, pytest
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P
from gensbi import sharding as sh
from gensbi.recipes import flow_pipeline
from gensbi.core import FlowMatchingMethod
from gensbi.recipes import (ConditionalFlowPipeline, ConditionalPipeline, Flux1DiffusionPipeline,
                            Flux1FlowPipeline, Flux1SMPipeline)
from gensbi.models import MAFlow, MAFlowParams, TarFlow, TarFlowParams
from .helpers import *
import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).parents[1] / "recipes"))
from mock_models import MockConditionalModel

def _pipe(cls, mesh, tmp_path, params=None):
    p = cls(train_dataset_cond, val_dataset_cond, dim_obs, dim_cond, ch_obs=2, ch_cond=2,
            params=params or small_flux1_params(), training_config=test_training_config(tmp_path), mesh=mesh)
    p.ema_model = p.model; p._wrap_model(); return p

x_o = jax.random.normal(jax.random.key(2), (1, dim_cond, 2))

def _on_data(x, mesh):
    return x.sharding.is_equivalent_to(NamedSharding(mesh, P("data")), x.ndim)

@requires_8_devices
@pytest.mark.parametrize("cls", [Flux1FlowPipeline, Flux1DiffusionPipeline])
def test_ode_and_edm_sampling_matches_no_mesh(cls, tmp_path, monkeypatch):
    a = _pipe(cls, None, tmp_path / "a"); b = _pipe(cls, sh.make_mesh(), tmp_path / "b")
    sa = a.sample(jax.random.key(1), x_o, nsamples=16, use_ema=False, show_progress_bars=False)
    # Spy on the x_init the (unchanged) jitted solver receives. Its output
    # sharding is the solver's business: diffrax gathers its SaveAt buffer,
    # so the ODE output comes back replicated even though the model runs
    # 2 rows per device.
    seen, build = [], b.method.build_sampler_fn
    def spy_build(*args, **kw):
        fn = build(*args, **kw)
        def sampler_fn(key, x_init, model_extras=None):
            seen.append(x_init); return fn(key, x_init, model_extras)
        return sampler_fn
    monkeypatch.setattr(b.method, "build_sampler_fn", spy_build)
    sb = b.sample(jax.random.key(1), x_o, nsamples=16, use_ema=False, show_progress_bars=False)
    np.testing.assert_allclose(np.asarray(sb), np.asarray(sa), rtol=1e-4, atol=1e-5)
    assert isinstance(sa.sharding, jax.sharding.SingleDeviceSharding)
    assert len(seen) == 1 and _on_data(seen[0], b.mesh)   # x_init placed on the batch axes

@requires_8_devices
def test_sharded_init_pads_places_and_trim_slices(tmp_path):
    b = _pipe(Flux1FlowPipeline, sh.make_mesh(), tmp_path)
    draw = lambda k, n: jax.random.normal(k, (n, dim_obs, 2))
    x, n_pad = b._sharded_init(draw, jax.random.key(0), 3)
    assert n_pad == 8 and x.shape == (8, dim_obs, 2) and _on_data(x, b.mesh)
    assert b._trim(x, 3, 0).shape == (3, dim_obs, 2)
    assert b._trim(jnp.zeros((4, 8, dim_obs, 2)), 3, 1).shape == (4, 3, dim_obs, 2)
    assert b._trim(x, 8, 0) is x
    a = _pipe(Flux1FlowPipeline, None, tmp_path / "a")
    x0, n0 = a._sharded_init(draw, jax.random.key(0), 3)
    assert n0 == 3 and x0.shape == (3, dim_obs, 2)
    np.testing.assert_array_equal(np.asarray(x0), np.asarray(draw(jax.random.key(0), 3)))

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

@requires_8_devices
def test_unannotated_user_model_samples_with_mesh(tmp_path):
    p = ConditionalPipeline(MockConditionalModel(), train_dataset_cond, val_dataset_cond, dim_obs, dim_cond,
                            method=FlowMatchingMethod(), ch_obs=2, ch_cond=2,
                            training_config=test_training_config(tmp_path, 10), mesh=sh.make_mesh())
    p.ema_model = p.model; p._wrap_model()
    assert p.sample(jax.random.key(1), x_o, nsamples=3, use_ema=False, show_progress_bars=False).shape == (3, dim_obs, 2)

def _flow_pipe(flow_fn, mesh, tmp_path):
    p = ConditionalFlowPipeline(flow_fn(), train_dataset_cond, val_dataset_cond, dim_obs, dim_cond, ch_obs=2, ch_cond=2,
                                training_config=test_training_config(tmp_path), mesh=mesh)
    p.ema_model = p.model; p._wrap_model(); return p

_maf = lambda: MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=dim_obs, cond_dim=dim_cond, n_layers=2, nn_width=16, channels=2, cond_channels=2, zero_init=False))
_tar = lambda: TarFlow(TarFlowParams(rngs=nnx.Rngs(0), dim=dim_obs, cond_dim=dim_cond, num_blocks=2, num_heads=2, head_dim=8, vec_channels=2, cond_channels=2, zero_init=False))

@pytest.fixture
def jit_calls(monkeypatch):
    """Record the condition passed to every jitted flow-sampling call."""
    calls, orig = [], flow_pipeline._jit_flow_sample
    def spy(flow, key, cond):
        calls.append(cond); return orig(flow, key, cond)
    monkeypatch.setattr(flow_pipeline, "_jit_flow_sample", spy)
    return calls

@requires_8_devices
@pytest.mark.parametrize("flow_fn", [_maf, _tar], ids=["maf", "tarflow"])
def test_flow_sampling_jitted_and_matches_no_mesh(flow_fn, tmp_path, jit_calls):
    a = _flow_pipe(flow_fn, None, tmp_path / "a"); b = _flow_pipe(flow_fn, sh.make_mesh(), tmp_path / "b")
    sa = a.sample(jax.random.key(1), x_o, nsamples=16, use_ema=False, show_progress_bars=False)
    sb = b.sample(jax.random.key(1), x_o, nsamples=16, use_ema=False, show_progress_bars=False)
    np.testing.assert_allclose(np.asarray(sb), np.asarray(sa), rtol=1e-4, atol=1e-5)
    assert b.sample(jax.random.key(1), x_o, nsamples=3, use_ema=False, show_progress_bars=False).shape == (3, dim_obs, 2)
    xb = jax.random.normal(jax.random.key(3), (4, dim_cond, 2))
    assert b.sample_batched(jax.random.key(1), xb, nsamples=4, use_ema=False, show_progress_bars=False).shape == (4, 4, dim_obs, 2)
    # Every draw went through the jitted path: no-mesh 16, mesh 16, mesh 3 (padded to 8), batched 16.
    assert [c.shape[0] for c in jit_calls] == [16, 16, 8, 16]
    assert isinstance(jit_calls[0].sharding, jax.sharding.SingleDeviceSharding)
    assert all(_on_data(c, b.mesh) for c in jit_calls[1:])
    assert _on_data(sb, b.mesh)
    np.testing.assert_allclose(np.asarray(b.sample_batched(jax.random.key(1), xb, nsamples=4, use_ema=False, show_progress_bars=False)),
                               np.asarray(a.sample_batched(jax.random.key(1), xb, nsamples=4, use_ema=False, show_progress_bars=False)),
                               rtol=1e-4, atol=1e-5)

@requires_8_devices
def test_flow_sample_batched_places_only_divisible_chunks(tmp_path, jit_calls):
    b = _flow_pipe(_maf, sh.make_mesh(), tmp_path)
    xb = jax.random.normal(jax.random.key(3), (4, dim_cond, 2))
    out = b.sample_batched(jax.random.key(1), xb, nsamples=5, chunk_size=8, use_ema=False, show_progress_bars=False)
    assert out.shape == (5, 4, dim_obs, 2)
    assert [c.shape[0] for c in jit_calls] == [8, 8, 4]
    assert _on_data(jit_calls[0], b.mesh) and _on_data(jit_calls[1], b.mesh)
    # The non-divisible tail is replicated over the mesh, never left on one device.
    assert jit_calls[2].sharding.is_equivalent_to(NamedSharding(b.mesh, P()), jit_calls[2].ndim)

# A condition committed to one device (e.g. built under jax.default_device or
# returned by a jitted function) must not clash with the mesh-placed model:
# the pipeline replicates the sampler extras over the mesh.
def _committed(x):
    return jax.device_put(x, jax.devices()[0])

@requires_8_devices
def test_committed_condition_samples_with_mesh(tmp_path):
    a = _pipe(Flux1FlowPipeline, None, tmp_path / "a"); b = _pipe(Flux1FlowPipeline, sh.make_mesh(), tmp_path / "b")
    sa = a.sample(jax.random.key(1), x_o, nsamples=16, use_ema=False, show_progress_bars=False)
    sb = b.sample(jax.random.key(1), _committed(x_o), nsamples=16, use_ema=False, show_progress_bars=False)
    np.testing.assert_allclose(np.asarray(sb), np.asarray(sa), rtol=1e-4, atol=1e-5)

@requires_8_devices
def test_committed_condition_samples_with_mesh_joint(tmp_path):
    from gensbi.recipes import SimformerFlowPipeline
    outs = []
    for name, mesh, x in [("a", None, x_o), ("b", sh.make_mesh(), _committed(x_o))]:
        p = SimformerFlowPipeline(train_dataset_joint, val_dataset_joint, dim_obs, dim_cond, ch_obs=2,
                                  params=small_simformer_params(), training_config=test_training_config(tmp_path / name), mesh=mesh)
        p.ema_model = p.model; p._wrap_model()
        outs.append(np.asarray(p.sample(jax.random.key(1), x, nsamples=16, use_ema=False, show_progress_bars=False)))
    np.testing.assert_allclose(outs[1], outs[0], rtol=1e-4, atol=1e-5)

@requires_8_devices
def test_flow_committed_condition_samples_with_mesh(tmp_path, jit_calls):
    a = _flow_pipe(_maf, None, tmp_path / "a"); b = _flow_pipe(_maf, sh.make_mesh(), tmp_path / "b")
    sa = a.sample(jax.random.key(1), x_o, nsamples=16, use_ema=False, show_progress_bars=False)
    sb = b.sample(jax.random.key(1), _committed(x_o), nsamples=16, use_ema=False, show_progress_bars=False)
    np.testing.assert_allclose(np.asarray(sb), np.asarray(sa), rtol=1e-4, atol=1e-5)
    # B * nsamples = 9 is not a multiple of 8: the condition is replicated, not sharded.
    xb = jax.random.normal(jax.random.key(3), (3, dim_cond, 2))
    ra = a.sample_batched(jax.random.key(1), xb, nsamples=3, use_ema=False, show_progress_bars=False)
    jit_calls.clear()
    rb = b.sample_batched(jax.random.key(1), _committed(xb), nsamples=3, use_ema=False, show_progress_bars=False)
    np.testing.assert_allclose(np.asarray(rb), np.asarray(ra), rtol=1e-4, atol=1e-5)
    assert [c.shape[0] for c in jit_calls] == [9]
    assert jit_calls[0].sharding.is_equivalent_to(NamedSharding(b.mesh, P()), jit_calls[0].ndim)

# Padding nsamples up to the data-parallel size must not change the samples:
# JAX's partitionable threefry makes the first n rows of an (n_pad, ...) draw
# equal to an (n, ...) draw.
@requires_8_devices
def test_padded_sampling_matches_no_mesh(tmp_path):
    kw = dict(use_ema=False, show_progress_bars=False)
    for name, make in [("flux", lambda m, d: _pipe(Flux1FlowPipeline, m, d)), ("maf", lambda m, d: _flow_pipe(_maf, m, d))]:
        a, b = make(None, tmp_path / f"{name}a"), make(sh.make_mesh(), tmp_path / f"{name}b")
        sa = a.sample(jax.random.key(1), x_o, nsamples=3, **kw)
        sb = b.sample(jax.random.key(1), x_o, nsamples=3, **kw)
        np.testing.assert_allclose(np.asarray(sb), np.asarray(sa), rtol=1e-4, atol=1e-5, err_msg=name)

@requires_8_devices
def test_joint_per_sample_condition_is_padded_with_mesh(tmp_path):
    import warnings
    from gensbi.recipes import SimformerFlowPipeline
    xb = jax.random.normal(jax.random.key(3), (3, dim_cond, 2))
    outs = []
    for name, mesh in [("a", None), ("b", sh.make_mesh())]:
        p = SimformerFlowPipeline(train_dataset_joint, val_dataset_joint, dim_obs, dim_cond, ch_obs=2,
                                  params=small_simformer_params(), training_config=test_training_config(tmp_path / name), mesh=mesh)
        p.ema_model = p.model; p._wrap_model()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # x_o batch > 1 warns by design
            outs.append(np.asarray(p.sample(jax.random.key(1), xb, nsamples=3, use_ema=False, show_progress_bars=False)))
    np.testing.assert_allclose(outs[1], outs[0], rtol=1e-4, atol=1e-5)
