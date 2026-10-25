# tests/sharding/test_checkpoint.py
import jax, jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from gensbi import sharding as sh
from gensbi.recipes import Flux1FlowPipeline
from .helpers import *
from .test_pipeline import _assert_params_close


def _flat_ndims(node):
    return {"/".join(map(str, p)): v[...].ndim for p, v in nnx.to_flat_state(nnx.state(node))}


def _assert_same_placement(before, after, node):
    nd = _flat_ndims(node)
    assert set(before) == set(after)
    for k in before:
        assert after[k].is_equivalent_to(before[k], nd[k]), k
    # non-vacuous: something is really sharded
    assert any(not before[k].is_fully_replicated for k in before)


def _flux(tc, **kw):
    return Flux1FlowPipeline(train_dataset_cond, val_dataset_cond, dim_obs, dim_cond, ch_obs=2, ch_cond=2,
                             params=small_flux1_params(), training_config=tc, **kw)


@requires_8_devices
def test_restore_follows_current_placement(tmp_path):
    tc = test_training_config(tmp_path, 10)
    a = _flux(tc)
    a.train(nnx.Rngs(0), save_model=True)
    mesh = sh.make_mesh()
    b = _flux(dict(tc), mesh=mesh, sharding_rule="fsdp")
    before = sh.sharding_of(b.model); b.restore_model()
    after = sh.sharding_of(b.model)
    _assert_same_placement(before, after, b.model)
    _assert_params_close(a.model, b.model, rtol=0, atol=0)
    b.save_model(experiment_id=2)
    c = _flux(dict(tc))
    c.restore_model(experiment_id=2)
    assert isinstance(c.model.obs_in.kernel[...].sharding, jax.sharding.SingleDeviceSharding)
    _assert_params_close(b.model, c.model, rtol=0, atol=0)


@requires_8_devices
def test_safetensors_roundtrip_keeps_placement(tmp_path):
    mesh = sh.make_mesh()
    p = _flux(test_training_config(tmp_path, 10), mesh=mesh, sharding_rule="fsdp")
    before = sh.sharding_of(p.ema_model)
    p.export_safetensors(tmp_path / "w.safetensors"); p.import_safetensors(tmp_path / "w.safetensors")
    after = sh.sharding_of(p.ema_model)
    _assert_same_placement(before, after, p.ema_model)


@requires_8_devices
def test_fit_standardization_keeps_placement(tmp_path):
    from gensbi.recipes import ConditionalFlowPipeline
    from gensbi.models import MAFlow, MAFlowParams
    mesh = sh.make_mesh()
    flow = MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=dim_obs, cond_dim=dim_cond, n_layers=2, nn_width=16, channels=2, cond_channels=2))
    p = ConditionalFlowPipeline(flow, train_dataset_cond, val_dataset_cond, dim_obs, dim_cond, ch_obs=2, ch_cond=2,
                                training_config=test_training_config(tmp_path, 10), mesh=mesh)
    p.fit_standardization(jnp.ones((64, dim_obs, 2)), axis=(0, 1))
    for node in (p.model, p.ema_model):
        for name, s in sh.sharding_of(node).items():
            assert isinstance(s, NamedSharding) and s.mesh == mesh, name
