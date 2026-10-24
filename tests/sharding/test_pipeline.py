# tests/sharding/test_pipeline.py
import numpy as np, jax, jax.numpy as jnp, pytest
from flax import nnx
from jax.sharding import NamedSharding, PartitionSpec as P
from gensbi import sharding as sh
from gensbi.recipes import ConditionalPipeline, Flux1FlowPipeline, SimformerFlowPipeline, Flux1JointFlowPipeline
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
    # Splitting the batch over 8 devices changes the float reduction order of
    # the batch mean; Adam amplifies that noise on near-zero coordinates. Measured
    # after 20 steps: max |dparam| 5.0e-5 (needs atol 3.4e-5 at rtol 1e-3) for the
    # live model, 5.4e-7 for EMA; a mesh run with a replicated batch differs by
    # only 3.9e-7 and two no-mesh runs are bit-identical.
    _assert_params_close(ref.model, p.model, atol=1e-4); _assert_params_close(ref.ema_model, p.ema_model)
    mesh = p.mesh
    for path, s in sh.sharding_of(p.model).items():
        assert s.is_equivalent_to(NamedSharding(mesh, P()), 1), path

# nsteps >= 10 in these tests: with decay_transition=0.8, nsteps=5 gives
# int(0.2 * 5) == 0 cosine decay steps, which optax rejects.
@requires_8_devices
def test_unannotated_user_model_trains_with_mesh(tmp_path):
    p = ConditionalPipeline(MockConditionalModel(), train_dataset_cond, val_dataset_cond, dim_obs, dim_cond,
                            method=FlowMatchingMethod(), ch_obs=2, ch_cond=2,
                            training_config=test_training_config(tmp_path, 10), mesh=sh.make_mesh())
    p.train(nnx.Rngs(0), save_model=False)
    assert sh.sharding_of(p.model)["dummy"].is_equivalent_to(NamedSharding(p.mesh, P()), 1)

@requires_8_devices
def test_validation_batch_not_divisible_names_loader(tmp_path):
    import grain
    bad_val = (grain.MapDataset.source(np.zeros((100, dim_obs + dim_cond, 2), np.float32)).repeat()
               .to_iter_dataset().batch(12).map(lambda d: (d[:, :dim_obs], d[:, dim_obs:])))
    p = Flux1FlowPipeline(train_dataset_cond, bad_val, dim_obs, dim_cond, ch_obs=2, ch_cond=2,
                          params=small_flux1_params(), training_config=test_training_config(tmp_path, 10), mesh=sh.make_mesh())
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

@requires_8_devices
def test_ema_model_follows_model_placement(tmp_path):
    with pytest.warns(UserWarning, match="experimental"):
        p = Flux1FlowPipeline(train_dataset_cond, val_dataset_cond, dim_obs, dim_cond, ch_obs=2, ch_cond=2,
                              params=small_flux1_params(), training_config=test_training_config(tmp_path),
                              mesh=sh.make_mesh(), sharding_rule="fsdp")
    model_s, ema_s = sh.sharding_of(p.model), sh.sharding_of(p.ema_model)
    ranks = {"/".join(map(str, k)): v[...].ndim for k, v in nnx.to_flat_state(nnx.state(p.model))}
    assert model_s.keys() == ema_s.keys()
    assert any(not s.is_equivalent_to(NamedSharding(p.mesh, P()), ranks[k]) for k, s in model_s.items())
    for k, s in model_s.items():
        assert ema_s[k].is_equivalent_to(s, ranks[k]), k


def _train_joint(cls, params, mesh, rule, tmp_path, nsteps=20):
    p = cls(train_dataset_joint, val_dataset_joint, dim_obs, dim_cond, ch_obs=2, params=params,
            training_config=test_training_config(tmp_path, nsteps), mesh=mesh, sharding_rule=rule)
    losses, _ = p.train(nnx.Rngs(0), save_model=False)
    return p, np.asarray([float(l) for l in losses])

@requires_8_devices
@pytest.mark.parametrize("rule", ["replicated", "fsdp"])
# Simformer needs atol 2e-4: the attention key bias has a mathematically zero
# gradient (softmax is shift-invariant), so Adam turns pure reduction-order noise
# into a drift. Measured max |dparam| after 20 steps: 1.2e-4 for that one bias
# (rule replicated and fsdp); every other parameter deviates by <= 2.1e-5.
@pytest.mark.parametrize("cls,params_fn,atol", [(SimformerFlowPipeline, small_simformer_params, 2e-4), (Flux1JointFlowPipeline, small_flux1joint_params, 1e-4)])
def test_joint_pipelines_match_no_mesh(cls, params_fn, atol, rule, tmp_path):
    ref, l_ref = _train_joint(cls, params_fn(), None, None, tmp_path / "a")
    p, l = _train_joint(cls, params_fn(), sh.make_mesh(), rule, tmp_path / "b")
    np.testing.assert_allclose(l, l_ref, rtol=1e-3, atol=1e-5)
    _assert_params_close(ref.model, p.model, atol=atol)  # R6: reduction-order noise over 8 devices

@requires_8_devices
def test_flux1_fsdp_matches_and_shards_params_opt_and_ema(tmp_path):
    ref, l_ref, _ = _train(Flux1FlowPipeline, small_flux1_params(), None, None, tmp_path / "a")
    mesh = sh.make_mesh()
    p = Flux1FlowPipeline(train_dataset_cond, val_dataset_cond, dim_obs, dim_cond, ch_obs=2, ch_cond=2,
                          params=small_flux1_params(), training_config=test_training_config(tmp_path / "b"), mesh=mesh, sharding_rule="fsdp")
    s = sh.sharding_of(p.model)
    ranks = {"/".join(map(str, k)): v[...].ndim for k, v in nnx.to_flat_state(nnx.state(p.model))}
    sharded = [k for k, v in s.items() if not v.is_equivalent_to(NamedSharding(mesh, P()), ranks[k])]
    assert sharded, "FSDP sharded nothing: hidden size must be divisible by 8"
    optimizer = p._get_optimizer(); sh.shard_model(optimizer, mesh, "fsdp")
    mu = {"/".join(map(str, k)): v for k, v in nnx.to_flat_state(nnx.state(optimizer)) if "mu" in map(str, k)}
    # Match the optimizer's mu twin by full path suffix on a "/" boundary.
    matched = 0
    for name in sharded:
        twins = [v for k, v in mu.items() if k == name or k.endswith("/" + name)]
        assert len(twins) == 1, (name, len(twins))
        assert twins[0][...].sharding.is_equivalent_to(s[name], twins[0][...].ndim), name
        matched += 1
    assert matched == len(sharded) > 0
    losses, _ = p.train(nnx.Rngs(0), save_model=False)
    np.testing.assert_allclose(np.asarray([float(l) for l in losses]), l_ref, rtol=1e-3, atol=1e-5)
    _assert_params_close(ref.model, p.model, atol=1e-4)  # R6
    for name, var in sh.sharding_of(p.ema_model).items():
        assert var.is_equivalent_to(s[name], ranks[name]), name  # EMA mirrors the model's placement after training

@requires_8_devices
def test_one_device_mesh_matches_no_mesh(tmp_path):
    ref, l_ref, _ = _train(Flux1FlowPipeline, small_flux1_params(), None, None, tmp_path / "a")
    one = sh.make_mesh(devices=jax.devices()[:1])
    p, l, _ = _train(Flux1FlowPipeline, small_flux1_params(), one, "replicated", tmp_path / "b")
    np.testing.assert_allclose(l, l_ref, rtol=1e-5, atol=1e-6)
