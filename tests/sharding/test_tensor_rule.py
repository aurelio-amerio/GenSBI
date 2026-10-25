# tests/sharding/test_tensor_rule.py
"""Smoke tests for the experimental ``"tensor"`` rule on a 4x2 mesh."""
import numpy as np, jax, pytest
from flax import nnx
from gensbi import sharding as sh
from gensbi.recipes import Flux1FlowPipeline, SimformerFlowPipeline
from .helpers import *
from .test_pipeline import _train, _train_joint


def _over_model(model):
    return sorted(k for k, v in sh.sharding_of(model).items() if "model" in str(v.spec))


# Real parameter paths (from ``sh.sharding_of``) that the tensor rule must
# shard over "model": attention (heads axis) and MLP (mlp axis) kernels.
_EXPECTED = {
    Flux1FlowPipeline: ["double_blocks/layers/0/obs_attn/qkv/kernel",
                        "double_blocks/layers/0/obs_mlp/layers/0/kernel",
                        "single_blocks/layers/0/linear1/kernel"],
    SimformerFlowPipeline: ["transformer/attention_blocks/0/attn/query/kernel",
                            "transformer/dense_blocks/0/hidden_blocks/0/kernel"],
}


@requires_8_devices
@pytest.mark.parametrize("cls,params_fn,joint", [(Flux1FlowPipeline, small_flux1_params, False),
                                                 (SimformerFlowPipeline, small_simformer_params, True)])
def test_tensor_rule_matches_no_mesh(cls, params_fn, joint, tmp_path):
    mesh = sh.make_mesh(model_axis=2)
    run = _train_joint if joint else (lambda c, p, m, r, t: _train(c, p, m, r, t)[:2])
    ref, l_ref = run(cls, params_fn(), None, None, tmp_path / "a")
    with pytest.warns(UserWarning, match="experimental"):
        p, l = run(cls, params_fn(), mesh, "tensor", tmp_path / "b")
    np.testing.assert_allclose(l, l_ref, rtol=1e-3, atol=1e-5)
    over_model = _over_model(p.model)
    missing = [k for k in _EXPECTED[cls] if k not in over_model]
    assert not missing, (missing, over_model)


@requires_8_devices
def test_tensor_rule_flux1_sampling_matches_no_mesh(tmp_path):
    x_o = jax.random.normal(jax.random.key(2), (1, dim_cond, 2))
    outs = []
    for name, mesh, rule in [("a", None, None), ("b", sh.make_mesh(model_axis=2), "tensor")]:
        p = Flux1FlowPipeline(train_dataset_cond, val_dataset_cond, dim_obs, dim_cond, ch_obs=2, ch_cond=2,
                              params=small_flux1_params(), training_config=test_training_config(tmp_path / name),
                              mesh=mesh, sharding_rule=rule)
        p.ema_model = p.model; p._wrap_model()
        outs.append(np.asarray(p.sample(jax.random.key(1), x_o, nsamples=16, use_ema=False,
                                        show_progress_bars=False)))
    np.testing.assert_allclose(outs[1], outs[0], rtol=1e-4, atol=1e-5)


@requires_8_devices
def test_tensor_rule_indivisible_heads_fall_back(tmp_path):
    params = small_simformer_params()
    params.num_heads = 1  # heads axis of size 1 on a model axis of 2
    with pytest.warns(UserWarning) as rec:
        p = SimformerFlowPipeline(train_dataset_joint, val_dataset_joint, dim_obs, dim_cond, ch_obs=2,
                                  params=params, training_config=test_training_config(tmp_path),
                                  mesh=sh.make_mesh(model_axis=2), sharding_rule="tensor")
    msgs = [str(w.message) for w in rec if "fell back to replicated" in str(w.message)]
    assert len(msgs) == 1, msgs
    assert "attn/query/kernel: dimension 1 not divisible by mesh axis 'model' of size 2" in msgs[0], msgs[0]
    # The fallback parameter is actually replicated, not sharded.
    s = sh.sharding_of(p.model)
    q = [k for k in s if k.endswith("attn/query/kernel")]
    assert q and all("model" not in str(s[k].spec) for k in q), {k: s[k].spec for k in q}
