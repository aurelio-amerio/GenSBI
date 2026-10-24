"""Tests for ``gensbi.sharding`` (logical axes, rules, mesh and placement helpers)."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

from gensbi import sharding as sh
from gensbi.normalizing_flows.bijections.base import Mask

from .helpers import has_healswin_sharding, requires_8_devices


class Toy(nnx.Module):
    def __init__(self, rngs):
        self.l1 = nnx.Linear(
            16, 32, rngs=rngs,
            kernel_metadata=sh.axes("embed", "mlp"), bias_metadata=sh.axes("mlp"),
        )
        self.l2 = nnx.Linear(
            32, 16, rngs=rngs,
            kernel_metadata=sh.axes("mlp", "embed"), bias_metadata=sh.axes("embed"),
        )
        self.odd = nnx.Param(jnp.ones((6, 3)), **sh.axes("embed", None))  # 6 % 8 != 0 -> fsdp fallback
        self.bad = nnx.Param(jnp.ones((4,)), **sh.axes("nonsense"))  # unknown name -> fallback
        self.buf = Mask(jnp.arange(5))
        self.drop = nnx.Dropout(0.1, rngs=rngs)


def test_rule_tables_cover_every_logical_name():
    for rule, table in sh.RULES.items():
        assert tuple(n for n, _ in table) == sh.LOGICAL_AXES, rule
    assert sh.RULES["replicated"] == tuple((n, None) for n in sh.LOGICAL_AXES)


def test_axes_and_partitioned_set_the_opt_out():
    assert sh.axes("embed", None) == {"out_sharding": ("embed", None), "eager_sharding": False}
    m = Toy(nnx.Rngs(0))  # must construct with NO mesh anywhere
    assert isinstance(m.l1.kernel[...].sharding, jax.sharding.SingleDeviceSharding)
    assert m.l1.kernel.get_metadata()["eager_sharding"] is False


def test_vocabulary_pinned_to_heal_swin():
    if not has_healswin_sharding():
        pytest.skip("installed heal_swin_nnx predates its sharding module")
    from heal_swin_nnx import sharding as hs

    assert hs.LOGICAL_AXES == sh.LOGICAL_AXES and hs.RULES == sh.RULES and hs.MODEL_AXIS == sh.MODEL_AXIS


@requires_8_devices
def test_make_mesh_shape_and_axis_types():
    mesh = sh.make_mesh()
    assert mesh.axis_names == ("data", "model") and dict(mesh.shape) == {"data": 8, "model": 1}
    assert all(t == AxisType.Auto for t in mesh.axis_types)
    mesh2 = sh.make_mesh(model_axis=2)
    assert dict(mesh2.shape) == {"data": 4, "model": 2}
    assert sh.batch_axes(mesh2) == ("data",) and sh.batch_size_divisor(mesh2) == 4
    with pytest.raises(ValueError):
        sh.make_mesh(model_axis=3)


@requires_8_devices
def test_shard_model_replicated_places_everything():
    m = Toy(nnx.Rngs(0))
    mesh = sh.make_mesh()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # replicated must not warn
        sh.shard_model(m, mesh, "replicated")
    for path, s in sh.sharding_of(m).items():
        assert isinstance(s, NamedSharding) and s.mesh == mesh, path
        assert s.is_equivalent_to(NamedSharding(mesh, P()), 1), path  # includes buf and dropout RngKey/RngCount


@requires_8_devices
def test_shard_model_fsdp_shards_embed_and_falls_back():
    m = Toy(nnx.Rngs(0))
    mesh = sh.make_mesh()
    with pytest.warns(UserWarning) as rec:
        sh.shard_model(m, mesh, "fsdp")
    msgs = " ".join(str(w.message) for w in rec)
    assert "odd" in msgs and "bad" in msgs
    assert "experimental" in msgs
    s = sh.sharding_of(m)
    assert s["l1/kernel"].is_equivalent_to(NamedSharding(mesh, P("data", None)), 2)
    assert s["l2/kernel"].is_equivalent_to(NamedSharding(mesh, P(None, "data")), 2)
    assert s["l2/bias"].is_equivalent_to(NamedSharding(mesh, P("data")), 1)
    assert s["odd"].is_equivalent_to(NamedSharding(mesh, P()), 2)
    assert s["bad"].is_equivalent_to(NamedSharding(mesh, P()), 1)
    with pytest.warns(UserWarning, match="experimental"):  # emitted on every call
        sh.shard_model(m, mesh, "fsdp")  # idempotent: second call is a no-op (no error)


@requires_8_devices
def test_shard_model_warns_on_unannotated_param():
    class M(nnx.Module):
        def __init__(self):
            self.w = nnx.Param(jnp.ones((8, 8)), **sh.axes("embed", None))
            self.naked = nnx.Param(jnp.ones((8,)))

    mesh = sh.make_mesh()
    with pytest.warns(UserWarning) as rec:
        sh.shard_model(M(), mesh, "fsdp")
    msgs = " ".join(str(w.message) for w in rec)
    assert "naked: no annotation" in msgs and "w:" not in msgs


@requires_8_devices
def test_shard_model_tensor_shards_mlp_over_model_axis():
    m = Toy(nnx.Rngs(0))
    mesh = sh.make_mesh(model_axis=2)
    with pytest.warns(UserWarning, match="experimental"):
        sh.shard_model(m, mesh, "tensor")
    assert sh.sharding_of(m)["l1/kernel"].is_equivalent_to(NamedSharding(mesh, P(None, "model")), 2)


def test_shard_model_rejects_unknown_rule():
    with pytest.raises(ValueError):
        sh.shard_model(Toy(nnx.Rngs(0)), sh.make_mesh(), "nope")


@requires_8_devices
def test_shard_model_accepts_optimizer():
    m = Toy(nnx.Rngs(0))
    import optax

    opt = nnx.Optimizer(m, optax.adam(1e-3), wrt=nnx.Param)
    mesh = sh.make_mesh()
    sh.shard_model(opt, mesh, "replicated")
    for path, s in sh.sharding_of(opt).items():
        assert s.is_equivalent_to(NamedSharding(mesh, P()), 1), path


@requires_8_devices
def test_shard_batch_numpy_jax_tuple_and_dict_leaves():
    mesh = sh.make_mesh()
    x = np.zeros((16, 5, 1), np.float32)
    y = jnp.zeros((16,), jnp.int32)
    out = sh.shard_batch({"image": x, "label": y, "meta": 3}, mesh)
    assert out["image"].sharding.is_equivalent_to(NamedSharding(mesh, P("data")), 3)
    assert out["label"].sharding.is_equivalent_to(NamedSharding(mesh, P("data")), 1)
    assert out["meta"] == 3
    a, b = sh.shard_batch((x, y), mesh)
    assert a.shape == (16, 5, 1) and b.shape == (16,)
    for r4 in (np.zeros((16, 2, 3, 4), np.float32), jnp.zeros((16, 2, 3, 4))):
        z = sh.shard_batch(r4, mesh)
        assert z.sharding.is_equivalent_to(NamedSharding(mesh, P("data")), 4)
        assert z.shape == (16, 2, 3, 4)


@requires_8_devices
def test_check_batch_divisible_message():
    mesh = sh.make_mesh()
    sh.check_batch_divisible((np.zeros((16, 4)),), mesh, "train")  # ok
    with pytest.raises(ValueError, match=r"train batch size 12 is not divisible by the data-parallel device count 8"):
        sh.check_batch_divisible((np.zeros((12, 4)),), mesh, "train")
    with pytest.raises(ValueError, match="differing leading sizes"):
        sh.check_batch_divisible((np.zeros((16, 4)), np.zeros((8,))), mesh, "train")


@requires_8_devices
def test_resolve_mesh_forms():
    assert sh.resolve_mesh(None) is None
    assert dict(sh.resolve_mesh("auto").shape) == {"data": 8, "model": 1}
    assert dict(sh.resolve_mesh({"model_axis": 2}).shape) == {"data": 4, "model": 2}
    m = sh.make_mesh()
    assert sh.resolve_mesh(m) is m
    with pytest.raises(ValueError):
        sh.resolve_mesh("eight")


@requires_8_devices
def test_pad_for_mesh():
    mesh = sh.make_mesh()
    assert sh.pad_for_mesh(16, mesh) == (16, 16)
    assert sh.pad_for_mesh(3, mesh) == (8, 3)
    assert sh.pad_for_mesh(9, mesh) == (16, 9)


@requires_8_devices
def test_shard_batch_jax_input_already_on_device():
    mesh = sh.make_mesh()
    x = jnp.ones((16, 4, 1))  # committed to device 0 as in notebook .map(split_data)
    y = sh.shard_batch(x, mesh)
    assert y.sharding.is_equivalent_to(NamedSharding(mesh, P("data")), 3)
    np.testing.assert_array_equal(np.asarray(y), np.asarray(x))


@requires_8_devices
def test_shard_sample_init():
    mesh = sh.make_mesh()
    y = sh.shard_sample_init(jnp.ones((16, 4, 1)), mesh)
    assert y.sharding.is_equivalent_to(NamedSharding(mesh, P("data")), 3)


def test_mesh_context_is_nullcontext_without_mesh():
    with sh.mesh_context(None):
        assert jax.sharding.get_mesh().empty
