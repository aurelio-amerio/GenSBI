import jax
import pytest
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from gensbi import sharding as sh
from gensbi.models import Flux1, Flux1Joint

from .helpers import (
    assert_every_param_annotated,
    has_healswin_sharding,
    requires_8_devices,
    small_flux1_params,
    small_flux1joint_params,
)


def test_flux1_every_param_annotated():
    m = Flux1(small_flux1_params())
    assert_every_param_annotated(m)
    k = m.obs_in.kernel
    assert isinstance(k[...].sharding, jax.sharding.SingleDeviceSharding)  # no mesh anywhere


def test_flux1joint_every_param_annotated():
    assert_every_param_annotated(Flux1Joint(small_flux1joint_params()))


def test_flux1joint_concat_every_param_annotated():
    import dataclasses

    params = dataclasses.replace(small_flux1joint_params(), id_merge_mode="concat", rngs=nnx.Rngs(0))
    assert_every_param_annotated(Flux1Joint(params))


def test_no_mesh_optimizer_ema_eval_shape():
    # nnx.get_abstract_model(init_fn, mesh) requires a mesh, so it is not exercised here.
    import optax

    from gensbi.recipes.pipeline import ModelEMA

    m = Flux1(small_flux1_params())
    nnx.Optimizer(m, optax.adamw(1e-3), wrt=nnx.Param)  # OptVariables copy eager_sharding=False
    ModelEMA(nnx.clone(m), optax.ema(0.999))
    p = small_flux1_params()

    def build():
        p.rngs = nnx.Rngs(0)  # fresh Rngs inside the trace; params are built outside it
        return Flux1(p)

    nnx.eval_shape(build)
    assert jax.sharding.get_mesh().empty


def test_building_block_composes_into_user_module_without_mesh():
    from gensbi.models.embedding import MLPEmbedder

    class Mine(nnx.Module):
        def __init__(self, rngs):
            self.emb = MLPEmbedder(in_dim=4, hidden_dim=16, rngs=rngs)
            self.head = nnx.Linear(16, 1, rngs=rngs)

    m = Mine(nnx.Rngs(0))
    assert isinstance(m.emb.in_layer.kernel[...].sharding, jax.sharding.SingleDeviceSharding)


def test_embed_wrapper_default_and_override():
    from gensbi.models.embedding.embedding import Embed

    e = Embed(num_embeddings=8, features=16, rngs=nnx.Rngs(0))
    assert e.embed.embedding.get_metadata()["out_sharding"] == ("vocab", "embed")
    e2 = Embed(num_embeddings=8, features=16, rngs=nnx.Rngs(0), embedding_metadata=sh.axes(None, "embed"))
    assert e2.embed.embedding.get_metadata()["out_sharding"] == (None, "embed")


@requires_8_devices
def test_shard_model_replicated_on_flux1_places_every_leaf():
    m = Flux1(small_flux1_params())
    mesh = sh.make_mesh()
    sh.shard_model(m, mesh, "replicated")
    for path, s in sh.sharding_of(m).items():
        assert isinstance(s, NamedSharding) and s.is_equivalent_to(NamedSharding(mesh, P()), 1), path


def test_simformer_every_param_annotated():
    from gensbi.models import Simformer

    from .helpers import small_simformer_params

    m = Simformer(small_simformer_params())
    assert_every_param_annotated(m)
    # transformer.rngs is part of the model state; RNG variables are not annotated.
    rng_vars = nnx.to_flat_state(nnx.state(m, nnx.Any(nnx.RngKey, nnx.RngCount)))
    assert len(rng_vars) > 0
    for path, var in rng_vars:
        assert "out_sharding" not in var.get_metadata(), path


@requires_8_devices
def test_shard_model_replicated_on_simformer_places_rng_state():
    from gensbi.models import Simformer

    from .helpers import small_simformer_params

    m = Simformer(small_simformer_params())
    mesh = sh.make_mesh()
    sh.shard_model(m, mesh, "replicated")
    shardings = sh.sharding_of(m)
    assert any("rngs" in str(p) for p in shardings)
    for path, s in shardings.items():
        assert isinstance(s, NamedSharding) and s.is_equivalent_to(NamedSharding(mesh, P()), 1), path


def test_tarflow_every_param_annotated():
    from gensbi.models import TarFlow, TarFlowParams

    assert_every_param_annotated(
        TarFlow(TarFlowParams(rngs=nnx.Rngs(0), dim=4, cond_dim=2, num_blocks=2, num_heads=2, head_dim=8))
    )


def test_maf_every_param_annotated():
    from gensbi.models import MAFlow, MAFlowParams

    assert_every_param_annotated(
        MAFlow(MAFlowParams(rngs=nnx.Rngs(0), dim=3, cond_dim=2, n_layers=2, nn_width=16))
    )


def test_healswin_encoder_annotated_via_gensbi():
    if not has_healswin_sharding():
        pytest.skip("installed heal_swin_nnx predates its sharding module")
    from gensbi.models import HealSwinEncoder, HealSwinParams

    p = HealSwinParams(
        nside=16, in_channels=1, out_channels=1, base_pixels=tuple(range(8)),
        embed_dim=16, depths=(2, 2), num_heads=(2, 4), drop_path_rate=0.0,
    )
    assert_every_param_annotated(HealSwinEncoder(p, rngs=nnx.Rngs(0)))
