"""Safetensors export / import through a real pipeline.

The model holds RNG state (dropout, drop-path) and a buffer, and the target
pipeline is built from scratch without data loaders: the "load weights and
sample" path of a published model.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from safetensors import safe_open

from gensbi.core import FlowMatchingMethod
from gensbi.recipes import ConditionalPipeline

from tests.recipes.mock_models import DropoutConditionalModel
from tests.recipes.model_test_helpers import (
    dim_cond,
    dim_obs,
    train_dataset_cond,
    val_dataset_cond,
    x,
)

x_o = x[:1]
NSAMPLES = 8


def _pipeline(checkpoint_dir, seed, *, loaders):
    cfg = ConditionalPipeline.get_default_training_config()
    cfg.update(nsteps=20, val_every=5, warmup_steps=2, early_stopping=False,
               checkpoint_dir=str(checkpoint_dir))
    return ConditionalPipeline(
        DropoutConditionalModel(seed),
        train_dataset_cond if loaders else None,
        val_dataset_cond if loaders else None,
        dim_obs,
        dim_cond,
        method=FlowMatchingMethod(),
        ch_obs=2,
        ch_cond=2,
        training_config=cfg,
    )


def _sample(pipeline, use_ema):
    out = pipeline.sample(jax.random.PRNGKey(3), x_o, nsamples=NSAMPLES, use_ema=use_ema)
    return np.asarray(out)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    """A briefly trained pipeline, its two weight files and its samples."""
    root = tmp_path_factory.mktemp("safetensors_pipeline")
    src = _pipeline(root / "src", seed=0, loaders=True)
    src.train(nnx.Rngs(0), save_model=False)
    paths = {True: root / "ema.safetensors", False: root / "final.safetensors"}
    src.export_safetensors(paths[True])
    src.export_safetensors(paths[False], ema=False)
    samples = {use_ema: _sample(src, use_ema) for use_ema in (True, False)}
    # non-vacuous: the two weight sets sample differently
    assert not np.array_equal(samples[True], samples[False])
    return root, paths, samples


def test_fresh_pipeline_samples_like_source_after_import(exported):
    root, paths, samples = exported
    dst = _pipeline(root / "dst", seed=1, loaders=False)
    dst.import_safetensors(paths[True])
    dst.import_safetensors(paths[False], ema=False)

    assert _sample(dst, True).shape == (NSAMPLES, dim_obs, 2)
    np.testing.assert_array_equal(_sample(dst, True), samples[True])
    np.testing.assert_array_equal(_sample(dst, False), samples[False])


def test_import_leaves_models_in_eval_mode(exported):
    root, paths, _ = exported
    dst = _pipeline(root / "dst_eval", seed=1, loaders=False)
    assert not dst.ema_model.drop.deterministic  # fresh models are in train mode
    dst.import_safetensors(paths[True])
    for model in (dst.model, dst.ema_model):
        assert model.drop.deterministic
        assert model.drop_path.deterministic


def test_import_wraps_models(exported):
    root, paths, _ = exported
    dst = _pipeline(root / "dst_wrap", seed=1, loaders=False)
    assert dst.ema_model_wrapped is None
    dst.import_safetensors(paths[True])
    assert dst.ema_model_wrapped is not None
    assert dst.model_wrapped is not None


def test_export_and_import_forward_wrt(exported, tmp_path):
    root, paths, _ = exported
    src = _pipeline(root / "src_wrt", seed=0, loaders=False)
    path = tmp_path / "params.safetensors"
    src.export_safetensors(path, wrt=nnx.Param)
    with safe_open(str(path), framework="flax") as f:
        assert "scale" not in f.keys()  # the buffer is not a Param

    dst = _pipeline(root / "dst_wrt", seed=1, loaders=False)
    dst.import_safetensors(path, wrt=nnx.Param)  # strict=True
    assert bool(jnp.array_equal(dst.ema_model.out.kernel[...], src.ema_model.out.kernel[...]))
    assert float(dst.ema_model.scale[...]) == 2.0  # buffer untouched (seed 1)


def test_train_without_loaders_raises(tmp_path):
    p = _pipeline(tmp_path / "ckpt", seed=0, loaders=False)
    with pytest.raises(ValueError, match="train_dataset"):
        p.train(nnx.Rngs(0), save_model=False)
