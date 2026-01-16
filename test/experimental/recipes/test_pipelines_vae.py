import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx

import warnings

import pytest

import tempfile

import grain
import numpy as np


import grain


from gensbi.experimental.models.autoencoders import AutoEncoderParams

from gensbi.experimental.recipes import VAE1DPipeline, VAE2DPipeline


df_train_1D = jax.random.normal(jax.random.PRNGKey(0), (1000, 128, 2))
df_val_1D = jax.random.normal(jax.random.PRNGKey(1), (200, 128, 2))

train_dataset_1D = (
    grain.MapDataset.source(np.array(df_train_1D))
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(128)
)

val_dataset_1D = (
    grain.MapDataset.source(np.array(df_val_1D))
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(128)
)

ae_params_1D = AutoEncoderParams(
    resolution=128,
    in_channels=2,
    ch=32,
    out_ch=2,
    ch_mult=[
        1,  # 64
        2,  # 32
        4,  # 16
        8,  # 8
        16,  # 4
    ],
    num_res_blocks=1,
    z_channels=128,
    scale_factor=0.3611,
    shift_factor=0.1159,
    rngs=nnx.Rngs(42),
    param_dtype=jnp.float32,
)

df_train_2D = jax.random.normal(jax.random.PRNGKey(0), (1000, 32, 32, 2))
df_val_2D = jax.random.normal(jax.random.PRNGKey(1), (200, 32, 32, 2))

train_dataset_2D = (
    grain.MapDataset.source(np.array(df_train_2D))
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(128)
)

val_dataset_2D = (
    grain.MapDataset.source(np.array(df_val_2D))
    .shuffle(42)
    .repeat()
    .to_iter_dataset()
    .batch(128)
)


ae_params_2D = AutoEncoderParams(
    resolution=32,
    in_channels=2,
    ch=32,
    out_ch=2,
    ch_mult=[
        1,  # 16
        2,  # 8
        4,  # 4
        8,  # 2
    ],
    num_res_blocks=1,
    z_channels=128,
    scale_factor=0.3611,
    shift_factor=0.1159,
    rngs=nnx.Rngs(42),
    param_dtype=jnp.float32,
)


@pytest.mark.parametrize(
    "pipeline_cls, params, train_dataset, val_dataset",
    [
        (VAE1DPipeline, ae_params_1D, train_dataset_1D, val_dataset_1D),
        (VAE2DPipeline, ae_params_2D, train_dataset_2D, val_dataset_2D),
    ],
)
def test_vae_pipeline(pipeline_cls, params, train_dataset, val_dataset):

    home = os.path.expanduser("~")
    with tempfile.TemporaryDirectory(dir=home) as model_dir:
        training_config = pipeline_cls.get_default_training_config()
        training_config["checkpoint_dir"] = model_dir
        training_config["val_every"] = 1  # validate every epoch

        pipeline = pipeline_cls(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            params=params,
            training_config=training_config,
        )

        # try to train the model
        pipeline.train(nnx.Rngs(0), 2, save_model=True)

        x_in = next(iter(val_dataset))
        key = jax.random.PRNGKey(0)
        x_out = pipeline.model(x_in, key)
        x_out_ema = pipeline.ema_model(x_in, key)

        assert (
            x_out.shape == x_in.shape
        ), "Output shape does not match input shape. Got {} expected {}".format(
            x_out.shape, x_in.shape
        )
        assert (
            x_out_ema.shape == x_in.shape
        ), "Output shape from the ema model does not match input shape. Got {} expected {}".format(
            x_out_ema.shape, x_in.shape
        )

        pipeline2 = pipeline_cls(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            params=params,
            training_config=training_config,
        )

        pipeline2.restore_model()

        out_restored = pipeline2.model(x_in, key)
        out_ema_restored = pipeline2.ema_model(x_in, key)

        assert jnp.allclose(
            x_out, out_restored
        ), "Restored model output does not match original model output."

        assert jnp.allclose(
            x_out_ema, out_ema_restored
        ), "Restored ema model output does not match original ema model output."
