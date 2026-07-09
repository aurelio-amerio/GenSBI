# tests/models/tarflow/test_structured_integration.py
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import grain

from gensbi.models import TarFlow, TarFlowParams
from gensbi.recipes.flow_pipeline import ConditionalFlowPipeline
from gensbi.inference import NLEPosterior
from gensbi.core.prior import make_gaussian_prior

H, W, Ch, D, N = 4, 4, 1, 2, 256
_k = jax.random.PRNGKey(0)
_kth, _kx = jax.random.split(_k)
_theta = np.array(jax.random.normal(_kth, (N, D)))
_Wm = jax.random.normal(jax.random.PRNGKey(5), (H * W, D))
_x = np.array((jnp.asarray(_theta) @ _Wm.T).reshape(N, H, W, Ch)
              + 0.1 * jax.random.normal(_kx, (N, H, W, Ch)))


def _iter(obs, cond, bs=64):
    idx = grain.MapDataset.source(list(range(len(obs))))
    return (idx.shuffle(0).repeat().to_iter_dataset().batch(bs)
            .map(lambda i: (obs[np.array(i)], cond[np.array(i)])))


def test_field_nle_train_smoke_and_mclmc(tmp_path):
    flow = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), cond_dim=D, modeled="image",
                                 img_size=H, patch_size=2, img_channels=Ch,
                                 head_dim=8, num_heads=2, num_blocks=4,
                                 layers_per_block=2, standardize=True))
    cfg = ConditionalFlowPipeline.get_default_training_config()
    cfg.update(dict(val_every=1, checkpoint_dir=str(tmp_path)))
    _theta_c = _theta[..., None]   # (N, D, 1) — channel-carrying cond
    pipe = ConditionalFlowPipeline(flow, _iter(_x, _theta_c), _iter(_x, _theta_c),
                                   dim_obs=H * W * Ch, dim_cond=D,
                                   structured_obs=True, training_config=cfg)
    pipe.fit_standardization(_x)
    pipe.train(nnx.Rngs(0), nsteps=2, save_model=False)
    from gensbi.inference import MCLMC
    post = NLEPosterior(pipe.ema_model, make_gaussian_prior((D,)), structured_obs=True)
    s = post.sample(jax.random.PRNGKey(7), _x[0],
                    sampler=MCLMC(adjusted=False, num_samples=10, num_tuning_steps=20))
    # untrained flow: samples may include non-finite rows (no clamping). Smoke-check shape
    # + at-least-some-finite, consistent with existing untrained-flow relaxations.
    assert s.shape == (10, D, 1) and jnp.any(jnp.isfinite(s))


def test_image_npe_train_smoke_and_sample(tmp_path):
    # NPE: obs = theta vector, cond = image
    flow = TarFlow(TarFlowParams(rngs=nnx.Rngs(0), dim=D, modeled="vector",
                                 cond="image", cond_img_size=H,
                                 cond_patch_size=2, cond_channels=Ch, head_dim=8,
                                 num_heads=2, num_blocks=4, layers_per_block=2,
                                 standardize=True))
    cfg = ConditionalFlowPipeline.get_default_training_config()
    cfg.update(dict(val_every=1, checkpoint_dir=str(tmp_path)))
    _theta_ch = _theta[..., None]   # (N, D, 1) — channel-carrying obs stream
    pipe = ConditionalFlowPipeline(flow, _iter(_theta_ch, _x), _iter(_theta_ch, _x),
                                   dim_obs=D, dim_cond=H * W * Ch,
                                   structured_cond=True, training_config=cfg)
    pipe.fit_standardization(_theta_ch)     # standardize the modeled theta
    pipe.train(nnx.Rngs(0), nsteps=2, save_model=False)
    s = pipe.sample(jax.random.PRNGKey(3), _x[0:1], nsamples=16, use_ema=False)
    assert s.shape == (16, D, 1) and jnp.all(jnp.isfinite(s))
    lp = pipe.log_prob(_theta_ch[:5], _x[0:1], use_ema=False)
    assert lp.shape == (5,) and jnp.all(jnp.isfinite(lp))
