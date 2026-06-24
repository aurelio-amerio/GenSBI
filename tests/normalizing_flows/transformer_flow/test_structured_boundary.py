# tests/normalizing_flows/transformer_flow/test_structured_boundary.py
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import grain

from gensbi.normalizing_flows import make_tarflow
from gensbi.recipes.flow_pipeline import ConditionalFlowPipeline
from gensbi.inference import NLEPosterior
from gensbi.core.prior import make_gaussian_prior

# Field NLE: obs = 4x4x1 image x, cond = 2-vector theta
H, W, Ch, D, N = 4, 4, 1, 2, 256
_k = jax.random.PRNGKey(0)
_kth, _kx = jax.random.split(_k)
_theta = jax.random.normal(_kth, (N, D))
# x_image[:, i, j, 0] = linear(theta) + noise
_W = jax.random.normal(jax.random.PRNGKey(5), (H * W, D))
_x = (_theta @ _W.T).reshape(N, H, W, Ch) + 0.1 * jax.random.normal(_kx, (N, H, W, Ch))


def _ds_field(bs=64):
    x, theta = np.array(_x), np.array(_theta)        # (obs=image, cond=theta)
    idx = grain.MapDataset.source(list(range(N)))
    return (idx.shuffle(0).repeat().to_iter_dataset().batch(bs)
            .map(lambda i: (x[np.array(i)], theta[np.array(i)])))


def _field_pipe(tmp_path):
    flow = make_tarflow(nnx.Rngs(0), cond_dim=D, modeled="image", img_size=H,
                        patch_size=2, img_channels=Ch, channels=16, num_blocks=4,
                        layers_per_block=2, head_dim=8, standardize=True)
    cfg = ConditionalFlowPipeline.get_default_training_config()
    cfg.update(dict(val_every=1, checkpoint_dir=str(tmp_path)))
    return ConditionalFlowPipeline(flow, _ds_field(), _ds_field(),
                                   dim_obs=H * W * Ch, dim_cond=D,
                                   structured_obs=True, structured_cond=False,
                                   training_config=cfg)


def test_field_loss_finite_and_grads(tmp_path):
    pipe = _field_pipe(tmp_path)
    loss_fn = pipe.get_loss_fn()
    obs = jnp.asarray(_x[:32])           # (32, H, W, Ch)
    cond = jnp.asarray(_theta[:32])      # (32, D)
    loss = loss_fn(pipe.model, (obs, cond), key=jax.random.PRNGKey(0))
    assert loss.shape == () and jnp.isfinite(loss)
    grads = nnx.grad(loss_fn)(pipe.model, (obs, cond), jax.random.PRNGKey(0))
    leaves = jax.tree_util.tree_leaves(grads)
    assert any(jnp.any(jnp.abs(g) > 0) for g in leaves)


def test_field_fit_standardization_image_shape(tmp_path):
    pipe = _field_pipe(tmp_path)
    pipe.fit_standardization(_x)         # (N, H, W, Ch)
    assert pipe.model.mean[...].shape == (H, W, Ch)
    assert pipe._standardized is True


def test_field_nle_potential_structured_xo(tmp_path):
    # zero_init=True: identity flow, finite log_prob for untrained weights on CPU
    flow = make_tarflow(nnx.Rngs(0), cond_dim=D, modeled="image", img_size=H,
                        patch_size=2, img_channels=Ch, channels=16, num_blocks=3,
                        layers_per_block=1, head_dim=8, zero_init=True)
    prior = make_gaussian_prior((D,))
    post = NLEPosterior(flow, prior, structured_obs=True)
    x_o = jnp.zeros((H, W, Ch))
    U = post.potential(x_o)
    theta = jnp.array([0.1, 0.2])
    val = U(theta)
    grad = jax.grad(U)(theta)
    assert val.shape == () and jnp.isfinite(val)
    assert grad.shape == (D,) and jnp.all(jnp.isfinite(grad))
