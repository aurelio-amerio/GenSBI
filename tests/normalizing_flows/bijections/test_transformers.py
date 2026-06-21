import jax.numpy as jnp
from gensbi.normalizing_flows.bijections.transformers import Affine


def test_affine_num_params():
    assert Affine().num_params == 2


def test_affine_roundtrip_and_logdet_signs():
    t = Affine()
    x = jnp.array([0.3, -1.2, 2.0])
    # params: (dim, 2) = (shift mu, log-scale a)
    params = jnp.array([[0.5, 0.1], [-0.2, -0.3], [1.0, 0.2]])
    u, logdet_inv = t.inverse(x, params)   # data -> noise
    x2, logdet_fwd = t.forward(u, params)  # noise -> data
    assert jnp.allclose(x, x2, atol=1e-6)
    # inverse logdet = -sum(a); forward logdet = +sum(a)
    a = params[:, 1]
    assert jnp.allclose(logdet_inv, -jnp.sum(a))
    assert jnp.allclose(logdet_fwd, jnp.sum(a))


def test_affine_clamps_log_scale():
    t = Affine(clamp_min=-5.0, clamp_max=3.0)
    x = jnp.array([1.0])
    params = jnp.array([[0.0, 100.0]])   # absurd log-scale
    u, logdet = t.inverse(x, params)
    # effective log-scale clamped to 3.0 -> logdet = -3.0
    assert jnp.allclose(logdet, -3.0)
