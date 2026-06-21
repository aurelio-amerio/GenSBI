import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.normalizing_flows.bijections.made import MADE


def _made(dim=5, cond_dim=3, num_params=2, seed=0):
    # zero_init=False so the output layer is live (non-identity), which is what
    # makes the autoregression and cond-dependence assertions meaningful.
    return MADE(dim=dim, cond_dim=cond_dim, num_params=num_params,
                nn_width=32, nn_depth=2, zero_init=False, rngs=nnx.Rngs(seed))


def test_made_output_shape():
    made = _made()
    x = jnp.linspace(-1, 1, 5)
    cond = jnp.array([0.1, -0.2, 0.3])
    out = made(x, cond)
    assert out.shape == (5, 2)  # (dim, num_params)


def test_made_is_autoregressive():
    """Output dim d must have ZERO Jacobian w.r.t. inputs x_j for j >= d."""
    made = _made(dim=5, num_params=2)
    cond = jnp.array([0.1, -0.2, 0.3])

    def out_flat(x):
        return made(x, cond).reshape(-1)   # (dim*num_params,)

    J = jax.jacobian(out_flat)(jnp.linspace(-1, 1, 5))  # (dim*np, dim)
    J = J.reshape(5, 2, 5)  # (out_dim, param, in_dim)
    for d in range(5):
        for j in range(d, 5):           # j >= d must be zero (strict autoregression)
            assert jnp.allclose(J[d, :, j], 0.0, atol=1e-6), (d, j)
        if d > 0:                        # must actually use the allowed prefix
            assert not jnp.allclose(J[d, :, :d], 0.0, atol=1e-6), d


def test_made_depends_on_cond_densely():
    """Every output must depend on the conditioning vector (FiLM is live)."""
    made = _made(dim=5, cond_dim=3, num_params=2)
    x = jnp.linspace(-1, 1, 5)

    def out_flat(cond):
        return made(x, cond).reshape(-1)

    J = jax.jacobian(out_flat)(jnp.array([0.1, -0.2, 0.3]))  # (dim*np, cond_dim)
    # no output row is entirely independent of cond
    assert jnp.all(jnp.any(jnp.abs(J) > 1e-6, axis=1))
