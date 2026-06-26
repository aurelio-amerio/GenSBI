import jax
import jax.numpy as jnp
from flax import nnx

from gensbi.normalizing_flows.bijections.base import Mask
from gensbi.models.maf.masked_linear import MaskedLinear


def test_masked_linear_zeros_out_masked_connections():
    # 3 inputs -> 2 outputs; only allow input 0 -> output 0
    mask = jnp.array([[True, False],
                      [False, False],
                      [False, False]])  # (in=3, out=2)
    layer = MaskedLinear(3, 2, mask, rngs=nnx.Rngs(0))
    x = jnp.ones((3,))
    y = layer(x)
    # output 1 receives nothing -> equals its bias only; perturbing inputs
    # must not change output 1.
    x2 = x.at[0].set(5.0)
    assert jnp.allclose(y[1], layer(x2)[1])


def test_masked_linear_grad_is_zero_on_masked_weights():
    mask = jnp.array([[True, False],
                      [True, False],
                      [True, False]])  # only output 0 is connected
    layer = MaskedLinear(3, 2, mask, rngs=nnx.Rngs(0))

    def loss(layer, x):
        return layer(x).sum()

    grads = nnx.grad(loss)(layer, jnp.ones((3,)))
    # gradient w.r.t. masked-out kernel entries (column 1) must be exactly zero
    assert jnp.all(grads["linear"]["kernel"][...][:, 1] == 0.0)


def test_mask_is_not_a_param():
    mask = jnp.ones((3, 2), dtype=bool)
    layer = MaskedLinear(3, 2, mask, rngs=nnx.Rngs(0))
    params = nnx.state(layer, nnx.Param)
    # the mask must NOT appear among Params
    flat = jax.tree_util.tree_leaves(params)
    assert all(leaf.dtype != bool for leaf in flat)
    # but it IS reachable as a Mask buffer
    masks = nnx.state(layer, Mask)
    assert len(jax.tree_util.tree_leaves(masks)) == 1
