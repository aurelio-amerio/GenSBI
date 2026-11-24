import jax
import jax.numpy as jnp
from jax import Array
from typing import Optional
from flax import nnx


class MockParams:
    def __init__(self):
        self.param_dtype = jnp.float32


class MockConditionalModel(nnx.Module):
    """Mock matching Flux1.__call__ signature (no *args, **kwargs)."""

    def __init__(self):
        super().__init__()
        self.params = MockParams()
        self.dummy = nnx.Param(jnp.zeros(1))

    def __call__(
        self,
        t: Array,
        obs: Array,
        obs_ids: Array,
        cond: Array,
        cond_ids: Array,
        conditioned: bool | Array = True,
        guidance: Array | None = None,
    ) -> Array:
        # Mimic real models: broadcast cond to obs batch dim.
        # This will fail if cond.shape[0] > obs.shape[0], catching
        # batch semantics bugs early (e.g. SDE solvers expect B=1).
        cond = jnp.broadcast_to(cond, (obs.shape[0], *cond.shape[1:]))
        return jnp.zeros_like(obs)


class MockJointModel(nnx.Module):
    """Mock matching Flux1Joint/Simformer.__call__ signature (no *args, **kwargs)."""

    def __init__(self):
        super().__init__()
        self.params = MockParams()
        self.dummy = nnx.Param(jnp.zeros(1))

    def __call__(
        self,
        t: Array,
        obs: Array,
        node_ids: Array,
        condition_mask: Array,
        guidance: Array | None = None,
        edge_mask: Optional[Array] = None,
    ) -> Array:
        return jnp.zeros_like(obs)


class MockUnconditionalModel(nnx.Module):
    """Mock matching the joint interface used via UnconditionalWrapper (no *args, **kwargs)."""

    def __init__(self):
        super().__init__()
        self.params = MockParams()
        self.dummy = nnx.Param(jnp.zeros(1))

    def __call__(
        self,
        t: Array,
        obs: Array,
        node_ids: Array,
        condition_mask: Array = None,
        edge_mask: Optional[Array] = None,
    ) -> Array:
        return jnp.zeros_like(obs)


class Const(nnx.Variable):
    """Non-trainable constant (a buffer), like the index tables of HEAL-SWIN."""


class DropPath(nnx.Module):
    """Stochastic depth: draws from its own RNG stream in train mode."""

    def __init__(self, rate, *, rngs):
        self.rate = rate
        self.deterministic = False
        self.rngs = rngs

    def __call__(self, x):
        if self.deterministic:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        keep = jax.random.bernoulli(self.rngs.dropout(), 1.0 - self.rate, shape)
        return x * keep / (1.0 - self.rate)


class DropoutConditionalModel(nnx.Module):
    """Tiny conditional model with RNG state (dropout, drop-path) and a buffer.

    Unlike the zero-output mocks above, its output depends on the weights, so
    it can tell two sets of weights apart when sampling.
    """

    def __init__(self, seed, ch=2, width=8):
        rngs = nnx.Rngs(seed)
        self.lin_obs = nnx.Linear(ch, width, rngs=rngs)
        self.lin_cond = nnx.Linear(ch, width, rngs=rngs)
        self.lin_t = nnx.Linear(1, width, rngs=rngs)
        self.drop = nnx.Dropout(0.1, rngs=rngs)
        self.hidden = nnx.Linear(width, width, rngs=rngs)
        self.drop_path = DropPath(0.1, rngs=nnx.Rngs(seed + 100))
        self.out = nnx.Linear(width, ch, rngs=rngs)
        self.scale = Const(jnp.full((), 1.0 + seed, dtype=jnp.float32))

    def __call__(
        self,
        t: Array,
        obs: Array,
        obs_ids: Array,
        cond: Array,
        cond_ids: Array,
        conditioned: bool | Array = True,
        guidance: Array | None = None,
    ) -> Array:
        t = jnp.reshape(jnp.asarray(t, obs.dtype), (-1, 1, 1))
        h = self.lin_obs(obs) + self.lin_cond(cond.mean(axis=1, keepdims=True)) + self.lin_t(t)
        h = self.drop(jnp.tanh(h))
        h = h + self.drop_path(jnp.tanh(self.hidden(h)))
        return self.out(h) * self.scale[...]
