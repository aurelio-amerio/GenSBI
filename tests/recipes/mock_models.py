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
