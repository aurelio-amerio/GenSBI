import jax.numpy as jnp
from flax import nnx


class MockParams:
    def __init__(self):
        self.param_dtype = jnp.float32


class MockConditionalModel(nnx.Module):
    def __init__(self):
        super().__init__()
        self.params = MockParams()
        self.dummy = nnx.Param(jnp.zeros(1))

    def __call__(self, t, obs, obs_ids, cond, cond_ids, *args, **kwargs):
        # Mimic real models: broadcast cond to obs batch dim.
        # This will fail if cond.shape[0] > obs.shape[0], catching
        # batch semantics bugs early (e.g. SDE solvers expect B=1).
        cond = jnp.broadcast_to(cond, (obs.shape[0], *cond.shape[1:]))
        return jnp.zeros_like(obs)


class MockJointModel(nnx.Module):
    def __init__(self):
        super().__init__()
        self.params = MockParams()
        self.dummy = nnx.Param(jnp.zeros(1))

    def __call__(self, t, obs, node_ids, condition_mask, *args, **kwargs):
        return jnp.zeros_like(obs)


class MockUnconditionalModel(nnx.Module):
    def __init__(self):
        super().__init__()
        self.params = MockParams()
        self.dummy = nnx.Param(jnp.zeros(1))

    def __call__(self, t, obs, *args, **kwargs):
        return jnp.zeros_like(obs)
