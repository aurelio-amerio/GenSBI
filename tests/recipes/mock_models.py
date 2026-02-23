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
