#%%
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp
from flax import nnx

from gensbi.models.flux1joint.model import Flux1Joint, Flux1JointParams
from gensbi.models.wrappers import JointWrapper

def get_rngs():
	return nnx.Rngs(0)

def get_params():
	return Flux1JointParams(
		in_channels=1,
		vec_in_dim=None,
		mlp_ratio=3.0,
		num_heads=2,
		depth_single_blocks=2,
		axes_dim=[4],
		condition_dim=[2],
		qkv_bias=True,
		rngs=nnx.Rngs(0),
		joint_dim=4,
		theta=16,
		guidance_embed=False,
		param_dtype=jnp.float32
	)

#%%
def test_flux1joint_forward_shape():
	params = get_params()
	model = Flux1Joint(params)
	x = jnp.ones((1, 4, 1))
	t = jnp.ones((1, 1))
	node_ids = jnp.arange(4).reshape(1, -1, 1)
	condition_mask = jnp.zeros((1, 4, 1))
	out = model(t=t, obs=x, node_ids=node_ids, condition_mask=condition_mask)
	assert out.shape == (1, 4, 1), f"Output shape is incorrect, got {out.shape}"

def test_flux1joint_wrapper():
	params = get_params()
	model = Flux1Joint(params)
	wrapper = JointWrapper(model)

	obs = jnp.ones((12, 2, 1))
	cond = jnp.ones((12, 2, 1))
	obs_ids = jnp.arange(2).reshape(1, -1, 1)
	cond_ids = jnp.arange(2).reshape(1, -1, 1)
	t = jnp.ones((12, 1))

	extra_args = {"cond": cond, "cond_ids": cond_ids, "obs_ids": obs_ids, "conditioned": True}

	out = wrapper(
		t=t,
		obs=obs,
		**extra_args,
	)

	assert out.shape == (12, 2, 1), f"1 - Wrapper output shape is incorrect, got {out.shape}"

	vf = wrapper.get_vector_field(**extra_args)
	out = vf(t, obs, None)

	assert out.shape == (12, 2, 1), f"2 - Vector field output shape is incorrect, got {out.shape}"

	vf = wrapper.get_vector_field()
	out = vf(t, obs, args=extra_args)

	assert out.shape == (12, 2, 1), f"3 - Vector field output shape is incorrect, got {out.shape}"