import jax
from jax import numpy as jnp
import numpy as np
from typing import Union, Tuple
from einops import repeat, rearrange
from jax import Array


def init_ids_joint(dim_obs: int, dim_cond: int):
    dim_joint = dim_obs + dim_cond
    node_ids = jnp.arange(dim_joint).reshape((1, -1, 1))
    obs_ids = jnp.arange(dim_obs).reshape((1, -1, 1))  # observation ids
    cond_ids = jnp.arange(dim_obs, dim_joint).reshape((1, -1, 1))  # conditional ids
    return node_ids, obs_ids, cond_ids


def init_ids_1d(dim: int, semantic_id: Union[int, None] = None):
    if semantic_id is None:
        ids = np.zeros((1, dim, 1), dtype=np.int32)
    else:
        ids = np.zeros((1, dim, 2), dtype=np.int32)
        ids[..., 1] = semantic_id

    ids[0, :, 0] = np.arange(dim)

    return jnp.array(ids, dtype=jnp.int32)


def init_ids_2d(dim: Tuple[int, int], semantic_id: int = 0):
    img_ids = np.zeros((dim[0] // 2, dim[1] // 2, 3), dtype=np.int32)
    img_ids[..., 0] = semantic_id
    img_ids[..., 1] = img_ids[..., 1] + np.arange(dim[0] // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + np.arange(dim[1] // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=1)

    return jnp.array(img_ids, dtype=jnp.int32)


@jax.jit
def patchify_2d(x: Array):
    return rearrange(x, "b (h ph) (w pw) c -> b (h w) (c ph pw)", ph=2, pw=2)
