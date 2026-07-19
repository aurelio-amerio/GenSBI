import os

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import jax.numpy as jnp
import pytest

from gensbi.recipes.utils import healpix_rope_theta


def test_healpix_rope_theta_follows_project_convention():
    # Project convention: theta = 10 * token count (Flux1Params defaults to
    # 10 * (dim_obs + dim_cond) at model.py:184). Full sky has 12*nside^2 tokens.
    assert healpix_rope_theta(2) == 480
    assert healpix_rope_theta(4) == 1920
    assert healpix_rope_theta(4) > healpix_rope_theta(2)
    assert isinstance(healpix_rope_theta(2), int)
