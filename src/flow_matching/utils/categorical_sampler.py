#FIXME: first pass

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import jax
import jax.numpy as jnp
from jax import Array


def categorical(key, probs: Array) -> Array:
    r"""Categorical sampler according to weights in the last dimension of ``probs`` using JAX's random.categorical.

    Args:
        probs (Array): probabilities.

    Returns:
        Array: Samples.
    """
    flat_probs = jnp.reshape(probs, (-1, probs.shape[-1]))
    samples = jax.random.categorical(key, flat_probs)
    return jnp.reshape(samples, probs.shape[:-1])
