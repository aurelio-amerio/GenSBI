#FIXME: not implemented yet

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import jax
import jax.numpy as jnp
from jax import random
from flax import struct
from typing import Any

from flow_matching.path.path import ProbPath
from flow_matching.path.path_sample import DiscretePathSample
from flow_matching.path.scheduler import ConvexScheduler
from flow_matching.utils import expand_tensor_like, unsqueeze_to_match


class MixtureDiscreteProbPath(ProbPath):
    """The MixtureDiscreteProbPath class defines a factorized discrete probability path.
    
    This path remains constant at the source data point :math:`X_0` until a random time, determined by the scheduler, when it flips to the target data point :math:`X_1`.
    The scheduler determines the flip probability using the parameter :math:`\sigma_t`, which is a function of time `t`. Specifically, :math:`\sigma_t` represents the probability of remaining at :math:`X_0`, while :math:`1 - \sigma_t` is the probability of flipping to :math:`X_1`:

    """
    
    def __init__(self, scheduler: ConvexScheduler):
        assert isinstance(
            scheduler, ConvexScheduler
        ), "Scheduler for ConvexProbPath must be a ConvexScheduler."
        self.scheduler = scheduler

    def sample(self, x_0: jnp.ndarray, x_1: jnp.ndarray, t: jnp.ndarray, 
               key: random.PRNGKey) -> DiscretePathSample:
        """Sample from the affine probability path.
        
        Args:
            x_0 (jnp.ndarray): source data point, shape (batch_size, ...).
            x_1 (jnp.ndarray): target data point, shape (batch_size, ...).
            t (jnp.ndarray): times in [0,1], shape (batch_size).
            key (PRNGKey): JAX random key for sampling.

        Returns:
            DiscretePathSample: a conditional sample at X_t ~ p_t.
        """
        self.assert_sample_shape(x_0=x_0, x_1=x_1, t=t)

        sigma_t = self.scheduler(t).sigma_t
        sigma_t = expand_tensor_like(input_tensor=sigma_t, expand_to=x_1)

        # Generate random values using JAX
        random_values = random.uniform(key, shape=x_1.shape)
        source_indices = random_values < sigma_t
        x_t = jnp.where(source_indices, x_0, x_1)

        return DiscretePathSample(x_t=x_t, x_1=x_1, x_0=x_0, t=t)

    def posterior_to_velocity(
        self, posterior_logits: jnp.ndarray, x_t: jnp.ndarray, t: jnp.ndarray
    ) -> jnp.ndarray:
        """Convert the factorized posterior to velocity.
        
        Args:
            posterior_logits (jnp.ndarray): logits of the x_1 posterior conditional on x_t, 
                                          shape (..., vocab size).
            x_t (jnp.ndarray): path sample at time t, shape (...).
            t (jnp.ndarray): time in [0,1].

        Returns:
            jnp.ndarray: velocity.
        """
        # Use JAX's softmax
        posterior = jax.nn.softmax(posterior_logits, axis=-1)
        vocabulary_size = posterior.shape[-1]
        
        # One-hot encoding in JAX
        x_t = jax.nn.one_hot(x_t, vocabulary_size)
        t = unsqueeze_to_match(source=t, target=x_t)

        scheduler_output = self.scheduler(t)

        kappa_t = scheduler_output.alpha_t
        d_kappa_t = scheduler_output.d_alpha_t

        return (d_kappa_t / (1 - kappa_t)) * (posterior - x_t)