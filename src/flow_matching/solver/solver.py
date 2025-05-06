# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC, abstractmethod

from jax import Array


class Solver(ABC):
    """Abstract base class for solvers."""

    @abstractmethod
    def sample(self, x_0: Array = None) -> Array:
        ...
