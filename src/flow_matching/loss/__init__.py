# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

# from .generalized_loss import MixturePathGeneralizedKL

from .continuous_loss import ContinuousFMLoss

__all__ = [
    # "MixturePathGeneralizedKL",
    "ContinuousFMLoss",
]
