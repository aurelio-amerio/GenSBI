# :no-index:
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from .utils import expand_tensor_like, unsqueeze_to_match

__all__ = [
    "unsqueeze_to_match",
    "expand_tensor_like",
]
