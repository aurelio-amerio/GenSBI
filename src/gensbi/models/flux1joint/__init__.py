"""
Flux1Joint Model in GenSBI
-------------------------

This package provides the Flux1Joint transformer-based model and related loss functions for simulation-based inference. The architecture is derived from the following foundational work:

* M. Gloeckler et al. "All-in-one simulation-based inference." `arXiv:2404.09636 <https://arxiv.org/abs/2404.09636>`_
* `mackelab/simformer <https://github.com/mackelab/simformer>`_
"""

# This file is a derivative work based on the Simformer architecture from the "All-in-one simulation-based inference" paper.
# Substantial modifications and extensions by Aurelio Amerio, 2025.
# If you use this package, please consider citing the original Simformer paper.

from .model import Flux1Joint, Flux1JointParams

__all__ = [
    "Flux1Joint",
    "Flux1JointParams",
]
