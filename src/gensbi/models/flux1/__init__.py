"""
Flux1 Model in GenSBI
---------------------

This package provides the Flux1 transformer-based model and related loss functions for flow matching and simulation-based inference. The implementation is derived and adapted from the jflux library and inspired by the following foundational work:

* Black Forest Labs et al. "FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space." `arXiv:2506.15742 <https://arxiv.org/abs/2506.15742>`_
* `ml-gde/jflux <https://github.com/ml-gde/jflux>`_
"""

# This file is a derivative work based on the jflux library (https://github.com/ml-gde/jflux).
# Substantial modifications and extensions by Aurelio Amerio, 2025.
# If you use this package, please consider citing the original jflux library and the FLUX.1 Kontext paper.


from .model import Flux1, Flux1Params

__all__ = [
    "Flux1",
    "Flux1Params",
]
