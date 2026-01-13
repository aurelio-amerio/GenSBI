"""
GenSBI: A library for Simulation-Based Inference (SBI) using Optimal Transport Flow Matching and Diffusion models in JAX.

Provides tools for probabilistic modeling, simulation, and training of generative models, including:

- Flow Matching techniques
- Diffusion models (EDM, score matching)
- Transformer-based models (Flux1, Simformer)

See the documentation for details and usage examples.
"""

__version__ = "0.0.1"

import warnings

# Suppress protobuf runtime version warning (grain dependency)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.runtime_version"
)


# coverage report:
# diffusion: 100%
# flow_matching: 100%
# recipes: 97%
# models: 87%
# utils: 100%
