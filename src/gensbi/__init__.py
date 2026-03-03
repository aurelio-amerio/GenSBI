"""
GenSBI: A library for Simulation-Based Inference (SBI) using Optimal Transport Flow Matching and Diffusion models in JAX.

Provides tools for probabilistic modeling, simulation, and training of generative models, including:

- Flow Matching techniques
- Diffusion models (EDM, score matching)
- Transformer-based models (Flux1, Simformer)

See the documentation for details and usage examples.
"""

from importlib.metadata import version, PackageNotFoundError

# try to automatically fetch the current gensbi version from pyproject.toml
try:
    __version__ = version("gensbi")
except PackageNotFoundError:
    # Fallback if the package is being run without being installed
    __version__ = "unknown"

import warnings

# Suppress protobuf runtime version warning (grain dependency)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.runtime_version"
)

