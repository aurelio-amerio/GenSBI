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
except PackageNotFoundError:  # pragma: no cover
    # Fallback if the package is being run without being installed
    __version__ = "unknown"  # pragma: no cover

import warnings

# Suppress protobuf runtime version warning (grain dependency)
warnings.filterwarnings(
    "ignore", category=UserWarning, module="google.protobuf.runtime_version"
)

# grain 0.2.18 reads an absl flag inside mp_prefetch on jax >= 0.11 and crashes
# in any non-absl process (scripts, notebooks); see gensbi._compat.
import grain  # noqa: E402,F401  (registers grain's absl flags)
from gensbi._compat import guard_grain_profiler_flag as _guard_grain_profiler_flag  # noqa: E402

_guard_grain_profiler_flag()
