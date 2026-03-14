"""
Core module for GenSBI.

Provides the ``GenerativeMethod`` strategy pattern and its concrete
implementations for flow matching, EDM diffusion, and score matching.

These strategy objects encapsulate the generative framework (path, loss,
solver, batch preparation) and are composed into mode-specific pipelines
(Conditional, Joint, Unconditional) in the ``recipes`` module.

Public API (all lazy-loaded to avoid circular imports with solver
subclasses that inherit from core base classes)::

    from gensbi.core import FlowMatchingMethod
    from gensbi.core import DiffusionEDMMethod
    from gensbi.core import ScoreMatchingMethod
    from gensbi.core import GenerativeMethod
"""

from gensbi.core.generative_method import GenerativeMethod

_LAZY_IMPORTS = {
    "FlowMatchingMethod": "gensbi.core.flow_matching",
    "DiffusionEDMMethod": "gensbi.core.diffusion_edm",
    "ScoreMatchingMethod": "gensbi.core.score_matching",
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        import importlib
        module = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(module, name)
    raise AttributeError(f"module 'gensbi.core' has no attribute {name!r}")


__all__ = [
    "GenerativeMethod",
    "FlowMatchingMethod",
    "DiffusionEDMMethod",
    "ScoreMatchingMethod",
]
