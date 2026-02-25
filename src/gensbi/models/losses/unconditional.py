"""
Deprecated unconditional loss classes.

These classes have been replaced by the canonical loss classes
(:class:`~gensbi.flow_matching.loss.FMLoss`,
:class:`~gensbi.diffusion.loss.EDMLoss`,
:class:`~gensbi.diffusion.loss.SMLoss`) accessed via the strategy API
(``method.build_loss(path)``).
"""
from gensbi.models.losses.conditional import _DeprecatedLoss


class UnconditionalCFMLoss(_DeprecatedLoss):
    _message = (
        "UnconditionalCFMLoss has been removed. "
        "Use FlowMatchingMethod().build_loss(path) or "
        "UnconditionalPipeline(method=FlowMatchingMethod(), ...) instead."
    )


class UnconditionalEDMLoss(_DeprecatedLoss):
    _message = (
        "UnconditionalEDMLoss has been removed. "
        "Use DiffusionEDMMethod().build_loss(path) or "
        "UnconditionalPipeline(method=DiffusionEDMMethod(), ...) instead."
    )


class UnconditionalSMLoss(_DeprecatedLoss):
    _message = (
        "UnconditionalSMLoss has been removed. "
        "Use ScoreMatchingMethod().build_loss(path) or "
        "UnconditionalPipeline(method=ScoreMatchingMethod(), ...) instead."
    )
