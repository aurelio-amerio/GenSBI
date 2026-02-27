"""
Deprecated joint loss classes.

These classes have been replaced by the canonical loss classes
(:class:`~gensbi.flow_matching.loss.FMLoss`,
:class:`~gensbi.diffusion.loss.EDMLoss`,
:class:`~gensbi.diffusion.loss.SMLoss`) accessed via the strategy API
(``method.build_loss(path)``).
"""
from gensbi.models.losses.conditional import _DeprecatedLoss


class JointCFMLoss(_DeprecatedLoss):
    _message = (
        "JointCFMLoss has been removed. "
        "Use FlowMatchingMethod().build_loss(path) or "
        "JointPipeline(method=FlowMatchingMethod(), ...) instead."
    )


class JointEDMLoss(_DeprecatedLoss):
    _message = (
        "JointEDMLoss has been removed. "
        "Use DiffusionEDMMethod().build_loss(path) or "
        "JointPipeline(method=DiffusionEDMMethod(), ...) instead."
    )


class JointSMLoss(_DeprecatedLoss):
    _message = (
        "JointSMLoss has been removed. "
        "Use ScoreMatchingMethod().build_loss(path) or "
        "JointPipeline(method=ScoreMatchingMethod(), ...) instead."
    )
