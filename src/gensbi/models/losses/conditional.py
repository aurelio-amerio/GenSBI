"""
Deprecated conditional loss classes.

These classes have been replaced by the canonical loss classes
(:class:`~gensbi.flow_matching.loss.FMLoss`,
:class:`~gensbi.diffusion.loss.EDMLoss`,
:class:`~gensbi.diffusion.loss.SMLoss`) accessed via the strategy API
(``method.build_loss(path)``).
"""


class _DeprecatedLoss:
    """Base for deprecated loss stubs — raises on instantiation."""

    _message = ""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(self._message)


class ConditionalCFMLoss(_DeprecatedLoss):
    _message = (
        "ConditionalCFMLoss has been removed. "
        "Use FlowMatchingMethod().build_loss(path) or "
        "ConditionalPipeline(method=FlowMatchingMethod(), ...) instead."
    )


class ConditionalEDMLoss(_DeprecatedLoss):
    _message = (
        "ConditionalEDMLoss has been removed. "
        "Use DiffusionEDMMethod().build_loss(path) or "
        "ConditionalPipeline(method=DiffusionEDMMethod(), ...) instead."
    )


class ConditionalSMLoss(_DeprecatedLoss):
    _message = (
        "ConditionalSMLoss has been removed. "
        "Use ScoreMatchingMethod().build_loss(path) or "
        "ConditionalPipeline(method=ScoreMatchingMethod(), ...) instead."
    )
