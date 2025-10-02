from dataclasses import dataclass, field
from jax import Array


@dataclass
class PathSample:
    r"""Represents a sample of a conditional-flow generated probability path.

    Attributes:
        x_1 (Array): the target sample :math:`X_1`.
        x_0 (Array): the source sample :math:`X_0`.
        t (Array): the time sample :math:`t`.
        x_t (Array): samples :math:`X_t \sim p_t(X_t)`, shape (batch_size, ...).
        dx_t (Array): conditional target :math:`\frac{\partial X}{\partial t}`, shape: (batch_size, ...).

    """

    x_1: Array = field(metadata={"help": "target samples X_1 (batch_size, ...)."})
    x_0: Array = field(metadata={"help": "source samples X_0 (batch_size, ...)."})
    t: Array = field(metadata={"help": "time samples t (batch_size, ...)."})
    x_t: Array = field(
        metadata={"help": "samples x_t ~ p_t(X_t), shape (batch_size, ...)."}
    )
    dx_t: Array = field(
        metadata={"help": "conditional target dX_t, shape: (batch_size, ...)."}
    )


# @dataclass
# class DiscretePathSample:
#     r"""
#     Represents a sample of a conditional-flow generated discrete probability path.

#     Attributes:
#         x_1 (Array): the target sample :math:`X_1`.
#         x_0 (Array): the source sample :math:`X_0`.
#         t (Array): the time sample  :math:`t`.
#         x_t (Array): the sample along the path  :math:`X_t \sim p_t`.
#     """

#     x_1: Array = field(metadata={"help": "target samples X_1 (batch_size, ...)."})
#     x_0: Array = field(metadata={"help": "source samples X_0 (batch_size, ...)."})
#     t: Array = field(metadata={"help": "time samples t (batch_size, ...)."})
#     x_t: Array = field(
#         metadata={"help": "samples X_t ~ p_t(X_t), shape (batch_size, ...)."}
#     )
