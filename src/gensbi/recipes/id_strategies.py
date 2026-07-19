"""Structured id-builder strategies for the recipe pipelines.

Two DIFFERENT vocabularies share the word "rope" — they collided once
(2026-07-19 handoff) and are deliberately kept distinct:

- **Model-side** strategy strings (e.g. ``Flux1Params.id_embedding_strategy``):
  ``"rope"`` means *apply RoPE to whatever ids arrive at the forward pass*.
- **Pipeline-side** builder strategies (``ConditionalPipeline``'s
  ``id_embedding_strategy``): strings like ``"rope1d"``/``"rope2d"`` — and the
  objects in this module — *build* the id arrays themselves.

A :class:`HealpixRope` pipeline strategy pairs with model-side
``("absolute", "rope")`` plus a 3-entry even ``axes_dim`` summing to the
per-head dim (e.g. ``(22, 22, 20)`` for 64).
"""

from dataclasses import dataclass
from typing import ClassVar, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

from gensbi.recipes.utils import (
    _validate_base_pixels,
    healpix_rope_theta,
    init_ids_healpix,
)


@runtime_checkable
class IdStrategy(Protocol):
    """Structural interface for pipeline id-builder strategy objects.

    Any object with a ``name`` and a ``build(dim) -> (ids, resolved_dim)``
    method can be passed in an ``id_embedding_strategy`` tuple slot; the
    pipeline calls ``build`` with the corresponding ``dim_obs``/``dim_cond``.
    Strategies own their full geometry — unlike the string strategies they
    receive no ``semantic_id``/``size``.
    """

    name: str

    def build(self, dim):
        """Return ``(ids, resolved_dim)`` for a stream of ``dim`` tokens."""
        ...


@dataclass(frozen=True)
class HealpixRope:
    """Spherical RoPE ids for tokens on a HEALPix grid (name: "healpix-rope").

    Wraps :func:`gensbi.recipes.utils.init_ids_healpix` (see there for the
    method and its rationale) with the geometry needed to build the ids —
    which the string-enum API cannot carry, since the pipeline only passes a
    token count.

    Parameters
    ----------
    nside : int
        HEALPix resolution of the *token* grid (power of 2).
    base_pixels : sequence of int, optional
        Base pixels (0..11) covered by the grid; ``None`` = full sky.
    """

    nside: int
    base_pixels: Optional[Union[Tuple[int, ...], Sequence[int]]] = None

    name: ClassVar[str] = "healpix-rope"

    def __post_init__(self):
        if self.nside < 1 or (self.nside & (self.nside - 1)) != 0:
            raise ValueError(f"nside must be a power of 2, got {self.nside}")
        if self.base_pixels is not None:
            object.__setattr__(
                self, "base_pixels", tuple(_validate_base_pixels(self.base_pixels))
            )

    @property
    def num_tokens(self) -> int:
        """Token count of the grid: ``n_faces * nside**2``."""
        n_faces = 12 if self.base_pixels is None else len(self.base_pixels)
        return n_faces * self.nside**2

    @property
    def theta(self) -> int:
        """Suggested model-side RoPE theta: :func:`healpix_rope_theta`."""
        return healpix_rope_theta(self.nside)

    def build(self, dim):
        """Return ``(ids, num_tokens)``; ``dim`` must match the grid."""
        if dim != self.num_tokens:
            sky = (
                "full sky"
                if self.base_pixels is None
                else f"{len(self.base_pixels)} base pixels"
            )
            raise ValueError(
                f"{self.name}: the pipeline stream has dim={dim} tokens, but "
                f"nside={self.nside} ({sky}) implies {self.num_tokens} tokens"
            )
        return init_ids_healpix(self.nside, self.base_pixels)
