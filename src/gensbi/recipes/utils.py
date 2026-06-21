import jax
from jax import numpy as jnp
import numpy as np
from typing import Union, Tuple
from einops import repeat, rearrange
from jax import Array

from gensbi.diffusion.path import EDMPath
from gensbi.diffusion.path.scheduler import (
    EDMScheduler,
    VEEdmScheduler,
    VPEdmScheduler,
)
from gensbi.diffusion.path.sm_path import SMPath
from gensbi.diffusion.path.scheduler import VPSmScheduler, VESmScheduler


def init_ids_joint(dim_obs: int, dim_cond: int):
    dim_joint = dim_obs + dim_cond
    node_ids = jnp.arange(dim_joint).reshape((1, -1, 1))
    obs_ids = jnp.arange(dim_obs).reshape((1, -1, 1))  # observation ids
    cond_ids = jnp.arange(dim_obs, dim_joint).reshape((1, -1, 1))  # conditional ids
    return node_ids, obs_ids, cond_ids


def init_ids_1d(dim: int, semantic_id: Union[int, None] = None):
    """Build 1D positional IDs, returning ``(ids, dim)``.

    ``ids`` is ``(1, dim, 1)`` when ``semantic_id is None`` (position only), or
    ``(1, dim, 2)`` otherwise, with the position at axis 0 and the semantic id
    at axis 1.

    FIXME (axis-order footgun): this places the semantic id on the LAST axis,
    which is the *reverse* of :func:`init_ids_2d` (semantic at axis 0, then h, w
    -- the established convention). The two are safe in isolation, but they do
    NOT line up if 1D and 2D ids are ever fed into the same RoPE grid (e.g.
    FieldDiT Phase-2 obs+cond co-tokenization with a shared ``EmbedND``, where
    ``axes_dim[i]`` is matched to ids axis ``i``). This should be unified to the
    2D convention (semantic at axis 0). It is not changed here because callers
    that pass ``semantic_id`` -- notably the Flux1 ``rope1d`` path via
    :func:`init_ids` -- and any code indexing ``ids[..., k]`` must be updated in
    lockstep.
    """
    if semantic_id is None:
        ids = np.zeros((1, dim, 1), dtype=np.int32)
    else:
        ids = np.zeros((1, dim, 2), dtype=np.int32)
        ids[..., 1] = semantic_id

    ids[0, :, 0] = np.arange(dim)

    return jnp.array(ids, dtype=jnp.int32), dim


def _normalize_patch_size(size):
    """Normalize a patch-size spec into an ``(obs, cond)`` tuple.

    Parameters
    ----------
    size : int or tuple of int
        A single int is broadcast to both inputs (``8 -> (8, 8)``). A
        length-2 tuple is taken as ``(obs_size, cond_size)`` so the two
        inputs can use different patch sizes. Use ``1`` for an input that
        is not patchified.

    Returns
    -------
    tuple of int
        ``(obs_size, cond_size)``.
    """
    if isinstance(size, int):
        return (size, size)
    size = tuple(size)
    if len(size) != 2:
        raise ValueError(
            f"size must be an int or a length-2 (obs, cond) tuple, got {size!r}"
        )
    return size


def init_ids_2d(dim: Tuple[int, int], semantic_id: int = 0, size: int = 2):
    """Build 2D positional IDs for a patchified image grid.

    The grid has one entry per patch, i.e. ``(dim[0] // size, dim[1] // size)``,
    matching ``patchify_2d(x, size=size)``. ``size`` is the patch edge length;
    use ``size=1`` for no patchification (one token per pixel).
    """
    img_ids = np.zeros((dim[0] // size, dim[1] // size, 3), dtype=np.int32)
    img_ids[..., 0] = semantic_id
    img_ids[..., 1] = img_ids[..., 1] + np.arange(dim[0] // size)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + np.arange(dim[1] // size)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=1)

    dim = (dim[0] // size) * (dim[1] // size)

    return jnp.array(img_ids, dtype=jnp.int32), dim


@jax.jit(static_argnames=["size"])
def patchify_2d(x: Array, size=2):
    return rearrange(x, "b (h ph) (w pw) c -> b (h w) (c ph pw)", ph=size, pw=size)


@jax.jit(static_argnames=["size", "grid"])
def depatchify_2d(x: Array, size=2, grid=None):
    """Inverse of :func:`patchify_2d`.

    Parameters
    ----------
    x : Array
        Patchified tensor of shape ``(B, h*w, C*size*size)``.
    size : int
        Patch edge length used by :func:`patchify_2d`.
    grid : tuple of int, optional
        The ``(h, w)`` patch grid. The grid cannot be inferred from the token
        count alone, so it is required for non-square grids. If ``None``, a
        square grid (``h == w``) is assumed.
    """
    if grid is None:
        n = x.shape[1]
        side = int(round(n ** 0.5))
        if side * side != n:
            raise ValueError(
                f"Cannot infer a square grid from {n} tokens; pass grid=(h, w)."
            )
        h = w = side
    else:
        h, w = grid
    return rearrange(
        x, "b (h w) (c ph pw) -> b (h ph) (w pw) c", h=h, w=w, ph=size, pw=size
    )


def scale_lr(batch_size, base_lr=1e-4, reference_batch_size=256):
    """Scale learning rate based on batch size using square root scaling.

    Parameters
    ----------
        batch_size : int
            The current batch size.
        base_lr : float
            The base learning rate for the reference batch size.
        reference_batch_size : int, optional
            The reference batch size. Defaults to 256.

    Returns
    -------
        float
            The adjusted learning rate.
    """
    import math

    return base_lr * math.sqrt(batch_size / reference_batch_size)


_EMBEDDINGS_1D = {"absolute", "pos1d", "rope1d"}
_EMBEDDINGS_2D = {"pos2d", "rope2d"}


def _resolve_embedding_ids(dim, strategy: str, semantic_id: int, size: int = 2):
    """Resolve ID embeddings by strategy name.

    Parameters
    ----------
    dim : int or tuple of int
        Dimension specification (number of tokens, or (H, W) for 2D images).
    strategy : str
        Embedding strategy name (e.g., "absolute", "pos1d", "rope1d",
        "pos2d", "rope2d").
    semantic_id : int
        Semantic identifier for the token group (0=obs, 1=cond).
    size : int, optional
        Patch edge length for 2D strategies (default 2). Ignored for 1D
        strategies. Use 1 for no patchification.

    Returns
    -------
    ids : Array
        Token ID array.
    resolved_dim : int
        Resolved flat dimension.

    Raises
    ------
    ValueError
        If ``strategy`` is not recognized.
    """
    if strategy in _EMBEDDINGS_1D:
        return init_ids_1d(dim, semantic_id=semantic_id)
    elif strategy in _EMBEDDINGS_2D:
        return init_ids_2d(dim, semantic_id=semantic_id, size=size)
    else:
        raise ValueError(f"Unknown id embedding strategy: {strategy}")


def build_edm_path(sde: str, config: dict) -> EDMPath:
    """Build an EDM-family diffusion path from an SDE type string and config.

    Parameters
    ----------
    sde : str
        SDE type: ``"EDM"``, ``"VE"``, or ``"VP"``.
    config : dict
        Training configuration dict; scheduler hyperparameters are read from
        here with sensible defaults.

    Returns
    -------
    EDMPath
        Configured diffusion path.

    Raises
    ------
    ValueError
        If ``sde`` is not one of ``{"EDM", "VE", "VP"}``.
    """
    if sde == "EDM":
        return EDMPath(
            scheduler=EDMScheduler(
                sigma_min=config.get("sigma_min", 0.002),
                sigma_max=config.get("sigma_max", 80.0),
            )
        )
    elif sde == "VE":
        return EDMPath(
            scheduler=VEEdmScheduler(
                sigma_min=config.get("sigma_min", 0.02),
                sigma_max=config.get("sigma_max", 100.0),
            )
        )
    elif sde == "VP":
        return EDMPath(
            scheduler=VPEdmScheduler(
                beta_min=config.get("beta_min", 0.1),
                beta_max=config.get("beta_max", 19.9),
            )
        )
    else:
        raise ValueError(f"Unknown sde type: {sde}")


def build_sm_path(sde_type: str, config: dict) -> SMPath:
    """Build a score-matching path from an SDE type string and config.

    Parameters
    ----------
    sde_type : str
        SDE type: ``"VP"`` or ``"VE"``.
    config : dict
        Training configuration dict; scheduler hyperparameters are read from
        here with sensible defaults.

    Returns
    -------
    SMPath
        Configured score-matching path.

    Raises
    ------
    ValueError
        If ``sde_type`` is not one of ``{"VP", "VE"}``.
    """
    if sde_type == "VP":
        return SMPath(
            VPSmScheduler(
                beta_min=config.get("beta_min", 0.001),
                beta_max=config.get("beta_max", 3.0),
            )
        )
    elif sde_type == "VE":
        return SMPath(
            VESmScheduler(
                sigma_min=config.get("sigma_min", 0.001),
                sigma_max=config.get("sigma_max", 15.0),
            )
        )
    else:
        raise ValueError(f"sde_type must be one of ['VP', 'VE'], got {sde_type}.")


def parse_training_config(config_path: str):
    """Parse training and optimizer configuration from a YAML config file.

    Reads the ``training`` and ``optimizer`` sections of the config and
    returns a flat dictionary consumed by :class:`AbstractPipeline`.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    training_config : dict
        Parsed training configuration dictionary.
    """
    import yaml

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Training parameters
    train_params = config.get("training", {})
    multistep = train_params.get("multistep", 1)

    training_config = {
        "nsteps": train_params.get("nsteps", 30000) * multistep,
        "ema_decay": train_params.get("ema_decay", 0.999),
        "multistep": multistep,
        "experiment_id": train_params.get("experiment_id", 1),
        "early_stopping": train_params.get("early_stopping", True),
        "val_every": train_params.get("val_every", 100) * multistep,
        "val_error_ratio": train_params.get("val_error_ratio", 1.3),
        # Optional method-specific parameters (override strategy defaults)
        "sigma_min": train_params.get("sigma_min", 0.002),
        "sigma_max": train_params.get("sigma_max", 80.0),
    }

    # Optimizer parameters
    opt_params = config.get("optimizer", {})

    MAX_LR = opt_params.get("max_lr", 1e-3)
    MIN_LR = opt_params.get("min_lr", 0.0)

    training_config["max_lr"] = MAX_LR
    training_config["min_lr"] = MIN_LR
    training_config["min_scale"] = MIN_LR / MAX_LR if MAX_LR > 0 else 0.0
    training_config["warmup_steps"] = opt_params.get("warmup_steps", 500)
    training_config["decay_transition"] = opt_params.get("decay_transition", 0.85)

    # ema_decay can also be specified in optimizer section (backward compat)
    if "ema_decay" in opt_params:
        training_config["ema_decay"] = opt_params["ema_decay"]

    return training_config
