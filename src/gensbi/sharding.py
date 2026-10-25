"""Multi-device sharding helpers for GenSBI.

Models annotate their variables with *logical* axis names through :func:`axes`
(or :func:`partitioned` for explicit initialisers).  Annotations are inert:
nothing is placed on a device until :func:`shard_model` maps the logical names
onto the axes of a :class:`jax.sharding.Mesh` according to a named rule
(``"replicated"``, ``"fsdp"`` or ``"tensor"``).  With no mesh, nothing changes.

Cross-repo contract
-------------------
``LOGICAL_AXES``, ``RULES``, ``MODEL_AXIS`` and the semantics of ``axes``,
``partitioned``, ``make_mesh``, ``batch_axes``, ``batch_size_divisor``,
``shard_model``, ``shard_batch``, ``check_batch_divisible`` and ``sharding_of``
are identical to ``heal_swin_nnx/sharding.py``.  Neither package imports the
other; a test pins the vocabulary against the installed ``heal_swin_nnx``.
Change both files together.
"""

import contextlib
import math
import warnings

import jax
import numpy as np
from flax import nnx
from jax.sharding import AxisType, NamedSharding
from jax.sharding import PartitionSpec as P

LOGICAL_AXES: tuple[str, ...] = ("embed", "mlp", "heads", "vocab", "in", "out")
RULES: dict[str, tuple[tuple[str, str | None], ...]] = {
    "replicated": (("embed", None), ("mlp", None), ("heads", None), ("vocab", None), ("in", None), ("out", None)),
    "fsdp":       (("embed", "data"), ("mlp", None), ("heads", None), ("vocab", None), ("in", None), ("out", None)),
    "tensor":     (("embed", None), ("mlp", "model"), ("heads", "model"), ("vocab", "model"), ("in", None), ("out", None)),
}
MODEL_AXIS = "model"


def axes(*names: str | None) -> dict:
    """Variable metadata annotating each array dimension with a logical axis.

    Parameters
    ----------
    *names : str or None
        One logical axis name (see ``LOGICAL_AXES``) or ``None`` per array
        dimension.

    Returns
    -------
    dict
        Keyword arguments for an ``nnx.Variable`` constructor, or the value of
        a ``*_metadata`` argument of an NNX layer.  ``eager_sharding=False``
        keeps construction independent of any mesh.
    """
    return {"out_sharding": names, "eager_sharding": False}


def partitioned(init, names: tuple[str | None, ...]):
    """Wrap an initialiser so its variable carries logical axis names.

    Parameters
    ----------
    init : callable
        Initialiser to wrap.
    names : tuple of (str or None)
        Logical axis name per array dimension.

    Returns
    -------
    callable
        ``nnx.with_partitioning(init, names, eager_sharding=False)``.
    """
    return nnx.with_partitioning(init, names, eager_sharding=False)


def make_mesh(devices=None, model_axis: int = 1) -> jax.sharding.Mesh:
    """Build a ``("data", "model")`` mesh with Auto axis types.

    Parameters
    ----------
    devices : sequence of jax.Device, optional
        Devices to use; defaults to ``jax.devices()``.
    model_axis : int, optional
        Size of the ``"model"`` axis; the ``"data"`` axis gets
        ``len(devices) // model_axis``.

    Returns
    -------
    jax.sharding.Mesh

    Raises
    ------
    ValueError
        If ``model_axis`` does not divide the device count.
    """
    devices = list(jax.devices() if devices is None else devices)
    if model_axis < 1 or len(devices) % model_axis:
        raise ValueError(
            f"model_axis={model_axis} must be a positive divisor of the device count {len(devices)}."
        )
    grid = np.asarray(devices).reshape(len(devices) // model_axis, model_axis)
    return jax.sharding.Mesh(
        grid, ("data", MODEL_AXIS), axis_types=(AxisType.Auto, AxisType.Auto)
    )


def batch_axes(mesh) -> tuple[str, ...]:
    """Mesh axis names that carry the batch (every axis except ``"model"``)."""
    return tuple(a for a in mesh.axis_names if a != MODEL_AXIS)


def batch_size_divisor(mesh) -> int:
    """Number of data-parallel shards a batch is split into."""
    return math.prod(mesh.shape[a] for a in batch_axes(mesh))


def _is_array(x) -> bool:
    return isinstance(x, (jax.Array, np.ndarray))


def shard_model(node, mesh, rule: str = "replicated") -> None:
    """Place every variable of ``node`` on ``mesh`` according to ``rule``.

    Works in place and is idempotent.  ``node`` is any NNX graph node
    (modules, ``nnx.Optimizer``, EMA containers).  Variables without
    annotation are replicated; variables with an unknown logical name, a
    dimension not divisible by the mapped mesh axis, or a mapped axis missing
    from ``mesh`` fall back to replicated and are reported in a single warning.  ``"replicated"`` ignores annotations
    (so never warns about them); under other rules unannotated ``nnx.Param`` s
    are replicated and listed in that warning too.

    Parameters
    ----------
    node : nnx graph node
        Object whose state is placed.
    mesh : jax.sharding.Mesh
        Target mesh.
    rule : {"replicated", "fsdp", "tensor"}, optional
        Logical-to-mesh axis rule.  Anything but ``"replicated"`` is
        experimental and warns on every call.

    Raises
    ------
    ValueError
        If ``rule`` is not a key of ``RULES``.
    """
    if rule not in RULES:
        raise ValueError(f"Unknown sharding rule '{rule}'. Available rules: {sorted(RULES)}.")
    if rule != "replicated":
        warnings.warn(
            f"sharding rule '{rule}' is experimental and not covered by convergence tests; "
            "use 'replicated' for production runs.",
            UserWarning,
            stacklevel=2,
        )
    if any(t == AxisType.Explicit for t in mesh.axis_types):
        warnings.warn(
            "shard_model expects a mesh with Auto axis types; Explicit axes are not supported "
            "and may fail inside jitted code.",
            UserWarning,
            stacklevel=2,
        )
    table = dict(RULES[rule])
    problems: list[str] = []
    # A user-built mesh may lack an axis the rule maps to (e.g. no "data"):
    # those logical names are replicated instead of raising KeyError.
    missing = sorted({a for a in table.values() if a is not None} - set(mesh.axis_names))
    if missing:
        problems.append(f"mesh {tuple(mesh.axis_names)} has no axis {missing} used by rule '{rule}'")
        table = {k: (None if a in missing else a) for k, a in table.items()}
    flat = []
    for path, var in nnx.to_flat_state(nnx.state(node)):
        value = var[...]
        names = var.get_metadata().get("out_sharding")
        spec = P()
        where = "/".join(map(str, path))
        if names is None and rule != "replicated" and isinstance(var, nnx.Param):
            problems.append(f"{where}: no annotation")
        if names is not None and rule != "replicated":  # replicated ignores annotations
            ndim = np.ndim(value)
            unknown = [n for n in names if n is not None and n not in LOGICAL_AXES]
            if len(names) != ndim:
                problems.append(f"{where}: annotation rank {len(names)} != array rank {ndim}")
            elif unknown:
                problems.append(f"{where}: unknown logical axis {unknown}")
            else:
                mapped = tuple(None if n is None else table[n] for n in names)
                bad = [
                    (d, a)
                    for d, a in zip(np.shape(value), mapped)
                    if a is not None and d % mesh.shape[a]
                ]
                if bad:
                    problems.append(
                        f"{where}: dimension {bad[0][0]} not divisible by mesh axis "
                        f"'{bad[0][1]}' of size {mesh.shape[bad[0][1]]}"
                    )
                else:
                    spec = P(*mapped)
        flat.append((path, var.replace(jax.device_put(value, NamedSharding(mesh, spec)))))
    if problems:
        warnings.warn(
            "shard_model fell back to replicated placement for: " + "; ".join(problems),
            UserWarning,
            stacklevel=2,
        )
    nnx.update(node, nnx.from_flat_state(flat))


def shard_batch(batch, mesh):
    """Shard the leading axis of every array leaf of ``batch`` over the batch axes.

    Non-array leaves are returned untouched.
    """
    sharding = NamedSharding(mesh, P(batch_axes(mesh)))
    return jax.tree.map(
        lambda x: jax.device_put(x, sharding) if _is_array(x) else x, batch
    )


def check_batch_divisible(batch, mesh, what: str) -> None:
    """Raise ``ValueError`` if the batch cannot be split evenly across the mesh.

    Parameters
    ----------
    batch : pytree
        Batch whose array leaves share the same leading size.
    mesh : jax.sharding.Mesh
    what : str
        Label used in the error message (e.g. ``"train"``).
    """
    sizes = [x.shape[0] for x in jax.tree.leaves(batch) if _is_array(x) and x.ndim > 0]
    if not sizes:
        return
    if len(set(sizes)) > 1:
        raise ValueError(f"{what} batch leaves have differing leading sizes: {sizes}")
    B, d = sizes[0], batch_size_divisor(mesh)
    if B % d:
        raise ValueError(
            f"{what} batch size {B} is not divisible by the data-parallel device count {d} "
            f"(mesh axes {batch_axes(mesh)}). Use a batch size that is a multiple of {d}."
        )


def sharding_of(node) -> dict[str, jax.sharding.Sharding]:
    """Map ``"a/b/c"`` state paths of ``node`` to the sharding of each leaf."""
    return {
        "/".join(map(str, path)): var[...].sharding
        for path, var in nnx.to_flat_state(nnx.state(node))
    }


def resolve_mesh(spec) -> jax.sharding.Mesh | None:
    """Turn a user-facing mesh spec into a mesh.

    ``None`` -> ``None``; a ``Mesh`` -> itself; ``"auto"`` -> :func:`make_mesh`;
    a ``dict`` -> ``make_mesh(**spec)``.  Anything else raises ``ValueError``.
    """
    if spec is None:
        return None
    if isinstance(spec, jax.sharding.Mesh):
        return spec
    if isinstance(spec, str) and spec == "auto":
        return make_mesh()
    if isinstance(spec, dict):
        return make_mesh(**spec)
    raise ValueError(
        f"Invalid mesh spec {spec!r}: expected None, a jax.sharding.Mesh, 'auto' or a dict of make_mesh kwargs."
    )


def pad_for_mesh(n: int, mesh) -> tuple[int, int]:
    """Round ``n`` up to a multiple of the data-parallel size.

    Returns
    -------
    tuple of int
        ``(padded, n)``.
    """
    d = batch_size_divisor(mesh)
    return -(-n // d) * d, n


def shard_sample_init(x_init, mesh):
    """Shard a sampler's initial state (array or pytree) over the batch axes."""
    return shard_batch(x_init, mesh)


def replicate(tree, mesh):
    """Replicate every array leaf of ``tree`` over every device of ``mesh``.

    Used for inputs that must live on the mesh without being split, such as
    a sampler's condition and node ids.  Arrays committed to a single device
    are moved; non-array leaves are returned untouched.

    Parameters
    ----------
    tree : pytree
        Arrays (``jax.Array`` or ``numpy.ndarray``) and arbitrary other leaves.
    mesh : jax.sharding.Mesh
        Target mesh.

    Returns
    -------
    pytree
        Same structure, array leaves placed with ``NamedSharding(mesh, P())``.
    """
    sharding = NamedSharding(mesh, P())
    return jax.tree.map(
        lambda x: jax.device_put(x, sharding) if _is_array(x) else x, tree
    )


@contextlib.contextmanager
def mesh_context(mesh):
    """Make ``mesh`` the active mesh for the body of a ``with`` block.

    ``jax.set_mesh(mesh)`` is entered when the ``with`` block is entered (not
    when this function is called) and left on exit.  With ``mesh=None`` this
    is a no-op context.

    Parameters
    ----------
    mesh : jax.sharding.Mesh or None
        Mesh to activate.
    """
    if mesh is None:
        yield
        return
    with jax.set_mesh(mesh):
        yield
