"""Small shims for third-party packages that misbehave on the supported stack.

Each shim is a no-op once the upstream fix ships, so they can be deleted
when the corresponding floor is raised past it.
"""

_GRAIN_PROFILER_FLAG = "grain_enable_multiprocess_worker_profiling"


def guard_grain_profiler_flag() -> None:
    """Let grain read its worker-profiling absl flag in a non-absl process.

    grain 0.2.18 (``grain/_src/core/profiler.py``) reads
    ``--grain_enable_multiprocess_worker_profiling`` inside ``mp_prefetch``
    whenever ``jaxlib_extension_version >= 448`` (jax >= 0.11). absl raises
    ``UnparsedFlagAccessError`` on that read in any process that never called
    ``absl.app.run()`` -- scripts, notebooks, pytest -- so the data pipeline
    in ``docs/basics/training.md`` dies on its first batch. Marking the single
    flag as present makes absl return its default without touching the global
    "parsed" state; it is the same trick grain uses for its own debug flags.
    grain's main branch already guards the read; this is a no-op once that
    ships, and a no-op if the flag is absent or flags were really parsed.
    """
    try:
        from absl import flags
    except ImportError:  # pragma: no cover - absl is a grain dependency
        return
    if flags.FLAGS.is_parsed() or _GRAIN_PROFILER_FLAG not in flags.FLAGS:
        return
    flags.FLAGS[_GRAIN_PROFILER_FLAG].present = True
