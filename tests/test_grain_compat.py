"""`import gensbi` must let grain's ``mp_prefetch`` run in a plain process.

grain 0.2.18 reads the absl flag ``--grain_enable_multiprocess_worker_profiling``
inside ``mp_prefetch`` once ``jaxlib_extension_version >= 448`` (jax >= 0.11).
absl raises ``UnparsedFlagAccessError`` in any process that never ran
``absl.app.run()`` (scripts, notebooks, pytest), which kills the data pipeline
documented in ``docs/basics/training.md``. The check runs in a fresh
interpreter because absl flag state is process-global and sticky.
"""

import subprocess
import sys

_SCRIPT = """
import gensbi  # noqa: F401  (must install the guard as a side effect)
from grain._src.core import profiler

if not profiler.is_worker_profiling_supported():
    print("unsupported")  # jax < 0.11: grain never reads the flag
else:
    assert profiler.is_worker_profiling_enabled() is False
    print("guarded")
"""


def test_import_gensbi_guards_grain_profiler_flag_in_fresh_process():
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() in {"guarded", "unsupported"}
