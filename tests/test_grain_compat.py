"""grain's ``mp_prefetch`` must run in a plain (non-absl) process.

grain 0.2.18 read the absl flag ``--grain_enable_multiprocess_worker_profiling``
inside ``mp_prefetch`` once ``jaxlib_extension_version >= 448`` (jax >= 0.11), and
absl raises ``UnparsedFlagAccessError`` in any process that never ran
``absl.app.run()`` (scripts, notebooks, pytest). That killed the data pipeline
documented in ``docs/basics/training.md``. grain 0.2.19 catches the error itself,
so GenSBI requires ``grain>=0.2.19`` instead of carrying a shim; this test fails if
the floor ever slips back. It runs in a fresh interpreter because absl flag state
is process-global and sticky.
"""

import subprocess
import sys

_SCRIPT = """
import gensbi  # noqa: F401
from grain._src.core import profiler

if not profiler.is_worker_profiling_supported():
    print("unsupported")  # jax < 0.11: grain never reads the flag
else:
    assert profiler.is_worker_profiling_enabled() is False
    print("ok")
"""


def test_grain_profiler_flag_read_does_not_crash_in_fresh_process():
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT], capture_output=True, text=True, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() in {"ok", "unsupported"}
