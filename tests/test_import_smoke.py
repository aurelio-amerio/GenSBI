import subprocess
import sys


def _fresh_import(module: str):
    r = subprocess.run([sys.executable, "-c", f"import {module}"],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"import {module} failed:\n{r.stderr}"


def test_import_models_clean():
    _fresh_import("gensbi.models")


def test_import_normalizing_flows_clean():
    _fresh_import("gensbi.normalizing_flows")
