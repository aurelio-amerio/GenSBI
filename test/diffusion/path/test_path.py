import os
os.environ['JAX_PLATFORMS'] = "cpu"

import pytest
from gensbi.diffusion.path.path import Path


def test_path_initialization():
    path = Path()
    assert isinstance(path, Path)
