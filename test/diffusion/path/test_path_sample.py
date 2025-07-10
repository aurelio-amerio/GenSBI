import os
os.environ['JAX_PLATFORMS'] = "cpu"

import pytest
from gensbi.diffusion.path.path_sample import PathSample


def test_path_sample_initialization():
    # Minimal test, as PathSample likely requires more context
    sample = PathSample()
    assert isinstance(sample, PathSample)
