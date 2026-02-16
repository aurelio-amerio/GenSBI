import numpy as np
import pytest
from gensbi.diagnostics.utils import jefferys_interval

def test_jefferys_interval_basic():
    """Test Jeffrey's interval for known values."""
    k = 5
    n = 10
    lower, upper = jefferys_interval(k, n)

    # Expected values for z=1 (approx 68% CI)
    # k=5, n=10 -> Beta(5.5, 5.5)
    # interval is centered at 0.5
    assert np.isclose(lower, 0.34940656)
    assert np.isclose(upper, 0.65059344)
    assert lower < upper
    assert 0 <= lower <= 1
    assert 0 <= upper <= 1

def test_jefferys_interval_edge_cases():
    """Test Jeffrey's interval for edge cases (k=0, k=n)."""
    n = 10

    # k = 0
    lower, upper = jefferys_interval(0, n)
    assert lower == 0.0
    assert upper > 0.0
    assert upper < 1.0

    # k = n
    lower, upper = jefferys_interval(n, n)
    assert lower > 0.0
    assert lower < 1.0
    assert upper == 1.0

def test_jefferys_interval_vectorized():
    """Test Jeffrey's interval with vectorized inputs."""
    k = np.array([0, 5, 10])
    n = 10
    lower, upper = jefferys_interval(k, n)

    assert lower.shape == (3,)
    assert upper.shape == (3,)

    # k=0
    assert lower[0] == 0.0
    assert upper[0] > 0.0

    # k=5
    assert lower[1] > 0.0
    assert upper[1] < 1.0

    # k=10
    assert lower[2] > 0.0
    assert upper[2] == 1.0

def test_jefferys_interval_z_score():
    """Test Jeffrey's interval with different z-scores."""
    k = 5
    n = 10

    # z=1 (approx 68%)
    l1, u1 = jefferys_interval(k, n, z=1)

    # z=1.96 (approx 95%)
    l2, u2 = jefferys_interval(k, n, z=1.96)

    # Higher confidence (larger z) should result in wider interval
    assert l2 < l1
    assert u2 > u1

def test_jefferys_interval_input_types():
    """Test Jeffrey's interval with different input types."""
    # Lists
    l, u = jefferys_interval([5], [10])
    assert isinstance(l, np.ndarray)
    assert isinstance(u, np.ndarray)

    # Scalars
    l, u = jefferys_interval(5, 10)
    # Should return 0-d arrays or scalars, effectively float-like
    assert np.ndim(l) == 0
    assert np.ndim(u) == 0
