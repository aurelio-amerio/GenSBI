import pytest
import jax
import jax.numpy as jnp
from unittest.mock import patch
from gensbi.recipes.pipeline import _get_batch_sampler

# Fixture to provide a simple mock sampler function
@pytest.fixture
def mock_sampler_fn():
    def sampler(key, ncond):
        # Return key value + ncond as a jnp array
        return jnp.sum(key) + ncond
    return sampler

def test_chunking_logic(mock_sampler_fn):
    """Test that the sampler processes data in the correct number of chunks."""
    ncond = 10
    chunk_size = 2
    n_samples = 5

    # Expected chunks: ceil(5/2) = 3
    # Chunks sizes: 2, 2, 1

    # We patch jax.vmap to count calls. We must disable JIT for this to work
    # because process_chunk is jitted.
    with jax.disable_jit():
        with patch('jax.vmap', side_effect=jax.vmap) as mock_vmap:
            sampler = _get_batch_sampler(
                sampler_fn=mock_sampler_fn,
                ncond=ncond,
                chunk_size=chunk_size,
                show_progress_bars=False
            )

            keys = jax.random.split(jax.random.PRNGKey(0), n_samples)
            results = sampler(keys)

            # Verify vmap was called 3 times (once per chunk)
            assert mock_vmap.call_count == 3

            # Verify results match expected output
            expected_results = jnp.array([jnp.sum(k) + ncond for k in keys])
            assert jnp.allclose(results, expected_results)

def test_chunking_exact_division(mock_sampler_fn):
    """Test chunking when n_samples is exactly divisible by chunk_size."""
    ncond = 5
    chunk_size = 2
    n_samples = 4

    # Expected chunks: 4/2 = 2

    with jax.disable_jit():
        with patch('jax.vmap', side_effect=jax.vmap) as mock_vmap:
            sampler = _get_batch_sampler(
                sampler_fn=mock_sampler_fn,
                ncond=ncond,
                chunk_size=chunk_size,
                show_progress_bars=False
            )

            keys = jax.random.split(jax.random.PRNGKey(1), n_samples)
            results = sampler(keys)

            assert mock_vmap.call_count == 2

            expected_results = jnp.array([jnp.sum(k) + ncond for k in keys])
            assert jnp.allclose(results, expected_results)

def test_chunking_single_chunk(mock_sampler_fn):
    """Test chunking when n_samples is less than chunk_size."""
    ncond = 5
    chunk_size = 10
    n_samples = 5

    # Expected chunks: 1

    with jax.disable_jit():
        with patch('jax.vmap', side_effect=jax.vmap) as mock_vmap:
            sampler = _get_batch_sampler(
                sampler_fn=mock_sampler_fn,
                ncond=ncond,
                chunk_size=chunk_size,
                show_progress_bars=False
            )

            keys = jax.random.split(jax.random.PRNGKey(2), n_samples)
            results = sampler(keys)

            assert mock_vmap.call_count == 1

            expected_results = jnp.array([jnp.sum(k) + ncond for k in keys])
            assert jnp.allclose(results, expected_results)

def test_progress_bar_enabled(mock_sampler_fn):
    """Test that tqdm is used when show_progress_bars is True."""
    ncond = 1
    chunk_size = 2
    n_samples = 5

    # Expected chunks: 3

    with patch('gensbi.recipes.pipeline.tqdm') as mock_tqdm:
        # We need to make sure the mocked tqdm behaves like an iterator
        # so the loop runs correctly.
        mock_tqdm.return_value = range(0, n_samples, chunk_size)

        sampler = _get_batch_sampler(
            sampler_fn=mock_sampler_fn,
            ncond=ncond,
            chunk_size=chunk_size,
            show_progress_bars=True
        )

        keys = jax.random.split(jax.random.PRNGKey(3), n_samples)
        sampler(keys)

        # Verify tqdm was initialized
        mock_tqdm.assert_called_once()

        # Check arguments: range(0, n_samples, chunk_size), total=3, desc="Sampling"
        args, kwargs = mock_tqdm.call_args
        assert args[0] == range(0, n_samples, chunk_size)
        assert kwargs['total'] == 3
        assert kwargs['desc'] == "Sampling"

def test_progress_bar_disabled(mock_sampler_fn):
    """Test that tqdm is NOT used when show_progress_bars is False."""
    ncond = 1
    chunk_size = 2
    n_samples = 5

    with patch('gensbi.recipes.pipeline.tqdm') as mock_tqdm:
        sampler = _get_batch_sampler(
            sampler_fn=mock_sampler_fn,
            ncond=ncond,
            chunk_size=chunk_size,
            show_progress_bars=False
        )

        keys = jax.random.split(jax.random.PRNGKey(4), n_samples)
        sampler(keys)

        # Verify tqdm was NOT called
        mock_tqdm.assert_not_called()

def test_results_concatenation(mock_sampler_fn):
    """Test that results are concatenated correctly even with multidimensional output."""

    # Sampler returning multidimensional array
    def multidim_sampler(key, ncond):
        return jnp.ones((ncond, 2)) * jnp.sum(key)

    ncond = 3
    chunk_size = 2
    n_samples = 5

    # Expected output shape: (5, 3, 2)

    sampler = _get_batch_sampler(
        sampler_fn=multidim_sampler,
        ncond=ncond,
        chunk_size=chunk_size,
        show_progress_bars=False
    )

    keys = jax.random.split(jax.random.PRNGKey(5), n_samples)
    results = sampler(keys)

    assert results.shape == (n_samples, ncond, 2)

    # Verify values
    for i in range(n_samples):
        expected_val = jnp.sum(keys[i])
        assert jnp.allclose(results[i], expected_val)

def test_pytree_output():
    """Test that the sampler works with PyTree outputs (e.g., dict of arrays)."""

    # Sampler returning a dict
    def pytree_sampler(key, ncond):
        val = jnp.sum(key) + ncond
        # Ensure outputs are at least 0-d arrays so vmap works nicely,
        # but here val is scalar (0-d array). vmap adds a batch dim.
        return {'a': val, 'b': val * 2}

    ncond = 1
    chunk_size = 2
    n_samples = 3

    # We patch jax.vmap to count calls. We must disable JIT for this to work
    # because process_chunk is jitted.
    with jax.disable_jit():
        sampler = _get_batch_sampler(
            sampler_fn=pytree_sampler,
            ncond=ncond,
            chunk_size=chunk_size,
            show_progress_bars=False
        )

        keys = jax.random.split(jax.random.PRNGKey(0), n_samples)

        # This should succeed now
        results = sampler(keys)

        assert isinstance(results, dict)
        assert 'a' in results
        assert 'b' in results
        assert results['a'].shape == (n_samples,)
        assert results['b'].shape == (n_samples,)

        # Check values
        for i in range(n_samples):
            val = jnp.sum(keys[i]) + ncond
            assert jnp.allclose(results['a'][i], val)
            assert jnp.allclose(results['b'][i], val * 2)
