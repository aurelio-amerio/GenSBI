import pytest
import jax
import jax.numpy as jnp
from flax import nnx
from gensbi.diffusion.solver.edm_samplers import edm_sampler, edm_ablation_sampler
from gensbi.diffusion.path.scheduler import EDMScheduler, VEEdmScheduler, VPEdmScheduler


class MockSDE:
    """Mock SDE for testing."""

    def __init__(self, timesteps=None):
        if timesteps is None:
            # Default to 5 steps from 10.0 to 0.0
            self._timesteps = jnp.linspace(10.0, 0.0, 6)[:-1]
        else:
            self._timesteps = timesteps

    def timesteps(self, step_indices, n_steps):
        # We assume step_indices map to indices in _timesteps
        # Or simpler, just return linspace regardless of indices for mock purposes
        return self._timesteps[step_indices]

    def denoise(self, model, x, t, **kwargs):
        # Pass through to model
        return model(x, t)


class MockModel(nnx.Module):
    """Mock model that returns input as denoised output."""

    def __call__(self, x, t):
        # Returning x effectively means the model predicts the clean image is the current noisy image.
        # This prevents the solver from collapsing to zero if we simply returned zeros.
        return x


@pytest.fixture
def mock_sde():
    return MockSDE()


@pytest.fixture
def mock_model():
    return MockModel()


def test_edm_sampler_deterministic(mock_sde, mock_model):
    """Test that edm_sampler is deterministic when S_churn=0."""
    x_1 = jnp.ones((1, 10))
    key1 = jax.random.PRNGKey(0)
    key2 = jax.random.PRNGKey(1)

    # S_churn=0 means no stochasticity added during sampling
    out1 = edm_sampler(
        mock_sde,
        mock_model,
        x_1,
        key=key1,
        n_steps=5,
        S_churn=0,
        return_intermediates=False,
    )

    out2 = edm_sampler(
        mock_sde,
        mock_model,
        x_1,
        key=key2,
        n_steps=5,
        S_churn=0,
        return_intermediates=False,
    )

    # Check that outputs are identical
    diff = jnp.max(jnp.abs(out1 - out2))
    assert diff < 1e-5, f"Expected deterministic output, but got diff {diff}"


def test_edm_sampler_stochasticity(mock_sde, mock_model):
    """Test that edm_sampler is stochastic when S_churn > 0."""
    x_1 = jnp.ones((1, 10))
    key1 = jax.random.PRNGKey(0)
    key2 = jax.random.PRNGKey(1)

    # S_churn > 0 introduces randomness
    out1 = edm_sampler(
        mock_sde,
        mock_model,
        x_1,
        key=key1,
        n_steps=5,
        S_churn=10.0,
        S_min=0.0,
        S_max=20.0,
        S_noise=1.0,
        return_intermediates=False,
    )

    out2 = edm_sampler(
        mock_sde,
        mock_model,
        x_1,
        key=key2,
        n_steps=5,
        S_churn=10.0,
        S_min=0.0,
        S_max=20.0,
        S_noise=1.0,
        return_intermediates=False,
    )

    # Check that outputs are different
    diff = jnp.max(jnp.abs(out1 - out2))
    assert diff > 1e-3, f"Expected stochastic output, but got diff {diff}"

# we removed conditioning from the EDM sampler, as it is not done at the level of the model wrapper.abs
# this way, EDM always sees a "conditional" model, without any masking needed.
# def test_edm_sampler_conditioning(mock_sde, mock_model):
#     """Test that edm_sampler respects conditioning mask and value."""
#     x_1 = jnp.zeros((1, 10))
#     key = jax.random.PRNGKey(0)

#     # Mask indices 1 and 3
#     mask = jnp.array([0, 1, 0, 1, 0, 0, 0, 0, 0, 0]).reshape(1, 10)
#     value = jnp.array([0, 5, 0, 5, 0, 0, 0, 0, 0, 0]).reshape(1, 10)

#     # Run sampler with conditioning
#     # Use S_churn > 0 to ensure values change if not conditioned
#     out = edm_sampler(
#         mock_sde,
#         mock_model,
#         x_1,
#         key=key,
#         condition_mask=mask,
#         condition_value=value,
#         n_steps=5,
#         S_churn=10.0,
#         S_min=0.0,
#         S_max=20.0,
#     )

#     # Check masked positions match condition_value
#     masked_out = out * mask
#     expected_masked = value * mask

#     assert jnp.allclose(
#         masked_out, expected_masked
#     ), "Conditioned values do not match expected values"

#     # Check unmasked positions are not all equal to condition_value (sanity check)
#     # Since model returns input, and we add noise, unmasked values should drift
#     # Step 0: x_next = x_1 * t_steps[0] = 0 * 10 = 0.
#     # Then noise added.
#     unmasked_out = out * (1 - mask)
#     # It shouldn't be all zeros either because of noise
#     assert not jnp.allclose(
#         unmasked_out, 0.0
#     ), "Unmasked values should not be zero due to noise"


# ---------------------------------------------------------------------------
# edm_ablation_sampler tests
# ---------------------------------------------------------------------------


class MockDenoiseModel(nnx.Module):
    """Mock model compatible with BaseSDE.denoise preconditioning."""

    def __call__(self, *, obs, t, **kwargs):
        return obs


@pytest.fixture
def mock_denoise_model():
    return MockDenoiseModel()


@pytest.mark.parametrize(
    "sampling_scheduler_cls",
    [VEEdmScheduler, VPEdmScheduler, EDMScheduler],
)
def test_ablation_sampler_runs(sampling_scheduler_cls, mock_denoise_model):
    """Test that edm_ablation_sampler runs with each scheduler for sampling
    dynamics while using EDMScheduler for denoising (preconditioning)."""
    sampling_scheduler = sampling_scheduler_cls()
    denoise_scheduler = EDMScheduler()

    x_1 = jnp.ones((4, 3))
    key = jax.random.PRNGKey(0)

    out = edm_ablation_sampler(
        sampling_scheduler,
        denoise_scheduler,
        mock_denoise_model,
        x_1,
        key=key,
        n_steps=5,
        S_churn=0,
        method="Heun",
    )
    assert out.shape == x_1.shape


@pytest.mark.parametrize(
    "sampling_scheduler_cls",
    [VEEdmScheduler, VPEdmScheduler],
)
def test_ablation_sampler_deterministic(sampling_scheduler_cls, mock_denoise_model):
    """Ablation sampler is deterministic when S_churn=0."""
    sampling_scheduler = sampling_scheduler_cls()
    denoise_scheduler = EDMScheduler()

    x_1 = jnp.ones((4, 3))

    out1 = edm_ablation_sampler(
        sampling_scheduler,
        denoise_scheduler,
        mock_denoise_model,
        x_1,
        key=jax.random.PRNGKey(0),
        n_steps=5,
        S_churn=0,
        S_noise=1,
    )
    out2 = edm_ablation_sampler(
        sampling_scheduler,
        denoise_scheduler,
        mock_denoise_model,
        x_1,
        key=jax.random.PRNGKey(1),
        n_steps=5,
        S_churn=0,
        S_noise=1,
    )
    assert jnp.isclose(
        out1, out2, rtol=1e-5
    ).all(), f"Expected deterministic output, got diff {jnp.max(jnp.abs((out1 - out2)/out1))}"


@pytest.mark.parametrize(
    "sampling_scheduler_cls",
    [VEEdmScheduler, VPEdmScheduler],
)
def test_ablation_sampler_stochastic(sampling_scheduler_cls, mock_denoise_model):
    """Ablation sampler is stochastic when S_churn > 0."""
    sampling_scheduler = sampling_scheduler_cls()
    denoise_scheduler = EDMScheduler()

    x_1 = jnp.ones((4, 3))

    out1 = edm_ablation_sampler(
        sampling_scheduler,
        denoise_scheduler,
        mock_denoise_model,
        x_1,
        key=jax.random.PRNGKey(0),
        n_steps=5,
        S_churn=10.0,
        S_min=0.0,
        S_max=200.0,
        S_noise=1.0,
    )
    out2 = edm_ablation_sampler(
        sampling_scheduler,
        denoise_scheduler,
        mock_denoise_model,
        x_1,
        key=jax.random.PRNGKey(1),
        n_steps=5,
        S_churn=10.0,
        S_min=0.0,
        S_max=200.0,
        S_noise=1.0,
    )
    diff = jnp.max(jnp.abs(out1 - out2))
    assert diff > 1e-3, f"Expected stochastic output, got diff {diff}"


def test_ablation_sampler_euler(mock_denoise_model):
    """Ablation sampler works with Euler method."""
    sampling_scheduler = VEEdmScheduler()
    denoise_scheduler = EDMScheduler()

    x_1 = jnp.ones((4, 3))
    key = jax.random.PRNGKey(0)

    out = edm_ablation_sampler(
        sampling_scheduler,
        denoise_scheduler,
        mock_denoise_model,
        x_1,
        key=key,
        n_steps=5,
        S_churn=0,
        method="Euler",
    )
    assert out.shape == x_1.shape
