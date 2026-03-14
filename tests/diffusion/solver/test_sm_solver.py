import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.diffusion.solver.sm_solver import SMSolver, SMPFSolver
from gensbi.diffusion.solver.sm_ode_solver_new import NewSMODESolver
from gensbi.diffusion.solver.sm_sde_solver_new import NewSMSDESolver
from gensbi.utils.model_wrapping import ModelWrapper, ScoreToODEDrift
from gensbi.diffusion.path.sm_path import SMPath
from gensbi.diffusion.path.scheduler.sm_sde import VPSmScheduler, VESmScheduler


class DummyScoreModel(nnx.Module):
    def __call__(self, obs, t, **kwargs):
        return jnp.zeros_like(obs) + t


def _build_smpf_solver(score_model, path):
    """Construct SMPFSolver via ScoreToODEDrift + ModelWrapper."""
    sde = path.scheduler
    drift_model = ScoreToODEDrift(score_model=score_model, sde=sde)
    wrapper = ModelWrapper(model=drift_model)
    return SMPFSolver(velocity_model=wrapper)


def _build_smpf_sampler(score_model, path, nsteps, return_intermediates=False):
    """Build a sampler via SMPFSolver, matching what the pipeline does."""
    solver = _build_smpf_solver(score_model, path)
    sde = path.scheduler
    T = sde.T
    eps = 1e-3

    if return_intermediates:
        time_grid = jnp.linspace(T, eps, nsteps + 1)
    else:
        time_grid = jnp.array([T, eps])

    step_size = -(T - eps) / nsteps

    return solver.get_sampler(
        step_size=step_size,
        method="Euler",
        time_grid=time_grid,
        return_intermediates=return_intermediates,
    )


# =========================================================
# Basic initialization tests
# =========================================================


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_solver_initialization(sde_cls):
    sde = sde_cls()
    path = SMPath(sde)
    solver = SMSolver(score_model=None, path=path)
    assert isinstance(solver, SMSolver)
    assert solver.path is path


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_pf_solver_initialization(sde_cls):
    sde = sde_cls()
    path = SMPath(sde)
    solver = _build_smpf_solver(DummyScoreModel(), path)
    assert isinstance(solver, SMPFSolver)


# =========================================================
# 3D shape tests: (batch, features, channels)
# =========================================================


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_solver_sample_3d(sde_cls):
    """Test sampling with input shape (batch, features, channels)."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    solver = SMSolver(score_model=score_model, path=path)
    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (5, 3, 4))

    samples = solver.sample(key, x_init, nsteps=5, return_intermediates=False)
    assert samples.shape == (5, 3, 4)


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_pf_solver_sample_3d(sde_cls):
    """Test sampling with 3D input via ScoreToODEDrift + SMPFSolver."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    sampler = _build_smpf_sampler(score_model, path, nsteps=5)

    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (5, 3, 4))

    samples = sampler(x_init)
    assert samples.shape == (5, 3, 4)


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_solver_sample_3d_intermediates(sde_cls):
    """Test sampling with intermediates for 3D input."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    solver = SMSolver(score_model=score_model, path=path)
    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (5, 3, 4))

    samples = solver.sample(key, x_init, nsteps=5, return_intermediates=True)
    assert samples.shape == (6, 5, 3, 4)  # (n_steps+1, batch, features, channels)


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_pf_solver_sample_3d_intermediates(sde_cls):
    """Test sampling with intermediates via ScoreToODEDrift + SMPFSolver."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    sampler = _build_smpf_sampler(
        score_model, path, nsteps=5, return_intermediates=True
    )

    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (5, 3, 4))

    samples = sampler(x_init)
    assert samples.shape == (6, 5, 3, 4)  # (n_steps+1, batch, features, channels)


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_solver_sample_channel1(sde_cls):
    """Test with channel=1 (user adds trailing dim to 2D data)."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    solver = SMSolver(score_model=score_model, path=path)
    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (5, 3, 1))

    samples = solver.sample(key, x_init, nsteps=5)
    assert samples.shape == (5, 3, 1)


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_pf_solver_sample_channel1(sde_cls):
    """Test with channel=1 via ScoreToODEDrift + SMPFSolver."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    sampler = _build_smpf_sampler(score_model, path, nsteps=5)

    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (5, 3, 1))

    samples = sampler(x_init)
    assert samples.shape == (5, 3, 1)


# =========================================================
# 2D inputs should raise an error (SMSolver only)
# =========================================================


def test_sm_solver_rejects_2d_input():
    """2D inputs (batch, features) should raise AssertionError."""
    score_model = DummyScoreModel()
    sde = VPSmScheduler()
    path = SMPath(sde)
    solver = SMSolver(score_model=score_model, path=path)
    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (5, 3))

    with pytest.raises(AssertionError, match="batch, features, channels"):
        solver.sample(key, x_init, nsteps=5)


# =========================================================
# CFG scale not implemented test (SMSolver only)
# =========================================================


def test_sm_solver_cfg_scale_not_implemented():
    sde = VPSmScheduler()
    path = SMPath(sde)
    solver = SMSolver(score_model=DummyScoreModel(), path=path)
    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (2, 2, 1))
    with pytest.raises(NotImplementedError):
        solver.sample(key, x_init, nsteps=2, cfg_scale=1.0)


# =========================================================
# Batch independence tests (regression for shared-RNG bug)
# =========================================================


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_solver_batch_independence(sde_cls):
    """Verify all samples in a batch are independent (not identical).

    Regression test: a previous bug caused all samples to share the same
    Brownian motion, producing identical outputs.
    """
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    solver = SMSolver(score_model=score_model, path=path)
    key = jax.random.PRNGKey(42)
    n_batch = 8
    x_init = path.sample_prior(key, (n_batch, 3, 4))

    samples = solver.sample(key, x_init, nsteps=10)
    assert samples.shape == (n_batch, 3, 4)

    for i in range(n_batch):
        for j in range(i + 1, n_batch):
            assert not jnp.allclose(
                samples[i], samples[j], atol=1e-6
            ), f"Samples {i} and {j} are identical — batch independence violated"


# =========================================================
# Adaptive step sizing tests (automatic from method choice)
# =========================================================


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_solver_shark(sde_cls):
    """ShARK method automatically uses PIDController adaptive stepping."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    solver = SMSolver(score_model=score_model, path=path)
    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (5, 3, 4))

    samples = solver.sample(key, x_init, nsteps=5, method="ShARK")
    assert samples.shape == (5, 3, 4)


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_pf_solver_dopri5(sde_cls):
    """Dopri5 method automatically uses PIDController adaptive stepping."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    solver = _build_smpf_solver(score_model, path)

    sde = path.scheduler
    T = sde.T
    eps = 1e-3
    time_grid = jnp.array([T, eps])

    sampler = solver.get_sampler(
        step_size=None,
        method="Dopri5",
        time_grid=time_grid,
    )

    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (5, 3, 4))

    samples = sampler(x_init)
    assert samples.shape == (5, 3, 4)


# =========================================================
# Condition mask tests (SMSolver only — conditioning is a
# wrapper concern for SMPFSolver)
# =========================================================


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_sm_solver_with_condition_mask(sde_cls):
    """Test SDE sampler with condition_mask and condition_value."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)
    solver = SMSolver(score_model=score_model, path=path)
    key = jax.random.PRNGKey(0)

    x_init = path.sample_prior(key, (3, 4, 1))
    condition_mask = jnp.zeros((3, 4, 1))
    condition_mask = condition_mask.at[:, 0:2, :].set(1.0)
    condition_value = jnp.ones((3, 4, 1)) * 2.0

    samples = solver.sample(
        key, x_init, nsteps=5,
        condition_mask=condition_mask,
        condition_value=condition_value,
    )
    assert samples.shape == (3, 4, 1)


# =========================================================
# Prior distribution tests
# =========================================================


def test_vp_prior_log_prob():
    """VP prior log_prob should match standard normal."""
    from gensbi.diffusion.sm_prior import VPPrior

    prior = VPPrior()
    key = jax.random.PRNGKey(0)
    x = prior.sample(key, (8, 3, 1))
    lp = prior.log_prob(x)
    assert lp.shape == (8,)
    assert jnp.all(jnp.isfinite(lp))


def test_ve_prior_log_prob():
    """VE prior log_prob should match N(0, sigma_max^2 I)."""
    from gensbi.diffusion.sm_prior import VEPrior

    sigma_max = 15.0
    prior = VEPrior(sigma_max=sigma_max)
    key = jax.random.PRNGKey(0)
    x = prior.sample(key, (8, 3, 1))
    lp = prior.log_prob(x)
    assert lp.shape == (8,)
    assert jnp.all(jnp.isfinite(lp))

    # VP has higher density at zero (narrower distribution)
    from gensbi.diffusion.sm_prior import VPPrior

    x_zero = jnp.zeros((1, 3, 1))
    vp_lp = VPPrior().log_prob(x_zero)
    ve_lp = prior.log_prob(x_zero)
    assert vp_lp > ve_lp


# =========================================================
# Pipeline integration test
# =========================================================


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_score_matching_method_smpf_solver(sde_cls):
    """ScoreMatchingMethod.build_sampler_fn with SMPFSolver produces valid samples."""
    from gensbi.core.score_matching import ScoreMatchingMethod

    sde_type = "VP" if sde_cls is VPSmScheduler else "VE"
    method = ScoreMatchingMethod(sde_type=sde_type)

    config = method.get_extra_training_config()
    path = method.build_path(config, event_shape=(3, 1))

    score_model = DummyScoreModel()

    sampler_fn = method.build_sampler_fn(
        model_wrapped=score_model,
        path=path,
        model_extras={},
        nsteps=10,
        solver=(NewSMODESolver, {}),
    )

    key = jax.random.PRNGKey(0)
    x_init = path.sample_prior(key, (4, 3, 1))
    result = sampler_fn(key, x_init)

    assert result.shape == (4, 3, 1)
    assert jnp.all(jnp.isfinite(result))

    # Verify prior was set correctly
    assert method.prior is not None
    lp = method.prior.log_prob(x_init)
    assert lp.shape == (4,)


# =========================================================
# Log-probability tests (SMPFSolver / ODESolver.get_log_prob)
# =========================================================


def _build_smpf_log_prob(score_model, path, nsteps=100, exact_divergence=True):
    """Build a log_prob callable via SMPFSolver, matching SM time conventions."""
    solver = _build_smpf_solver(score_model, path)
    sde = path.scheduler
    T = sde.T
    eps = 1e-3
    time_grid = jnp.array([T, eps])
    step_size = (T - eps) / nsteps

    from gensbi.diffusion.sm_prior import VPPrior, VEPrior

    if path.name == "SM-VP":
        prior = VPPrior()
    else:
        prior = VEPrior(sigma_max=sde.sigma_max)

    return solver.get_log_prob(
        log_p0=prior.log_prob,
        step_size=step_size,
        method="Euler",
        time_grid=time_grid,
        exact_divergence=exact_divergence,
    )


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_smpf_log_prob_shape(sde_cls):
    """SMPFSolver.get_log_prob produces log-prob with shape (batch,)."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)

    log_prob_fn = _build_smpf_log_prob(score_model, path, nsteps=10)

    key = jax.random.PRNGKey(0)
    x_1 = path.sample_prior(key, (5, 3, 1))

    logp = log_prob_fn(x_1)
    assert logp.shape == (5,)
    assert jnp.all(jnp.isfinite(logp))


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_smpf_log_prob_exact_vs_hutchinson(sde_cls):
    """Hutchinson log-prob should match exact log-prob for the dummy score model."""
    score_model = DummyScoreModel()
    sde = sde_cls()
    path = SMPath(sde)

    key = jax.random.PRNGKey(0)
    x_1 = path.sample_prior(key, (4, 2, 1))

    # Exact divergence
    logp_exact_fn = _build_smpf_log_prob(
        score_model, path, nsteps=10, exact_divergence=True
    )
    logp_exact = logp_exact_fn(x_1)

    # Hutchinson divergence
    hutch_key = jax.random.PRNGKey(42)
    logp_hutch_fn = _build_smpf_log_prob(
        score_model, path, nsteps=10, exact_divergence=False
    )
    logp_hutch = logp_hutch_fn(x_1, key=hutch_key)

    assert logp_hutch.shape == logp_exact.shape
    assert jnp.allclose(logp_exact, logp_hutch, rtol=1e-3), (
        f"Exact vs Hutchinson mismatch: exact={logp_exact}, hutch={logp_hutch}"
    )


# =========================================================
# Pipeline-level log-prob tests
# =========================================================


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_score_matching_method_build_log_prob_fn(sde_cls):
    """ScoreMatchingMethod.build_log_prob_fn with SMPFSolver produces valid log-prob."""
    from gensbi.core.score_matching import ScoreMatchingMethod

    sde_type = "VP" if sde_cls is VPSmScheduler else "VE"
    method = ScoreMatchingMethod(sde_type=sde_type)

    config = method.get_extra_training_config()
    path = method.build_path(config, event_shape=(3, 1))

    score_model = DummyScoreModel()

    log_prob_fn = method.build_log_prob_fn(
        model_wrapped=score_model,
        path=path,
        model_extras={},
        nsteps=10,
        method="Euler",
        solver=(NewSMODESolver, {}),
    )

    key = jax.random.PRNGKey(0)
    x_1 = path.sample_prior(key, (4, 3, 1))
    logp = log_prob_fn(x_1)

    assert logp.shape == (4,)
    assert jnp.all(jnp.isfinite(logp))


def test_score_matching_method_log_prob_rejects_sde_solver():
    """build_log_prob_fn should reject SMSolver (SDE, not ODE)."""
    from gensbi.core.score_matching import ScoreMatchingMethod

    method = ScoreMatchingMethod(sde_type="VP")
    config = method.get_extra_training_config()
    path = method.build_path(config, event_shape=(3, 1))

    with pytest.raises(NotImplementedError, match="SMODESolver"):
        method.build_log_prob_fn(
            model_wrapped=DummyScoreModel(),
            path=path,
            model_extras={},
            solver=(NewSMSDESolver, {}),
        )


