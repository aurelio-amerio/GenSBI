import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from gensbi.diffusion.solver.sm_ode_solver_new import NewSMODESolver
from gensbi.diffusion.solver.sm_sde_solver_new import NewSMSDESolver
from gensbi.utils.model_wrapping import ModelWrapper, ScoreToODEDrift
from gensbi.diffusion.path.sm_path import SMPath
from gensbi.diffusion.path.scheduler.sm_sde import VPSmScheduler, VESmScheduler


class DummyScoreModel(nnx.Module):
    def __call__(self, obs, t, **kwargs):
        return jnp.zeros_like(obs) + t


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
    """ScoreMatchingMethod.build_sampler_fn with NewSMODESolver produces valid samples."""
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
# Pipeline-level log-prob tests
# =========================================================


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
def test_score_matching_method_build_log_prob_fn(sde_cls):
    """ScoreMatchingMethod.build_log_prob_fn with NewSMODESolver produces valid log-prob."""
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
    """build_log_prob_fn should reject SMSDESolver (SDE, not ODE)."""
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
