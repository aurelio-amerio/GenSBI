import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import pytest
from gensbi.diffusion.path.sm_path import SMPath
from gensbi.diffusion.path.path_sample import SMPathSample
from gensbi.diffusion.path.scheduler.sm_sde import VPSmScheduler, VESmScheduler


@pytest.mark.parametrize("sde_cls", [VPSmScheduler, VESmScheduler])
class TestSMPath:
    def test_initialization(self, sde_cls):
        sde = sde_cls()
        path = SMPath(sde)
        assert path.name == sde.name

    def test_sample_returns_sm_path_sample(self, sde_cls):
        sde = sde_cls()
        path = SMPath(sde)
        key = jax.random.PRNGKey(0)
        x_1 = jnp.ones((4, 3))
        t = jnp.ones((4, 1))
        sample = path.sample(key, x_1, t)
        assert isinstance(sample, SMPathSample)

    def test_sample_shapes(self, sde_cls):
        sde = sde_cls()
        path = SMPath(sde)
        key = jax.random.PRNGKey(0)
        x_1 = jnp.ones((4, 3))
        t = jnp.ones((4, 1))
        sample = path.sample(key, x_1, t)

        assert sample.x_1.shape == (4, 3)
        assert sample.x_t.shape == (4, 3)
        assert sample.t.shape == (4, 1)
        assert sample.noise.shape == (4, 3)
        assert sample.std_t.shape == (4, 1)

    def test_sample_prior_shape(self, sde_cls):
        sde = sde_cls()
        path = SMPath(sde)
        key = jax.random.PRNGKey(0)
        prior = path.sample_prior(key, (10, 5))
        assert prior.shape == (10, 5)

    def test_get_loss_fn_callable(self, sde_cls):
        sde = sde_cls()
        path = SMPath(sde)
        loss_fn = path.get_loss_fn()
        assert callable(loss_fn)

    def test_get_batch(self, sde_cls):
        sde = sde_cls()
        path = SMPath(sde)
        key = jax.random.PRNGKey(0)
        x_1 = jnp.ones((4, 3))
        t = jnp.ones((4, 1))
        sample = path.sample(key, x_1, t)
        batch = sample.get_batch()
        assert len(batch) == 5
        assert batch[0] is sample.x_1
        assert batch[1] is sample.x_t
        assert batch[2] is sample.t
        assert batch[3] is sample.noise
        assert batch[4] is sample.std_t


class TestSMPathLoss:
    def test_loss_runs_vp(self):
        sde = VPSmScheduler()
        path = SMPath(sde)
        loss_fn = path.get_loss_fn()

        batch_size = 4
        x_1 = jnp.ones((batch_size, 3))
        t = jnp.ones((batch_size, 1))

        # Now we construct the batch as passed from the pipeline to the loss function
        # The pipeline samples t and constructs batch = (x_1, t)
        # However, the path.get_loss_fn() expects the UNPACKED sample batch: (x_1, x_t, t, noise, std_t)
        # Wait, the pipeline calls path.sample(key, x_1, t) which returns a Sample object.
        # The Loss class (e.g. UnconditionalSMLoss) calls path.sample(), gets the sample object,
        # calls sample.get_batch() which returns (x_1, x_t, t, noise, std_t),
        # and THEN passes this 5-element tuple to loss_fn.

        key = jax.random.PRNGKey(0)
        path_sample = path.sample(key, x_1, t)
        batch = path_sample.get_batch()

        def model(obs, t, **kwargs):
            return obs + t

        loss = loss_fn(model, batch)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_loss_runs_ve(self):
        sde = VESmScheduler()
        path = SMPath(sde)
        loss_fn = path.get_loss_fn()

        batch_size = 4
        x_1 = jnp.ones((batch_size, 3))
        t = jnp.ones((batch_size, 1))

        key = jax.random.PRNGKey(0)
        path_sample = path.sample(key, x_1, t)
        batch = path_sample.get_batch()

        def model(obs, t, **kwargs):
            return obs + t

        loss = loss_fn(model, batch)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_loss_with_mask(self):
        sde = VPSmScheduler()
        path = SMPath(sde)
        loss_fn = path.get_loss_fn()

        batch_size = 4
        x_1 = jnp.ones((batch_size, 3))
        t = jnp.ones((batch_size, 1))

        key = jax.random.PRNGKey(0)
        path_sample = path.sample(key, x_1, t)
        batch = path_sample.get_batch()

        mask = jnp.array([[True, False, False]] * batch_size)

        def model(obs, t, **kwargs):
            return obs + t

        loss = loss_fn(model, batch, condition_mask=mask)
        assert loss.shape == ()
        assert jnp.isfinite(loss)
