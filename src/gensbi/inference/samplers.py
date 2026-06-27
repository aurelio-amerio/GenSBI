"""blackjax samplers consumed by NLEPosterior. blackjax imported lazily in run()."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp


def _rescale(mu):
    """Map a mean trajectory length to a uniform-integer draw scale.

    From the blackjax adjusted-MCLMC tutorial: choosing the number of integration
    steps as ceil(U(0,1) * _rescale(L/step_size)) keeps the average near the tuned L.
    """
    k = jax.lax.max(1, jnp.round(jnp.log(2 * mu - 1) / jnp.log(2)).astype(int))
    return mu / k


class Sampler(ABC):
    """Turns a PosteriorTarget into posterior samples."""

    @abstractmethod
    def run(self, key, target):
        """Return (samples (n, dim), info)."""
        raise NotImplementedError


@dataclass(frozen=True)
class MclmcInfo:
    L: float
    step_size: float
    acceptance_rate: float
    num_samples: int
    num_chains: int


def _inference_loop(rng_key, step_fn, initial_state, num_samples):
    """Run a blackjax SamplingAlgorithm.step for num_samples via lax.scan."""
    @jax.jit
    def one_step(state, k):
        state, info = step_fn(k, state)
        return state, (state, info)
    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
    return states, infos


class MCLMC(Sampler):
    """Microcanonical Langevin Monte Carlo.

    adjusted=True (default, added in Task 3) is MH-corrected / asymptotically exact.
    adjusted=False is the faster unadjusted variant (biased by the discretization).
    """

    def __init__(self, *, adjusted=True, num_samples=1000, num_tuning_steps=5000,
                 num_chains=1, target_acceptance=0.9, diagonal_preconditioning=True):
        self.adjusted = adjusted
        self.num_samples = num_samples
        self.num_tuning_steps = num_tuning_steps
        self.num_chains = num_chains
        self.target_acceptance = target_acceptance
        self.diagonal_preconditioning = diagonal_preconditioning

    def run(self, key, target):
        # Python loop over chains (not vmap): _run_single returns a plain MclmcInfo
        # dataclass, which is not a registered pytree and cannot be a vmap output.
        # num_chains is small, so the loop cost is negligible.
        keys = jax.random.split(key, self.num_chains)
        results = [self._run_single(k, target) for k in keys]
        samples = jnp.concatenate([r[0] for r in results], axis=0)  # (num_chains*num_samples, dim)
        if self.num_chains == 1:
            return samples, results[0][1]
        infos = [r[1] for r in results]
        info = MclmcInfo(
            L=float(jnp.mean(jnp.array([i.L for i in infos]))),
            step_size=float(jnp.mean(jnp.array([i.step_size for i in infos]))),
            acceptance_rate=float(jnp.mean(jnp.array([i.acceptance_rate for i in infos]))),
            num_samples=self.num_samples, num_chains=self.num_chains,
        )
        return samples, info

    def _run_single(self, key, target):
        if self.adjusted:
            return self._run_adjusted(key, target)
        return self._run_unadjusted(key, target)

    def _run_unadjusted(self, key, target):
        import blackjax
        from blackjax.mcmc.integrators import isokinetic_mclachlan

        pos_key, mom_key, tune_key, run_key = jax.random.split(key, 4)
        position = target.prior.sample(pos_key, ())
        init_state = blackjax.mcmc.mclmc.init(
            position=position, logdensity_fn=target.log_posterior, rng_key=mom_key)
        kernel = lambda inverse_mass_matrix: blackjax.mcmc.mclmc.build_kernel(
            logdensity_fn=target.log_posterior, integrator=isokinetic_mclachlan,
            inverse_mass_matrix=inverse_mass_matrix)
        state, params, _ = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel, num_steps=self.num_tuning_steps, state=init_state,
            rng_key=tune_key, diagonal_preconditioning=self.diagonal_preconditioning)
        alg = blackjax.mclmc(target.log_posterior, L=params.L, step_size=params.step_size,
                             inverse_mass_matrix=params.inverse_mass_matrix)
        states, _ = _inference_loop(run_key, alg.step, state, self.num_samples)
        info = MclmcInfo(L=params.L, step_size=params.step_size,
                         acceptance_rate=jnp.nan, num_samples=self.num_samples,
                         num_chains=self.num_chains)
        return states.position, info

    def _run_adjusted(self, key, target):
        import blackjax
        from blackjax.mcmc.integrators import isokinetic_mclachlan

        pos_key, init_key, tune_key, run_key = jax.random.split(key, 4)
        position = target.prior.sample(pos_key, ())
        init_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
            position=position, logdensity_fn=target.log_posterior,
            random_generator_arg=init_key)

        def kernel(rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix):
            return blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
                integration_steps_fn=lambda k: jnp.ceil(
                    jax.random.uniform(k) * _rescale(avg_num_integration_steps)),
                inverse_mass_matrix=inverse_mass_matrix,
            )(rng_key=rng_key, state=state, step_size=step_size,
              logdensity_fn=target.log_posterior)

        state, params, _ = blackjax.adjusted_mclmc_find_L_and_step_size(
            mclmc_kernel=kernel, num_steps=self.num_tuning_steps, state=init_state,
            rng_key=tune_key, target=self.target_acceptance,
            diagonal_preconditioning=self.diagonal_preconditioning)

        alg = blackjax.adjusted_mclmc_dynamic(
            logdensity_fn=target.log_posterior, step_size=params.step_size,
            integration_steps_fn=lambda k: jnp.ceil(
                jax.random.uniform(k) * _rescale(params.L / params.step_size)),
            inverse_mass_matrix=params.inverse_mass_matrix)

        states, infos = _inference_loop(run_key, alg.step, state, self.num_samples)
        info = MclmcInfo(L=params.L, step_size=params.step_size,
                         acceptance_rate=jnp.mean(infos.acceptance_rate),
                         num_samples=self.num_samples, num_chains=self.num_chains)
        return states.position, info


@dataclass(frozen=True)
class SmcInfo:
    log_evidence: float
    num_temperature_steps: int
    final_tempering_param: float


class TemperedSMC(Sampler):
    """Adaptive tempered SMC for (possibly multimodal) posteriors.

    Walks particles along p(theta) * q(x_o|theta)^beta for beta: 0 -> 1, choosing the
    beta-ladder adaptively to hold target_ess. Inner rejuvenation kernel is adjusted
    MCLMC by default (Task 5); NUTS is the fallback.
    """

    def __init__(self, *, num_particles=1000, target_ess=0.5, num_mcmc_steps=10,
                 inner_kernel="mclmc", inner_step_size=0.1,
                 inner_num_integration_steps=5, inner_inverse_mass_matrix=None):
        self.num_particles = num_particles
        self.target_ess = target_ess
        self.num_mcmc_steps = num_mcmc_steps
        self.inner_kernel = inner_kernel
        self.inner_step_size = inner_step_size
        self.inner_num_integration_steps = inner_num_integration_steps
        self.inner_inverse_mass_matrix = inner_inverse_mass_matrix

    def _inner(self, target):
        """Return (mcmc_step_fn, mcmc_init_fn, mcmc_parameters) for the inner kernel."""
        import blackjax
        imm = self.inner_inverse_mass_matrix
        if imm is None:
            imm = jnp.ones(target.dim)
        if self.inner_kernel == "nuts":
            step_fn = blackjax.nuts.build_kernel()
            init_fn = blackjax.nuts.init
            params = dict(step_size=self.inner_step_size, inverse_mass_matrix=imm)
            return step_fn, init_fn, params
        if self.inner_kernel == "mclmc":
            raise NotImplementedError("mclmc inner kernel added in Task 5")
        raise ValueError(f"unknown inner_kernel {self.inner_kernel!r}")

    def run(self, key, target):
        import blackjax

        init_key, smc_key = jax.random.split(key)
        step_fn, init_fn, params = self._inner(target)
        smc = blackjax.adaptive_tempered_smc(
            logprior_fn=target.log_prior, loglikelihood_fn=target.log_likelihood,
            mcmc_step_fn=step_fn, mcmc_init_fn=init_fn,
            mcmc_parameters=blackjax.smc.extend_params(params),
            resampling_fn=blackjax.smc.resampling.systematic,
            target_ess=self.target_ess, num_mcmc_steps=self.num_mcmc_steps,
        )
        init_particles = target.prior.sample(init_key, (self.num_particles,))
        state = smc.init(init_particles)

        def cond(carry):
            _, st, _, _ = carry
            return st.tempering_param < 1.0

        def body(carry):
            k, st, n, logZ = carry
            k, sub = jax.random.split(k)
            st, info = smc.step(sub, st)
            return k, st, n + 1, logZ + info.log_likelihood_increment

        _, final, nsteps, logZ = jax.lax.while_loop(
            cond, body, (smc_key, state, 0, 0.0))
        info = SmcInfo(log_evidence=logZ, num_temperature_steps=nsteps,
                       final_tempering_param=final.tempering_param)
        return final.particles, info
