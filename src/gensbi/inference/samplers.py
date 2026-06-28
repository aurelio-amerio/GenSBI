"""blackjax samplers consumed by NLEPosterior. blackjax imported lazily in run()."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp


def _rescale(mu):
    """Map a mean trajectory length to a uniform-integer draw scale.

    From the blackjax adjusted-MCLMC tutorial: choosing the number of integration
    steps as ceil(U(0,1) * _rescale(L/step_size)) keeps the average near the tuned L.

    ``mu`` must satisfy ``2 * mu - 1 > 0`` (i.e. ``mu > 0.5``); see
    :func:`_check_rescale_domain` for the host-side guard applied to the tuned value.
    """
    k = jax.lax.max(1, jnp.round(jnp.log(2 * mu - 1) / jnp.log(2)).astype(int))
    return mu / k


def _check_rescale_domain(mu):
    """Raise if the tuned ``mu = L / step_size`` is outside ``_rescale``'s domain.

    ``_rescale`` takes ``log(2 * mu - 1)``, which is non-finite for ``mu <= 0.5``.
    A host-side check on the tuned value turns an otherwise silent all-NaN
    posterior into an explicit error. (The in-tuning average is left to blackjax;
    this is a convenience sampler, not a fully hardened MCMC engine.)
    """
    mu = float(mu)
    if 2.0 * mu - 1.0 <= 0.0:
        raise ValueError(
            f"adjusted-MCLMC tuning produced L/step_size = {mu:.4g} <= 0.5, for "
            f"which the integration-step rescaling log(2*mu - 1) is undefined; "
            f"the run would yield all-NaN samples. This usually means tuning did "
            f"not converge — try increasing num_tuning_steps, increasing "
            f"num_samples, or using MCLMC(adjusted=False).")


class Sampler(ABC):
    """Abstract base class for posterior samplers.

    Subclasses consume a :class:`~gensbi.inference.posterior.PosteriorTarget`
    and return an array of posterior samples together with a sampler-specific
    info object.
    """

    @abstractmethod
    def run(self, key, target):
        """Draw posterior samples from a log-density target.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        target : PosteriorTarget
            Posterior target produced by
            :meth:`~gensbi.inference.posterior.NLEPosterior.build_target`.

        Returns
        -------
        samples : Array
            Posterior samples of shape ``(n, dim)``.
        info : object
            Sampler-specific diagnostics.

        Raises
        ------
        NotImplementedError
            This is an abstract method; subclasses must override it.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class MclmcInfo:
    """Tuning parameters and diagnostics from an MCLMC run.

    Parameters
    ----------
    L : float
        Tuned trajectory length.
    step_size : float
        Tuned integrator step size.
    acceptance_rate : float
        Mean Metropolis acceptance rate over the sampling run.
        ``float('nan')`` for the unadjusted variant.
    num_samples : int
        Number of samples drawn per chain.
    num_chains : int
        Number of independent chains.
    """

    L: float
    step_size: float
    acceptance_rate: float
    num_samples: int
    num_chains: int


def _inference_loop(rng_key, step_fn, initial_state, num_samples):
    """Run a blackjax SamplingAlgorithm.step for num_samples via lax.scan."""
    def one_step(state, k):
        state, info = step_fn(k, state)
        return state, (state, info)
    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)
    return states, infos


class MCLMC(Sampler):
    """Microcanonical Langevin Monte Carlo sampler.

    ``adjusted=True`` (the default) applies an MH correction for
    asymptotically exact sampling.  ``adjusted=False`` uses the faster
    unadjusted variant, which is biased by the discretization error.

    Parameters
    ----------
    adjusted : bool, optional
        Whether to apply an MH correction.  Default is ``True``.
    num_samples : int, optional
        Number of posterior samples to collect per chain.  Default is 1000.
    num_tuning_steps : int, optional
        Number of warmup steps for the L / step-size tuning loop.
        Default is 5000.
    num_chains : int, optional
        Number of independent MCLMC chains.  Default is 1.
    target_acceptance : float, optional
        Target Metropolis acceptance rate for the adjusted variant.
        Default is 0.9.
    diagonal_preconditioning : bool, optional
        Whether to use diagonal preconditioning during tuning.
        Default is ``True``.
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
        """Draw posterior samples using MCLMC.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        target : PosteriorTarget
            Posterior target produced by
            :meth:`~gensbi.inference.posterior.NLEPosterior.build_target`.

        Returns
        -------
        samples : Array
            Posterior samples of shape ``(num_chains * num_samples, dim)``.
        info : MclmcInfo
            Tuning parameters and diagnostic information.
        """
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
        info = MclmcInfo(L=float(params.L), step_size=float(params.step_size),
                         acceptance_rate=float(jnp.nan), num_samples=self.num_samples,
                         num_chains=self.num_chains)
        return states.position, info

    def _run_adjusted(self, key, target):
        import blackjax

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
        _check_rescale_domain(params.L / params.step_size)

        alg = blackjax.adjusted_mclmc_dynamic(
            logdensity_fn=target.log_posterior, step_size=params.step_size,
            integration_steps_fn=lambda k: jnp.ceil(
                jax.random.uniform(k) * _rescale(params.L / params.step_size)),
            inverse_mass_matrix=params.inverse_mass_matrix)

        states, infos = _inference_loop(run_key, alg.step, state, self.num_samples)
        info = MclmcInfo(L=float(params.L), step_size=float(params.step_size),
                         acceptance_rate=float(jnp.mean(infos.acceptance_rate)),
                         num_samples=self.num_samples, num_chains=self.num_chains)
        return states.position, info


@dataclass(frozen=True)
class SmcInfo:
    """Diagnostics from an adaptive tempered SMC run.

    Parameters
    ----------
    log_evidence : float
        Log marginal likelihood estimate accumulated over tempering steps.
    num_temperature_steps : int
        Number of temperature increments taken from beta=0 to beta=1.
    final_tempering_param : float
        Final value of the tempering parameter beta (should be 1.0).
    """

    log_evidence: float
    num_temperature_steps: int
    final_tempering_param: float


class TemperedSMC(Sampler):
    """Adaptive tempered Sequential Monte Carlo for (possibly multimodal) posteriors.

    Walks particles along ``p(theta) * q(x_o | theta)^beta`` for beta from 0
    to 1, choosing the beta ladder adaptively to maintain ``target_ess``.  The
    inner rejuvenation kernel is adjusted MCLMC by default; NUTS is available
    as a fallback.

    Parameters
    ----------
    num_particles : int, optional
        Number of SMC particles.  Default is 1000.
    target_ess : float, optional
        Target effective sample size ratio in ``(0, 1)``, used to adapt the
        temperature increments.  Default is 0.5.
    num_mcmc_steps : int, optional
        Number of inner MCMC rejuvenation steps per temperature.  Default is 10.
    inner_kernel : str, optional
        Inner MCMC kernel: ``"mclmc"`` (adjusted MCLMC, default) or
        ``"nuts"``.
    inner_step_size : float, optional
        Step size for the inner MCMC kernel.  Default is 0.1.
    inner_num_integration_steps : int, optional
        Number of integration steps for the inner MCLMC kernel.  Default is 5.
    inner_inverse_mass_matrix : Array or None, optional
        Inverse mass matrix for the inner kernel.  If ``None``, defaults to a
        vector of ones of length ``target.dim``.  Default is ``None``.
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
            from blackjax.mcmc.integrators import isokinetic_mclachlan

            def step_fn(rng_key, state, logdensity_fn, step_size,
                        num_integration_steps, inverse_mass_matrix):
                # Build the adjusted-MCLMC kernel bound to the *tempered* logdensity
                # SMC injects at each temperature.
                kernel = blackjax.mcmc.adjusted_mclmc.build_kernel(
                    logdensity_fn=logdensity_fn, integrator=isokinetic_mclachlan,
                    inverse_mass_matrix=inverse_mass_matrix)
                return kernel(rng_key, state, step_size=step_size,
                              num_integration_steps=num_integration_steps)

            init_fn = blackjax.mcmc.adjusted_mclmc.init   # (position, logdensity_fn) -> HMCState
            params = dict(step_size=self.inner_step_size,
                          num_integration_steps=self.inner_num_integration_steps,
                          inverse_mass_matrix=imm)
            return step_fn, init_fn, params
        raise ValueError(f"unknown inner_kernel {self.inner_kernel!r}")

    def run(self, key, target):
        """Draw posterior samples using adaptive tempered SMC.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random key.
        target : PosteriorTarget
            Posterior target produced by
            :meth:`~gensbi.inference.posterior.NLEPosterior.build_target`.

        Returns
        -------
        samples : Array
            Posterior samples of shape ``(num_particles, dim)``.
        info : SmcInfo
            SMC diagnostics including log evidence and tempering statistics.
        """
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
        info = SmcInfo(log_evidence=float(logZ), num_temperature_steps=int(nsteps),
                       final_tempering_param=float(final.tempering_param))
        return final.particles, info
