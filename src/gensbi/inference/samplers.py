"""blackjax samplers consumed by NLEPosterior. blackjax imported lazily in run()."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax
import jax.numpy as jnp


def _rescale(mu):
    """Map a mean trajectory length to a uniform-integer draw scale.

    From blackjax's ``adjusted_mclmc_dynamic``: drawing the number of
    integration steps as ``ceil(U(0,1) * _rescale(L/step_size))`` makes the
    average number of steps exactly ``mu = L / step_size``.

    ``mu`` must satisfy ``mu >= 1``; see :func:`_check_rescale_domain` for the
    host-side guard applied to the tuned value.
    """
    k = jnp.floor(2 * mu - 1)
    x = k * (mu - 0.5 * (k + 1)) / (k + 1 - mu)
    return k + x


def _check_rescale_domain(mu):
    """Raise if the tuned ``mu = L / step_size`` is outside ``_rescale``'s domain.

    For ``mu < 1``, ``floor(2 * mu - 1) == 0`` and ``_rescale`` returns 0, so
    the integration-step draw ``ceil(U(0,1) * 0)`` is 0 — a chain that never
    moves. A host-side check on the tuned value turns that silent failure into
    an explicit error. (The in-tuning average is left to blackjax; this is a
    convenience sampler, not a fully hardened MCMC engine.)
    """
    mu = float(mu)
    if mu < 1.0:
        raise ValueError(
            f"adjusted-MCLMC tuning produced L/step_size = {mu:.4g} < 1, for "
            f"which the randomized integration-step count rounds to zero and "
            f"the chain would never move. This usually means tuning did not "
            f"converge — try increasing num_tuning_steps, increasing "
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
        # blackjax >= 1.6: build_kernel no longer binds logdensity_fn /
        # inverse_mass_matrix; the tuner threads them through per call.
        kernel = blackjax.mcmc.mclmc.build_kernel(integrator=isokinetic_mclachlan)
        state, params, _ = blackjax.mclmc_find_L_and_step_size(
            mclmc_kernel=kernel, logdensity_fn=target.log_posterior,
            num_steps=self.num_tuning_steps, state=init_state,
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
        from blackjax.mcmc.integrators import isokinetic_mclachlan

        pos_key, init_key, tune_key, run_key = jax.random.split(key, 4)
        position = target.prior.sample(pos_key, ())
        init_state = blackjax.mcmc.adjusted_mclmc_dynamic.init(
            position=position, logdensity_fn=target.log_posterior,
            random_generator_arg=init_key)

        # blackjax >= 1.6: the tuner drives the kernel as
        # kernel(rng_key=..., state=..., logdensity_fn=..., step_size=...,
        # inverse_mass_matrix=..., integration_steps_params=(avg,)); the
        # running average number of integration steps arrives as
        # integration_steps_fn's second positional argument.
        kernel = blackjax.mcmc.adjusted_mclmc_dynamic.build_kernel(
            integration_steps_fn=lambda k, avg: jnp.ceil(
                jax.random.uniform(k) * _rescale(avg)),
            integrator=isokinetic_mclachlan,
        )

        state, params, _ = blackjax.adjusted_mclmc_find_L_and_step_size(
            mclmc_kernel=kernel, logdensity_fn=target.log_posterior,
            num_steps=self.num_tuning_steps, state=init_state,
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
    inner rejuvenation kernel is adjusted MCLMC by default; fixed-trajectory
    HMC is available as an alternative. (NUTS is deliberately not offered: its
    data-dependent trajectory length does not vectorize cleanly across SMC
    particles, and rejuvenation does not need NUTS's full-mixing guarantee.)

    ``target_ess`` and ``num_mcmc_steps`` default to values calibrated for
    blackjax >= 1.6, whose adaptive-tempering ESS solver was corrected
    (upstream #914 fixed a sign bug in the bisection target). The fix
    changes how many temperature steps are needed to anneal from beta=0 to
    1: with the pre-1.6-era ``target_ess=0.5`` default, the corrected
    solver can collapse the schedule to a single step, leaving too few
    temperature increments for rejuvenation to move particles onto the
    posterior.

    Parameters
    ----------
    num_particles : int, optional
        Number of SMC particles.  Default is 1000.
    target_ess : float, optional
        Target effective sample size ratio in ``(0, 1)``, used to adapt the
        temperature increments.  Default is 0.9.
    num_mcmc_steps : int, optional
        Number of inner MCMC rejuvenation steps per temperature.  Default is 10.
    inner_kernel : str, optional
        Inner MCMC kernel: ``"mclmc"`` (adjusted MCLMC, default) or
        ``"hmc"`` (fixed-trajectory HMC).
    inner_step_size : float, optional
        Step size for the inner MCMC kernel.  Default is 0.1.
    inner_num_integration_steps : int, optional
        Number of integration steps for the inner MCLMC or HMC kernel.
        Default is 5.
    inner_inverse_mass_matrix : Array or None, optional
        Inverse mass matrix for the inner kernel.  If ``None``, defaults to a
        vector of ones of length ``target.dim``.  Default is ``None``.
    """

    def __init__(self, *, num_particles=1000, target_ess=0.9, num_mcmc_steps=10,
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
        if self.inner_kernel == "hmc":
            # blackjax >= 1.6: build_kernel() is a pure factory; the returned
            # kernel's signature is (rng_key, state, logdensity_fn, step_size,
            # inverse_mass_matrix, num_integration_steps), so SMC can call it
            # directly with these params as kwargs -- no wrapper needed. HMC's
            # fixed num_integration_steps gives static per-particle cost that
            # vectorizes cleanly across the particle population (unlike NUTS's
            # data-dependent trajectory length).
            step_fn = blackjax.hmc.build_kernel()
            init_fn = blackjax.hmc.init
            params = dict(step_size=self.inner_step_size,
                          num_integration_steps=self.inner_num_integration_steps,
                          inverse_mass_matrix=imm)
            return step_fn, init_fn, params
        if self.inner_kernel == "mclmc":
            from blackjax.mcmc.integrators import isokinetic_mclachlan

            # blackjax >= 1.6: build_kernel is a pure factory (no logdensity_fn /
            # inverse_mass_matrix binding), so build it once; SMC injects the
            # tempered logdensity per call.
            kernel = blackjax.mcmc.adjusted_mclmc.build_kernel(
                integrator=isokinetic_mclachlan)

            def step_fn(rng_key, state, logdensity_fn, step_size,
                        num_integration_steps, inverse_mass_matrix):
                return kernel(rng_key, state, logdensity_fn, step_size,
                              integration_steps_params=(num_integration_steps,),
                              inverse_mass_matrix=inverse_mass_matrix)

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


@dataclass(frozen=True)
class NestedSamplerInfo:
    """Diagnostics from a nested sampling run.

    Parameters
    ----------
    log_evidence : float
        Log marginal likelihood estimate (mean over stochastic
        prior-volume draws).
    log_evidence_err : float
        Standard deviation of the log-evidence estimate over the
        stochastic prior-volume draws.
    ess : float
        Effective sample size of the weighted dead-point set.
    num_dead : int
        Total number of points in the finalised run (dead points plus
        the final live set).
    dead : object
        Raw finalised ``blackjax.ns.base.NSInfo`` carrying the full
        point history (positions, log-likelihoods, birth contours).
        Kept for downstream re-weighting or anesthetic-style analysis.
    """

    log_evidence: float
    log_evidence_err: float
    ess: float
    num_dead: int
    dead: object


class NestedSampler(Sampler):
    """Nested slice sampling (blackjax ``nss``) posterior sampler.

    Runs blackjax's Nested Slice Sampling from prior-drawn live points,
    accumulating dead points until the live set's evidence share is
    negligible, then resamples the dead-point history into equal-weight
    posterior draws.  Unlike the MCMC samplers this also estimates the
    log evidence, and handles multimodal posteriors without tempering.

    Parameters
    ----------
    num_live : int, optional
        Number of live points.  Default is 500.
    num_delete : int or None, optional
        Number of lowest-likelihood points replaced per step (device
        batching).  If ``None``, defaults to ``max(1, num_live // 10)``.
    num_inner_steps : int or None, optional
        Constrained slice moves per replacement.  If ``None``, resolved
        at run time to ``max(5, 2 * target.dim)`` (blackjax's rule of
        thumb for reliable mixing).  Default is ``None``.
    num_samples : int, optional
        Number of equal-weight posterior draws returned.  Default is 1000.
    dlogz : float, optional
        Termination threshold (blackjax convention): stop once
        ``logZ_live - logZ < dlogz``.  Default is -3.0; use e.g. -10.0
        near phase transitions.
    max_iterations : int, optional
        Safety cap on the number of outer NS steps.  Default is 100_000.
    num_rejuvenation_steps : int, optional
        Posterior-invariant hit-and-run slice moves applied to each
        equal-weight draw after resampling.  The with-replacement resampling
        duplicates draws whenever ``num_samples`` is comparable to the run's
        ESS; a few slice moves break the duplicated atoms without changing
        the sampled distribution.  Default is 0 (no rejuvenation).
    """

    def __init__(self, *, num_live=500, num_delete=None, num_inner_steps=None,
                 num_samples=1000, dlogz=-3.0, max_iterations=100_000,
                 num_rejuvenation_steps=0):
        if num_live <= 0:
            raise ValueError(f"num_live must be positive, got {num_live}")
        if num_delete is None:
            num_delete = max(1, num_live // 10)
        if not 1 <= num_delete < num_live:
            raise ValueError(
                f"num_delete must be in [1, num_live), got num_delete="
                f"{num_delete} with num_live={num_live}")
        if num_inner_steps is not None and num_inner_steps <= 0:
            raise ValueError(
                f"num_inner_steps must be positive or None, got {num_inner_steps}")
        if num_rejuvenation_steps < 0:
            raise ValueError(
                f"num_rejuvenation_steps must be non-negative, got "
                f"{num_rejuvenation_steps}")
        self.num_live = num_live
        self.num_delete = num_delete
        self.num_inner_steps = num_inner_steps
        self.num_samples = num_samples
        self.dlogz = dlogz
        self.max_iterations = max_iterations
        self.num_rejuvenation_steps = num_rejuvenation_steps

    def _resolve_num_inner_steps(self, dim):
        """``num_inner_steps`` if set, else blackjax's ``max(5, 2 * dim)``."""
        if self.num_inner_steps is not None:
            return self.num_inner_steps
        return max(5, 2 * dim)

    def run(self, key, target):
        """Draw posterior samples using nested slice sampling.

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
            Equal-weight posterior samples of shape ``(num_samples, dim)``.
        info : NestedSamplerInfo
            Evidence estimate, ESS and the raw finalised run.
        """
        import blackjax
        from blackjax.ns.utils import ess, finalise, log_weights
        from blackjax.ns.utils import sample as ns_sample

        init_key, run_key, weights_key, ess_key, resample_key, rejuv_key = \
            jax.random.split(key, 6)
        particles = target.prior.sample(init_key, (self.num_live,))
        algo = blackjax.nss(
            logprior_fn=target.log_prior,
            loglikelihood_fn=target.log_likelihood,
            num_inner_steps=self._resolve_num_inner_steps(target.dim),
            num_delete=self.num_delete)
        state = algo.init(particles)
        step = jax.jit(algo.step)

        dead = []
        while state.integrator.logZ_live - state.integrator.logZ >= self.dlogz:
            if len(dead) >= self.max_iterations:
                raise RuntimeError(
                    f"nested sampling did not terminate within max_iterations="
                    f"{self.max_iterations} steps (logZ_live - logZ = "
                    f"{float(state.integrator.logZ_live - state.integrator.logZ):.3g}"
                    f" >= dlogz = {self.dlogz}). Consider a looser dlogz, more "
                    f"num_live points, or more num_inner_steps.")
            run_key, subkey = jax.random.split(run_key)
            state, step_info = step(subkey, state)
            dead.append(step_info)

        ns_run = finalise(state, dead)
        logw = log_weights(weights_key, ns_run, shape=100)   # (num_points, 100)
        logz_draws = jax.scipy.special.logsumexp(logw, axis=0)
        samples = ns_sample(resample_key, ns_run, self.num_samples).position
        if self.num_rejuvenation_steps > 0:
            samples = self._rejuvenate(rejuv_key, samples, target)
        info = NestedSamplerInfo(
            log_evidence=float(jnp.mean(logz_draws)),
            log_evidence_err=float(jnp.std(logz_draws)),
            ess=float(ess(ess_key, ns_run)),
            num_dead=int(ns_run.particles.loglikelihood.shape[0]),
            dead=ns_run,
        )
        return samples, info

    def _rejuvenate(self, key, positions, target):
        """Break duplicated equal-weight draws with posterior-invariant moves.

        Runs ``num_rejuvenation_steps`` hit-and-run slice moves on every
        resampled draw (one vmapped chain per draw), targeting the
        unconstrained log-posterior.  Directions are shaped by the empirical
        covariance of the resampled cloud and scaled to Mahalanobis norm 2,
        the same proposal the NS run's inner kernel uses -- so the moves are
        local decorrelation only; mode coverage and weights stay as the NS
        run left them.
        """
        from blackjax.mcmc import slice as slice_mcmc
        from blackjax.ns.nss import sample_direction_from_covariance

        cov = jnp.atleast_2d(jnp.cov(positions, rowvar=False))

        def proposal_generator(rng_key, position, logdensity_fn):
            direction = sample_direction_from_covariance(rng_key, position, cov)

            def slice_fn(t):
                x = jax.tree.map(lambda p, d: p + t * d, position, direction)
                return slice_mcmc.SliceState(x, logdensity_fn(x)), True

            return slice_fn

        algo = slice_mcmc.as_top_level_api(
            target.log_posterior, proposal_generator=proposal_generator)

        def chain(chain_key, position):
            def body(state, step_key):
                state, _ = algo.step(step_key, state)
                return state, None

            state, _ = jax.lax.scan(
                body, algo.init(position),
                jax.random.split(chain_key, self.num_rejuvenation_steps))
            return state.position

        keys = jax.random.split(key, positions.shape[0])
        return jax.jit(jax.vmap(chain))(keys, positions)
