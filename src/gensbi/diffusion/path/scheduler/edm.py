"""
EDM schedulers and noise schedules.

This module implements various noise schedulers for diffusion models based on the
EDM framework, including variance-preserving (VP), variance-exploding (VE), and
EDM-specific schedules. These schedulers define the noise schedule and preconditioning
functions used during training and sampling.

Based on the paper "Elucidating the Design Space of Diffusion-Based Generative Models"
by Karras et al., 2022. https://arxiv.org/abs/2206.00364
"""
import abc
import jax
import jax.numpy as jnp
from jax import Array
from typing import Callable, Any

# from .samplers import sampler #moved to samplers module

# we will create an abstract SDE class which can implement VP, VE, and EDM methods, following https://github.com/NVlabs/edm/
# we will then define a precondition function for each method

# TODO still need to test


class BaseSDE(abc.ABC):
    def __init__(self) -> None:
        """Base class for SDE schedulers."""
        return

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Returns the name of the SDE scheduler."""
        ...  # pragma: no cover

    @abc.abstractmethod
    def time_schedule(self, u: Array) -> Array:
        """
        Given the value of the random uniform variable u ~ U(0,1), return the time t in the schedule.

        Parameters
        ----------
            u : Array
                Uniform random variable in [0, 1].

        Returns
        -------
            Array
                Time in the schedule.
        """
        ...  # pragma: no cover

    def timesteps(self, i: Array, N: int) -> Array:
        """
        Compute the time steps for a given index array and total number of steps.

        Parameters
        ----------
            i : Array
                Step indices.
            N : int
                Total number of steps.

        Returns
        -------
            Array
                Time steps.
        """
        u = i / (N - 1)
        return self.time_schedule(u)

    @abc.abstractmethod
    def sigma(self, t: Array) -> Array:
        """
        Returns the noise scale (schedule) at time t.

        Parameters
        ----------
            t : Array
                Time.

        Returns
        -------
            Array
                Noise scale.
        """
        ...  # pragma: no cover

    @abc.abstractmethod
    def sigma_inv(self, sigma: Array) -> Array:
        """
        Inverse of the noise scale function.

        Parameters
        ----------
            sigma : Array
                Noise scale.

        Returns
        -------
            Array
                Time corresponding to the given sigma.
        """
        ...  # pragma: no cover

    @abc.abstractmethod
    def sigma_deriv(self, t: Array) -> Array:
        """
        Derivative of the noise scale with respect to time.

        Parameters
        ----------
            t : Array
                Time.

        Returns
        -------
            Array
                Derivative of sigma.
        """
        ...  # pragma: no cover

    @abc.abstractmethod
    def s(self, t: Array) -> Array:
        """
        Scaling function as in EDM paper.

        Parameters
        ----------
            t : Array
                Time.

        Returns
        -------
            Array
                Scaling value.
        """
        ...  # pragma: no cover

    @abc.abstractmethod
    def s_deriv(self, t: Array) -> Array:
        """
        Derivative of the scaling function.

        Parameters
        ----------
            t : Array
                Time.

        Returns
        -------
            Array
                Derivative of scaling.
        """
        ...  # pragma: no cover

    @abc.abstractmethod
    def c_skip(self, sigma: Array) -> Array:
        """
        Preconditioning skip connection coefficient.

        Parameters
        ----------
            sigma : Array
                Noise scale.

        Returns
        -------
            Array
                Skip coefficient.
        """
        ...  # pragma: no cover

    @abc.abstractmethod
    def c_out(self, sigma: Array) -> Array:
        """
        Preconditioning output coefficient.

        Parameters
        ----------
            sigma : Array
                Noise scale.

        Returns
        -------
            Array
                Output coefficient.
        """
        ...  # pragma: no cover

    @abc.abstractmethod
    def c_in(self, sigma: Array) -> Array:
        """
        Preconditioning input coefficient.

        Parameters
        ----------
            sigma : Array
                Noise scale.

        Returns
        -------
            Array
                Input coefficient.
        """
        ...  # pragma: no cover

    @abc.abstractmethod
    def c_noise(self, sigma: Array) -> Array:
        """
        Preconditioning noise coefficient.

        Parameters
        ----------
            sigma : Array
                Noise scale.

        Returns
        -------
            Array
                Noise coefficient.
        """
        ...  # pragma: no cover

    @abc.abstractmethod
    def sample_sigma(self, key: Array, shape: Any) -> Array:
        """
        Sample sigma from the prior noise distribution.

        Parameters
        ----------
            key : Array
                JAX random key.
            shape : Any
                Shape of the output.

        Returns
        -------
            Array
                Sampled sigma.
        """
        ...  # pragma: no cover

    def sample_noise(self, key: Array, shape: Any, sigma: Array) -> Array:
        """
        Sample noise from the prior noise distribution with noise scale sigma(t).

        Parameters
        ----------
            key : Array
                JAX random key.
            shape : Any
                Shape of the output.
            sigma : Array
                Noise scale.

        Returns
        -------
            Array
                Sampled noise.
        """
        n = jax.random.normal(key, shape) * sigma
        return n

    def sample_prior(self, key: Array, shape: Any) -> Array:
        """
        Sample x from the prior distribution.

        Parameters
        ----------
            key : Array
                JAX random key.
            shape : Any
                Shape of the output.

        Returns
        -------
            Array
                Sampled prior.
        """
        return jax.random.normal(key, shape)

    @abc.abstractmethod
    def loss_weight(self, sigma: Array) -> Array:
        """
        Weight for the loss function, for MLE estimation, also known as λ(σ) in the EDM paper.

        Parameters
        ----------
            sigma : Array
                Noise scale.

        Returns
        -------
            Array
                Loss weight.
        """
        ...  # pragma: no cover

    def f(self, x: Array, t: Array) -> Array:
        r"""
        Drift term for the forward diffusion process.

        Computes the drift term :math:`f(x, t) = x \frac{ds}{dt} / s(t)` as used in the SDE formulation.

        Parameters
        ----------
            x : Array
                Input data.
            t : Array
                Time.

        Returns
        -------
            Array
                Drift term.
        """
        t = self.time_schedule(t)
        return x * self.s_deriv(t) / self.s(t)

    def g(self, x: Array, t: Array) -> Array:
        r"""
        Diffusion term for the forward diffusion process.

        Computes the diffusion term :math:`g(x, t) = s(t) \sqrt{2 \frac{d\sigma}{dt} \sigma(t)}` as used in the SDE formulation.

        Parameters
        ----------
            x : Array
                Input data.
            t : Array
                Time.

        Returns
        -------
            Array
                Diffusion term.
        """
        t = self.time_schedule(t)
        return self.s(t) * jnp.sqrt(2 * self.sigma_deriv(t) * self.sigma(t))

    def denoise(self, F: Callable, x: Array, sigma: Array, *args, **kwargs) -> Array:
        r"""
        Denoise function, :math:`D` in the EDM paper, which shares a connection with the score function:

        .. math::
            \nabla_x \log p(x; \sigma) = \frac{D(x; \sigma) - x}{\sigma^2}

        This function includes the preconditioning and is connected to the NN objective :math:`F`:

        .. math::
            D_\theta(x; \sigma) = c_\text{skip}(\sigma) x + c_\text{out}(\sigma) F_\theta (c_\text{in}(\sigma) x; c_\text{noise}(\sigma))

        Parameters
        ----------
            F : Callable
                Model function.
            x : Array
                Input data.
            sigma : Array
                Noise scale.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns
        -------
            Array
                Denoised output.
        """
        return self.c_skip(sigma) * x + self.c_out(sigma) * F(
            obs=self.c_in(sigma) * x, t=self.c_noise(sigma), *args, **kwargs
        )

    # FIXME: for some reason, when sampling using the score, instead of the algorithm provided by the EDM paper, the sample quality is very bad
    # there must be something wrong in the function signatures, but the diffusion and drift terms for the forward process seem to be correct
    # def get_score_function(self, F: Callable) -> Callable:
    #     r"""
    #     Returns the score function :math:`\nabla_x \log p(x; \sigma)` as described in the EDM paper.

    #     The score function is computed as:

    #     .. math::
    #         \nabla_x \log p(x; \sigma) = \frac{D(x; \sigma) - x}{\sigma^2}

    #     where :math:`D(x; \sigma)` is the denoised output (see `denoise` method).

    #     Args:
    #         F (Callable): Model function.

    #     Returns:
    #         Callable: Score function.
    #     """
    #     def score(x: Array, u: Array, *args, **kwargs) -> Array:
    #         t = self.time_schedule(u)
    #         sigma = self.sigma(t)
    #         x = x/self.s(t) # todo: check this
    #         return (self.denoise(F, x, sigma, *args, **kwargs) - x) / (sigma**2)
    #     return score

    def get_loss_fn(self) -> Callable:
        r"""
        Returns the loss function for EDM training, as described in the EDM paper.

        The loss is computed as (see Eq. 8 in the EDM paper):

        .. math::
            \lambda(\sigma) \, c_\text{out}^2(\sigma) \left[
                F(c_\text{in}(\sigma) x_t, c_\text{noise}(\sigma), \ldots)
                - \frac{1}{c_\text{out}(\sigma)} (x_1 - c_\text{skip}(\sigma) x_t)
            \right]^2

        Parameters
        ----------
            None directly; returns a function that computes the loss.

        Returns
        -------
            Callable
                Loss function.
        """

        def loss_fn(
            F: Callable, batch: tuple, loss_mask: Any = None, model_extras: dict = {}
        ) -> Array:
            (x_1, x_t, sigma) = batch

            lam = self.loss_weight(sigma)
            c_out = self.c_out(sigma)
            c_in = self.c_in(sigma)
            c_noise = self.c_noise(sigma)
            c_skip = self.c_skip(sigma)

            if loss_mask is not None:
                loss_mask = jnp.broadcast_to(loss_mask, x_1.shape)
                x_t = jnp.where(loss_mask, x_1, x_t)

            loss = (
                lam
                * c_out**2
                * (
                    F(obs=c_in * (x_t), t=c_noise, **model_extras)
                    - 1 / c_out * (x_1 - c_skip * (x_t))
                )
                ** 2
            )
            if loss_mask is not None:
                loss = jnp.where(loss_mask, 0.0, loss)
            # we sum the loss on any dimension that is not the batch dimentsion, and then we compute the mean over the batch dimension (the first)
            return jnp.mean(jnp.sum(loss, axis=tuple(range(1, len(x_1.shape)))))  # type: ignore

        return loss_fn


class VPScheduler(BaseSDE):
    """
    Variance Preserving (VP) SDE scheduler as described in the EDM paper.

    Parameters
    ----------
        beta_min : float
            Minimum beta value.
        beta_max : float
            Maximum beta value.
        e_s : float
            Starting epsilon value for time schedule.
        e_t : float
            Ending epsilon value for time schedule.
        M : int
            Scaling factor for noise preconditioning.


    References:
        - Karras, Tero, et al. "Elucidating the design space of diffusion-based generative models." `arXiv:2206.00364 <https://arxiv.org/abs/2206.00364>`_
    """

    def __init__(self, beta_min=0.1, beta_max=20.0, e_s=1e-3, e_t=1e-5, M=1000):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_d = beta_max - beta_min
        self.e_s = e_s
        self.e_t = e_t
        self.M = M
        return

    @property
    def name(self):
        return "EDM-VP"

    def time_schedule(self, u):
        return 1 + u * (self.e_s - 1)

    def sigma(self, t):
        # also known as the schedule, as in tab 1 of EDM paper
        return jnp.sqrt(jnp.exp(0.5 * self.beta_d * t**2 + self.beta_min * t) - 1)

    def sigma_inv(self, sigma):
        return (
            jnp.sqrt(self.beta_min**2 + 2 * self.beta_d * jnp.log(1 + sigma**2))
            - self.beta_min
        ) / self.beta_d

    def sigma_deriv(self, t):
        # also known as the schedule derivative
        return (
            0.5
            * (self.beta_min + self.beta_d * t)
            * (self.sigma(t) + 1 / self.sigma(t))
        )

    def s(self, t):
        # also known as scaling, as in tab 1 of EDM paper
        return 1 / jnp.sqrt(jnp.exp(0.5 * self.beta_d * t**2 + self.beta_min * t))

    def s_deriv(self, t):
        # also known as scaling derivative
        return -self.sigma(t) * self.sigma_deriv(t) * (self.s(t) ** 3)

    def f(self, x, t):
        # f(x, sigma) in the SDE, also known as drift term for the forward diffusion process
        return -x * 0.5 * (self.beta_min + self.beta_d * t)

    def g(self, x, t):
        # g(sigma) in the SDE, also known as diffusion term for the forward diffusion process
        return jnp.sqrt(self.beta_min + self.beta_d * t)

    def c_skip(self, sigma):
        # c_skip for preconditioning
        return jnp.ones_like(sigma)

    def c_out(self, sigma):
        # c_out for preconditioning
        return -sigma

    def c_in(self, sigma):
        # c_in for preconditioning
        return 1 / jnp.sqrt(sigma**2 + 1)

    def c_noise(self, sigma):
        # c_noise for preconditioning
        return (self.M - 1) * self.sigma_inv(sigma)

    def loss_weight(self, sigma):
        return 1 / sigma**2

    def sample_sigma(self, key, shape):
        # sample sigma from the prior noise distribution
        u = jax.random.uniform(key, shape, minval=self.e_t, maxval=1)
        return self.sigma(u)


class VEScheduler(BaseSDE):
    """
    Variance Exploding (VE) SDE scheduler as described in the EDM paper.

    Parameters
    ----------
        sigma_min : float
            Minimum sigma value.
        sigma_max : float
            Maximum sigma value.
    """

    def __init__(self, sigma_min=1e-3, sigma_max=15.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        return

    @property
    def name(self):
        return "EDM-VE"

    def time_schedule(self, u):
        return self.sigma_max**2 * (self.sigma_min / self.sigma_max) ** (2 * u)

    def sigma(self, t):
        # also known as the schedule, as in tab 1 of EDM paper
        return jnp.sqrt(t)

    def sigma_inv(self, sigma):
        return sigma**2

    def sigma_deriv(self, t):
        return 1 / (2 * jnp.sqrt(t))

    def s(self, t):
        # also known as scaling, as in tab 1 of EDM paper
        return jnp.ones_like(t)

    def s_deriv(self, t):
        # also known as scaling derivative
        return jnp.zeros_like(t)

    def c_skip(self, sigma):
        # c_skip for preconditioning
        return jnp.ones_like(sigma)

    def c_out(self, sigma):
        # c_out for preconditioning
        return sigma

    def c_in(self, sigma):
        # c_in for preconditioning
        return jnp.ones_like(sigma)

    def c_noise(self, sigma):
        # c_noise for preconditioning
        return jnp.log(0.5 * sigma)

    def loss_weight(self, sigma):
        return 1 / sigma**2

    def sample_sigma(self, key, shape):
        # sample sigma from the prior noise distribution
        log_sigma = jax.random.uniform(
            key, shape, minval=jnp.log(self.sigma_min), maxval=jnp.log(self.sigma_max)
        )
        return jnp.exp(log_sigma)


class EDMScheduler(BaseSDE):
    def __init__(
        self,
        sigma_min=0.002,
        sigma_max=80.0,
        sigma_data=1.0,
        rho=7,
        P_mean=-1.2,
        P_std=1.2,
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.P_mean = P_mean
        self.P_std = P_std
        return

    @property
    def name(self):
        return "EDM"

    def time_schedule(self, u):
        return (
            self.sigma_max ** (1 / self.rho)
            + u * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho

    def sigma(self, t):
        # also known as the schedule, as in tab 1 of EDM paper
        return t

    def sigma_inv(self, sigma):
        return sigma

    def sigma_deriv(self, t):
        # also known as the schedule derivative
        return jnp.ones_like(t)

    def s(self, t):
        # also known as scaling, as in tab 1 of EDM paper
        return jnp.ones_like(t)

    def s_deriv(self, t):
        # also known as scaling derivative
        return jnp.zeros_like(t)

    def c_skip(self, sigma):
        # c_skip for preconditioning
        return self.sigma_data**2 / jnp.sqrt(sigma**2 + self.sigma_data**2)

    def c_out(self, sigma):
        # c_out for preconditioning
        return sigma * self.sigma_data / jnp.sqrt(sigma**2 + self.sigma_data**2)

    def c_in(self, sigma):
        # c_in for preconditioning
        return 1 / jnp.sqrt(sigma**2 + self.sigma_data**2)

    def c_noise(self, sigma):
        # c_noise for preconditioning
        return 0.25 * jnp.log(sigma)

    def loss_weight(self, sigma):
        # weight for the loss function, for MLE estimation, also known as λ(σ) in the EDM paper
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

    def sample_sigma(self, key, shape):
        # sample sigma from the prior noise distribution, in this case it is not anymore a uniform distribution, see https://github.com/NVlabs/edm/blob/008a4e5316c8e3bfe61a62f874bddba254295afb/training/loss.py#L66
        rnd_normal = jax.random.normal(key, shape)
        sigma = jnp.exp(rnd_normal * self.P_std + self.P_mean)
        return sigma
