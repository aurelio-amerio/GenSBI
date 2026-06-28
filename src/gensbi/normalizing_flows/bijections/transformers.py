"""Elementwise transformers parameterised per-dimension by a conditioner.

Pure functions of (value, params) — no learnable state of their own.
"""

import jax
import jax.numpy as jnp
from jax import Array


def _clamp(a: Array, lo: float, hi: float) -> Array:
    """Clamp with a straight-through gradient (NumPyro IAF trick)."""
    return a + jax.lax.stop_gradient(jnp.clip(a, lo, hi) - a)


class Affine:
    """Elementwise affine transform with log-scale clamping.

    The parameter layout per dimension is ``[shift mu, log-scale a]``
    (``num_params == 2``).  Log-scale values are clamped to
    ``[clamp_min, clamp_max]`` before exponentiation using a straight-through
    gradient (NumPyro IAF trick).

    Forward (noise→data): ``x = u * exp(a) + mu``; logabsdet = ``+sum(a)``.
    Inverse (data→noise): ``u = (x - mu) * exp(-a)``; logabsdet = ``-sum(a)``.

    Parameters
    ----------
    clamp_min : float, optional
        Lower bound for log-scale clamping.  Default is -5.0.
    clamp_max : float, optional
        Upper bound for log-scale clamping.  Default is 3.0.
    """

    num_params = 2

    def __init__(self, clamp_min: float = -5.0, clamp_max: float = 3.0):
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def _split(self, params: Array) -> tuple[Array, Array]:
        mu = params[..., 0]
        a = _clamp(params[..., 1], self.clamp_min, self.clamp_max)
        return mu, a

    def forward(self, u: Array, params: Array) -> tuple[Array, Array]:
        """Map noise to data: ``x = u * exp(a) + mu``.

        Parameters
        ----------
        u : Array
            Noise-space input of shape ``(dim,)``.
        params : Array
            Per-dimension parameters of shape ``(dim, 2)``; column 0 is the
            shift ``mu`` and column 1 is the log-scale ``a``.

        Returns
        -------
        x : Array
            Data-space output.
        logabsdet : Array
            Sum of clamped log-scales: ``sum(a)``.
        """
        mu, a = self._split(params)
        x = u * jnp.exp(a) + mu
        return x, jnp.sum(a)

    def inverse(self, x: Array, params: Array) -> tuple[Array, Array]:
        """Map data to noise: ``u = (x - mu) * exp(-a)``.

        Parameters
        ----------
        x : Array
            Data-space input of shape ``(dim,)``.
        params : Array
            Per-dimension parameters of shape ``(dim, 2)``; column 0 is the
            shift ``mu`` and column 1 is the log-scale ``a``.

        Returns
        -------
        u : Array
            Noise-space output.
        logabsdet : Array
            Negative sum of clamped log-scales: ``-sum(a)``.
        """
        mu, a = self._split(params)
        u = (x - mu) * jnp.exp(-a)
        return u, -jnp.sum(a)

    def forward_dim(self, u_i: Array, params_i: Array) -> tuple[Array, Array]:
        """Scalar noise-to-data transform for a single dimension.

        Used by the sequential sampling scan in the autoregressive loop, which
        accumulates the per-dimension log-determinant as it goes.

        Parameters
        ----------
        u_i : Array
            Scalar noise-space value for one dimension.
        params_i : Array
            Parameter vector of length 2 for the same dimension; index 0 is
            the shift ``mu``, index 1 is the log-scale ``a``.

        Returns
        -------
        x_i : Array
            Scalar data-space output: ``u_i * exp(a) + mu``.
        logabsdet_i : Array
            Forward log-determinant contribution for this dimension: ``a``.
        """
        mu = params_i[0]
        a = _clamp(params_i[1], self.clamp_min, self.clamp_max)
        return u_i * jnp.exp(a) + mu, a


def _inv_softplus(y: Array) -> Array:
    """Inverse of softplus: ``x`` such that ``softplus(x) == y`` (y > 0)."""
    return jnp.log(jnp.expm1(y))


def _rqs_apply(z: Array, x_knots: Array, y_knots: Array, derivatives: Array,
               inverse: bool):
    """Apply the RQ spline (or its inverse) to a scalar ``z``.

    Returns ``(out, logderiv)`` where ``logderiv = log(dy/dx)`` evaluated at the
    relevant x (the same forward derivative is used for both directions; the
    caller flips its sign for ``forward``). Outside ``[-B, B]`` the map is the
    identity (logderiv 0).
    """
    lo, hi = x_knots[0], x_knots[-1]
    in_bounds = (z >= lo) & (z <= hi)
    n_bins = x_knots.shape[0] - 1

    if not inverse:                                  # z is x; bins on x_knots
        k = jnp.clip(jnp.searchsorted(x_knots, z) - 1, 0, n_bins - 1)
    else:                                            # z is y; bins on y_knots
        k = jnp.clip(jnp.searchsorted(y_knots, z) - 1, 0, n_bins - 1)

    xk, xk1 = x_knots[k], x_knots[k + 1]
    yk, yk1 = y_knots[k], y_knots[k + 1]
    dk, dk1 = derivatives[k], derivatives[k + 1]
    w = xk1 - xk
    s = (yk1 - yk) / w                               # bin slope

    if not inverse:
        xi = jnp.clip((z - xk) / w, 0.0, 1.0)
        num = (yk1 - yk) * (s * xi ** 2 + dk * xi * (1 - xi))
        den = s + (dk1 + dk - 2 * s) * xi * (1 - xi)
        out_in = yk + num / den
    else:
        dy = z - yk
        c2 = dk1 + dk - 2 * s
        a = (yk1 - yk) * (s - dk) + dy * c2
        b = (yk1 - yk) * dk - dy * c2
        c = -s * dy
        disc = jnp.clip(b ** 2 - 4 * a * c, 0.0)
        xi = jnp.clip((2 * c) / (-b - jnp.sqrt(disc)), 0.0, 1.0)
        out_in = xk + xi * w

    out = jnp.where(in_bounds, out_in, z)
    num_d = s ** 2 * (dk1 * xi ** 2 + 2 * s * xi * (1 - xi) + dk * (1 - xi) ** 2)
    den_d = (s + (dk1 + dk - 2 * s) * xi * (1 - xi)) ** 2
    deriv = jnp.where(in_bounds, num_d / den_d, 1.0)
    return out, jnp.log(deriv)


class RQSpline:
    """Elementwise monotonic rational-quadratic spline on ``[-B, B]``.

    Linear (identity) tails outside the spline interval.  The mapping is
    parameterised per dimension; with zero-initialised parameters the spline
    reduces to the identity, so the flow warm-starts as a standard normal
    (same warm-start contract as :class:`Affine`).

    The parameter layout per dimension is
    ``[widths(K), heights(K), inner_derivatives(K-1)]`` of total length
    ``3K - 1``.  Reference: Durkan et al. 2019
    (https://arxiv.org/abs/1906.04032).

    Parameters
    ----------
    num_bins : int, optional
        Number of spline bins ``K``.  Default is 8.
    range_bound : float, optional
        Half-width ``B`` of the spline interval ``[-B, B]``.  Default is 5.0.
    min_bin_width : float, optional
        Minimum fractional bin width after softmax normalisation.
        Default is 1e-3.
    min_bin_height : float, optional
        Minimum fractional bin height after softmax normalisation.
        Default is 1e-3.
    min_derivative : float, optional
        Minimum knot derivative value.  Default is 1e-3.
    """

    def __init__(self, num_bins: int = 8, range_bound: float = 5.0,
                 min_bin_width: float = 1e-3, min_bin_height: float = 1e-3,
                 min_derivative: float = 1e-3):
        self.num_bins = num_bins
        self.B = range_bound
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.num_params = 3 * num_bins - 1

    def _knots(self, params: Array):
        """Raw params -> (x_knots, y_knots, derivatives), each over K+1 knots."""
        K, B = self.num_bins, self.B
        raw_w = params[:K]
        raw_h = params[K:2 * K]
        raw_d = params[2 * K:3 * K - 1]                  # (K-1,)

        w = jax.nn.softmax(raw_w)
        w = self.min_bin_width + (1.0 - self.min_bin_width * K) * w
        h = jax.nn.softmax(raw_h)
        h = self.min_bin_height + (1.0 - self.min_bin_height * K) * h

        x_knots = -B + 2.0 * B * jnp.concatenate([jnp.zeros(1), jnp.cumsum(w)])
        y_knots = -B + 2.0 * B * jnp.concatenate([jnp.zeros(1), jnp.cumsum(h)])

        # offset so raw_d == 0 -> derivative == 1 (identity warm-start)
        d_inner = self.min_derivative + jax.nn.softplus(
            raw_d + _inv_softplus(1.0 - self.min_derivative))
        derivatives = jnp.concatenate([jnp.ones(1), d_inner, jnp.ones(1)])
        return x_knots, y_knots, derivatives

    def _fwd_scalar(self, x: Array, params: Array):
        x_knots, y_knots, d = self._knots(params)
        return _rqs_apply(x, x_knots, y_knots, d, inverse=False)

    def _inv_scalar(self, u: Array, params: Array):
        x_knots, y_knots, d = self._knots(params)
        return _rqs_apply(u, x_knots, y_knots, d, inverse=True)

    def inverse(self, x: Array, params: Array):
        """Map data to noise (fast vectorised spline pass).

        Parameters
        ----------
        x : Array
            Data-space input of shape ``(dim,)``.
        params : Array
            Per-dimension spline parameters of shape ``(dim, 3K-1)``.

        Returns
        -------
        u : Array
            Noise-space output.
        logabsdet : Array
            Positive sum of log-derivatives of the spline at each input:
            ``+sum(log g'(x))``, where ``g'`` is the forward spline derivative
            (data→noise direction).
        """
        u, logderiv = jax.vmap(self._fwd_scalar)(x, params)
        return u, jnp.sum(logderiv)

    def forward(self, u: Array, params: Array):
        """Map noise to data (vectorised inverse spline pass).

        Parameters
        ----------
        u : Array
            Noise-space input of shape ``(dim,)``.
        params : Array
            Per-dimension spline parameters of shape ``(dim, 3K-1)``.

        Returns
        -------
        x : Array
            Data-space output.
        logabsdet : Array
            Negative sum of log-derivatives of the spline: ``-sum(log g'(x))``,
            where ``g'`` is the forward spline derivative evaluated at the
            corresponding data-space position.
        """
        x, logderiv = jax.vmap(self._inv_scalar)(u, params)
        return x, -jnp.sum(logderiv)

    def forward_dim(self, u_i: Array, params_i: Array) -> tuple[Array, Array]:
        """Scalar noise-to-data transform for a single dimension.

        Used by the sequential sampling scan in the autoregressive loop, which
        accumulates the per-dimension log-determinant as it goes.

        Parameters
        ----------
        u_i : Array
            Scalar noise-space value for one dimension.
        params_i : Array
            Spline parameter vector of length ``3K-1`` for the same dimension.

        Returns
        -------
        x_i : Array
            Scalar data-space output.
        logabsdet_i : Array
            Forward log-determinant contribution for this dimension:
            ``-log g'(x_i)`` (matching :meth:`forward`).
        """
        x_i, logderiv = self._inv_scalar(u_i, params_i)
        return x_i, -logderiv
