"""Training-time timestep samplers for flow matching.

A small, pure helper so the timestep distribution is a configurable knob on
:class:`~gensbi.core.flow_matching.FlowMatchingMethod` without touching the
loss, path, or models.
"""

import jax


def sample_time(key, n, *, dist="uniform", logitnorm_mean=0.0, logitnorm_std=1.0):
    """Sample ``n`` flow-matching timesteps in ``(0, 1)``.

    Parameters
    ----------
    key : jax.random.PRNGKey
    n : int
        Number of timesteps (batch size).
    dist : str
        ``"uniform"`` (default) -> ``jax.random.uniform(key, (n,))``, bit-identical
        to the previous inline sampling so existing runs are unchanged.
        ``"logitnormal"`` -> ``sigmoid(logitnorm_mean + logitnorm_std * N(0, 1))``
        (SD3 / Esser et al.); concentrates mass near ``sigmoid(logitnorm_mean)``.
        The reference's ``lognorm_t`` flag is a misnomer for this logit-normal sampler.
    logitnorm_mean, logitnorm_std : float
        Mean/std of the underlying normal (used only for ``"logitnormal"``).

    Returns
    -------
    jax.Array
        Shape ``(n,)`` timesteps.
    """
    if dist == "uniform":
        return jax.random.uniform(key, (n,))
    if dist == "logitnormal":
        eps = jax.random.normal(key, (n,))
        return jax.nn.sigmoid(logitnorm_mean + logitnorm_std * eps)
    raise ValueError(
        f"unknown time dist {dist!r}; expected 'uniform' or 'logitnormal'"
    )
