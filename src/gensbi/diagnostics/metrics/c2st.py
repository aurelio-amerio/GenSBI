# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
#
# --------------------------------------------------------------------------
# MODIFICATION NOTICE:
# This file was modified by Aurelio Amerio on 01-2026.
# Description: Ported implementation to use JAX instead of PyTorch.
# --------------------------------------------------------------------------


from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import jax 
from jax import Array
import jax.numpy as jnp

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier


def c2st(
    X: Array,
    Y: Array,
    seed: int = 1,
    n_folds: int = 5,
    metric: str = "accuracy",
    classifier: Union[str, Callable] = "rf",
    classifier_kwargs: Optional[Dict[str, Any]] = None,
    z_score: bool = True,
    noise_scale: Optional[float] = None,
    verbosity: int = 0,
) -> Array:
    """
    Compute classifier-based two-sample test accuracy between X and Y.

    This method uses a classifier to distinguish between two sets of samples. 
    If the returned accuracy is 0.5, X and Y are considered to be from the same generating distribution. 
    If the accuracy is close to 1, X and Y are considered to be from different distributions.

    Training of the classifier is performed with N-fold cross-validation using scikit-learn. 
    By default, a `RandomForestClassifier` is used (``classifier='rf'``). 
    Alternatively, a multi-layer perceptron is available (``classifier='mlp'``).

    Both sets of samples are normalized (z-scored) using the mean and standard deviation of X, 
    unless ``z_score=False``. If features in X are close to constant, the standard deviation 
    is set to 1 to avoid division by zero.


    Parameters
    ----------
    X : jax.Array
        Samples from one distribution. Shape: (n_samples, n_features).
    Y : jax.Array
        Samples from another distribution. Shape: (n_samples, n_features).
    seed : int, optional
        Seed for the sklearn classifier and the KFold cross-validation. Default is 1.
    n_folds : int, optional
        Number of folds to use for cross-validation. Default is 5.
    metric : str, optional
        Scikit-learn metric to use for scoring. Default is 'accuracy'.
    classifier : str or Callable, optional
        Classification architecture to use. 'rf' for RandomForestClassifier, 'mlp' for MLPClassifier, or a scikit-learn compatible classifier. Default is 'rf'.
    classifier_kwargs : dict, optional
        Additional keyword arguments for the classifier.
    z_score : bool, optional
        Whether to z-score X and Y using the mean and std of X. Default is True.
    noise_scale : float, optional
        If provided, adds Gaussian noise with standard deviation ``noise_scale`` to X and Y.
    verbosity : int, optional
        Controls the verbosity of scikit-learn's cross_val_score. Default is 0.

    Returns
    -------
    float
        Mean accuracy score over the test sets from cross-validation.

    Examples
    --------
    >>> c2st(X, Y)
    0.519  # X and Y likely come from the same distribution
    >>> c2st(P, Q)
    0.998  # P and Q likely come from different distributions

    References
    ----------
    [1] http://arxiv.org/abs/1610.06545
    [2] https://www.osti.gov/biblio/826696/
    [3] https://scikit-learn.org/stable/modules/cross_validation.html
    [4] https://github.com/psteinb/c2st/
    """
    
    key = jax.random.PRNGKey(seed)

    # the default configuration
    if classifier == "rf":
        clf_class = RandomForestClassifier
        clf_kwargs = classifier_kwargs or {}  # use sklearn defaults
    elif classifier == "mlp":
        ndim = X.shape[-1]
        clf_class = MLPClassifier
        # set defaults for the MLP
        clf_kwargs = classifier_kwargs or {
            "activation": "relu",
            "hidden_layer_sizes": (10 * ndim, 10 * ndim),
            "max_iter": 1000,
            "solver": "adam",
            "early_stopping": True,
            "n_iter_no_change": 50,
        }

    if z_score:
        X_mean = jnp.mean(X, axis=0)
        X_std = jnp.std(X, axis=0)
        # Set std to 1 if it is close to zero.
        X_std = jnp.where(X_std < 1e-14, 1, X_std)
        assert not jnp.any(jnp.isnan(X_mean)), "X_mean contains NaNs"
        assert not jnp.any(jnp.isnan(X_std)), "X_std contains NaNs"
        X = (X - X_mean) / X_std
        Y = (Y - X_mean) / X_std

    if noise_scale is not None:
        key1, key2 = jax.random.split(key)
        X += noise_scale * jax.random.normal(key1, X.shape)
        Y += noise_scale * jax.random.normal(key2, Y.shape)

    clf = clf_class(random_state=seed, **clf_kwargs)

    # prepare data, convert to numpy
    data = np.concatenate((np.array(X), np.array(Y)))
    # labels
    target = np.concatenate((np.zeros((X.shape[0],)), np.ones((Y.shape[0],))))

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(
        clf, data, target, cv=shuffle, scoring=metric, verbose=verbosity
    )

    return jnp.mean(scores)


def check_c2st(x: Array, y: Array, alg: str, tol: float = 0.1) -> None:
    """Compute classification based two-sample test accuracy and assert it close to
    chance."""

    score = c2st(x, y).item()

    assert (0.5 - tol) <= score <= (0.5 + tol), (
        f"{alg}'s c2st={score:.2f} is too far from the desired near-chance performance."
    )