# This file is part of sbi, a toolkit for simulation-based inference. sbi is licensed
# under the Apache License Version 2.0, see <https://www.apache.org/licenses/>
#
# --------------------------------------------------------------------------
# MODIFICATION NOTICE:
# This file was modified by Aurelio Amerio on 01-2026.
# Description: Ported implementation to use JAX instead of PyTorch.
# --------------------------------------------------------------------------


from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import jax
from jax import Array
import jax.numpy as jnp

from flax import nnx

from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure, FigureBase


class LC2ST:
    r"""L-C2ST: Local Classifier Two-Sample Test.

    Implementation based on the official code from [1] and the exisiting C2ST
    metric [2], using scikit-learn classifiers.

    L-C2ST tests the local consistency of a posterior estimator :math:`q` with
    respect to the true posterior :math:`p`, at a fixed observation :math:`x_o`,
    i.e., whether the following null hypothesis holds:

    :math:`H_0(x_o) := q(\theta \mid x_o) = p(\theta \mid x_o)`.

    L-C2ST proceeds as follows:

    1. It first trains a classifier to distinguish between samples from two joint
       distributions :math:`[\theta_p, x_p]` and :math:`[\theta_q, x_q]`, and
       evaluates the L-C2ST statistic at a given observation :math:`x_o`.

    2. The L-C2ST statistic is the mean squared error between the predicted
       probabilities of being in p (class 0) and a Dirac at 0.5, which corresponds to
       the chance level of the classifier, unable to distinguish between p and q.

    - If ``num_ensemble>1``, the average prediction over all classifiers is used.
    - If ``num_folds>1`` the average statistic over all cv-folds is used.

    To evaluate the test, steps 1 and 2 are performed over multiple trials under the
    null hypothesis (H0). If the null distribution is not known, it is estimated
    using the permutation method, i.e. by training the classifier on the permuted
    data. The statistics obtained under (H0) is then compared to the one obtained
    on observed data to compute the p-value, used to decide whether to reject (H0)
    or not.

    Parameters
    ----------
    thetas : Array
        Samples from the prior, of shape (sample_size, dim).
    xs : Array
        Corresponding simulated data, of shape (sample_size, dim_x).
    posterior_samples : Array
        Samples from the estiamted posterior, of shape (sample_size, dim).
    seed : int, optional
        Seed for the sklearn classifier and the KFold cross validation. Default is 1.
    num_folds : int, optional
        Number of folds for the cross-validation. Default is 1 (no cross-validation).
        This is useful to reduce variance coming from the data.
    num_ensemble : int, optional
        Number of classifiers for ensembling. Default is 1.
        This is useful to reduce variance coming from the classifier.
    classifier : str or Type[BaseEstimator], optional
        Classification architecture to use, can be one of the following:
        - "random_forest" or "mlp", defaults to "mlp" or
        - A classifier class (e.g., RandomForestClassifier, MLPClassifier).
    z_score : bool, optional
        Whether to z-score to normalize the data. Default is False.
    classifier_kwargs : Dict[str, Any], optional
        Custom kwargs for the sklearn classifier. Default is None.
    num_trials_null : int, optional
        Number of trials to estimate the null distribution. Default is 100.
    permutation : bool, optional
        Whether to use the permutation method for the null hypothesis. Default is True.

    References
    ----------
    [1] : https://arxiv.org/abs/2306.03580, https://github.com/JuliaLinhart/lc2st
    [2] : https://github.com/sbi-dev/sbi/blob/main/sbi/utils/metrics.py

    """

    def __init__(
        self,
        thetas: Array,
        xs: Array,
        posterior_samples: Array,
        seed: int = 1,
        num_folds: int = 1,
        num_ensemble: int = 1,
        classifier: Union[str, Type[BaseEstimator]] = MLPClassifier,
        z_score: bool = False,
        classifier_kwargs: Optional[Dict[str, Any]] = None,
        num_trials_null: int = 100,
        permutation: bool = True,
    ) -> None:

        assert (
            thetas.shape[0] == xs.shape[0] == posterior_samples.shape[0]
        ), f"Number of samples must match, got {thetas.shape[0]}, {xs.shape[0]}, {posterior_samples.shape[0]}"

        # set observed data for classification
        self.theta_p = posterior_samples
        self.x_p = xs
        self.theta_q = thetas
        self.x_q = xs

        # z-score normalization parameters
        self.z_score = z_score
        self.theta_p_mean = jnp.mean(self.theta_p, axis=0)
        self.theta_p_std = jnp.std(self.theta_p, axis=0)
        self.x_p_mean = jnp.mean(self.x_p, axis=0)
        self.x_p_std = jnp.std(self.x_p, axis=0)

        # set parameters for classifier training
        self.seed = seed
        self.rngs = nnx.Rngs(seed)
        self.num_folds = num_folds
        self.num_ensemble = num_ensemble

        # initialize classifier
        if isinstance(classifier, str):
            if classifier.lower() == "mlp":
                classifier = MLPClassifier
            elif classifier.lower() == "random_forest":
                classifier = RandomForestClassifier
            else:
                raise ValueError(
                    f'Invalid classifier: "{classifier}".'
                    'Expected "mlp", "random_forest", '
                    "or a valid scikit-learn classifier class."
                )
        assert issubclass(
            classifier, BaseEstimator
        ), "classier must either be a string or a subclass of BaseEstimator."
        self.clf_class = classifier

        # for MLPClassifier, set default parameters
        if classifier_kwargs is None:
            if self.clf_class == MLPClassifier:
                ndim = thetas.shape[-1]
                self.clf_kwargs = {
                    "activation": "relu",
                    "hidden_layer_sizes": (10 * ndim, 10 * ndim),
                    "max_iter": 1000,
                    "solver": "adam",
                    "early_stopping": True,
                    "n_iter_no_change": 50,
                }
            else:
                self.clf_kwargs: Dict[str, Any] = {}

        # initialize classifiers, will be set after training
        self.trained_clfs = None
        self.trained_clfs_null = None

        # parameters for the null hypothesis testing
        self.num_trials_null = num_trials_null
        self.permutation = permutation
        # can be specified if known and independent of x (see `LC2ST-NF`)
        self.null_distribution = None

    def _train(
        self,
        theta_p: Array,
        theta_q: Array,
        x_p: Array,
        x_q: Array,
        verbosity: int = 0,
    ) -> List[Any]:
        """Returns the classifiers trained on observed data.

        Parameters
        ----------
        theta_p : Array
            Samples from P, of shape (sample_size, dim).
        theta_q : Array
            Samples from Q, of shape (sample_size, dim).
        x_p : Array
            Observations corresponding to P, of shape (sample_size, dim_x).
        x_q : Array
            Observations corresponding to Q, of shape (sample_size, dim_x).
        verbosity : int, optional
            Verbosity level. Default is 0.

        Returns
        -------
        List[Any]
            List of trained classifiers for each cv fold.
        """

        # prepare data

        if self.z_score:
            theta_p = (theta_p - self.theta_p_mean) / self.theta_p_std
            theta_q = (theta_q - self.theta_p_mean) / self.theta_p_std
            x_p = (x_p - self.x_p_mean) / self.x_p_std
            x_q = (x_q - self.x_p_mean) / self.x_p_std

        # initialize classifier
        clf = self.clf_class(**self.clf_kwargs or {})

        if self.num_ensemble > 1:
            clf = EnsembleClassifier(clf, self.num_ensemble, verbosity=verbosity)

        # cross-validation
        if self.num_folds > 1:
            trained_clfs = []
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)
            cv_splits = kf.split(np.array(theta_p))
            for train_idx, _ in tqdm(
                cv_splits, desc="Cross-validation", disable=verbosity < 1
            ):
                # get train split
                theta_p_train, theta_q_train = theta_p[train_idx], theta_q[train_idx]
                x_p_train, x_q_train = x_p[train_idx], x_q[train_idx]

                # train classifier
                clf_n = train_lc2st(
                    theta_p_train, theta_q_train, x_p_train, x_q_train, clf
                )

                trained_clfs.append(clf_n)
        else:
            # train single classifier
            clf = train_lc2st(theta_p, theta_q, x_p, x_q, clf)
            trained_clfs = [clf]

        return trained_clfs

    def get_scores(
        self,
        theta_o: Array,
        x_o: Array,
        trained_clfs: List[Any],
        return_probs: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Computes the L-C2ST scores given the trained classifiers.

        Mean squared error (MSE) between 0.5 and the predicted probabilities
        of being in class 0 over the dataset (`theta_o`, `x_o`).

        Parameters
        ----------
        theta_o : Array
            Samples from the posterior conditioned on the observation `x_o`,
            of shape (sample_size, dim).
        x_o : Array
            The observation, of shape (,dim_x).
        trained_clfs : List[Any]
            List of trained classifiers, of length `num_folds`.
        return_probs : bool, optional
            Whether to return the predicted probabilities of being in P. Default is False.

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            - scores: L-C2ST scores at `x_o`, of shape (`num_folds`,).
            - (probs, scores): Predicted probabilities and L-C2ST scores at `x_o`,
              each of shape (`num_folds`,).
        """
        if x_o.shape == self.x_p_mean.shape:
            x_o = x_o[None, ...]

        # prepare data
        if self.z_score:
            theta_o = (theta_o - self.theta_p_mean) / self.theta_p_std
            x_o = (x_o - self.x_p_mean) / self.x_p_std

        probs, scores = [], []

        # evaluate classifiers
        for clf in trained_clfs:
            proba, score = eval_lc2st(theta_o, x_o, clf, return_proba=True)
            probs.append(proba)
            scores.append(score)
        probs, scores = np.array(probs), np.array(scores)

        if return_probs:
            return probs, scores
        else:
            return scores

    def train_on_observed_data(
        self, seed: Optional[int] = None, verbosity: int = 1
    ) -> Union[None, List[Any]]:
        """Trains the classifier on the observed data.

        Saves the trained classifier(s) as a list of length `num_folds`.

        Parameters
        ----------
        seed : int, optional
            Random state of the classifier. Default is None.
        verbosity : int, optional
            Verbosity level. Default is 1.
        """
        # set random state
        if seed is not None:
            if "random_state" in self.clf_kwargs:
                print("WARNING: changing the random state of the classifier.")
            self.clf_kwargs["random_state"] = seed

        # train the classifier
        trained_clfs = self._train(
            self.theta_p, self.theta_q, self.x_p, self.x_q, verbosity=verbosity
        )
        self.trained_clfs = trained_clfs

    def get_statistic_on_observed_data(
        self,
        theta_o: Array,
        x_o: Array,
    ) -> float:
        """Computes the L-C2ST statistics for the observed data.

        Mean over all cv-scores.

        Parameters
        ----------
        theta_o : Array
            Samples from the posterior conditioned on the observation `x_o`,
            of shape (sample_size, dim).
        x_o : Array
            The observation, of shape (, dim_x)

        Returns
        -------
        float
            L-C2ST statistic at `x_o`.
        """
        assert (
            self.trained_clfs is not None
        ), "No trained classifiers found. Run `train_on_observed_data` first."
        _, scores = self.get_scores(
            theta_o=theta_o,
            x_o=x_o,
            trained_clfs=self.trained_clfs,
            return_probs=True,
        )
        return float(scores.mean())

    def p_value(
        self,
        theta_o: Array,
        x_o: Array,
    ) -> float:
        r"""Computes the p-value for L-C2ST.

        The p-value is the proportion of times the L-C2ST statistic under the null
        hypothesis is greater than the L-C2ST statistic at the observation `x_o`.
        It is computed by taking the empirical mean over statistics computed on
        several trials under the null hypothesis: $1/H \sum_{h=1}^{H} I(T_h < T_o)$.

        Parameters
        ----------
        theta_o : Array
            Samples from the posterior conditioned on the observation `x_o`,
            of dhape (sample_size, dim).
        x_o : Array
            The observation, of shape (, dim_x).

        Returns
        -------
        float
            p-value for L-C2ST at `x_o`.
        """
        stat_data = self.get_statistic_on_observed_data(theta_o=theta_o, x_o=x_o)
        _, stats_null = self.get_statistics_under_null_hypothesis(
            theta_o=theta_o, x_o=x_o, return_probs=True, verbosity=0
        )
        return float((stat_data < stats_null).mean())

    def reject_test(
        self,
        theta_o: Array,
        x_o: Array,
        alpha: float = 0.05,
    ) -> bool:
        """Computes the test result for L-C2ST at a given significance level.

        Parameters
        ----------
        theta_o : Array
            Samples from the posterior conditioned on the observation `x_o`,
            of shape (sample_size, dim).
        x_o : Array
            The observation, of shape (, dim_x).
        alpha : float, optional
            Significance level. Default is 0.05.

        Returns
        -------
        bool
            The L-C2ST result: True if rejected, False otherwise.
        """
        return bool(self.p_value(theta_o=theta_o, x_o=x_o) < alpha)

    def train_under_null_hypothesis(
        self,
        verbosity: int = 1,
    ) -> None:
        """Computes the L-C2ST scores under the null hypothesis (H0).
        Saves the trained classifiers for each null trial.

        Parameters
        ----------
        verbosity : int, optional
            Verbosity level. Default is 1.
        """

        trained_clfs_null = {}
        for t in tqdm(
            range(self.num_trials_null),
            desc=f"Training the classifiers under H0, permutation = {self.permutation}",
            disable=verbosity < 1,
        ):
            # prepare data
            if self.permutation:
                joint_p = jnp.concatenate([self.theta_p, self.x_p], axis=1)
                joint_q = jnp.concatenate([self.theta_q, self.x_q], axis=1)
                # permute data (same as permuting the labels)
                joint_p_perm, joint_q_perm = permute_data(joint_p, joint_q, seed=t)
                # extract the permuted P and Q and x
                theta_p_t, x_p_t = (
                    joint_p_perm[:, : self.theta_p.shape[-1]],
                    joint_p_perm[:, self.theta_p.shape[1] :],
                )
                theta_q_t, x_q_t = (
                    joint_q_perm[:, : self.theta_q.shape[-1]],
                    joint_q_perm[:, self.theta_q.shape[1] :],
                )
            else:
                assert (
                    self.null_distribution is not None
                ), "You need to provide a null distribution"
                theta_p_t = self.null_distribution.sample(
                    self.rngs.sample(), (self.theta_p.shape[0],)
                )
                theta_q_t = self.null_distribution.sample(
                    self.rngs.sample(), (self.theta_p.shape[0],)
                )
                x_p_t, x_q_t = self.x_p, self.x_q

            if self.z_score:
                theta_p_t = (theta_p_t - self.theta_p_mean) / self.theta_p_std
                theta_q_t = (theta_q_t - self.theta_p_mean) / self.theta_p_std
                x_p_t = (x_p_t - self.x_p_mean) / self.x_p_std
                x_q_t = (x_q_t - self.x_p_mean) / self.x_p_std

            # train
            clf_t = self._train(theta_p_t, theta_q_t, x_p_t, x_q_t, verbosity=0)
            trained_clfs_null[t] = clf_t

        self.trained_clfs_null = trained_clfs_null

    def get_statistics_under_null_hypothesis(
        self,
        theta_o: Array,
        x_o: Array,
        return_probs: bool = False,
        verbosity: int = 0,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Computes the L-C2ST scores under the null hypothesis.

        Parameters
        ----------
        theta_o : Array
            Samples from the posterior conditioned on the observation `x_o`,
            of shape (sample_size, dim).
        x_o : Array
            The observation, of shape (, dim_x).
        return_probs : bool, optional
            Whether to return the predicted probabilities of being in P. Default is False.
        verbosity : int, optional
            Verbosity level. Default is 1.

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            - scores: L-C2ST scores under (H0).
            - (probs, scores): Predicted probabilities and L-C2ST scores under (H0).
        """

        if self.trained_clfs_null is None:
            raise ValueError(
                "You need to train the classifiers under (H0). \
                    Run `train_under_null_hypothesis`."
            )
        else:
            assert (
                len(self.trained_clfs_null) == self.num_trials_null
            ), "You need one classifier per trial."

        probs_null, stats_null = [], []
        for t in tqdm(
            range(self.num_trials_null),
            desc=f"Computing T under (H0) - permutation = {self.permutation}",
            disable=verbosity < 1,
        ):
            # prepare data
            if self.permutation:
                theta_o_t = theta_o
            else:
                assert (
                    self.null_distribution is not None
                ), "You need to provide a null distribution"

                theta_o_t = self.null_distribution.sample(
                    self.rngs.sample(), (theta_o.shape[0],)
                )

            if self.z_score:
                theta_o_t = (theta_o_t - self.theta_p_mean) / self.theta_p_std
                x_o = (x_o - self.x_p_mean) / self.x_p_std

            # evaluate
            clf_t = self.trained_clfs_null[t]
            probs, scores = self.get_scores(
                theta_o=theta_o_t, x_o=x_o, trained_clfs=clf_t, return_probs=True
            )
            probs_null.append(probs)
            stats_null.append(scores.mean())

        probs_null, stats_null = np.array(probs_null), np.array(stats_null)

        if return_probs:
            return probs_null, stats_null
        else:
            return stats_null


def train_lc2st(
    theta_p: Array, theta_q: Array, x_p: Array, x_q: Array, clf: BaseEstimator
) -> Any:
    """Trains the classifier on the joint data for the L-C2ST.

    Parameters
    ----------
    theta_p : Array
        Samples from P, of shape (sample_size, dim).
    theta_q : Array
        Samples from Q, of shape (sample_size, dim).
    x_p : Array
        Observations corresponding to P, of shape (sample_size, dim_x).
    x_q : Array
        Observations corresponding to Q, of shape (sample_size, dim_x).
    clf : BaseEstimator
        Classifier to train.

    Returns
    -------
    Any
        Trained classifier.
    """
    # concatenate to get joint data
    joint_p = np.concatenate([np.array(theta_p), np.array(x_p)], axis=1)
    joint_q = np.concatenate([np.array(theta_q), np.array(x_q)], axis=1)

    # prepare data
    data = np.concatenate((joint_p, joint_q))
    # labels
    target = np.concatenate(
        (
            np.zeros((joint_p.shape[0],)),
            np.ones((joint_q.shape[0],)),
        )
    )

    # train classifier
    clf_ = clone(clf)
    clf_.fit(data, target)  # type: ignore

    return clf_


def eval_lc2st(
    theta_p: Array, x_o: Array, clf: BaseEstimator, return_proba: bool = False
) -> Union[float, Tuple[np.ndarray, float]]:
    """Evaluates the classifier returned by `train_lc2st` for one observation
    `x_o` and over the samples `P`.

    Parameters
    ----------
    theta_p : Array
        Samples from p (class 0), of shape (sample_size, dim).
    x_o : Array
        The observation, of shape (1, dim_x).
    clf : BaseEstimator
        Trained classifier.
    return_proba : bool, optional
        Whether to return the predicted probabilities of being in P. Default is False.

    Returns
    -------
    Union[float, Tuple[np.ndarray, float]]
        L-C2ST score at `x_o`: MSE between 0.5 and the predicted classifier
        probability for class 0 on `theta_p`.
    """
    # concatenate to get joint data
    joint_p = np.concatenate(
        [np.array(theta_p), np.array(x_o).repeat(theta_p.shape[0], 0)], axis=1
    )

    # evaluate classifier
    # probability of being in P (class 0)
    proba = clf.predict_proba(joint_p)[:, 0]  # type: ignore
    # mean squared error between proba and dirac at 0.5
    score = float(((proba - [0.5] * len(proba)) ** 2).mean())

    if return_proba:
        return proba, score
    else:
        return score


def permute_data(theta_p: Array, theta_q: Array, seed: int = 1) -> Tuple[Array, Array]:
    """Permutes the concatenated data [P,Q] to create null samples.

    Parameters
    ----------
    theta_p : Array
        Samples from P, of shape (sample_size, dim).
    theta_q : Array
        Samples from Q, of shape (sample_size, dim).
    seed : int, optional
        Random seed. Default is 1.

    Returns
    -------
    Tuple[Array, Array]
        Permuted data [theta_p, theta_q].
    """
    key = jax.random.PRNGKey(seed)
    # check inputs
    assert theta_p.shape[0] == theta_q.shape[0]

    sample_size = theta_p.shape[0]
    X = jnp.concatenate([theta_p, theta_q], axis=0)
    x_perm = X[jax.random.permutation(key, sample_size * 2)]
    return x_perm[:sample_size], x_perm[sample_size:]


class EnsembleClassifier(BaseEstimator):
    def __init__(self, clf, num_ensemble=1, verbosity=1):
        self.clf = clf
        self.num_ensemble = num_ensemble
        self.trained_clfs = []
        self.verbosity = verbosity

    def fit(self, X, y):
        for n in tqdm(
            range(self.num_ensemble),
            desc="Ensemble training",
            disable=self.verbosity < 1,
        ):
            clf = clone(self.clf)
            if clf.random_state is not None:  # type: ignore
                clf.random_state += n  # type: ignore
            else:
                clf.random_state = n + 1  # type: ignore
            clf.fit(X, y)  # type: ignore
            self.trained_clfs.append(clf)

    def predict_proba(self, X):
        probas = [clf.predict_proba(X) for clf in self.trained_clfs]
        return np.mean(probas, axis=0)


def plot_lc2st(
    lc2st: LC2ST,
    post_samples_star: Array,
    x_o: Array,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    conf_alpha = 0.05
) -> Tuple[Figure, Axes]:
    
    probs_data, scores_data = lc2st.get_scores(
        theta_o=post_samples_star,
        x_o=x_o,
        return_probs=True,
        trained_clfs=lc2st.trained_clfs,
    )
    probs_null, scores_null = lc2st.get_statistics_under_null_hypothesis(
        theta_o=post_samples_star,
        x_o=x_o,
        return_probs=True,
    )

    p_value = lc2st.p_value(post_samples_star, x_o)
    reject = lc2st.reject_test(post_samples_star, x_o, alpha=conf_alpha)

    if fig is None or ax is None:
        fig, ax = plt.subplots(1,1,
            figsize=(5, 3)
        )
        
    quantiles = np.quantile(scores_null, [0, 1 - conf_alpha])
    ax.hist(scores_null, bins=50, density=True, alpha=0.5, label="Null")
    ax.axvline(np.mean(scores_data), color="red", label="Observed")
    ax.axvline(quantiles[0], color="black", linestyle="--", label=f"{(1 - conf_alpha) * 100:.0f}% CI")
    ax.axvline(quantiles[1], color="black", linestyle="--")
    ax.set_xlabel("Test statistic")
    ax.set_ylabel("Density")
    ax.set_title(f"p-value = {p_value:.3f}, reject = {reject}")

    return fig, ax
