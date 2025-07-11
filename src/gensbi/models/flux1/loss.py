import jax.numpy as jnp
import jax
from flax import nnx

from gensbi.flow_matching.loss import ContinuousFMLoss


class FluxCFMLoss(ContinuousFMLoss):
    """
    FluxCFMLoss is a class that computes the continuous flow matching loss for the Flux model.

    Args:
        path: Probability path (x-prediction training).
        reduction (str, optional): Specify the reduction to apply to the output ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction is applied to the output, ``'mean'``: the output is reduced by mean over sequence elements, ``'sum'``: the output is reduced by sum over sequence elements. Defaults to 'mean'.
    """
    def __init__(self, path, reduction="mean", cfg_scale=None):
        # self.path = path
        # if reduction not in ["None", "mean", "sum"]:
        #     raise ValueError(f"{reduction} is not a valid value for reduction")

        # if reduction == "mean":
        #     self.reduction = jnp.mean
        # elif reduction == "sum":
        #     self.reduction = jnp.sum
        # else:
        #     self.reduction = lambda x: x

        super().__init__(path, reduction)

        self.cfg_scale = cfg_scale

    def __call__(self, vf, batch, cond, obs_ids, cond_ids):
        """
        Evaluates the continuous flow matching loss.

        Args:
            vf (callable): The vector field model to evaluate.
            batch (tuple): A tuple containing the input data (x_0, x_1, t).
            cond (jnp.ndarray): The conditioning data.
            obs_ids (jnp.ndarray): The observation IDs.
            cond_ids (jnp.ndarray): The conditioning IDs.

        Returns:
            jnp.ndarray: The computed loss.
        """

        x_0, x_1, t = batch

        path_sample = self.path.sample(x_0, x_1, t)

        x_t = path_sample.x_t


        if self.cfg_scale is not None:
            key = jax.random.PRNGKey(0)
            conditioned = jax.random.bernoulli(key, p=self.cfg_scale, shape=(x_t.shape[0],))
        else:
            conditioned = jnp.ones((x_t.shape[0],), dtype=jnp.bool_)


        model_output = vf(x_t, obs_ids, cond, cond_ids, t, conditioned=conditioned)
        loss = model_output - path_sample.dx_t
        loss = jnp.square(loss)
            
        return self.reduction(loss)
