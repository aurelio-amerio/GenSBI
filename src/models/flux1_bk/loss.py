import jax.numpy as jnp
import jax
from flax import nnx


class FluxCFMLoss(nnx.Module):
    def __init__(self, path, reduction="mean", cfg_scale=0.5):
        """
        ContinuousFMLoss is a class that computes the continuous flow matching loss.

        Args:
            path: Probability path (x-prediction training).
            reduction (str, optional): Specify the reduction to apply to the output ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction is applied to the output, ``'mean'``: the output is reduced by mean over sequence elements, ``'sum'``: the output is reduced by sum over sequence elements. Defaults to 'mean'.
        """
        self.path = path
        if reduction not in ["None", "mean", "sum"]:
            raise ValueError(f"{reduction} is not a valid value for reduction")

        if reduction == "mean":
            self.reduction = jnp.mean
        elif reduction == "sum":
            self.reduction = jnp.sum
        else:
            self.reduction = lambda x: x

        self.cfg_scale = cfg_scale

    def __call__(self, vf, batch, cond, obs_ids, cond_ids, **kwargs):
        """
        Evaluates the continuous flow matching loss.

        Args:
            vf (callable): The vector field model to evaluate.
            batch (tuple): A tuple containing the input data (x_0, x_1, t).
            cond (jnp.ndarray): The conditioning data.
            obs_ids (jnp.ndarray): The observation IDs.
            cond_ids (jnp.ndarray): The conditioning IDs.
            **kwargs: Additional keyword arguments for the function.

        Returns:
            jnp.ndarray: The computed loss.
        """

        x_0, x_1, t = batch

        path_sample = self.path.sample(x_0, x_1, t)

        x_t = path_sample.x_t

        model_output = vf(x_t, obs_ids, cond, cond_ids, t, conditioned=True)
        model_output_uncond = vf(x_t, obs_ids, cond, cond_ids, t, conditioned=False)

        loss_cond = model_output - path_sample.dx_t
        loss_uncond = model_output_uncond - path_sample.dx_t

        weight = self.cfg_scale

        loss = weight*jnp.square(loss_cond) + (1-weight)*jnp.square(loss_uncond)
        return self.reduction(loss)
