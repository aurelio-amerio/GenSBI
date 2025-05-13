import jax.numpy as jnp
from flax import nnx


class ContinuousFMLoss(nnx.Module):
    def __init__(self, path, reduction="mean"):
        """
        ContinuousFMLoss is a class that computes the continuous flow matching loss.

        Args:
            path (MixtureDiscreteProbPath): Probability path (x-prediction training).
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

    def __call__(self, vf, batch, args=None, condition_mask=None, **kwargs):
        """
        Evaluates the continuous flow matching loss.

        Args:
            vf (callable): The vector field model to evaluate.
            batch (tuple): A tuple containing the input data (x_0, x_1, t).
            args (optional): Additional arguments for the function.
            condition_mask (optional): A mask to apply to the input data.
            **kwargs: Additional keyword arguments for the function.

        Returns:
            jnp.ndarray: The computed loss.
        """
        _, x_1, _ = batch
        path_sample = self.path.sample(*batch)

        if condition_mask is not None:
            kwargs["condition_mask"] = condition_mask

        x_t = path_sample.x_t

        if condition_mask is not None:
            condition_mask = condition_mask.reshape(x_t.shape)
            x_t = jnp.where(condition_mask, x_1, x_t)
        model_output = vf(path_sample.x_t, path_sample.t, args=args, **kwargs)
        loss = model_output - path_sample.dx_t
        if condition_mask is not None:
            loss = jnp.where(condition_mask, 0.0, loss)

        return self.reduction(jnp.square(loss))
