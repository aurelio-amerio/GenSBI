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

    def __call__(self, vf, batch, args=None, **kwargs):
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

        path_sample = self.path.sample(*batch)

        x_t = path_sample.x_t

        model_output = vf(x_t, path_sample.t, args=args, **kwargs)
        
        loss = model_output - path_sample.dx_t

        return self.reduction(jnp.square(loss))
