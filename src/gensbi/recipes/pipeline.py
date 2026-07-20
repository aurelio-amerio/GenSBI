"""
Pipeline module for GenSBI.

This module provides an abstract pipeline class for training and evaluating conditional generative models
(such as conditional flow matching or diffusion models) in the GenSBI framework. It handles model creation,
training loop, optimizer setup, checkpointing, and evaluation utilities.

For practical implementations, subclasses should implement specific model architectures, loss functions, and sampling methods.
See `JointPipeline` and `ConditionalPipeline` for concrete examples.

"""

from flax import nnx
import jax
from jax import numpy as jnp
from typing import Any, Callable, Optional, Tuple
from jax import Array

from numpyro import distributions as dist
from gensbi.utils.math import _expand_dims

import abc
from functools import partial

import optax
from optax.contrib import reduce_on_plateau

import orbax.checkpoint as ocp

from tqdm import tqdm

import os
import warnings
import flax.traverse_util as tu


from gensbi.utils.misc import get_colored_value
from gensbi.utils.serialization import save_safetensors, load_safetensors


def _warn_if_not_fp32_master_weights(model):
    """Warn when trainable params are not fp32 master weights.

    Mixed precision in GenSBI stores master weights in fp32 and selects the
    compute dtype via each model's ``dtype`` knob; bf16 master weights break
    AdamW moment accumulation and make optax.ema unable to integrate
    (1 - decay)-scale updates.
    """
    flat = tu.flatten_dict(nnx.to_pure_dict(nnx.state(model, nnx.Param)))
    bad = {
        ".".join(str(p) for p in k): str(v.dtype)
        for k, v in flat.items()
        if jnp.issubdtype(v.dtype, jnp.floating) and v.dtype != jnp.float32
    }
    if bad:
        warnings.warn(
            "model has non-fp32 master weights (training will be numerically "
            f"degraded; set param_dtype=jnp.float32 and use the dtype knob "
            f"for compute instead): {bad}",
            UserWarning,
            stacklevel=3,
        )


def _cast_state_to_target_dtypes(state: nnx.State, target_state: nnx.State) -> nnx.State:
    """Cast ``state``'s leaves to match ``target_state``'s dtypes, in place.

    Old checkpoints (e.g. bf16 master weights from a pre-mixed-precision run)
    must still be restorable into a model whose current dtype is fp32;
    mirrors the ``arr.astype(want.dtype)`` loop already used by
    :func:`gensbi.utils.serialization.load_safetensors`.
    """
    flat = tu.flatten_dict(nnx.to_pure_dict(state))
    target_flat = tu.flatten_dict(nnx.to_pure_dict(target_state))
    new = {}
    for k, arr in flat.items():
        want = target_flat.get(k)
        if want is not None and hasattr(arr, "dtype") and hasattr(want, "dtype"):
            new[k] = arr.astype(want.dtype)
        else:
            new[k] = arr
    nnx.replace_by_pure_dict(state, tu.unflatten_dict(new))
    return state


class ModelEMA(nnx.Optimizer):
    """
    Exponential Moving Average (EMA) optimizer for maintaining a smoothed version of model parameters.

    This optimizer keeps an exponential moving average of the model parameters, which can help stabilize training
    and improve evaluation performance. The EMA parameters are updated at each training step.

    Parameters
    ----------
    model : nnx.Module
        The model whose parameters will be tracked.
    tx : optax.GradientTransformation
        The Optax transformation defining the EMA update rule.

    """

    def __init__(
        self,
        model: nnx.Module,
        tx: optax.GradientTransformation,
    ):
        super().__init__(model, tx, wrt=[nnx.Param, nnx.BatchStat])

    def update(self, model, model_orginal: nnx.Module):
        """
        Update the EMA parameters using the current model parameters.
        Parameters
        ----------
        model : nnx.Module
            The model with EMA parameters to be updated.
        model_orginal : nnx.Module
            The original model with current parameters.
        """
        params = nnx.state(model_orginal, self.wrt)
        ema_params = nnx.state(model, self.wrt)
        self.step[...] += 1

        ema_state = optax.EmaState(count=self.step, ema=ema_params)

        _, new_ema_state = self.tx.update(params, ema_state)

        nnx.update(model, new_ema_state.ema)


@nnx.jit
def ema_step(ema_model, model, ema_optimizer: nnx.Optimizer):
    """Update EMA model with current model parameters."""
    ema_optimizer.update(ema_model, model)


def _get_batch_sampler(
    sampler_fn: Callable,
    ncond: int,
    chunk_size: int,
    show_progress_bars: bool = True,
):
    """
    Create a batch sampler that processes samples in chunks.

    Parameters
    ----------
    sampler_fn : Callable
        Sampling function.
    ncond : int
        Number of conditions.
    chunk_size : int
        Size of each chunk.
    show_progress_bars : bool, optional
        Whether to show progress bars.

    Returns
    -------
    Callable
        Batch sampler function.
    """

    # JIT the chunk processor
    @jax.jit
    def process_chunk(key_batch):
        """Process a batch of keys."""
        return jax.vmap(lambda k: sampler_fn(k, ncond))(key_batch)

    def sampler(keys):
        """Sample in batches with optional progress bar."""
        n_samples = keys.shape[0]
        results = []

        # Calculate total chunks for tqdm
        # We use ceil division to handle remainders
        n_chunks = (n_samples + chunk_size - 1) // chunk_size

        # Using a tqdm loop

        if show_progress_bars:
            loop = tqdm(
                range(0, n_samples, chunk_size),
                total=n_chunks,
                desc="Sampling",
            )
        else:
            loop = range(0, n_samples, chunk_size)

        for i in loop:
            batch_keys = keys[i : i + chunk_size]

            chunk_out = process_chunk(batch_keys)

            # CRITICAL: Wait for GPU to finish this chunk before updating bar
            # This makes the progress bar accurate.
            chunk_out.block_until_ready()

            results.append(chunk_out)

        return jnp.concatenate(results, axis=0)

    return sampler


class AbstractPipeline(abc.ABC):
    """
    Abstract base class for GenSBI training pipelines.

    This class provides a template for implementing training and evaluation pipelines for conditional generative models.
    Subclasses should implement model creation, default parameter setup, loss function, sampling, and evaluation methods.

    Parameters
    ----------
    train_dataset : iterable
        Training dataset, should yield batches of data.
    val_dataset : iterable
        Validation dataset, should yield batches of data.
    dim_obs : int
        Dimensionality of the parameter (theta) space.
    dim_cond : int
        Dimensionality of the observation (x) space.
    model : nnx.Module, optional
        The model to be trained. If None, the model is created using `_make_model`.
    params : dict, optional
        Model parameters. If None, uses defaults from `_get_default_params`.
    ch_obs : int, optional
        Number of channels in the observation data. Default is 1.
    ch_cond : int, optional
        Number of channels in the conditional data (if applicable). Default is None.
    training_config : dict, optional
        Training configuration. If None, uses defaults from `get_default_training_config`.

    """

    def __init__(
        self,
        model: nnx.Module,
        train_dataset,
        val_dataset,
        dim_obs: int,
        dim_cond: int,
        ch_obs=1,
        ch_cond=None,
        params=None,
        training_config=None,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_dataset_iter = iter(self.train_dataset)
        self.val_dataset_iter = iter(self.val_dataset)

        self.dim_obs = dim_obs
        self.dim_cond = dim_cond
        # test test
        # self.dim_joint = dim_obs + dim_cond

        self.ch_obs = ch_obs
        self.ch_cond = ch_cond

        # self.node_ids = None # to be set in subclass
        # self.obs_ids = None # to be set in subclass
        # self.cond_ids = None # to be set in subclass

        self.params = params

        self.training_config = training_config
        if training_config is None:
            self.training_config = self.get_default_training_config()

        self.training_config["min_scale"] = (
            self.training_config["min_lr"] / self.training_config["max_lr"]
            if self.training_config["max_lr"] > 0
            else 0.0
        )

        os.makedirs(self.training_config["checkpoint_dir"], exist_ok=True)

        self.model = model
        if model is not None:
            _warn_if_not_fp32_master_weights(model)
        self.model_wrapped = None  # to be set in subclass

        if model is None:
            self.ema_model = None
        else:
            self.ema_model = nnx.clone(model)
        self.ema_model_wrapped = None  # to be set in subclass

        self.p0_dist_model = None  # to be set in subclass
        self.loss_fn = None  # to be set in subclass
        self.path = None  # to be set in subclass

    @abc.abstractmethod
    def init_pipeline_from_config(
        cls,
        train_dataset,
        val_dataset,
        dim_obs: int,
        dim_cond: int,
        config_path: str,
        checkpoint_dir: str,
    ):
        """
        Initialize the pipeline from a configuration file.

        Parameters
        ----------
        train_dataset : iterable
            Training dataset.
        val_dataset : iterable
            Validation dataset.
        dim_obs : int
            Dimensionality of the parameter (theta) space.
        dim_cond : int
            Dimensionality of the observation (x) space.
        config_path : str
            Path to the configuration file.
        checkpoint_dir : str
            Directory for saving checkpoints.

        Returns
        -------
        pipeline : AbstractPipeline
            An instance of the pipeline initialized from the configuration.
        """
        ...  # pragma: no cover

    @abc.abstractmethod
    def _make_model(self, params):
        """
        Create and return the model to be trained.
        """
        ...  # pragma: no cover

    def _get_ema_optimizer(self):
        """
        Construct the EMA optimizer for maintaining an exponential moving average of model parameters.
        Returns
        -------
        ema_optimizer : ModelEMA
            The EMA optimizer instance.
        """
        ema_tx = optax.ema(self.training_config["ema_decay"])
        ema_optimizer = ModelEMA(self.ema_model, ema_tx)
        return ema_optimizer

    def _get_optimizer(self):
        """
        Construct the optimizer for training, including learning rate scheduling and gradient clipping.

        Returns
        -------
        optimizer : nnx.Optimizer
            The optimizer instance for the model.
        """
        warmup_steps = (
            self.training_config["warmup_steps"] * self.training_config["multistep"]
        )
        nsteps = self.training_config["nsteps"]
        max_lr = self.training_config["max_lr"]
        min_lr = self.training_config["min_lr"]

        # we define the following schedule using join schedules: warmup for warmup_steps, then constant LR until 90% of the training steps, then cosine decay to min_lr
        decay_transition = self.training_config["decay_transition"]

        warmup_schedule = optax.linear_schedule(
            init_value=1e-7, end_value=max_lr, transition_steps=warmup_steps
        )
        constant_schedule = optax.constant_schedule(value=max_lr)
        decay_schedule = optax.cosine_decay_schedule(
            init_value=max_lr,
            decay_steps=int((1 - decay_transition) * nsteps),
            alpha=min_lr / max_lr,
        )
        schedule = optax.join_schedules(
            schedules=[
                warmup_schedule,
                constant_schedule,
                decay_schedule,
            ],
            boundaries=[warmup_steps, int(decay_transition * nsteps)],
        )

        # define the weight decay mask to avoid applying weight decay to bias and norm parameters
        def decay_mask_fn(params):
            return jax.tree_util.tree_map(lambda x: x.ndim > 1, params)

        opt = optax.chain(
            optax.adaptive_grad_clip(10.0),
            optax.adamw(schedule, mask=decay_mask_fn),
        )
        if self.training_config["multistep"] > 1:
            opt = optax.MultiSteps(opt, self.training_config["multistep"])

        optimizer = nnx.Optimizer(self.model, opt, wrt=nnx.Param)
        return optimizer

    @abc.abstractmethod
    def get_default_params(cls, dim_obs, dim_cond, ch_obs, ch_cond):
        raise NotImplementedError(
            "Default parameters not implemented for ConditionalFlowPipeline."
        )

    @classmethod
    def get_default_training_config(cls):
        """
        Return a dictionary of default training configuration parameters.

        Returns
        -------
        training_config : dict
            Default training configuration.
        """
        training_config = {}

        training_config["nsteps"] = 50_000

        training_config["ema_decay"] = 0.999
        
        training_config["decay_transition"] = 0.80
        training_config["warmup_steps"] = 500

        training_config["max_lr"] = 1e-4
        training_config["min_lr"] = 1e-6
        training_config["val_every"] = 100
        training_config["early_stopping"] = True
        training_config["experiment_id"] = 1
        training_config["multistep"] = 1
        training_config["checkpoint_dir"] = os.path.join(os.getcwd(), "checkpoints")
        training_config["val_error_ratio"] = 1.3

        return training_config

    def update_training_config(self, new_config):
        """
        Update the training configuration with new parameters.

        Parameters
        ----------
        new_config : dict
            New training configuration parameters.
        """
        self.training_config.update(new_config)
        self.training_config["min_scale"] = (
            self.training_config["min_lr"] / self.training_config["max_lr"]
            if self.training_config["max_lr"] > 0
            else 0.0
        )
        return

    # def update_params(self, new_params):
    #     """
    #     Update the model parameters and re-initialize the model.

    #     Parameters
    #     ----------
    #     new_params : dict
    #         New model parameters.
    #     """
    #     self.params = new_params
    #     self.model = self._make_model(self.params)
    #     self.model_wrapped = None  # to be set in subclass
    #     return

    @abc.abstractmethod
    def get_loss_fn(self):
        """
        Return the loss function for training/validation.
        """
        ...  # pragma: no cover

    def get_train_step_fn(self, loss_fn):
        """
        Return the training step function, which performs a single optimization step.

        Returns
        -------
        train_step : Callable
            JIT-compiled training step function.
        """

        @nnx.jit
        def train_step(model, optimizer, batch, key: jax.random.PRNGKey):
            """Perform single training step with gradient update."""
            loss, grads = nnx.value_and_grad(loss_fn)(model, batch, key)
            optimizer.update(model, grads, value=loss)
            return loss

        return train_step

    def get_val_step_fn(self, loss_fn):
        """
        Return the validation step function, which performs a single optimization step.

        Returns
        -------
        val_step : Callable
            JIT-compiled validation step function.
        """

        @nnx.jit
        def val_step(model, batch, key: jax.random.PRNGKey):
            """Compute validation loss for a batch."""
            loss = loss_fn(model, batch, key)
            return loss

        return val_step

    def save_model(self, experiment_id=None):
        """
        Save model and EMA model checkpoints.

        Parameters
        ----------
        experiment_id : str, optional
            Experiment identifier. If None, uses training_config value.
        """
        if experiment_id is None:
            experiment_id = self.training_config["experiment_id"]

        checkpoint_dir = self.training_config["checkpoint_dir"]
        checkpoint_dir_ema = os.path.join(self.training_config["checkpoint_dir"], "ema")

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(checkpoint_dir_ema, exist_ok=True)

        # Save the model
        checkpoint_manager = ocp.CheckpointManager(
            checkpoint_dir,
            options=ocp.CheckpointManagerOptions(
                max_to_keep=None,
                keep_checkpoints_without_metrics=True,
                create=True,
            ),
        )
        _, state = nnx.split(self.model)
        checkpoint_manager.save(
            experiment_id,
            args=ocp.args.Composite(state=ocp.args.StandardSave(state)),
        )
        checkpoint_manager.close()

        # now we create the ema model and save it
        _, ema_state = nnx.split(self.ema_model)

        # save the ema model
        checkpoint_manager_ema = ocp.CheckpointManager(
            checkpoint_dir_ema,
            options=ocp.CheckpointManagerOptions(
                max_to_keep=None,
                keep_checkpoints_without_metrics=True,
                create=True,
            ),
        )

        checkpoint_manager_ema.save(
            experiment_id,
            args=ocp.args.Composite(state=ocp.args.StandardSave(ema_state)),
        )

        checkpoint_manager_ema.close()

        print("Saved model to checkpoint")
        return

    def restore_model(self, experiment_id=None):
        """
        Restore model and EMA model from checkpoints.

        Parameters
        ----------
        experiment_id : str, optional
            Experiment identifier. If None, uses training_config value.
        """
        if experiment_id is None:
            experiment_id = self.training_config["experiment_id"]

        graphdef, model_state = nnx.split(self.model)

        with ocp.CheckpointManager(
            self.training_config["checkpoint_dir"],
            options=ocp.CheckpointManagerOptions(read_only=True),
        ) as read_mgr:
            restored = read_mgr.restore(
                experiment_id,
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(item=model_state)
                ),
            )
        _cast_state_to_target_dtypes(restored["state"], model_state)
        self.model = nnx.merge(graphdef, restored["state"])

        # restore the ema model
        graphdef, model_state_ema = nnx.split(self.ema_model)

        with ocp.CheckpointManager(
            os.path.join(self.training_config["checkpoint_dir"], "ema"),
            options=ocp.CheckpointManagerOptions(read_only=True),
        ) as read_mgr_ema:
            restored_ema = read_mgr_ema.restore(
                experiment_id,
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(item=model_state_ema)
                ),
            )
        _cast_state_to_target_dtypes(restored_ema["state"], model_state_ema)
        self.ema_model = nnx.merge(graphdef, restored_ema["state"])

        self.model.eval()
        self.ema_model.eval()

        # wrap models
        self._wrap_model()

        print("Restored model from checkpoint")
        return

    def export_safetensors(self, path, *, ema=True, metadata=None):
        """Export trained weights to a single ``.safetensors`` file.

        ``ema=True`` (default) exports the EMA model -- usually the weights you
        want for inference and for sharing. Pass ``ema=False`` for the primary
        model. This is a thin wrapper over
        :func:`gensbi.utils.serialization.save_safetensors`.
        """
        model = self.ema_model if ema else self.model
        save_safetensors(model, path, metadata=metadata)

    def import_safetensors(self, path, *, ema=True, strict=True):
        """Load weights from a ``.safetensors`` file into this pipeline in place.

        ``ema=True`` (default) loads into the EMA model. Thin wrapper over
        :func:`gensbi.utils.serialization.load_safetensors`.
        """
        model = self.ema_model if ema else self.model
        load_safetensors(model, path, strict=strict)

    @abc.abstractmethod
    def _wrap_model(self):
        """
        Wrap the model for evaluation (either using JointWrapper or ConditionalWrapper).
        """
        ...  # pragma: no cover

    def _restore_best_state(self, best_state, best_state_ema):
        """Restore the best model and EMA states (used after early stopping).

        Parameters
        ----------
        best_state : nnx.State
            Best model state recorded during training.
        best_state_ema : nnx.State
            Best EMA model state recorded during training.
        """
        graphdef = nnx.graphdef(self.model)
        self.model = nnx.merge(graphdef, best_state)
        self.ema_model = nnx.merge(graphdef, best_state_ema)

    def _run_validation(self, val_step, batch_val, rng_val, min_val,
                        best_state, best_state_ema, counter, val_error_ratio,
                        loss_array, val_loss_array, l_train):
        """Run a validation step and update early-stopping bookkeeping.

        Parameters
        ----------
        val_step : Callable
            Validation step function.
        batch_val : Any
            Fixed validation batch.
        rng_val : jax.random.PRNGKey
            Fixed validation RNG key.
        min_val : float
            Best validation loss seen so far.
        best_state : nnx.State
            Current best model state.
        best_state_ema : nnx.State
            Current best EMA model state.
        counter : int
            Early-stopping patience counter.
        val_error_ratio : float
            Threshold ratio for incrementing the counter.
        loss_array : list
            Training loss history (mutated in place).
        val_loss_array : list
            Validation loss history (mutated in place).
        l_train : float
            Current smoothed training loss.

        Returns
        -------
        l_val : float
            Validation loss for this step.
        ratio : float
            Ratio of current validation loss to best.
        min_val : float
            Updated best validation loss.
        best_state : nnx.State
            Updated best model state.
        best_state_ema : nnx.State
            Updated best EMA model state.
        counter : int
            Updated early-stopping counter.
        """
        # Use a fixed val batch and rng for validation, to avoid noise
        l_val = val_step(self.model, batch_val, rng_val)

        ratio = l_val / min_val
        if ratio > val_error_ratio:
            counter += 1
        else:
            counter = 0

        loss_array.append(l_train)
        val_loss_array.append(l_val)

        if l_val < min_val:
            min_val = l_val
            best_state = nnx.state(self.model)
            best_state_ema = nnx.state(self.ema_model)

        return l_val, ratio, min_val, best_state, best_state_ema, counter

    def train(
        self, rngs: nnx.Rngs, nsteps: Optional[int] = None, save_model=True
    ) -> Tuple[list, list]:
        """
        Run the training loop for the model.

        Parameters
        ----------
        rngs : nnx.Rngs
            Random number generators for training/validation steps.

        Returns
        -------
        loss_array : list
            List of training losses.
        val_loss_array : list
            List of validation losses.
        """

        optimizer = self._get_optimizer()
        ema_optimizer = self._get_ema_optimizer()

        best_state = nnx.state(self.model)
        best_state_ema = nnx.state(self.ema_model)

        loss_fn = self.get_loss_fn()

        train_step = self.get_train_step_fn(loss_fn)
        val_step = self.get_val_step_fn(loss_fn)

        rng_val = rngs.val_step()
        batch_val = next(self.val_dataset_iter)
        min_val = val_step(self.model, batch_val, rng_val)

        val_error_ratio = self.training_config.get("val_error_ratio", 1.3)
        counter = 0
        cmax = 10

        loss_array = []
        val_loss_array = []

        self.model.train()
        self.ema_model.train()

        if nsteps is None:
            nsteps = self.training_config["nsteps"]
        early_stopping = self.training_config["early_stopping"]
        val_every = self.training_config["val_every"]

        experiment_id = self.training_config["experiment_id"]

        pbar = tqdm(range(nsteps))
        l_train = None
        ratio = 0  # initialize ratio
        l_val = min_val  # initialize l_val

        for j in pbar:
            if counter > cmax and early_stopping:
                print("Early stopping")
                self._restore_best_state(best_state, best_state_ema)
                break

            batch = next(self.train_dataset_iter)

            loss = train_step(self.model, optimizer, batch, rngs.train_step())
            # update the parameters ema
            if j % self.training_config["multistep"] == 0:
                ema_step(self.ema_model, self.model, ema_optimizer)

            decay = 0.99

            if j == 0:
                l_train = loss
            else:
                l_train = decay * l_train + (1 - decay) * loss

            if j > 0 and j % val_every == 0:
                l_val, ratio, min_val, best_state, best_state_ema, counter = (
                    self._run_validation(
                        val_step, batch_val, rng_val, min_val,
                        best_state, best_state_ema, counter, val_error_ratio,
                        loss_array, val_loss_array, l_train,
                    )
                )

            # print stats
            if j > 0 and j % 10 == 0:
                pbar.set_postfix(
                    loss=f"{l_train:.4f}",
                    ratio=get_colored_value(ratio, thresholds=(1.1, 1.3)),
                    counter=counter,
                    val_loss=f"{l_val:.4f}",
                )

        self.model.eval()
        self.ema_model.eval()

        if save_model:
            self.save_model(experiment_id)

        self._wrap_model()

        return loss_array, val_loss_array

    @abc.abstractmethod
    def get_sampler(
        self,
        key,
        x_o,
        step_size=0.01,
        use_ema=True,
        time_grid=None,
        **model_extras,
    ):
        """
        Get a sampler function for generating samples from the trained model.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        x_o : array-like
            Conditioning variable.
        step_size : float, optional
            Step size for the sampler.
        use_ema : bool, optional
            Whether to use the EMA model for sampling.
        time_grid : array-like, optional
            Time grid for the sampler (if applicable).
        model_extras : dict, optional
            Additional model-specific parameters.

        Returns
        -------
        sampler : Callable: key, nsamples -> samples
            A function that generates samples when called with a random key and number of samples.
        """
        ...  # pragma: no cover

    @abc.abstractmethod
    def sample(self, key, x_o, nsamples=10_000):
        """
        Generate samples from the trained model.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        x_o : array-like
            Conditioning variable (e.g., observed data).
        nsamples : int, optional
            Number of samples to generate.

        Returns
        -------
        samples : array-like
            Generated samples of size (nsamples, dim_obs, ch_obs).
        """
        ...  # pragma: no cover

    @abc.abstractmethod
    def get_log_prob_fn(self, *args, **kwargs):
        """Get a log-probability function for evaluating data under the model.

        Returns
        -------
        log_prob_fn : Callable
            ``(x_1) -> log_prob``
        """
        ...  # pragma: no cover

    @abc.abstractmethod
    def log_prob(self, x_1, *args, **kwargs):
        """Compute log-probability of data under the trained model.

        Returns
        -------
        Array
            Log-probabilities.
        """
        ...  # pragma: no cover

    def sample_batched(
        self,
        key,
        x_o: Array,
        nsamples: int,
        *args,
        chunk_size: Optional[int] = 50,
        show_progress_bars=True,
        **kwargs,
    ):
        """
        Generate samples from the trained model in batches.

        Parameters
        ----------
        key : jax.random.PRNGKey
            Random number generator key.
        x_o : array-like
            Conditioning variable (e.g., observed data).
        nsamples : int
            Number of samples to generate.
        chunk_size : int, optional
            Size of each batch for sampling. Default is 50.
        show_progress_bars : bool, optional
            Whether to display progress bars during sampling. Default is True.
        args : tuple
            Additional positional arguments for the sampler.
        kwargs : dict
            Additional keyword arguments for the sampler.

        Returns
        -------
        samples : array-like
            Generated samples of shape (nsamples, batch_size_cond, dim_obs, ch_obs).
        """

        # Build the sampler once using the first condition for shape.
        # The sampler's JIT compilation traces model_extras by shape/dtype,
        # so calling it with different cond values (same shape) reuses the
        # compiled function — no recompilation per condition.
        sampler = self.get_sampler(x_o[0:1], *args, **kwargs)

        # Retrieve the default extras that get_sampler baked in, then swap
        # cond for each condition in the loop below.
        B = x_o.shape[0]
        keys_per_cond = jax.random.split(key, B)

        results = []
        for i in range(B):
            cond_i = _expand_dims(x_o[i : i + 1])
            extras_i = {
                "cond": cond_i,
                "obs_ids": self.obs_ids,
                "cond_ids": self.cond_ids,
            }
            samples_i = sampler(keys_per_cond[i], nsamples, model_extras=extras_i)
            results.append(samples_i)

        return jnp.stack(results, axis=1)  # (nsamples, B, dim_obs, ch_obs)
