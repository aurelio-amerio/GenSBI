"""
Pipeline module for GenSBI.

This module provides an abstract pipeline class for training and evaluating conditional generative models
(such as conditional flow matching or diffusion models) in the GenSBI framework. It handles model creation,
training loop, optimizer setup, checkpointing, and evaluation utilities.

Example:
    .. code-block:: python

    from gensbi.recipes.pipeline import AbstractPipeline
    # Implement a subclass with your model and loss definition
    class MyPipeline(AbstractPipeline):
        def _make_model(self):
            ...
        def _get_default_params(self, rngs):
            ...
        def get_loss_fn(self):
            ...
        def sample(self, rng, x_o, nsamples=10000, step_size=0.01):
            ...

    # Instantiate and train
    pipeline = MyPipeline(train_dataset, val_dataset, dim_theta=2, dim_x=2)
    pipeline.train(rngs)
"""

from flax import nnx
import jax
from jax import numpy as jnp
from typing import Any, Callable, Optional, Tuple
from jax import Array

from numpyro import distributions as dist

import abc
from functools import partial

import optax
from optax.contrib import reduce_on_plateau

import orbax.checkpoint as ocp

from tqdm import tqdm

import os


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
        self.step.value += 1

        ema_state = optax.EmaState(count=self.step, ema=ema_params)

        _, new_ema_state = self.tx.update(params, ema_state)

        nnx.update(model, new_ema_state.ema)


@nnx.jit
def ema_step(ema_model, model, ema_optimizer: nnx.Optimizer):
    ema_optimizer.update(ema_model, model)


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
    dim_theta : int
        Dimensionality of the parameter (theta) space.
    dim_x : int
        Dimensionality of the observation (x) space.
    params : dict, optional
        Model parameters. If None, uses defaults from `_get_default_params`.
    training_config : dict, optional
        Training configuration. If None, uses defaults from `_get_default_training_config`.

    """

    def __init__(
        self,
        train_dataset,
        val_dataset,
        dim_theta: int,
        dim_x: int,
        params=None,
        training_config=None,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_dataset_iter = iter(self.train_dataset)
        self.val_dataset_iter = iter(self.val_dataset)

        self.dim_theta = dim_theta
        self.dim_x = dim_x
        self.dim_joint = dim_theta + dim_x

        self.node_ids = jnp.arange(self.dim_joint)
        self.obs_ids = jnp.arange(self.dim_theta)  # observation ids
        self.cond_ids = jnp.arange(self.dim_theta, self.dim_joint)  # conditional ids

        self.params = params
        if params is None:
            self.params = self._get_default_params()

        self.training_config = training_config
        if training_config is None:
            self.training_config = self._get_default_training_config()

        self.training_config["min_scale"] = (
            self.training_config["min_lr"] / self.training_config["max_lr"]
            if self.training_config["max_lr"] > 0
            else 0.0
        )

        os.makedirs(self.training_config["checkpoint_dir"], exist_ok=True)

        self.model = self._make_model()
        self.model_wrapped = None  # to be set in subclass

        self.ema_model = nnx.clone(self.model)
        self.ema_model_wrapped = None  # to be set in subclass

        self.p0_dist_model = None  # to be set in subclass
        self.loss_fn = None  # to be set in subclass
        self.path = None  # to be set in subclass

    @abc.abstractmethod
    def init_pipeline_from_config(
        cls,
        train_dataset,
        val_dataset,
        dim_theta: int,
        dim_x: int,
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
        dim_theta : int
            Dimensionality of the parameter (theta) space.
        dim_x : int
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
    def _make_model(self):
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
        opt = optax.chain(
            optax.adaptive_grad_clip(10.0),
            optax.adamw(self.training_config["max_lr"]),
            reduce_on_plateau(
                patience=self.training_config["patience"],
                cooldown=self.training_config["cooldown"],
                factor=self.training_config["factor"],
                rtol=self.training_config["rtol"],
                accumulation_size=self.training_config["accumulation_size"],
                min_scale=self.training_config["min_scale"],
            ),
        )
        if self.training_config["multistep"] > 1:
            opt = optax.MultiSteps(opt, self.training_config["multistep"])

        optimizer = nnx.Optimizer(self.model, opt, wrt=nnx.Param)
        return optimizer

    @abc.abstractmethod
    def _get_default_params(self, rngs: nnx.Rngs):
        """
        Return a dictionary of default model parameters.
        """
        ...  # pragma: no cover

    @classmethod
    def _get_default_training_config(cls):
        """
        Return a dictionary of default training configuration parameters.

        Returns
        -------
        training_config : dict
            Default training configuration.
        """
        training_config = {}

        training_config["num_steps"] = 30_000

        training_config["ema_decay"] = 0.99

        training_config["patience"] = 10
        training_config["cooldown"] = 2
        training_config["factor"] = 0.5
        training_config["accumulation_size"] = 100
        training_config["rtol"] = 1e-4
        training_config["max_lr"] = 1e-3
        training_config["min_lr"] = 1e-8
        training_config["val_every"] = 100
        training_config["early_stopping"] = True
        training_config["experiment_id"] = 1
        training_config["multistep"] = 1
        training_config["checkpoint_dir"] = os.path.join(os.getcwd(), "checkpoints")

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

    def update_params(self, new_params):
        """
        Update the model parameters and re-initialize the model.

        Parameters
        ----------
        new_params : dict
            New model parameters.
        """
        self.params = new_params
        self.model = self._make_model()
        self.model_wrapped = None  # to be set in subclass
        return


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
        def train_step(model, optimizer, batch, rng: jax.random.PRNGKey):
            loss, grads = nnx.value_and_grad(loss_fn)(model, batch, rng)
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
        def val_step(model, batch, rng: jax.random.PRNGKey):
            loss = loss_fn(model, batch, rng)
            return loss

        return val_step

    def save_model(self, experiment_id=None):
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
            experiment_id, args=ocp.args.Composite(state=ocp.args.StandardSave(ema_state)))
        
        checkpoint_manager_ema.close()

        print("Saved model to checkpoint")
        return
    
    def restore_model(self, experiment_id=None):
        if experiment_id is None:
            experiment_id = self.training_config["experiment_id"]

        graphdef, model_state = nnx.split(self.model)

        with ocp.CheckpointManager(
            self.training_config["checkpoint_dir"],
            options=ocp.CheckpointManagerOptions(read_only=True),
        ) as read_mgr:
            restored = read_mgr.restore(
                experiment_id,
                args=ocp.args.Composite(state=ocp.args.StandardRestore(item=model_state)),
            )
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
        self.ema_model = nnx.merge(graphdef, restored_ema["state"])

        # wrap models
        self._wrap_model()

        print("Restored model from checkpoint")
        return

    @abc.abstractmethod
    def _wrap_model(self):
        """
        Wrap the model for evaluation (either using SimformerWrapper or Flux1Wrapper).
        """
        ...  # pragma: no cover


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

        batch_val = next(self.val_dataset_iter)
        min_val = val_step(self.model, batch_val, rngs.val_step())

        val_error_ratio = 1.1
        counter = 0
        cmax = 10

        loss_array = []
        val_loss_array = []

        self.model.train()

        if nsteps is None:
            nsteps = self.training_config["num_steps"]
        early_stopping = self.training_config["early_stopping"]
        val_every = self.training_config["val_every"]

        experiment_id = self.training_config["experiment_id"]

        pbar = tqdm(range(nsteps))
        l_train = None

        for j in pbar:
            if counter > cmax and early_stopping:
                print("Early stopping")
                graphdef = nnx.graphdef(self.model)
                self.model = nnx.merge(graphdef, best_state)
                self.ema_model = nnx.merge(graphdef, best_state_ema)

                break

            batch = next(self.train_dataset_iter)

            loss = train_step(
                self.model, optimizer, batch, rngs.train_step()
            )
            # update the parameters ema
            if j % self.training_config["multistep"] == 0:
                ema_step(self.ema_model, self.model, ema_optimizer)

            if j == 0:
                l_train = loss
            else:
                l_train = 0.9 * l_train + 0.1 * loss

            if j > 0 and j % val_every == 0:
                batch_val = next(self.val_dataset_iter)
                l_val = val_step(self.model, batch_val, rngs.val_step())

                ratio = l_val / l_train
                if ratio > val_error_ratio:
                    counter += 1
                else:
                    counter = 0

                pbar.set_postfix(
                    loss=f"{l_train:.4f}",
                    ratio=f"{ratio:.4f}",
                    counter=counter,
                    val_loss=f"{l_val:.4f}",
                )
                loss_array.append(l_train)
                val_loss_array.append(l_val)

                if l_val < min_val:
                    min_val = l_val
                    best_state = nnx.state(self.model)
                    best_state_ema = nnx.state(self.ema_model)

                l_val = 0
                l_train = 0

        self.model.eval()

        if save_model:
            self.save_model(experiment_id)

        self._wrap_model()

        return loss_array, val_loss_array

    @abc.abstractmethod
    def sample(self, rng, x_o, nsamples=10_000, step_size=0.01):
        """
        Generate samples from the trained model.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            Random number generator key.
        x_o : array-like
            Conditioning variable (e.g., observed data).
        nsamples : int, optional
            Number of samples to generate.
        step_size : float, optional
            Step size for the sampler.

        Returns
        -------
        samples : array-like
            Generated samples.
        """
        ...  # pragma: no cover
