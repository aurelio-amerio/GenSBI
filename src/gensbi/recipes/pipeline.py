
"""
Pipeline module for GenSBI.

This module provides an abstract pipeline class for training and evaluating conditional generative models
(such as conditional flow matching or diffusion models) in the GenSBI framework. It handles model creation,
training loop, optimizer setup, checkpointing, and evaluation utilities.

Example usage::

    from gensbi.cookies.pipeline import AbstractPipeline
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
        def evaluate_c2st(self, rng, x_o, reference_samples):
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
    def __init__(self, train_dataset, val_dataset, dim_theta: int, dim_x: int, params=None, training_config=None):
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

        self.training_config["min_scale"] = self.training_config["min_lr"] / self.training_config["max_lr"] if self.training_config["max_lr"] > 0 else 0.0

        os.makedirs(self.training_config["checkpoint_dir"], exist_ok=True)

        self.vf_model = self._make_model()
        self.vf_model_wrapped = None # to be set in subclass

        self.p0_dist_model = None # to be set in subclass
        self.loss_fn_cfm = None  # to be set in subclass
        self.path = None  # to be set in subclass

        # self.

    @abc.abstractmethod
    def _make_model(self):
        """
        Create and return the model to be trained.
        """
        return 

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

        optimizer = nnx.Optimizer(self.vf_model, opt, wrt=nnx.Param)
        return optimizer
    
    @abc.abstractmethod
    def _get_default_params(self, rngs: nnx.Rngs):
        """
        Return a dictionary of default model parameters.
        """
        return 
    
    def _get_default_training_config(self):
        """
        Return a dictionary of default training configuration parameters.

        Returns
        -------
        training_config : dict
            Default training configuration.
        """
        training_config = {}

        training_config["num_steps"] = 30_000

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
    
    def update_params(self, new_params):
        """
        Update the model parameters and re-initialize the model.

        Parameters
        ----------
        new_params : dict
            New model parameters.
        """
        self.params = new_params
        self.vf_model = self._make_model()
        self.vf_model_wrapped = None # to be set in subclass
        return

    def _next_batch(self):
        """
        Return the next batch from the training dataset.
        """
        return next(self.train_dataset_iter)

    def _next_val_batch(self):
        """
        Return the next batch from the validation dataset.
        """
        return next(self.val_dataset_iter)

    def get_train_loss_fn(self, loss_fn_):
        """
        Wrap the training loss function for use in the training loop.

        Parameters
        ----------
        loss_fn_ : Callable
            The loss function to use.

        Returns
        -------
        train_loss : Callable
            JIT-compiled training loss function.
        """
        @nnx.jit
        def train_loss(vf_model, key: jax.random.PRNGKey):
            x_1 = self._next_batch()
            return loss_fn_(vf_model, x_1, key)

        return train_loss
    
    @abc.abstractmethod
    def get_loss_fn(self):
        """
        Return the loss function for training/validation.
        """
        return

    def get_val_loss_fn(self, loss_fn_):
        """
        Wrap the validation loss function for use in the training loop.

        Parameters
        ----------
        loss_fn_ : Callable
            The loss function to use.

        Returns
        -------
        val_loss : Callable
            JIT-compiled validation loss function.
        """
        @nnx.jit
        def val_loss(vf_model, key: jax.random.PRNGKey):
            x_1 = self._next_val_batch()
            return loss_fn_(vf_model, x_1, key)

        return val_loss
    
    def get_train_step_fn(self):
        """
        Return the training step function, which performs a single optimization step.

        Returns
        -------
        train_step : Callable
            JIT-compiled training step function.
        """
        train_loss = self.get_train_loss_fn(self.get_loss_fn())

        @nnx.jit
        def train_step(model, optimizer, rng):
            loss_fn = lambda model: train_loss(model, rng)
            loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads, value=loss)
            return loss

        return train_step

    @abc.abstractmethod
    def restore_model(self, experiment_id=None):
        """
        Restore model parameters from a checkpoint.

        Parameters
        ----------
        experiment_id : int, optional
            Experiment ID to restore. If None, uses the one in training_config.
        """
        return
    
    def train(self, rngs: nnx.Rngs):
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

        best_state = nnx.state(self.vf_model)
        # best_state_ema = nnx.state(self.ema_model)

        val_loss = self.get_val_loss_fn(self.get_loss_fn())
        train_step = self.get_train_step_fn()

        min_val = val_loss(self.vf_model, rngs.val_step())
        val_error_ratio = 1.1
        counter = 0
        cmax = 10

        loss_array = []
        val_loss_array = []

        self.vf_model.train()

        nsteps = self.training_config["num_steps"]
        early_stopping = self.training_config["early_stopping"]
        val_every = self.training_config["val_every"]

        checkpoint_dir = self.training_config["checkpoint_dir"]
        experiment_id = self.training_config["experiment_id"]

        pbar = tqdm(range(nsteps)) 
        l_train = None

        for j in pbar:
            if counter > cmax and early_stopping:
                print("Early stopping")
                graphdef, abstract_state = nnx.split(self.vf_model)
                self.vf_model = nnx.merge(graphdef, best_state)
                # ema_params = best_state_ema

                break
            loss = train_step(self.vf_model, optimizer, rngs.train_step())
            # update the parameters ema
            # ema_step(ema_model, vf_model, ema_optimizer)  # Update the EMA model.

            if j == 0:
                l_train = loss
            else:
                l_train = 0.9 * l_train + 0.1 * loss

            if j > 50 and j % val_every == 0:
                l_val = val_loss(self.vf_model, rngs.val_step())
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
                    best_state = nnx.state(self.vf_model)
                    # best_state_ema = nnx.state(ema_model)

                l_val = 0
                l_train = 0

        self.vf_model.eval()
        # Save the model
        checkpoint_manager = ocp.CheckpointManager(
            checkpoint_dir,
            options=ocp.CheckpointManagerOptions(
                max_to_keep=None,
                keep_checkpoints_without_metrics=True,
                create=True,
            ),
        )
        model_state = nnx.state(self.vf_model)
        checkpoint_manager.save(
            experiment_id, args=ocp.args.Composite(state=ocp.args.PyTreeSave(model_state))
        )
        checkpoint_manager.close()

        print("Training complete and model saved.")
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
        return

    @abc.abstractmethod
    def evaluate_c2st(self, rng, x_o, reference_samples):
        """
        Evaluate the classifier two-sample test (C2ST) between model and reference samples.

        Parameters
        ----------
        rng : jax.random.PRNGKey
            Random number generator key.
        x_o : array-like
            Conditioning variable.
        reference_samples : array-like
            Reference samples to compare against.

        Returns
        -------
        c2st_score : float
            C2ST score.
        """
        return
    
