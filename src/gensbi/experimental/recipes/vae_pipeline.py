"""
VAE Pipeline module for GenSBI.

This module provides an abstract pipeline class and concrete implementations for training and evaluating Variational Autoencoders (VAEs)
within the GenSBI framework. It manages model instantiation, training and validation loops, optimizer and EMA setup, checkpointing,
and utility functions for saving and restoring models.

The `AbstractVAEPipeline` class defines the general workflow for VAE-based models, including optimizer configuration, KL annealing,
and early stopping. Subclasses such as `VAE1DPipeline` and `VAE2DPipeline` implement pipelines for 1D and 2D autoencoder architectures, respectively.

Typical usage involves subclassing or instantiating the provided pipelines with appropriate datasets, model parameters, and (optionally) training configurations.

Key features:
- Model and EMA initialization
- Training and validation step functions (JIT-compiled)
- Learning rate scheduling and gradient clipping
- Early stopping and checkpoint management
- Support for custom training configurations

See the `VAE1DPipeline` and `VAE2DPipeline` classes for concrete examples.
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

from optax.schedules import linear_schedule, constant_schedule

import yaml

import orbax.checkpoint as ocp

from tqdm import tqdm

import os

from gensbi.experimental.models.autoencoders import (
    AutoEncoderParams,
    vae_loss_fn,
    AutoEncoder1D,
    AutoEncoder2D,
)
from gensbi.recipes.pipeline import ema_step, ModelEMA


def parse_training_config(config_path: str):
    """
    Parse a training configuration file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.

    Returns
    -------
    config : dict
        Parsed configuration dictionary.

    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Training parameters
    train_params = config.get("training", {})
    multistep = train_params.get("multistep", 1)
    experiment_id = train_params.get("experiment_id", 1)
    early_stopping = train_params.get("early_stopping", True)
    nsteps = train_params.get("nsteps", 30000) * multistep
    val_every = train_params.get("val_every", 100) * multistep

    # Optimizer parameters
    opt_params = config.get("optimizer", {})

    MAX_LR = opt_params.get("max_lr", 1e-3)
    MIN_LR = opt_params.get("min_lr", 0.0)
    MIN_SCALE = MIN_LR / MAX_LR if MAX_LR > 0 else 0.0

    warmup_steps = opt_params.get("warmup_steps", 500)

    ema_decay = opt_params.get("ema_decay", 0.999)

    decay_transition = opt_params.get("decay_transition", 0.85)

    training_config = {}
    # overwrite the defaults with the config file values
    training_config["nsteps"] = nsteps
    training_config["ema_decay"] = ema_decay
    training_config["decay_transition"] = decay_transition

    training_config["max_lr"] = MAX_LR
    training_config["min_lr"] = MIN_LR
    training_config["min_scale"] = MIN_SCALE
    training_config["val_every"] = val_every
    training_config["early_stopping"] = early_stopping
    training_config["experiment_id"] = experiment_id
    training_config["multistep"] = multistep
    training_config["warmup_steps"] = warmup_steps

    return training_config


# AutoEncoderParams:

#     resolution: int
#     in_channels: int
#     ch: int
#     out_ch: int
#     ch_mult: list[int]
#     num_res_blocks: int
#     z_channels: int
#     scale_factor: float
#     shift_factor: float
#     rngs: nnx.Rngs
#     param_dtype: DTypeLike


def parse_autoencoder_params(config_path: str):
    """
    Parse a VAE configuration file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.

    Returns
    -------
    config : dict
        Parsed configuration dictionary.

    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_params = config.get("vae_model", {})

    params_dict = dict(
        resolution=model_params.get("resolution"),
        in_channels=model_params.get("in_channels"),
        ch=model_params.get("ch"),
        out_ch=model_params.get("out_ch"),
        ch_mult=model_params.get("ch_mult"),
        num_res_blocks=model_params.get("num_res_blocks"),
        z_channels=model_params.get("z_channels"),
        scale_factor=model_params.get("scale_factor", 1.0),
        shift_factor=model_params.get("shift_factor", 0.0),
        param_dtype=getattr(jnp, model_params.get("param_dtype", "float32")),
    )

    return params_dict


class AbstractVAEPipeline:
    """
    Abstract pipeline for training and evaluating Variational Autoencoders (VAEs) in GenSBI.

    This class manages model creation, optimizer and EMA setup, training and validation loops, checkpointing,
    and utility functions. It is designed to be subclassed for specific VAE architectures.
    """

    def __init__(
        self,
        model_cls,
        train_dataset,
        val_dataset,
        params: AutoEncoderParams,
        training_config=None,
    ):
        """
        Initialize the VAE pipeline.

        Parameters
        ----------
        model_cls : type
            The class of the VAE model to instantiate (e.g., AutoEncoder1D or AutoEncoder2D).
        train_dataset : iterable
            Training dataset.
        val_dataset : iterable
            Validation dataset.
        params : AutoEncoderParams
            Model hyperparameters and configuration.
        training_config : dict, optional
            Training configuration dictionary. If None, defaults are used.
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_dataset_iter = iter(self.train_dataset)
        self.val_dataset_iter = iter(self.val_dataset)

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

        self.model = model_cls(self.params)

        self.ema_model = nnx.clone(self.model)

        self.loss_fn = vae_loss_fn

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
        config_path : str
            Path to the configuration file.

        """

        params_dict = parse_autoencoder_params(config_path)

        params = AutoEncoderParams(
            rngs=nnx.Rngs(0),
            **params_dict,
        )

        # Training parameters
        training_config = cls.get_default_training_config()
        training_config["checkpoint_dir"] = checkpoint_dir

        training_config_ = parse_training_config(config_path)

        for key, value in training_config_.items():
            training_config[key] = value  # update with config file values

        pipeline = cls(
            train_dataset,
            val_dataset,
            params=params,
            training_config=training_config,
        )

        return pipeline

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

    def _get_kl_schedule(self, nsteps):
        """
        Construct a KL annealing schedule for training.

        Parameters
        ----------
        nsteps : int
            Number of training steps.

        Returns
        -------
        schedule : Callable
            KL weight schedule function.
        """
        # schedule = linear_schedule(0.1, 1, nsteps)
        schedule = constant_schedule(0.1)
        return schedule

    # def _get_optimizer(self):
    #     """
    #     Construct the optimizer for training, including learning rate scheduling and gradient clipping.

    #     Returns
    #     -------
    #     optimizer : nnx.Optimizer
    #         The optimizer instance for the model.
    #     """
    #     opt = optax.chain(
    #         optax.adaptive_grad_clip(10.0),
    #         optax.adamw(self.training_config["max_lr"]),
    #         reduce_on_plateau(
    #             patience=self.training_config["patience"],
    #             cooldown=self.training_config["cooldown"],
    #             factor=self.training_config["factor"],
    #             rtol=self.training_config["rtol"],
    #             accumulation_size=self.training_config["accumulation_size"],
    #             min_scale=self.training_config["min_scale"],
    #         ),
    #     )
    #     if self.training_config["multistep"] > 1:
    #         opt = optax.MultiSteps(opt, self.training_config["multistep"])

    #     optimizer = nnx.Optimizer(self.model, opt, wrt=nnx.Param)
    #     return optimizer

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

    @classmethod
    def get_default_training_config(cls):
        """
        Return a dictionary of default training configuration parameters for VAE training.

        Returns
        -------
        training_config : dict
            Default training configuration.
        """
        training_config = {}

        training_config["nsteps"] = 50_000

        training_config["ema_decay"] = 0.999
        training_config["warmup_steps"] = 500
        training_config["decay_transition"] = 0.70

        training_config["max_lr"] = 1e-4
        training_config["min_lr"] = 1e-6
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

    def get_train_step_fn(self):
        """
        Return the training step function, which performs a single optimization step.

        Returns
        -------
        train_step : Callable
            JIT-compiled training step function.
        """

        @nnx.jit
        def train_step(model, optimizer, batch, key, kl_weight):
            loss, grads = nnx.value_and_grad(self.loss_fn)(model, batch, key, kl_weight)
            optimizer.update(model, grads, value=loss)

            return loss

        return train_step

    def get_val_step_fn(self):
        """
        Return the validation step function, which computes the loss on a validation batch.

        Returns
        -------
        val_step : Callable
            JIT-compiled validation step function.
        """

        @nnx.jit
        def val_step(model, batch, key, kl_weight):
            loss = self.loss_fn(model, batch, key, kl_weight)
            return loss

        return val_step

    def save_model(self, experiment_id=None):
        """
        Save the current model and EMA model to checkpoint directories.

        Parameters
        ----------
        experiment_id : int, optional
            Identifier for the experiment/checkpoint. If None, uses the current training config value.
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

        # Save the EMA model
        _, ema_state = nnx.split(self.ema_model)
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
        Restore the model and EMA model from checkpoint directories.

        Parameters
        ----------
        experiment_id : int, optional
            Identifier for the experiment/checkpoint. If None, uses the current training config value.
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
        self.model = nnx.merge(graphdef, restored["state"])
        self.model.training = False

        # Restore the EMA model
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
        self.ema_model.training = False

        print("Restored model from checkpoint")
        return

    def train(
        self, rngs: nnx.Rngs, nsteps: Optional[int] = None, save_model=True
    ) -> Tuple[list, list]:
        """
        Run the training loop for the VAE model.

        Parameters
        ----------
        rngs : nnx.Rngs
            Random number generators for training/validation steps.
        nsteps : int, optional
            Number of training steps. If None, uses the value from training config.
        save_model : bool, optional
            Whether to save the model after training.

        Returns
        -------
        loss_array : list
            List of training losses.
        val_loss_array : list
            List of validation losses.
        """

        self.model.train(update_KL=True)

        optimizer = self._get_optimizer()
        ema_optimizer = self._get_ema_optimizer()

        best_state = nnx.state(self.model)
        best_state_ema = nnx.state(self.ema_model)

        train_step = self.get_train_step_fn()
        val_step = self.get_val_step_fn()

        batch_val = next(self.val_dataset_iter)
        min_val = val_step(self.model, batch_val, rngs.val_step(), 1.0)

        val_error_ratio = 0.1
        counter = 0
        cmax = 10

        loss_array = []
        val_loss_array = []

        if nsteps is None:
            nsteps = self.training_config["nsteps"]
        early_stopping = self.training_config["early_stopping"]
        val_every = self.training_config["val_every"]

        kl_schedule = self._get_kl_schedule(nsteps)

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

            kl_weight = kl_schedule(j)

            batch = next(self.train_dataset_iter)

            loss = train_step(
                self.model, optimizer, batch, rngs.train_step(), kl_weight
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
                l_val = val_step(self.model, batch_val, rngs.val_step(), kl_weight)

                if (
                    l_val < l_train
                ):  # TODO figure out something more clever to do, since the loss may be negative here
                    ratio = 0.0
                else:
                    ratio = jnp.abs((l_train - l_val) / (l_train + 1e-8))

                if ratio > val_error_ratio:
                    counter += 1
                else:
                    counter = 0

                pbar.set_postfix(
                    loss=f"{l_train:.4f}",
                    diff_ratio=f"{ratio:.4f}",
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

        self.model.eval(update_KL=False)

        if save_model:
            self.save_model(experiment_id)

        return loss_array, val_loss_array


class VAE1DPipeline(AbstractVAEPipeline):
    """
    Pipeline for training and evaluating 1D Variational Autoencoders (VAE1D) in GenSBI.

    Inherits from AbstractVAEPipeline and uses the AutoEncoder1D model class.
    """

    def __init__(
        self,
        train_dataset,
        val_dataset,
        params: AutoEncoderParams,
        training_config=None,
    ):
        """
        Initialize the 1D VAE pipeline.

        Parameters
        ----------
        train_dataset : iterable
            Training dataset.
        val_dataset : iterable
            Validation dataset.
        params : AutoEncoderParams
            Model hyperparameters and configuration.
        training_config : dict, optional
            Training configuration dictionary. If None, defaults are used.
        """
        super().__init__(
            AutoEncoder1D,
            train_dataset,
            val_dataset,
            params,
            training_config,
        )
        return


class VAE2DPipeline(AbstractVAEPipeline):
    """
    Pipeline for training and evaluating 2D Variational Autoencoders (VAE2D) in GenSBI.

    Inherits from AbstractVAEPipeline and uses the AutoEncoder2D model class.
    """

    def __init__(
        self,
        train_dataset,
        val_dataset,
        params: AutoEncoderParams,
        training_config=None,
    ):
        """
        Initialize the 2D VAE pipeline.

        Parameters
        ----------
        train_dataset : iterable
            Training dataset.
        val_dataset : iterable
            Validation dataset.
        params : AutoEncoderParams
            Model hyperparameters and configuration.
        training_config : dict, optional
            Training configuration dictionary. If None, defaults are used.
        """
        super().__init__(
            AutoEncoder2D,
            train_dataset,
            val_dataset,
            params,
            training_config,
        )
