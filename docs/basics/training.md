# Training Guide

This guide details how the training pipeline works in `GenSBI`, best practices for training flow and diffusion models, and how to customize the training loop.

## Training 101

Training a flow matching model is extremely easy using the default pipeline. 
For detailed parameter initialization arguments, see the [Model Cards](/basics/model_cards).
The basic training pipeline is as follows:

```python
from flax import nnx
from gensbi.recipes import Flux1FlowPipeline
from gensbi.models import Flux1Params

train_dataset = ... # define a training dataset (infinite iterator)
val_dataset = ...   # define a validation dataset (infinite iterator)
dim_obs = ...       # dimension of the parameters (theta)
dim_cond = ...      # dimension of the simulator observations (x)

params = Flux1Params(...) # parameters require specific arguments (see Model Cards)

# Instantiate the pipeline
pipeline = Flux1FlowPipeline(
    train_dataset,
    val_dataset,
    dim_obs,
    dim_cond,
    params=params,
)

# Train the model
# Note: GenSBI uses Flax NNX, so we pass a random key generator
pipeline.train(rngs=nnx.Rngs(0))
```

For a full example, see the [5-minute guide](/getting_started/quick_start) or the [full notebook example](/notebooks/my_first_model).

```{note}
GenSBI uses the tensor convention `(batch, dim, channels)`.

- `dim_obs` (a.k.a. `dim_obs`) is the number of parameters to infer (tokens in $\theta$).
- `dim_cond` (a.k.a. `dim_cond`) is the number of conditioning observables (tokens in $x$).
- `ch_obs` and `ch_cond` are the number of values carried by each token.

Most SBI problems use `ch_obs = 1` (one scalar per parameter), while `ch_cond` is often > 1 (e.g., multiple detectors or multiple features per measurement).
See [Troubleshooting: Shape Mismatch Errors](/basics/troubleshooting#shape-mismatch-errors) for a concrete example.
```

## Pipeline Overview

The default training pipeline (e.g., `Flux1FlowPipeline`, `SimformerFlowPipeline`) is built on the `AbstractPipeline` class. It manages the entire training lifecycle. For the default `Flux1` model, use the `Flux1FlowPipeline`. GenSBI also provides `SimformerFlowPipeline` (for low-dim) and `Flux1JointFlowPipeline`. See [Model Cards](/basics/model_cards) for details.

- **State Management**: Uses **Flax NNX** for managing model parameters and optimizer states.
- **Steps vs. Epochs**: Training runs for a fixed number of steps (`nsteps`), not epochs. This is common in generative modeling where datasets (like simulation outputs) might be effectively infinite.
- **EMA (Exponential Moving Average)**: The pipeline maintains a shadow copy of the model weights using EMA. This version is smoother and often yields better generation results. It is saved in a separate `checkpoints/ema` folder.
- **Early Stopping**: Validation runs every `val_every` steps. If the validation loss stops improving or diverges significantly from training loss (controlled by `val_error_ratio`), training stops early.
- **Checkpointing**: Models are automatically saved to the `checkpoints` directory. Both the latest training state and the EMA state are preserved.

## Data Preprocessing

To ensure efficient training, especially when your simulator is computationally expensive or when training on GPUs, it is highly advisable to use an optimized data loader. We recommend using `grain`, a high-performance data loader for JAX, with multiprocessing pre-fetching.

However, using multiprocessing in a Python script alongside JAX requires careful handling to prevent subprocesses from attempting to initialize GPU resources, which can lead to errors or hangs.

Here is a step-by-step example of how to guard your script and set up a performant data loader:

### 1. Guarding GPU Initialization

The very first lines of your script must control the `JAX_PLATFORMS` environment variable. This ensures that only the main process attempts to use the GPU (CUDA), while worker subprocesses (spawned by the data loader) are forced to use the CPU.

```python
import os

if __name__ != "__main__":
    # Worker processes must not touch the GPU
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    # Main process uses the GPU (change to 'cpu' if you don't have one)
    os.environ["JAX_PLATFORMS"] = "cuda"

import grain.python as grain # Import after setting env vars
import numpy as np
import jax
from jax import numpy as jnp
from numpyro import distributions as dist
```

### 2. Defining the Simulator

Next, define your simulator logic. In this example, we create a simple simulator that takes parameters ($\theta$) and generates observations ($x$).

```python
def _simulator(key, thetas):
    # Simulate data: x depends on theta with some noise
    xs = thetas + 1 + jax.random.normal(key, thetas.shape) * 0.1

    # Ensure correct shapes for concatenation
    thetas = thetas[..., None]
    xs = xs[..., None]

    # For Joint Pipelines, concatenating theta and x is common
    # Typically: data = [parameters, observations]
    data = jnp.concatenate([thetas, xs], axis=1)
    return data

# Define a prior distribution for parameters
theta_prior = dist.Uniform(
    low=jnp.array([-2.0, -2.0, -2.0]), high=jnp.array([2.0, 2.0, 2.0])
)

def simulator(key, nsamples):
    theta_key, sample_key = jax.random.split(key, 2)
    # Sample parameters from the prior
    thetas = theta_prior.sample(theta_key, (nsamples,))
    # Run simulation
    return _simulator(sample_key, thetas)
```

   

### 3. Setting up the Data Loader

In your main execution block, generate your dataset and configure the `grain` loader. We use `grain.MapDataset` for simple array-based data, shuffle it, and then use `mp_prefetch` to load batches in parallel.

```python
def main():
    # --- 1. Define Dimensions ---
    dim_obs = 3   # Dimension of simulator input (theta)
    dim_cond = 3  # Dimension of simulator output (x)
    
    # --- 2. Generate Data ---
    # In a real scenario, you might load this from disk
    train_data = jnp.asarray(simulator(jax.random.PRNGKey(0), 10_000), dtype=jnp.bfloat16)
    val_data = jnp.asarray(simulator(jax.random.PRNGKey(1), 2_000), dtype=jnp.bfloat16)

    # Helper to split the joint data back into (obs, cond) for the pipeline
    def split_obs_cond(data):
        # Assuming first 'dim_obs' columns are observations (theta)
        # and remaining columns are conditions (x)
        return data[:, :dim_obs], data[:, dim_obs:]

    batch_size = 1024
    
    # --- 3. Create Training Pipeline with Grain ---
    # Create basic source dataset
    train_ds = grain.MapDataset.source(np.array(train_data))
    train_ds = train_ds.shuffle(seed=42).repeat() # Infinite iterator
    
    # Convert to iter dataset for batching and prefetching
    train_iter_ds = train_ds.to_iter_dataset()

    # Automatically determine optimal multiprocessing settings
    performance_config = grain.experimental.pick_performance_config(
        ds=train_iter_ds,
        ram_budget_mb=1024 * 4, # Adjust based on available RAM
        max_workers=None,       # Defaults to CPU count
        max_buffer_size=None,
    )
    
    # Apply batching, splitting, and multiprocessing
    train_dataset_grain = (
        train_iter_ds
        .batch(batch_size)
        .map(split_obs_cond)
        .mp_prefetch(performance_config.multiprocessing_options)
    )

    # --- 4. Create Validation Pipeline ---
    # Validation usually doesn't need complex prefetching
    val_dataset_grain = (
        grain.MapDataset.source(np.array(val_data))
        .to_iter_dataset()
        .batch(batch_size)
        .map(split_obs_cond)
    )
    
    # ... Initialize and run your GenSBI pipeline here ...
    print("Data loaders ready. Starting training...")
    # ....

# Run main only if this is the entry point
if __name__ == "__main__":
    main()
```

To see a complete runnable script that puts all of this together, check out the [first_model.py example](https://github.com/aurelio-amerio/GenSBI-examples/blob/main/examples/getting_started/first_model.py).

## Configuration & Hyperparameters

You can control most training behaviors via the `training_config` dictionary without subclassing the pipeline.

### Default Configuration Keys

These are the standard hyperparameters available in `AbstractPipeline`:

| Key              | Default         | Description                                                  |
| ---------------- | --------------- | ------------------------------------------------------------ |
| `nsteps`      | `30,000`        | Total number of training steps.                              |
| `max_lr`         | `1e-3`          | Maximum learning rate.                                       |
| `patience`       | `10`            | Steps to wait for improvement before reducing LR (via `reduce_on_plateau`). |
| `multistep`      | `1`             | **Gradient Accumulation**. Accumulates gradients over $N$ steps before updating weights. |
| `ema_decay`      | `0.999`         | Decay rate for the Exponential Moving Average.               |
| `val_every`      | `100`           | How frequently (in steps) to run validation.                 |
| `early_stopping` | `True`          | Whether to enable early stopping.                            |
| `checkpoint_dir` | `./checkpoints` | Directory to save model states.                              |

### Modifying the Configuration

To change parameters from the default training configuration, you can pass a customized dictionary to the pipeline constructor. A common use case is changing the checkpoint directory or the number of training steps.

```python
from gensbi.recipes.flux1 import Flux1FlowPipeline

# 1. Retrieve the default configuration
training_config = Flux1FlowPipeline.get_default_training_config()

# 2. Modify specific settings
training_config["checkpoint_dir"] = "/path/to/custom/checkpoints"
training_config["nsteps"] = 50_000 # Train for longer

# 3. Instantiate the pipeline with the custom config
pipeline = Flux1FlowPipeline(
    ..., # Pass standard arguments (model, datasets, etc.) here
    training_config=training_config, # Pass the modified config here
)
```

## Best Practices

### 1. Prefer Flow Matching over Diffusion

**Flow Matching models are the recommended default in GenSBI.** While the library supports standard Diffusion, Flow Matching models are generally easier to train (straighter paths in latent space) and faster to sample from. Use `Flux1FlowPipeline` or similar classes unless you have a specific research need for Diffusion.

### 2. Use Large Effective Batch Sizes

Generative models benefit significantly from seeing many different time steps ($t \in [0, 1]$) in a single update.

- **Target:** 1,000 to 10,000 samples per batch (start at 1024).
- **Why:** Small batches lead to high variance gradients because they cover only a tiny slice of the time interval.

### 3. Gradient Accumulation (`multistep`)

If your GPU cannot fit a batch size of 1024+, use the `multistep` configuration to achieve a large **effective batch size**.

For example, if your GPU fits a batch of 128:

- Set physical `batch_size=128` in your DataLoader.
- Set `training_config["multistep"] = 8`.
- **Effective Batch Size** = $128 \times 8 = 1024$.

## Advanced Customization

If the options provided by `training_config` aren’t enough, you can subclass the pipeline to modify internal logic.

### Customizing the Optimizer

Override `_get_optimizer` to change the optimization algorithm or schedule.

```python
import optax
from flax import nnx
from gensbi.recipes.flux1 import Flux1FlowPipeline

class CustomFluxPipeline(Flux1FlowPipeline):
    def _get_optimizer(self):
        """
        Overrides the default optimizer to use a linear schedule with AdamW.
        """
        # Retrieve config values
        lr = self.training_config["max_lr"]
        steps = self.training_config["nsteps"]
        
        # Example: Linear decay schedule
        schedule = optax.linear_schedule(
            init_value=lr,
            end_value=0.0,
            transition_steps=steps
        )
        
        # Use AdamW with the schedule
        opt = optax.adamw(learning_rate=schedule, weight_decay=1e-4)
        
        # Return nnx.Optimizer wrapping the model
        return nnx.Optimizer(self.model, opt, wrt=nnx.Param)
```

### Writing a New Training Loop

If you need to change how batches are processed or add custom logging, override the `train` method.

```python
def train(
    self, rngs: nnx.Rngs, nsteps: Optional[int] = None, save_model=True
) -> Tuple[list, list]:
    """
    Run the training loop for the model.
    """
    # Initialize optimizers
    optimizer = self._get_optimizer()
    ema_optimizer = self._get_ema_optimizer()

    # Save initial state for best model tracking
    best_state = nnx.state(self.model)
    best_state_ema = nnx.state(self.ema_model)

    loss_fn = self.get_loss_fn()
    
    # JIT compile steps
    train_step = self.get_train_step_fn(loss_fn)
    val_step = self.get_val_step_fn(loss_fn)

    # Initial validation check
    batch_val = next(self.val_dataset_iter)
    min_val = val_step(self.model, batch_val, rngs.val_step())

    val_error_ratio = 1.1
    counter = 0
    cmax = 10

    loss_array = []
    val_loss_array = []

    self.model.train()

    if nsteps is None:
        nsteps = self.training_config["nsteps"]
    early_stopping = self.training_config["early_stopping"]
    val_every = self.training_config["val_every"]

    experiment_id = self.training_config["experiment_id"]

    pbar = tqdm(range(nsteps))
    l_train = None

    for j in pbar:
        # Check early stopping conditions
        if counter > cmax and early_stopping:
            print("Early stopping")
            # Restore best state
            graphdef = nnx.graphdef(self.model)
            self.model = nnx.merge(graphdef, best_state)
            self.ema_model = nnx.merge(graphdef, best_state_ema)
            break

        batch = next(self.train_dataset_iter)

        # Optimization step
        loss = train_step(
            self.model, optimizer, batch, rngs.train_step()
        )
        
        # EMA Update
        if j % self.training_config["multistep"] == 0:
            ema_step(self.ema_model, self.model, ema_optimizer)

        # Smoothing loss for display
        if j == 0:
            l_train = loss
        else:
            l_train = 0.9 * l_train + 0.1 * loss

        # Validation Loop
        if j > 0 and j % val_every == 0:
            batch_val = next(self.val_dataset_iter)
            l_val = val_step(self.model, batch_val, rngs.val_step())

            # Divergence check
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

            # Keep track of best model
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
```

## Further Help

If you have any questions not covered in this guide or encounter bugs while training your models, please feel free to open an issue on the [GitHub page](https://github.com/aurelio-amerio/GenSBI/issues).