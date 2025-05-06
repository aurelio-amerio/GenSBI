

import time
import torch

from torch import nn, Tensor

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

# visualization
import matplotlib.pyplot as plt

from matplotlib import cm


# To avoide meshgrid warning
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='torch')



     



#%%
import os
os.environ['JAX_PLATFORMS']="cpu"


import sys
sys.path.append("./src/cfm-jax")
import conditional_flow_matching as cfm

from flax import nnx
import jax 
import jax.numpy as jnp
#%%
a = jnp.array([[1, 2], [3, 4]])
# %%
# create a MLP model with flax nnx 
class MLP(nnx.Module):
  def __init__(self, din: int, dmid: int, *, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(din+1, dmid, rngs=rngs)
    self.bn1 = nnx.BatchNorm(dmid, rngs=rngs)
    self.linear2 = nnx.Linear(dmid, dmid, rngs=rngs)
    self.bn2 = nnx.BatchNorm(dmid, rngs=rngs)
    self.linear3 = nnx.Linear(dmid, din, rngs=rngs)

  def __call__(self, x: jax.Array):
    x = self.linear1(x)
    x = self.bn1(x)
    x = jax.nn.gelu(x)

    x = self.linear2(x)
    x = self.bn2(x)
    x = jax.nn.gelu(x)

    x = self.linear3(x)

    return x
# %%
mlp = MLP(2, 64, rngs=jax.random.PRNGKey(0))