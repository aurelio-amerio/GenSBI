


#%%
import os
os.environ['JAX_PLATFORMS']="cpu"


import sys
sys.path.append("./src")

from flax import nnx
import jax 
import jax.numpy as jnp
import optax

import time
#%%
# make dataset
from sklearn.datasets import make_moons


def inf_train_gen(batch_size: int = 200):
    x = make_moons(batch_size, noise=0.1)[0]

    return jnp.array(x)

#%%
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
# from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
#%%
class MLP(nnx.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128, *, rngs: nnx.Rngs):

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.linear1 = nnx.Linear(self.input_dim + self.time_dim, self.hidden_dim, rngs=rngs)
        self.bn1 = nnx.BatchNorm(self.hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(self.hidden_dim, self.hidden_dim, rngs=rngs)
        self.bn2 = nnx.BatchNorm(self.hidden_dim, rngs=rngs)
        self.linear3 = nnx.Linear(self.hidden_dim, self.hidden_dim, rngs=rngs)
        self.bn3 = nnx.BatchNorm(self.hidden_dim, rngs=rngs)
        self.linear4 = nnx.Linear(self.hidden_dim, self.hidden_dim, rngs=rngs)
        self.bn4 = nnx.BatchNorm(self.hidden_dim, rngs=rngs)
        self.linear5 = nnx.Linear(self.hidden_dim, self.input_dim, rngs=rngs)

    def __call__(self, x: jax.Array, t: jax.Array):
        sz = x.shape
        t = jnp.reshape(t, (-1, self.time_dim))
        t = jnp.broadcast_to(t, (sz[0], self.time_dim))
        h = jnp.concatenate([x, t], axis=1)

        x = self.linear1(h)
        x = self.bn1(x)
        x = jax.nn.swish(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = jax.nn.swish(x)

        x = self.linear3(x)
        x = self.bn3(x)
        x = jax.nn.swish(x)

        x = self.linear4(x)
        x = self.bn4(x)
        x = jax.nn.swish(x)

        x = self.linear5(x)

        return x.reshape(*sz)
# %%
# training arguments
lr = 0.001
batch_size = 4096
iterations = 20001
print_every = 100
hidden_dim = 512

# velocity field model init
vf = MLP(input_dim=2, time_dim=1, hidden_dim=hidden_dim, rngs=nnx.Rngs(0))
#%%

# instantiate an affine path object
path = AffineProbPath(scheduler=CondOTScheduler())

# init optimizer
optimizer = nnx.Optimizer(vf, optax.adam(lr))

def loss_fn(vf, batch):
    x_0, x_1, t = batch
    path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
    return jnp.mean(jnp.square(vf(path_sample.x_t, path_sample.t) - path_sample.dx_t))

@nnx.jit
def train_step(model, optimizer, batch):
    # sample data (user's responsibility): in this case, (X_0,X_1) ~ pi(X_0,X_1) = N(X_0|0,I)q(X_1)
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch)
    optimizer.update(grads)  # In-place updates.

    return loss
#%%
x_1 = inf_train_gen(batch_size=batch_size) # sample data
x_0 = jax.random.normal(jax.random.PRNGKey(0), x_1 .shape)
t = jax.random.uniform(jax.random.PRNGKey(0), x_1.shape[0])
#%%
loss_fn(vf, (x_0, x_1, t))
#%%
batch = (x_0, x_1, t)
train_step(vf, optimizer, batch)
#%%
# train
start_time = time.time()
key = jax.random.PRNGKey(0)
for i in range(iterations):
    key, subkey1, subkey2 = jax.random.split(key, 3)
    x_1 = inf_train_gen(batch_size=batch_size) # sample data
    x_0 = jax.random.normal(subkey1, x_1 .shape)
    t = jax.random.uniform(subkey2, x_1.shape[0])

    barch = (x_0, x_1, t)
    loss = train_step(vf, optimizer, batch)  # update model parameters

    # log loss
    if (i+1) % print_every == 0:
        elapsed = time.time() - start_time
        print('| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} '
              .format(i+1, elapsed*1000/print_every, loss.item()))
        start_time = time.time()
# %%
