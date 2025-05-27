


#%%
import os
# os.environ['JAX_PLATFORMS']="cpu"


import sys
sys.path.append("./src")

from flax import nnx
import jax 
import jax.numpy as jnp
import optax

# visualization
import matplotlib.pyplot as plt

from matplotlib import cm

import time
import diffrax
#%%
# make dataset
from sklearn.datasets import make_moons

key = jax.random.PRNGKey(0)
#%%

def inf_train_gen(key, batch_size: int = 200):
    x = make_moons(batch_size, noise=0.1, random_state=int(key[0]))[0]

    return jnp.array(x)
#%%
inf_train_gen(key, batch_size = 200).shape
#%%
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
#%%
class MLP(nnx.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 128, *, rngs: nnx.Rngs):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        din = input_dim + 1

        self.linear1 = nnx.Linear(din, self.hidden_dim, rngs=rngs)
        self.linear2 = nnx.Linear(self.hidden_dim, self.hidden_dim, rngs=rngs)
        self.linear3 = nnx.Linear(self.hidden_dim, self.hidden_dim, rngs=rngs)
        self.linear4 = nnx.Linear(self.hidden_dim, self.hidden_dim, rngs=rngs)
        self.linear5 = nnx.Linear(self.hidden_dim, self.input_dim, rngs=rngs)

    def __call__(self, x: jax.Array, t: jax.Array):
        if len(t.shape)<2:
            t = t[..., None]
 
        h = jnp.concatenate([x, t], axis=-1)

        x = self.linear1(h)
        # x = self.bn1(x)
        x = jax.nn.gelu(x)

        x = self.linear2(x)
        # x = self.bn2(x)
        x = jax.nn.gelu(x)

        x = self.linear3(x)
        # x = self.bn3(x)
        x = jax.nn.gelu(x)

        x = self.linear4(x)
        # x = self.bn4(x)
        x = jax.nn.gelu(x)

        x = self.linear5(x)

        return x
# %%
# training arguments
lr = 0.001
batch_size = 4096
iterations = 20001
print_every = 2000
hidden_dim = 512

# velocity field model init
vf = MLP(input_dim=2, hidden_dim=hidden_dim, rngs=nnx.Rngs(0))
#%%

# instantiate an affine path object
path = AffineProbPath(scheduler=CondOTScheduler())

# init optimizer
optimizer = nnx.Optimizer(vf, optax.adam(lr))

def loss_fn(vf, batch):
    path_sample = path.sample(*batch)
    return jnp.mean(jnp.square(vf(path_sample.x_t, path_sample.t) - path_sample.dx_t))

@nnx.jit
def train_step(vf, optimizer, batch):
    # sample data (user's responsibility): in this case, (X_0,X_1) ~ pi(X_0,X_1) = N(X_0|0,I)q(X_1)
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(vf, batch)
    optimizer.update(grads)  # In-place updates.

    return loss
#%%
# x_1 = jnp.array([0.62, 0.42]).reshape((1,2))
# x_0 = jnp.zeros((1,2))
# t = jnp.array([0.12])
# path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
# path_sample.x_t, path_sample.t, path_sample.dx_t
# #%%
key = jax.random.PRNGKey(0)
#%%
key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
x_1 = inf_train_gen(subkey1, batch_size=batch_size) # sample data
x_0 = jax.random.normal(subkey2, x_1 .shape)
t = jax.random.uniform(subkey3, x_1.shape[0])


batch_ = (x_0, x_1, t)
loss=train_step(vf, optimizer, batch_)
loss
#%%
for i in range(5):
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    x_1 = inf_train_gen(subkey1, batch_size=batch_size) # sample data
    x_0 = jax.random.normal(subkey2, x_1 .shape)
    t = jax.random.uniform(subkey3, x_1.shape[0])


    batch_ = (x_0, x_1, t)
    loss=train_step(vf, optimizer, batch_)
print(loss)    

#%%
a = jnp.zeros((3,2))
b = jnp.array((1,1))

a,b=jnp.broadcast_arrays(a,b)
a.shape, b.shape
#%%
# train
start_time = time.time()
key = jax.random.PRNGKey(0)
for i in range(iterations):
    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    x_1 = inf_train_gen(subkey1, batch_size=batch_size) # sample data
    x_0 = jax.random.normal(subkey2, x_1.shape)
    t = jax.random.uniform(subkey3, x_1.shape[0])

    batch = (x_0, x_1, t)
    loss = train_step(vf, optimizer, batch)  # update model parameters

    # log loss
    if (i+1) % print_every == 0:
        elapsed = time.time() - start_time
        print('| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} '
              .format(i+1, elapsed*1000/print_every, loss.item()))
        start_time = time.time()
# %%
vf.eval()
#%%
def vector_field(t, x, args):
    return vf(x=x, t=t)
#%%
batch_size
#%%
term = diffrax.ODETerm(vector_field)

solver = diffrax.Dopri5()


stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
#%%
T = jnp.linspace(0,1,10)  # sample times
x_init = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 2))
solution = diffrax.diffeqsolve(
    term,
    solver,
    t0=T[0],
    t1=T[1],
    dt0=0.05,
    y0=x_init,
    # saveat=diffrax.SaveAt(ts=T),
    stepsize_controller=stepsize_controller,
)
#%%
plt.scatter(solution.ys[0,:,0], solution.ys[0,:,1], s=1, c='r', alpha=0.5)
plt.show()
#%%
vector_field(jnp.array(1.0), x_init, None).shape

#%%
# step size for ode solver
step_size = jnp.array(0.05)

norm = cm.colors.Normalize(vmax=50, vmin=0)

batch_size = 50000  # batch size
# eps_time = 1e-2
T = jnp.linspace(0,1,10)  # sample times

# x_init = torch.randn((batch_size, 2), dtype=torch.float32, device=device)
x_init = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 2))  # initial conditions
solver = ODESolver(velocity_model=vf)  # create an ODESolver class
sol = solver.sample(time_grid=T, x_init=x_init, method='Dopri5', step_size=step_size, return_intermediates=True)  # sample from the model
#%%
sol.shape
#%%
idx=5
plt.scatter(sol[idx,:,0], sol[idx,:,1], s=1, c='r', alpha=0.5)
plt.show()
#%%
import numpy as np

sol = np.array(sol)  # convert to numpy array
T = np.array(T)  # convert to numpy array

fig, axs = plt.subplots(1, 10, figsize=(20,20))

for i in range(10):
    H = axs[i].hist2d(sol[i,:,0], sol[i,:,1], 300, range=((-5,5), (-5,5)))
    
    cmin = 0.0
    cmax = jnp.quantile(jnp.array(H[0]), 0.99).item()
    
    norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)
    
    _ = axs[i].hist2d(sol[i,:,0], sol[i,:,1], 300, range=((-5,5), (-5,5)), norm=norm)
    
    axs[i].set_aspect('equal')
    axs[i].axis('off')
    axs[i].set_title('t= %.2f' % (T[i]))
    
plt.tight_layout()
plt.show()


     


     

# %%
