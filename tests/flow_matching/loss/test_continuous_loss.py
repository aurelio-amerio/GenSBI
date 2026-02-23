import jax
import jax.numpy as jnp
import pytest
from gensbi.flow_matching.loss.continuous_loss import  ContinuousFMLoss

from gensbi.flow_matching.path import AffineProbPath
from gensbi.flow_matching.path.scheduler import CondOTScheduler

@pytest.mark.parametrize("reduction", ["None", "mean", "sum"])
def test_loss(reduction):

    path = AffineProbPath(scheduler=CondOTScheduler())
    loss = ContinuousFMLoss(path, reduction=reduction)

    vf = lambda x, t, args=None: x 

    x_0 = jnp.zeros((8, 2))
    x_1 = jnp.ones((8, 2))
    t = jnp.linspace(0, 1, 8)
    batch = (x_0, x_1, t)
    loss_value = loss(vf, batch)
    if reduction == "None":
        assert loss_value.shape == x_0.shape, f"loss: expected shape {x_0.shape}, got {loss_value.shape}"
    else:
        assert loss_value.shape == (), f"loss: expected shape (), got {loss_value.shape}"

def test_invalid_reduction():
    path = AffineProbPath(scheduler=CondOTScheduler())
    with pytest.raises(ValueError) as e:
        ContinuousFMLoss(path, reduction="invalid")
    assert "is not a valid value for reduction" in str(e.value)