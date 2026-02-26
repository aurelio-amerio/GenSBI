import os

os.environ["JAX_PLATFORMS"] = "cpu"

import pytest
from gensbi.models.losses import UnconditionalCFMLoss, UnconditionalEDMLoss, UnconditionalSMLoss


def test_unconditional_cfm_loss_raises():
    with pytest.raises(RuntimeError, match="UnconditionalCFMLoss has been removed"):
        UnconditionalCFMLoss("dummy_path")


def test_unconditional_edm_loss_raises():
    with pytest.raises(RuntimeError, match="UnconditionalEDMLoss has been removed"):
        UnconditionalEDMLoss("dummy_path")


def test_unconditional_sm_loss_raises():
    with pytest.raises(RuntimeError, match="UnconditionalSMLoss has been removed"):
        UnconditionalSMLoss("dummy_path")
