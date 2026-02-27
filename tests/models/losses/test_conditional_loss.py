import os

os.environ["JAX_PLATFORMS"] = "cpu"

import pytest
from gensbi.models.losses import ConditionalCFMLoss, ConditionalEDMLoss, ConditionalSMLoss


def test_conditional_cfm_loss_raises():
    with pytest.raises(RuntimeError, match="ConditionalCFMLoss has been removed"):
        ConditionalCFMLoss("dummy_path")


def test_conditional_edm_loss_raises():
    with pytest.raises(RuntimeError, match="ConditionalEDMLoss has been removed"):
        ConditionalEDMLoss("dummy_path")


def test_conditional_sm_loss_raises():
    with pytest.raises(RuntimeError, match="ConditionalSMLoss has been removed"):
        ConditionalSMLoss("dummy_path")
