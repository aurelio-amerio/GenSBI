import os

os.environ["JAX_PLATFORMS"] = "cpu"

import pytest
from gensbi.models.losses import JointCFMLoss, JointEDMLoss, JointSMLoss


def test_joint_cfm_loss_raises():
    with pytest.raises(RuntimeError, match="JointCFMLoss has been removed"):
        JointCFMLoss("dummy_path")


def test_joint_edm_loss_raises():
    with pytest.raises(RuntimeError, match="JointEDMLoss has been removed"):
        JointEDMLoss("dummy_path")


def test_joint_sm_loss_raises():
    with pytest.raises(RuntimeError, match="JointSMLoss has been removed"):
        JointSMLoss("dummy_path")
