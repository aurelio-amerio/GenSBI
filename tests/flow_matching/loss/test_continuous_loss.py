"""Test that ContinuousFMLoss has been removed."""
import pytest


def test_continuous_fm_loss_removed():
    """ContinuousFMLoss was moved out — importing it should fail."""
    with pytest.raises(ImportError):
        from gensbi.flow_matching.loss.continuous_loss import ContinuousFMLoss  # noqa: F401


def test_continuous_fm_loss_not_in_public_api():
    """ContinuousFMLoss should not be in the flow_matching.loss public API."""
    from gensbi.flow_matching import loss

    assert not hasattr(loss, "ContinuousFMLoss")