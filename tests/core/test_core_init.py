"""
Tests for ``gensbi.core.__init__`` — lazy import mechanism.
"""

import pytest


class TestCoreInitLazyImports:
    def test_lazy_import_flow_matching(self):
        from gensbi.core import FlowMatchingMethod
        assert FlowMatchingMethod is not None

    def test_lazy_import_diffusion_edm(self):
        from gensbi.core import DiffusionEDMMethod
        assert DiffusionEDMMethod is not None

    def test_lazy_import_score_matching(self):
        from gensbi.core import ScoreMatchingMethod
        assert ScoreMatchingMethod is not None

    def test_invalid_attribute_raises(self):
        import gensbi.core as core
        with pytest.raises(AttributeError, match="has no attribute"):
            _ = core.NonExistentClass
