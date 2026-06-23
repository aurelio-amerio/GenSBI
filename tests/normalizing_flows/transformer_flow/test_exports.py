def test_top_level_exports():
    from gensbi.normalizing_flows import make_tarflow, TransformerFlow
    from gensbi.normalizing_flows.transformer_flow import (
        make_tarflow as mt2, TransformerFlow as TF2,
    )
    assert make_tarflow is mt2
    assert TransformerFlow is TF2
