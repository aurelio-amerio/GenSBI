def test_models_export_nf_classes():
    from gensbi.models import MAFlow, MAFlowParams, TarFlow, TarFlowParams
    from gensbi.models.maf import MAFlow as M2
    from gensbi.models.tarflow import TarFlow as T2
    assert MAFlow is M2 and TarFlow is T2
    assert MAFlowParams is not None and TarFlowParams is not None
