from astro_cs_rag.atoms.schemas import Chunk, EvidenceAtom
from astro_cs_rag.selection.greedy import greedy_select


def test_coverage_trace_has_overlap_when_query_set() -> None:
    atoms = [
        EvidenceAtom(
            query_id="q1",
            chunk_id="d::0",
            raw_score=1.0,
            bg_mean=0.0,
            bg_std=1.0,
            snr=2.0,
            detector_rank=1,
        )
    ]
    ch = Chunk(
        chunk_id="d::0",
        doc_id="d",
        text="Crab Nebula pulsar",
        start_char=0,
        end_char=10,
        token_count=5,
    )
    cmap = {"d::0": ch}
    sel, dropped, trace = greedy_select(
        atoms,
        cmap,
        token_budget=100,
        query_text="Crab Nebula",
    )
    assert len(sel) == 1
    assert any(r.get("overlap_fraction", 0) > 0 for r in trace if r.get("action") == "select")
