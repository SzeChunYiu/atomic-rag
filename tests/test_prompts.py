from astro_cs_rag.generation.prompts import assemble


def test_assembled_prompt_inserts_evidence_markers() -> None:
    out = assemble(
        query="who?",
        evidence=[("c1", "Alpha sentence."), ("c2", "Beta sentence.")],
    )
    assert "[E1]" in out.user
    assert "[E2]" in out.user
    assert out.evidence_chunk_ids == ["c1", "c2"]
    assert "<<SYS>>" in out.full and "<<USER>>" in out.full


def test_assembled_prompt_handles_empty_evidence() -> None:
    out = assemble(query="?", evidence=[])
    assert "(no evidence retrieved)" in out.user
    assert out.evidence_chunk_ids == []
