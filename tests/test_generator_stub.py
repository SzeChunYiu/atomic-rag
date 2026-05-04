from astro_cs_rag.config.schema import GeneratorSettings
from astro_cs_rag.generation.generator import StubGenerator, build_generator


def test_stub_generator_cites_all_evidence() -> None:
    gen = StubGenerator()
    out = gen.answer(
        query_id="q1",
        query_text="What pulsar is in the Crab Nebula?",
        evidence=[
            ("c1", "The Crab Nebula contains a pulsar. It is the remnant of a supernova."),
            ("c2", "Distractor passage about something unrelated."),
        ],
    )
    assert out.query_id == "q1"
    assert "[E1]" in out.answer_text
    assert "[E2]" in out.answer_text
    # Stub cites every evidence chunk so selector differences are visible
    # in citation_accuracy. The original "cite first only" behavior masked
    # selector signal whenever two selectors picked the same top-1 chunk.
    assert out.cited_chunk_ids == ["c1", "c2"]
    assert out.selected_chunk_ids == ["c1", "c2"]


def test_build_generator_defaults_to_stub_when_disabled() -> None:
    settings = GeneratorSettings(enabled=False, provider="ollama")
    gen = build_generator(settings)
    assert gen.provider == "stub"


def test_stub_generator_handles_no_evidence() -> None:
    gen = StubGenerator()
    out = gen.answer(query_id="q1", query_text="anything", evidence=[])
    assert out.answer_text == "I don't know."
    assert out.cited_chunk_ids == []
