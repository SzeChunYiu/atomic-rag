from astro_cs_rag.atoms.decomposer import decompose_chunk
from astro_cs_rag.atoms.schemas import Chunk


def _chunk(text: str) -> Chunk:
    return Chunk(
        chunk_id="d1::0",
        doc_id="d1",
        text=text,
        start_char=0,
        end_char=len(text),
        token_count=len(text.split()),
    )


def test_sentence_split_and_atom_offsets() -> None:
    text = "The Crab Nebula is in Taurus. It contains a pulsar PSR B0531+21. NASA studies it."
    atoms = decompose_chunk(_chunk(text))
    assert len(atoms) == 3
    for a in atoms:
        assert text[a.span_start : a.span_end].strip() == a.text


def test_entity_number_date_extraction() -> None:
    text = "Crab Nebula has angular size 6 arcmin, observed since 1054 AD by NASA."
    atoms = decompose_chunk(_chunk(text))
    assert len(atoms) == 1
    a = atoms[0]
    assert "Crab Nebula" in a.entities
    assert any("6" in n for n in a.numbers)
    assert "1054" in a.dates


def test_decompose_handles_empty_and_punctuation() -> None:
    atoms = decompose_chunk(_chunk(""))
    assert atoms == []
    atoms = decompose_chunk(_chunk("Hi."))
    assert len(atoms) == 1
    assert atoms[0].text == "Hi."
