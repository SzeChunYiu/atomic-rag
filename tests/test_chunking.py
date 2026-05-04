from astro_cs_rag.atoms.schemas import Document
from astro_cs_rag.chunking.splitters import chunk_documents


def test_chunk_slices_match_source() -> None:
    doc = Document(doc_id="d1", text="abcdefghijklmnop")
    chunks = chunk_documents([doc], chunk_size=5, chunk_overlap=2)
    assert len(chunks) >= 2
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(doc.text)
    for c in chunks:
        assert doc.text[c.start_char : c.end_char] == c.text
