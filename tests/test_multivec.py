import numpy as np

from astro_cs_rag.indexing.multivec import MultiVecIndex
from astro_cs_rag.indexing.multivec_encoder import HashMultiVecEncoder


def test_maxsim_returns_score_per_chunk() -> None:
    enc = HashMultiVecEncoder()
    chunks = ["the crab nebula contains a pulsar", "cosmic microwave background two point seven kelvin"]
    idx = MultiVecIndex.from_texts(chunk_ids=["c1", "c2"], texts=chunks, encoder=enc)
    q_tokens = enc.encode_tokens(["pulsar in crab nebula"])[0]
    scores = idx.maxsim_scores(q_tokens)
    assert set(scores) == {"c1", "c2"}
    # The chunk that literally shares words "crab nebula pulsar" should score higher.
    assert scores["c1"] > scores["c2"]


def test_save_load_roundtrip(tmp_path) -> None:
    enc = HashMultiVecEncoder()
    idx = MultiVecIndex.from_texts(
        chunk_ids=["c1", "c2"], texts=["alpha beta", "gamma delta"], encoder=enc
    )
    idx.save(tmp_path)
    loaded = MultiVecIndex.load(tmp_path)
    assert loaded.chunk_ids == idx.chunk_ids
    for a, b in zip(loaded.chunks, idx.chunks, strict=True):
        np.testing.assert_array_equal(a.tokens, b.tokens)
