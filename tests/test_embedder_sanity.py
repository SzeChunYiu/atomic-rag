"""Pre-flight embedder content-blindness check."""

from __future__ import annotations

from astro_cs_rag.diagnostics.embedder_sanity import check
from astro_cs_rag.indexing.embedders import HashEmbedder, TrigramEmbedder


def test_hash_embedder_is_caught_by_check():
    """HashEmbedder is content-blind — must not pass the sanity check."""
    result = check(HashEmbedder())
    assert not result.passed, result.reason
    assert "content-blind" in result.reason


def test_trigram_embedder_passes_check():
    """TrigramEmbedder ranks hop1 well — must pass."""
    result = check(TrigramEmbedder())
    assert result.passed, result.reason
    assert result.median_hop1_rank < result.random_expected_rank
    assert result.z_score > 4.0


def test_check_reason_includes_z_and_n_probes():
    """Reason string is human-readable and includes the diagnostic numbers
    a future reader would need to interpret it."""
    r = check(TrigramEmbedder())
    assert "z=" in r.reason
    assert "median rank" in r.reason
