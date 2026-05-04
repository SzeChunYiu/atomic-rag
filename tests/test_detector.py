from astro_cs_rag.detection.snr import detect_evidence


def test_snr_threshold_removes_low_snr() -> None:
    scores = {f"c{i}": float(i) for i in range(30)}
    all_atoms = detect_evidence("q", scores, window=5, snr_threshold=0.0)
    filt = detect_evidence("q", scores, window=5, snr_threshold=1e9)
    assert len(all_atoms) == 30
    assert len(filt) < len(all_atoms)
