import json
from pathlib import Path

from astro_cs_rag.benchmarks.asimov import (
    stage_efficiency_decomposition,
    synthesize_asimov,
    write_asimov_jsonl,
)


def test_synthesize_and_write(tmp_path: Path) -> None:
    qa = [
        ("What is the speed of light?", "Light travels at 299792458 m/s in vacuum.", ["299792458"]),
        ("Where is the Crab Nebula?", "The Crab Nebula is in the constellation Taurus.", ["Taurus"]),
    ]
    distractors = [f"Distractor passage number {i}." for i in range(40)]
    bench = synthesize_asimov(
        qa_pairs=qa,
        distractor_pool=distractors,
        pool_size=10,
        seed=0,
        position_strategy="middle",
    )
    assert len(bench.queries) == 2
    assert all(q.gold_position == 5 for q in bench.queries)

    out = write_asimov_jsonl(bench, tmp_path / "asimov", distractors)
    corpus_lines = (out["corpus"]).read_text(encoding="utf-8").splitlines()
    queries_lines = (out["queries"]).read_text(encoding="utf-8").splitlines()
    assert len(queries_lines) == 2
    # Each query draws (pool_size - 1)=9 distractors + 1 gold; corpus dedups across queries.
    assert len(corpus_lines) <= 2 + 2 * 9
    for line in queries_lines:
        row = json.loads(line)
        assert row["metadata"]["asimov"] is True
        assert row["metadata"]["gold_position"] == 5
        assert row["metadata"]["pool_size"] == 10


def test_stage_decomposition_multiplies_correctly() -> None:
    decomp = stage_efficiency_decomposition(
        epsilon_retrieval=0.9, epsilon_select=0.8, epsilon_generate=0.7
    )
    assert abs(decomp["epsilon_e2e_predicted"] - 0.9 * 0.8 * 0.7) < 1e-9


def test_uniform_position_seeded(tmp_path: Path) -> None:
    qa = [(f"q{i}", f"gold-{i}", [f"a{i}"]) for i in range(10)]
    distractors = [f"d{i}" for i in range(20)]
    b1 = synthesize_asimov(qa_pairs=qa, distractor_pool=distractors, pool_size=5, seed=42, position_strategy="uniform")
    b2 = synthesize_asimov(qa_pairs=qa, distractor_pool=distractors, pool_size=5, seed=42, position_strategy="uniform")
    assert [q.gold_position for q in b1.queries] == [q.gold_position for q in b2.queries]
