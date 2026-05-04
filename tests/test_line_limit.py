"""No script over 300 lines (project rule)."""

from __future__ import annotations

from pathlib import Path


def test_python_sources_under_line_cap() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "astro_cs_rag"
    assert root.is_dir()
    cap = 300
    failures: list[str] = []
    for path in sorted(root.rglob("*.py")):
        n = sum(1 for _ in path.open(encoding="utf-8"))
        if n > cap:
            failures.append(f"{path.relative_to(root.parents[1])}: {n} lines")
    assert not failures, "\n".join(failures)

