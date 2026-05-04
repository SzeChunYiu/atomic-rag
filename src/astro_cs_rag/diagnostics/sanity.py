"""Sanity checks over run artifacts."""

from __future__ import annotations

from pathlib import Path


# Minimum trust after `rag-retrieve` (extend checklist after evaluate if desired).
EXPECTED_FILES = (
    "config.yaml",
    "manifest.json",
    "candidates.jsonl",
)


def load_run_checks(run_dir: Path) -> dict[str, object]:
    missing = [name for name in EXPECTED_FILES if not (run_dir / name).is_file()]
    ok = not missing
    return {"ok": ok, "missing": missing, "run_dir": str(run_dir)}


def sanity_report_payload(payload: dict[str, object]) -> str:
    lines = ["## Sanity check", "", f"- ok: {payload.get('ok')}", ""]
    missing = payload.get("missing") or []
    if missing:
        lines.append("Missing files:")
        for m in missing:
            lines.append(f"- {m}")
    else:
        lines.append("All expected artifact files present.")
    return "\n".join(lines) + "\n"
