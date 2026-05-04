"""Deterministic markdown summaries for human inspection."""

from __future__ import annotations

from pathlib import Path


def write_summary_report(
    path: Path,
    *,
    title: str,
    metrics: dict[str, float],
    notes: list[str] | None = None,
    reproduction_commands: list[str] | None = None,
) -> None:
    lines = [f"# {title}", ""]
    lines.append("## Metrics")
    lines.append("")
    for k in sorted(metrics):
        lines.append(f"- {k}: {metrics[k]:.6f}")
    lines.append("")
    if reproduction_commands:
        lines.append("## Reproduction")
        lines.append("")
        for cmd in reproduction_commands:
            lines.append(f"- `{cmd}`")
        lines.append("")
    if notes:
        lines.append("## Notes")
        lines.append("")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
