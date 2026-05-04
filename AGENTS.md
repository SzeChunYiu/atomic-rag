# Agents

Pointers for AI assistants in this repo. **Do not duplicate long policy here**—read the linked files.

## Read first

- [CLAUDE.md](CLAUDE.md) — non‑negotiable project rules and research discipline.
- [CURSOR.md](CURSOR.md) — Cursor-oriented notes for this workspace.
- [docs/](docs/) — goals, protocols, metrics, reproducibility.
- [implementation/](implementation/) — design notes that track the code layout.
- [tasks/cursor/](tasks/cursor/) — staged build tasks for implementation work.

## Parent workspace layout

Projects under `/Users/billy/Desktop/projects` share a standard structure (`AGENTS.md`, `.claude/`, `docs/`, etc.). If you need to **bootstrap** missing pieces, see [`../../CLAUDE.md`](../../CLAUDE.md) and run `bootstrap-ai-project` from that folder (optionally: `bootstrap-ai-project /absolute/path/to/this/repo`).

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Config: [pyproject.toml](pyproject.toml) (`testpaths`, `pythonpath`).

## CLI commands

After `pip install -e .`, console scripts are defined in [pyproject.toml](pyproject.toml): `rag-build-index`, `rag-retrieve`, `rag-detect`, `rag-select`, `rag-generate`, `rag-evaluate`, `rag-run-benchmark`, `rag-sanity-check`, `rag-prepare-data`. Use `--help` on each.

## Phased program

Status anchor for any agent. Do not start a later phase until the previous one
has shipped a runnable artifact + a `change_log/` entry.

- P0 — reproducibility floor (Ollama generation, BGE-M3, real datasets, full artifact set).
- P1 — atomic instrumentation + literature reconnaissance.
- P2 — strong baselines (ColBERT/RAPTOR/SPLADE) + Asimov benchmark + cross-section metric.
- P3 — anti-$k_T$ IRC-safe evidence-jet clustering (first novel method).
- P4 — lock-in coherent paraphrase retrieval (lock-in amplifier analogy).
- P5 — conservation-law faithfulness verification (HEP analogy).
- P6 — unification under scattering-theoretic framing (the paper).
- P7 — scale + multi-model robustness on LUNARC.
- P8 — falsification + paper writing.

Each phase ends with a *mechanism statement* (what we now understand about RAG)
plus a numerical artifact, both logged in `change_log/`.

## Operating principles

1. **Atomic outputs.** Inspectable artifacts before the next change.
2. **Schema discipline.** No raw `dict` across module boundaries (`src/astro_cs_rag/atoms/schemas.py`).
3. **File budget.** Every Python file < 300 lines. CI enforces (`tests/test_line_limit.py`).
4. **Mechanism over leaderboard.** Method ↔ atomic failure mode mapping in `change_log/atomic_failure_atlas.md` (P1).
5. **No claim without ablation.** Matched compute, reproduction command, falsification condition.

## Hard rules

- New dependency → add to `pyproject.toml`.
- Do not commit benchmark numbers without a one-line reproduction command.
- Do not introduce hidden global state.
- Do not skip the line-limit test.
