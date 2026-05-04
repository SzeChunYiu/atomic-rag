# Astro-CS RAG Project Docs v2

Goal: build a search algorithm for RAG that is more accurate, faster, and more efficient than strong baselines.

Core thesis:
RAG is not only nearest-neighbor search. It is noisy evidence detection plus sparse evidence reconstruction.

Main method families:
1. Astro-RAG: astronomy/source-detection inspired evidence scoring.
2. CS-RAG: compressed-sensing inspired sparse evidence selection.
3. Atomic RAG: decompose chunks into evidence atoms, debug every output, and build upward.

How to use this package:
1. Read `CLAUDE.md` before any research or coding.
2. Give `tasks/claude/*` to Claude for literature research.
3. Give `tasks/cursor/*` and `implementation/*` to Cursor/Composer for mechanical codebase building.
4. Use `docs/15_sanity_check_protocol.md` before trusting any benchmark improvement.

Hard rule:
Every script must be under 300 lines. Prefer more files, smaller modules, and deeper folders.
