# CURSOR.md

Cursor/Composer should build the codebase mechanically from these docs.

## Prime Directive
Build small, typed, testable modules. Do not improvise large scripts.

## File Limit
Every Python script must stay under 300 lines.
Break modules into folders before adding complexity.

## Cursor Work Order
1. Scaffold repository folders.
2. Add configs and schemas.
3. Implement baseline retrieval.
4. Implement artifact logging.
5. Implement Astro-RAG detector.
6. Implement sparse selector.
7. Implement benchmark runner.
8. Implement sanity checks and error analysis.
9. Only then add optimization.

## Forbidden
- No monolithic notebooks as source of truth.
- No untracked experiment outputs.
- No benchmark number without reproduction command.
- No method merged without ablation.

## Expected Result
A modular research codebase that supports fast iteration while preserving atomic traceability.
