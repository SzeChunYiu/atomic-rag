# CLAUDE.md

This file defines non-negotiable project rules.

## Mission
Develop an efficient, accurate, and speedy RAG search algorithm that can beat strong current baselines.

## Coding Rules
- Every script must be under 300 lines.
- If a file approaches 220 lines, split it before it becomes hard to reason about.
- Prefer more files over longer files.
- Organize recursively into folders by responsibility.
- No giant utilities file.
- No hidden global state.
- No experiment without logged config, inputs, outputs, and metrics.

## Research Rules
- Do not reinvent obvious prior work.
- For every proposed method, find adjacent work first.
- Separate novelty into: known, partially known, potentially novel, and benchmark-proven.
- Never claim breakthrough until strong baselines are beaten.

## Atomic Thinking
Every pipeline output must be explainable from first principles.
For every improvement, ask:
1. Which atom changed?
2. Which metric moved?
3. Which failure mode was fixed?
4. Which new failure mode appeared?
5. Is the improvement real or benchmark noise?

## Output Discipline
Every run must produce:
- config snapshot
- dataset manifest
- candidates
- selected evidence atoms
- model answer
- metrics
- sanity checks
- error-analysis notes

## Build Philosophy
Atoms become molecules.
Molecules become polymers.
Polymers become the final search algorithm.
