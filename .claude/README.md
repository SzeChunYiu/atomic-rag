# .claude/

Project-local Claude Code configuration for the Astro-CS-RAG project.

Currently intentionally empty. Add hooks, skills, and settings overrides here
as they become necessary; do not import from any parent `.claude/` folder.

Suggested future contents:

- `settings.json` — project-local hook config (e.g., a pre-commit hook that
  runs `pytest tests/test_line_limit.py` before any code change is accepted).
- `skills/` — project-specific skills (e.g., a "reproduce-baseline" skill that
  runs the canonical benchmark command and writes artifacts).
- `hooks/` — shell scripts triggered on Stop, SubagentStop, etc.
