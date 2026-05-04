# Logging and Artifacts

Each run folder should contain:

```text
config.yaml
manifest.json
candidates.jsonl
scores.jsonl
evidence_atoms.jsonl
selected_context.jsonl
answers.jsonl
metrics.json
sanity_checks.json
error_cases.jsonl
report.md
```

Log raw score components, not only final scores.
Store enough information to explain why each chunk was selected.
