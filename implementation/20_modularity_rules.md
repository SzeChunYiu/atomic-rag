# Modularity Rules

Each module has one job.

Bad:
`pipeline.py` does loading, retrieval, detection, selection, and evaluation.

Good:
- loader loads
- retriever retrieves
- detector scores
- selector selects
- evaluator evaluates
- reporter reports

When a module needs many helpers, create a subfolder.
Do not exceed 300 lines per script.
