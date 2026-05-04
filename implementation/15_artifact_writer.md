# Artifact Writer

Purpose:
Centralize file output.

Functions:
- create run directory
- write JSONL records
- write config snapshot
- write metrics
- write markdown report
- write manifest

Rules:
- no module writes random files directly
- all outputs go through artifact writer
- output schema must be stable
