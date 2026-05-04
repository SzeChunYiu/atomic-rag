# Evaluation Module

Evaluate separately:
- retrieval quality
- context quality
- answer quality
- efficiency

Inputs:
- run artifacts
- gold labels if available

Outputs:
- metrics.json
- metric table
- error cases

Do not compute metrics from in-memory state only.
Always evaluate from saved artifacts.
