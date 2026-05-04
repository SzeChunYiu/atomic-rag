# Data Model

Core data records:

```text
Document
Chunk
Span
Query
Candidate
EvidenceAtom
SelectedEvidenceSet
Answer
MetricRecord
RunManifest
```

Each record should be a typed schema.
Each record should have stable ids.
Never pass raw dictionaries between major modules.
Use schemas to prevent silent pipeline drift.
