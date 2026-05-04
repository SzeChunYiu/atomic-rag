# Cursor Task: Build Sparse Selector

Implement greedy selector.

Inputs:
- detected candidates
- token budget
- optional query facets

Outputs:
- selected_evidence.jsonl
- dropped_candidates.jsonl
- coverage_trace.jsonl

Tests:
- respects token budget
- avoids duplicates
- selects coverage when available
