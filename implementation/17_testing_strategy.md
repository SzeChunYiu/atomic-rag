# Testing Strategy

Unit tests:
- schemas validate
- chunk offsets correct
- SNR formula stable
- selector respects budget
- metrics compute known examples

Integration tests:
- tiny corpus end-to-end run
- artifact files exist
- sanity checks pass

Regression tests:
- saved tiny benchmark expected outputs
- no script over 300 lines
- no missing config snapshot
