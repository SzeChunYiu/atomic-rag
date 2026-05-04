# Generation Module

Purpose:
Generate answers from selected evidence.

Inputs:
- query
- selected evidence atoms
- prompt config

Outputs:
- answer text
- cited evidence ids
- unsupported claim markers if checked

Rules:
- keep generation separate from retrieval evaluation
- support no-generation retrieval-only mode
- store prompt and model settings
