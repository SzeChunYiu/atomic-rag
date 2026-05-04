# Atomic Claim Deblending

Problem:
One chunk can contain multiple claims.
Multiple chunks can describe the same claim.
Overlapping claims can confuse selection.

Deblending goal:
Separate candidate text into evidence atoms.

Initial approach:
- sentence split
- claim extraction with lightweight rules or LLM later
- entity/date/number tagging
- cluster near-duplicate claims
- mark support or contradiction links

Do not make deblending complex until SNR reranking works.
