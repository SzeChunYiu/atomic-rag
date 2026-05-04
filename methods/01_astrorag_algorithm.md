# Astro-RAG Algorithm

Purpose:
Detect evidence atoms as sources above local background noise.

Pipeline:
1. Retrieve top-N candidate chunks.
2. Compute raw relevance score for each candidate.
3. Estimate local background for each candidate.
4. Compute evidence-SNR.
5. Apply threshold or top-k by SNR.
6. Deblend overlapping candidate claims.
7. Emit evidence atoms with score components.

Key output:
A candidate evidence catalog, not just ranked chunks.

First implementation should be simple and inspectable.
