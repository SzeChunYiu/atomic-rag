# Project Goal

Build a search algorithm for RAG that improves the quality-cost frontier.

Target improvements:
- higher retrieval precision
- higher evidence recall
- fewer irrelevant chunks
- fewer tokens passed to the LLM
- lower latency
- stronger faithfulness
- better citation grounding

Main hypothesis:
Top-k semantic similarity confuses proximity with evidential usefulness.

New framing:
RAG should detect evidence atoms in a noisy corpus and reconstruct the smallest sufficient evidence set.

Breakthrough condition:
The method must beat strong baselines on accuracy while reducing cost, latency, or context size.
