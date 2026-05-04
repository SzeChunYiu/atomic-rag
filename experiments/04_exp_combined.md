# Experiment: Combined Astro-CS-RAG

Goal:
Test full system.

Pipeline:
Dense or hybrid retrieval
→ evidence-SNR detection
→ sparse evidence selection
→ generation
→ evaluation

Baselines:
- dense top-k
- hybrid top-k
- dense + reranker
- sparse selector without SNR

Success:
Better quality-cost frontier than baselines.
