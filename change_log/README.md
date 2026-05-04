# Change Log

What changed, what is planned, what is still not done.

One entry per phase, file per phase. Newest phase last.

Conventions:

- Filenames: `P{n}_{slug}.md`.
- Each file ends with three sections: **Done**, **Planned**, **Not done**.
- Numerical artifacts (benchmark numbers, ablation tables) live next to their
  source code as JSONL/JSON, not in this folder. This folder is the *narrative*.

## Index

- [P0_reproducibility_floor.md](P0_reproducibility_floor.md) — first runnable end-to-end stack (Ollama, BGE-M3, HF data).
- [P0_reproducibility_receipt.md](P0_reproducibility_receipt.md) — exact commands + smoke-run numbers.
- [P1_atomic_instrumentation.md](P1_atomic_instrumentation.md) — claim-atom decomposer + calorimetric profiler + failure atlas.
- [P2_strong_baselines_and_asimov.md](P2_strong_baselines_and_asimov.md) — cross-section metric, Asimov benchmark, late-interaction baseline, LUNARC SLURM (in progress).
- [P3_anti_kt_evidence_jet.md](P3_anti_kt_evidence_jet.md) — first novel selector with provable IRC-safety in RAG.
- [P3_5_through_P6_inference_sketches.md](P3_5_through_P6_inference_sketches.md) — full F1–F9 coverage: aperture/FDR/Cherenkov/coronagraph/VLBI/standard-candle/CLEAN/SBI/SMC/Wasserstein/TDA/OOD/unfolding/conservation. 77/77 tests.
- [atomic_failure_atlas.md](atomic_failure_atlas.md) — F1–F9 mechanism taxonomy and method assignment matrix.

## Atomic failure atlas (mechanism reference)

Every method must be mapped to one or more atomic failure modes. The taxonomy:

| ID | Failure | Atomic mechanism |
|----|---------|------------------|
| F1 | topical-but-wrong | encoder mixes topic + relation |
| F2 | distractor swarm | gold sits in dense cluster of similar non-gold |
| F3 | query degeneracy | query embedding is smeared image of latent need |
| F4 | split evidence | claim spans chunk boundary |
| F5 | popular-but-empty | high prior retrievability, zero info |
| F6 | lost-in-middle | LLM positional decay |
| F7 | multi-source contradiction | RAG averages, doesn't resolve |
| F8 | multi-hop bridge | gold chunk has bridge entity, no answer |
| F9 | generator drift | right context, wrong synthesis |

Reference this atlas when proposing or evaluating any method.
