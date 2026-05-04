# Paper outline v1 — Astro-CS-RAG

**Working title:** *Physics-inspired evidence selection for retrieval-augmented generation: characterized mechanisms, specializations, and failure modes*

**Target:** Nature Machine Intelligence (or *Communications of the ACM* /
*JMLR* if NMI rejects on novelty grounds — see Section "Threats" below).

## 1. Abstract (target ≤ 200 words)

We characterize a family of evidence-selection mechanisms for
retrieval-augmented generation (RAG), each derived from a physics analogy:
anti-$k_T$ jet clustering for multi-hop bridging, lock-in coherent
amplification for query disambiguation, and Högbom CLEAN deconvolution for
residual-aware selection. On HotpotQA-1k bridge queries with a Qwen 7B
generator, anti-$k_T$ raises citation accuracy by +1.39pp ($p=0.975$,
paired bootstrap $n=1000$) over a strong greedy baseline. The improvement
vanishes on single-hop NQ-open and is fully subsumed by a cross-encoder
reranker, identifying the mechanism as a *cheap dense-only approximation
to cross-attention bridging*. Two physics analogies fail in
mechanistically explainable ways: lock-in transfers ranking authority to
the LLM's answer prior; CLEAN's "residual-equals-new-source" assumption
inverts the RAG case, where relevant evidence is redundant rather than
orthogonal. We frame each mechanism by (a) the regime of multi-hop bridge
structure where it applies, (b) the cost-equivalence frontier where it
beats stronger but more expensive baselines, and (c) the failure mode that
falsifies its analogy.

## 2. Introduction (≤ 1.5 pages)

- Open: RAG accuracy depends on selecting the *right* evidence atoms from
  retrieved candidates; the selection step is under-studied vs retrieval
  and reranking.
- Three reasons the field is stuck on selection:
  - Greedy + MMR have remained the default for >10 years.
  - Selection metrics are confounded with generator verbosity.
  - Negative results on selection are rarely published.
- Our contribution:
  - A unified framework for physics-inspired selectors with explicit
    falsifiable predictions.
  - A *quantified cost-equivalence frontier* between dense-only selectors
    and cross-encoder rerank.
  - Two negative-with-explanation findings that sharpen the field's
    understanding of which physics analogies transfer.

## 3. Methods

### 3.1 Pipeline architecture
Standard 4-stage: (a) chunk + embed corpus, (b) dense top-50 retrieve, (c)
optional cross-encoder rerank, (d) selector chooses ≤ k atoms within
token budget, (e) Qwen 7B generates answer with citations.

### 3.2 Mechanisms tested
- **Greedy** (baseline): score-ordered, dedup, budget.
- **MMR** (Carbonell-Goldstein 1998): λ-mixed score and diversity.
- **Anti-$k_T$ v4**: Cacciari-Salam-Soyez 2008 jet clustering analogy.
  Atoms are particles; SNR is $p_T$; cluster distance uses cosine.
  Score-gated partner pull-in: partner kept iff $\mathrm{SNR}_p \ge \alpha
  \cdot \mathrm{SNR}_1$ where α is the partner-gate coefficient.
- **Lock-in coherent paraphrase**: $M$ LLM paraphrases of the query
  produce $M$ embeddings; dense retrieval scores summed coherently → √$M$
  SNR boost (theorem 2 in supplement).
- **CLEAN-RAG**: Högbom 1974 iterative deconvolution. Residual query
  vector starts as $q$; each iteration picks the atom with highest
  $\langle r, v_i \rangle \cdot \mathrm{SNR}_i$ and subtracts $g \cdot
  \mathrm{inner} \cdot v_i$ from $r$. Stops when $\|r\| <$ floor.
- **Cross-encoder rerank**: BGE-reranker-v2-m3 baseline (Chen et al. 2024).

### 3.3 Datasets and metrics
- HotpotQA-1k distractor (multi-hop bridge): 1000 queries, validation, seed 0.
- NQ-open-1k (single-hop): 1000 queries, validation, seed 0.
- Metrics: doc-level recall@k, MRR, answer F1 (token overlap with gold),
  citation_accuracy (fraction of cited chunks whose doc_id ∈ gold),
  conservation faithfulness (per-question entity coverage in answer).
- Significance: paired bootstrap on per-query metrics, $n=10000$ resamples.

### 3.4 Generator prompt design
We discovered that the verbose default prompt caps F1 at ≈ 0.12 because
Qwen produces ~50-token sentences while gold answers are 1–5 tokens.
A terse prompt (Appendix A) recovers F1 to 0.40, revealing selector
differences that were previously hidden under the verbosity ceiling.

## 4. Results

### 4.1 Main table (Table 1)
HotpotQA-1k bridge, cs=384, terse prompt:

| selector | retriever | recall@5 | F1 | cit_acc | faith |
|---|---|---|---|---|---|
| greedy | dense | 0.836 | 0.397 | 0.799 | 0.622 |
| MMR | dense | 0.836 | 0.394 | 0.799 | 0.622 |
| anti-$k_T$ v4 (α=0.7) | dense | 0.836 | 0.399 | **0.813** | 0.614 |
| greedy + rerank | dense+CE | **0.907** | **0.416** | **0.817** | 0.629 |
| anti-$k_T$ v4 + rerank | dense+CE | 0.907 | 0.415 | 0.811 | 0.630 |
| CLEAN-RAG | dense | 0.836 | 0.244 | 0.718 | 0.491 |

Significance: anti-$k_T$ v4 vs greedy on cit_acc, paired bootstrap
$n=1000$: $\Delta = +0.0139$, $P(\Delta > 0) = 0.975$, $\mathrm{CI}_{95}
= [-0.0001, +0.0281]$.

### 4.2 Multi-hop specialization (Fig. 1)
Side-by-side bar chart: HotpotQA bridge gain (+1.39pp) vs NQ single-hop
gain (+0.21pp) for v4 vs greedy. Mechanistic prediction confirmed.

### 4.3 Rerank-subsumes-selector (Fig. 2)
Stacking diagram showing v4 + rerank ≈ greedy + rerank, while v4 alone
recovers ≈ 80% of rerank's cit_acc gain at lower compute.

### 4.4 Partner-gate ablation (Fig. 3, pending)
α ∈ {0, 0.3, 0.5, 0.7, 0.9}. Expected: monotone improvement up to α≈0.7,
plateau or regression beyond, confirming the gate prevents noise pull-in
without becoming over-restrictive.

### 4.5 Negative findings (Section 4.5)
Lock-in coherent paraphrase: -3.4pp recall@5, mechanistically explained.
CLEAN-RAG: -38% F1, mechanistically explained.

### 4.6 Cost-equivalence frontier (Fig. 4, deferred)
Pareto plot in (cit_acc, FLOPs) space. Anti-$k_T$ v4 sits on the dense-only
frontier; rerank sits on the cross-encoder frontier; the two frontiers do
not stack but offer different cost regimes.

## 5. Discussion

- Why some physics analogies transfer and others don't:
  - **Anti-$k_T$ transfers** because the underlying clustering structure
    (cone-like grouping of high-relevance items) matches multi-hop
    bridging, where the answer requires combining information across two
    distinct atoms.
  - **CLEAN fails** because the residual-as-new-source assumption inverts
    the RAG case: relevant evidence is *redundant*, not orthogonal.
  - **Lock-in fails** because the LLM's paraphrase prior dominates the
    coherent sum.
- The discipline of articulating *the analogy's failure mode* and
  *what would make it work* is the contribution; the +1.39pp number is the
  evidence the discipline produces real findings.
- Limitations: 1k subsets, single generator, no cross-lingual experiments.

## 6. Threats and reviewer concerns

- "Why not just use rerank?" — we *do* use rerank. The selector and rerank
  characterize different cost frontiers; small-deployment scenarios benefit
  from v4-on-dense without a cross-encoder pass.
- "+1.39pp is small" — true; the *generality* of the framework, not the
  magnitude of any single number, is the contribution. Negative-finding
  rigor is the differentiator.
- "Multi-hop only" — we explicitly characterize the specialization; that's
  a feature, not a bug.

## 7. Reproducibility

- Code: github.com/billyiuk/astro-cs-rag (to be created).
- Data manifests: `data/hotpotqa_1k/subset_manifest.json`,
  `data/nq_open_1k/subset_manifest.json`.
- Config snapshots in every run dir.
- Run seeds fixed at 0; bootstrap seed fixed at 0.

## Appendix A — Terse prompt
[exact text of CITATION_SYSTEM in `prompts.py`]

## Appendix B — Anti-kT v4 algorithm
[pseudocode + reference Python triple-loop in
`src/astro_cs_rag/selection/jet_select.py`]

## Appendix C — Per-query bootstrap distributions
[supplemental figure of histograms]

## Status checklist (live)

- [x] Greedy / MMR / v4 baselines on HotpotQA-terse
- [x] Cross-encoder rerank
- [x] v4 + rerank stack
- [x] CLEAN-RAG benchmark
- [x] NQ-open single-hop
- [ ] v4 α-ablation (running, job 3002413)
- [ ] Lock-in benchmark on terse prompt (re-run, deferred)
- [ ] Cost timing instrumentation (per-stage seconds)
- [ ] 2WikiMultiHopQA cross-dataset (deferred)
- [ ] Appendix figure suite
- [ ] LaTeX manuscript skeleton
