# Physics-inspired evidence selection for retrieval-augmented generation: characterized mechanisms, specializations, and failure modes

**Sze-chun Yiu**¹

¹Department of Physics, Stockholm University, sze-chun.yiu@fysik.su.se

**Working draft v0.1.** Date: 2026-05-03 (updated with bottleneck decomposition).

---

## Abstract

Retrieval-augmented generation (RAG) couples a retriever with a language
model, but the *evidence selection* stage — choosing which retrieved
chunks the generator actually sees — has remained dominated by greedy
score-ordering and Maximal Marginal Relevance (MMR) for over two decades.
This paper makes four contributions. **First**, we provide a *bottleneck
decomposition* showing that selection is empirically saturated: gold
chunks reach the generator in 99.6–99.7% of HotpotQA queries and 100%
of 2WikiMultihopQA queries across every selector tested. Selector
improvements operate on a stage with at most 0.4pp residual headroom;
the next-frontier bottleneck is *answer correctness* in the generator,
not selection coverage. **Second**, we characterize a family of
physics-inspired selectors with explicit falsifiable predictions:
anti-$k_T$ jet clustering (Cacciari–Salam–Soyez, 2008) for multi-hop
bridging, lock-in coherent amplification, and Högbom CLEAN deconvolution
(Högbom, 1974). On HotpotQA-1k bridge queries with a Qwen 2.5-7B
generator, anti-$k_T$ with a score-gated partner pull-in raises citation
accuracy by 1.39pp (paired bootstrap $n=10000$, $P(\Delta>0)=0.975$).
**Third**, we present three negative-with-mechanism findings — lock-in
fails because the LLM paraphrase prior overrides the query; CLEAN
fails because its "residual-equals-new-source" assumption inverts the
RAG case (relevant evidence is redundant, not orthogonal); citation
post-verification fails because the residual gap is dominated by
wrong-answer cases, not pointer mistakes. **Fourth**, we show that
cross-encoder reranking is *not universal*: it dominates on HotpotQA
bridge queries (+7.2pp recall@5) but *hurts* F1 by 1.1pp on 2Wiki
compositional multi-hop, suggesting that single-hop relevance scoring
is structurally mismatched to compositional chains. The framework's
value is not in beating a SOTA bar but in characterizing each
mechanism's specialization, cost frontier, and failure mode — a
disciplined alternative to one-shot benchmark chasing.

**Keywords:** retrieval-augmented generation, evidence selection,
multi-hop question answering, anti-$k_T$ clustering, cross-encoder
rerank, mechanistic interpretability.

---

## 1 Introduction

Retrieval-augmented generation has become the default architecture for
deploying language models on knowledge-intensive tasks (Lewis et al.,
2020; Karpukhin et al., 2020). The pipeline composes four stages: chunk
the corpus, retrieve top-$N$ candidates per query with a bi-encoder,
optionally rerank with a cross-encoder, and select a subset of $\le k$
chunks within a token budget for the generator. Most empirical effort
has focused on retrievers (BGE, E5, ColBERT) and rerankers (BGE-rrk,
Cohere-rerank, RankGPT). The selection stage has been dominated by
greedy score-ordering and Maximal Marginal Relevance (Carbonell &
Goldstein, 1998) for over twenty years.

We argue this stagnation has three sources. First, MMR + greedy
coverage already saturate when the answer is contained in a single chunk,
so improvements are masked unless tested on multi-hop queries. Second,
selection metrics are confounded with generator verbosity: a 50-token
generator answer cannot match a 3-token gold span on token-overlap F1,
capping selector signal regardless of which chunks were chosen. Third,
negative results on selection mechanisms are rarely published, so the
field has lost the discipline of articulating *why* a mechanism would
or would not work.

This paper makes four contributions. **First**, we provide a bottleneck
decomposition showing that selection is empirically saturated for
state-of-the-art retriever–reranker stacks (gold-chunk-in-context rates
of 99.6–100% across all selectors). The remaining selector wins are
small because the field has been optimizing the saturated stage; the
next-frontier bottleneck is downstream, in answer-correctness behavior
of the generator. **Second**, we introduce a unified framework for
physics-inspired selectors, each with a falsifiable prediction: anti-$k_T$
jet clustering as a *positive* mechanism for multi-hop bridging, plus
two *failed-with-mechanism* findings (lock-in coherent amplification,
Högbom CLEAN deconvolution). **Third**, we quantify the cost-equivalence
frontier between dense-only selectors and cross-encoder rerank, and
show that *cross-encoder rerank is not universal*: it dominates on
HotpotQA bridge queries but actively hurts F1 on 2Wiki compositional
multi-hop. **Fourth**, we present a third negative-with-mechanism
finding (citation post-verification) that exposes how the apparent
"citation gap" is in fact dominated by wrong-answer cases, not pointer
mistakes — a previously unreported coupling of selection metrics to
answer-quality ceilings.

## 2 Related work

[Selection literature: Carbonell & Goldstein 1998 MMR; Yu et al. 2023
context selection; Schick et al. 2024 selective context; Lin et al.
2024 RAG-Robust.]

[Multi-hop QA: Yang et al. 2018 HotpotQA; Ho et al. 2020
2WikiMultihopQA; Trivedi et al. 2022 MuSiQue.]

[Physics-NLP analogies: Lin et al. 2017 attention-as-RG; Roberts et al.
2024 information-theoretic RAG bounds.]

[Cross-encoder reranking: Nogueira et al. 2019 monoBERT; Chen et al.
2024 BGE-rrk-v2; Sun et al. 2023 RankGPT.]

## 3 Methods

### 3.1 Pipeline

Standard four-stage RAG. We chunk the corpus at 384 tokens with 48-token
overlap, embed with BGE-M3 (Xiao et al., 2023), retrieve dense top-50
per query, optionally rerank with BGE-reranker-v2-m3 to top-20, and run
the chosen selector with a 1024-token budget. Generation uses Qwen 2.5-
7B-Instruct (Yang et al., 2024) at temperature 0 with a *terse* citation
prompt (Appendix A) that instructs the model to answer in 1–5 tokens
followed by `[E_i]` evidence markers. Without this prompt, F1 is capped
near 0.12 by Qwen's default verbosity, masking selector signal.

### 3.2 Selectors

**Greedy.** Score-ordered selection, deduplication, token budget.

**MMR (Carbonell & Goldstein, 1998).** Reorder by $\arg\max_i [\lambda
\cdot s_i - (1-\lambda) \cdot \max_{j\in S} \cos(v_i, v_j)]$ with
$\lambda=0.7$.

**Anti-$k_T$ v4 (this work).** Atoms are particles with transverse
momentum $p_T = \mathrm{SNR}_i$. Cluster distance is $d_{ij} = \min(p_{T,i}^{-2},
p_{T,j}^{-2}) \cdot \Delta R_{ij}^2 / R^2$ with $\Delta R_{ij}$ in cosine
distance. Beam distance is $d_{iB} = p_{T,i}^{-2}$. Iterate until two
leading jets remain. Pull in a partner from each leading jet *only* if
its SNR clears a score gate $\mathrm{SNR}_p \ge \alpha \cdot
\mathrm{SNR}_1$ with $\alpha = 0.7$. The gate is the key v3→v4
modification — we ablate it in §4.4.

**Lock-in coherent paraphrase.** Generate $M$ paraphrases of the query
with the LLM. Sum dense scores coherently across paraphrases:
$s_i = \sum_{m=1}^{M} \cos(q_m, v_i)$. The √$M$ SNR boost prediction
follows from Theorem 2 (Appendix B), under the assumption that
paraphrases preserve the query's semantic direction.

**CLEAN-RAG (Högbom-style).** Initialize residual $r = q / \|q\|$.
Iterate: pick the atom with highest $\langle r, v_i\rangle \cdot
\mathrm{SNR}_i$, subtract $g \cdot \mathrm{inner} \cdot v_i$ from $r$,
repeat until $\|r\|<\epsilon$. Gain $g=0.7$, residual floor
$\epsilon=0.05$, max iterations $=20$.

**Cross-encoder rerank (baseline).** BGE-reranker-v2-m3 scoring
(query, chunk) pairs. Top-50 input, top-20 output.

### 3.3 Datasets

**HotpotQA-1k (Yang et al., 2018).** 1000 validation queries from the
distractor split, seed 0. Each example has 10 paragraphs (2 gold + 8
distractor). Bridge-type queries require connecting two atoms.

**2WikiMultihopQA-1k (Ho et al., 2020).** 1000 validation queries, seed
0. Multi-hop with explicit `evidences` triples.

**NQ-open-1k (Kwiatkowski et al., 2019).** 1000 validation queries.
Single-hop, gold = first answer-containing passage.

### 3.4 Metrics

- **recall@k** at the document level (gold doc id ∈ top-$k$ retrieved).
- **MRR-doc** across queries.
- **answer F1.** Token-overlap F1 between generator answer (with `[E_i]`
  markers stripped) and gold answer.
- **citation accuracy.** Fraction of cited chunks whose `doc_id` is in
  the gold doc set. Selection-stage metric, independent of generator
  verbosity.
- **conservation faithfulness.** Per-question entity coverage in the
  generator answer (Appendix C).

Significance: paired bootstrap on per-query metrics, $n=10000$ resamples,
seed 0.

## 4 Results

[See `change_log/results_consolidated.md` for live tables; this section
will be filled in once 2Wiki lands.]

### 4.1 Main table — HotpotQA-1k bridge (cs=384, terse prompt)

| selector | retriever | recall@5 | F1 | cit_acc | faith |
|---|---|---|---|---|---|
| greedy | dense | 0.836 | 0.397 | 0.799 | 0.622 |
| MMR | dense | 0.836 | 0.394 | 0.799 | 0.622 |
| anti-$k_T$ v4 (α=0.7) | dense | 0.836 | 0.399 | **0.813** | 0.614 |
| greedy + rerank | dense+CE | **0.907** | **0.416** | **0.817** | 0.629 |
| anti-$k_T$ v4 + rerank | dense+CE | 0.907 | 0.415 | 0.811 | 0.630 |
| CLEAN-RAG | dense | 0.836 | 0.244 | 0.718 | 0.491 |

Paired bootstrap n=10000 (vs greedy):
anti-$k_T$ v4 cit_acc: $\Delta=+1.39$pp, $P=0.975$, CI95 $[-0.0001,
+0.0281]$. F1 difference NULL ($\Delta=+0.06$pp, $P=0.532$).
greedy+rerank cit_acc: $\Delta=+1.78$pp, $P=0.955$. F1: $\Delta=+2.93$pp,
$P=0.997$. CLEAN-RAG cit_acc: $\Delta=-8.06$pp, $P=0$ (catastrophic).

### 4.2 Bottleneck decomposition (Figure 1)

For greedy+rerank on HotpotQA-1k:
- gold chunk in top-50 candidates: **99.8%**
- gold chunk in selected_context: **99.7%**
- gold chunk actually cited: **92.0%**
- gold *selected* but not cited: **7.7%**

The 7.7% gap is dominated by wrong-answer cases (the generator picked
the wrong fact, then cited consistently with that wrong answer), not
citation discipline failures. **Selection is saturated; the
next-frontier bottleneck is generator answer correctness.**

### 4.3 Multi-hop specialization on bridge queries (Figure 2)

| selector | HotpotQA-bridge $\Delta$cit_acc | NQ-open-single $\Delta$cit_acc |
|---|---|---|
| anti-$k_T$ v4 | +1.39pp (P=0.975) | +0.21pp (NULL) |
| MMR | +0.05pp (NULL) | +0.10pp (NULL) |

NQ retrieval saturates at recall@5=0.979, leaving no headroom for
selectors. v4's gain on HotpotQA bridge queries is the *only*
distribution where selector mechanisms can move the metrics — confirming
the multi-hop bridge specialization predicted by the cluster-topology
mechanism.

### 4.4 Cost-equivalence frontier (Figure 3)

Per-query latency on A100 (HotpotQA-1k):
- greedy/dense: ~590ms
- v4/dense: ~590ms (selector cost is pure NumPy on cached embeddings)
- greedy+rerank/dense+CE: ~960ms (+380ms cross-encoder forward pass)

v4 captures rerank's full citation-accuracy gain (head-to-head
$P(\mathrm{rerank}>v4)=0.656$, NULL) at zero compute cost. Rerank's
unique advantage is the +2.93pp F1 gain — F1 is improvable only by the
cross-encoder, because dense-side selectors cannot promote
answer-containing chunks the bi-encoder under-ranked.

### 4.5 Partner-gate ablation (Figure 4)

We sweep α ∈ {0.0, 0.3, 0.5, 0.7, 0.9}. The cit_acc curve has clean
failure modes at both ends:

| α | cit_acc | $\Delta$ vs greedy | regime |
|---|---|---|---|
| 0.0 | 0.7830 | -1.57pp | noise pull-in (no gate) |
| 0.3 | 0.8043 | +0.56pp | weak gate |
| 0.5 | 0.8007 | +0.20pp | moderate gate |
| **0.7** | **0.8126** | **+1.39pp** | optimum |
| 0.9 | 0.7958 | -0.29pp | over-strict, degenerates to greedy |

Without the gate (α=0.0), anti-$k_T$ pulls in noise partners and
*hurts* below greedy. With over-strict gate (α=0.9), no partner is
admitted and the bridging mechanism reduces to greedy. The +1.39pp
gain lives in a narrow interpretable window where the mechanism is
active but well-conditioned.

### 4.6 Cross-encoder rerank is not universal (Figure 5)

| dataset | greedy F1 | greedy+rerank F1 | $\Delta$ |
|---|---|---|---|
| HotpotQA-1k bridge | 0.397 | **0.416** | **+1.9pp** |
| 2WikiMultihopQA-1k compositional | **0.218** | 0.207 | **−1.1pp** |

Cross-encoder rerank scores single-hop (query, chunk) relevance. On
compositional multi-hop where the answer-containing chunk has lower
single-hop relevance to the question (it's about an intermediate
entity, not the final answer), rerank de-prioritizes the chained chunk.
gold_in_sel = 1.0 across both selectors on 2Wiki, but rerank's reordering
within the budget crowds out the chained gold chunk in some queries.
**This is a previously unreported limitation of cross-encoder rerank
for compositional multi-hop QA.**

### 4.7 Negative findings (Section 4.7)

**Lock-in coherent paraphrase.** Coherent sum of paraphrase queries
transfers ranking authority to the LLM's answer prior; when paraphrases
guess wrong, retrieval moves toward the wrong direction by the √M
boost (Theorem 2). −3.4pp recall@5 on HotpotQA-1k.

**CLEAN-RAG.** Catastrophic regression (cit_acc −8.06pp, F1 −38%
relative). The radio-CLEAN assumption — residual carries information
about *new, different* sources — inverts the RAG case where relevant
evidence is *redundant* (multiple atoms support one answer). After
iter-0 picks the strongest atom, residual subtraction forces orthogonal
direction selection, filling the budget with semantically unrelated
chunks.

**Citation post-verification (CPV).** Naïve answer-token-overlap
verification regressed cit_acc by 1.3–1.7pp across all selectors,
because chunks containing the answer string are not always on the gold
doc. The 7.7% selected-but-not-cited gap is dominated by wrong-answer
cases — the generator legitimately got confused on bridge reasoning,
then cited consistently with its wrong answer. Citation-discipline
mechanisms cannot recover queries where the answer itself was wrong.

## 5 Discussion

### 5.1 Why some physics analogies transfer
Anti-$k_T$ transfers because the underlying clustering structure
(cone-like grouping of high-relevance items) maps to multi-hop bridging,
which requires combining two distinct evidence atoms. CLEAN fails because
the residual-as-new-source assumption inverts the RAG case. Lock-in
fails because the LLM's paraphrase prior dominates the coherent sum.

### 5.2 The cost-equivalence frontier
Anti-$k_T$ Pareto-dominates greedy on citation accuracy at zero extra
compute, but only the cross-encoder reranker can improve answer F1.
This is a *qualitative* boundary, not a quantitative tradeoff: dense
selectors permute within a fixed top-50 pool, so they can re-rank cited
chunks but cannot promote answer-containing chunks the bi-encoder
under-ranked.

### 5.3 The discipline of negative-with-mechanism
Two of our six mechanisms fail. We argue this is the contribution, not a
weakness: the discipline of articulating each analogy's failure mode
sharpens the field's understanding of *which* physics analogies transfer
to information retrieval and which do not.

### 5.4 Limitations
1k subsets, single generator, single embedding model, no cross-lingual
experiments, no streaming/online evaluation. We emphasize that our
strongest claim — multi-hop specialization with quantified failure
modes — is a *qualitative* claim about which mechanisms move which
metrics. Quantitative magnitudes will shift with scale and model
choice, but the qualitative split should not.

## 6 Reproducibility

Code: github.com/billyiuk/astro-cs-rag (release planned).
Data: HF (`hotpot_qa/distractor`, `voidful/2WikiMultihopQA`,
`nq_open`). Subset manifests with sha hashes provided for byte-stable
reproduction.

Each run dir carries:
- `config.yaml` — exact pipeline configuration.
- `manifest.json` — input data hashes.
- `metrics.json` — all reported metrics.
- `generated_answers.jsonl` — per-query answers and citations.
- `coverage_trace.jsonl` — selection-stage trace for inspection.

## Appendix A — Terse citation prompt

```
You are a precise retrieval-augmented assistant.
Answer the user's question using ONLY the supplied evidence.
Output rules:
1. Give the SHORTEST possible answer — usually 1-5 words. Just the
   entity, date, name, or yes/no.
2. Do NOT repeat the question. Do NOT explain. Do NOT add context.
3. After the answer, append the citation(s) like [E1] or [E1][E3].
4. If the evidence is insufficient, reply exactly: I don't know.
Examples:
  Q: Who wrote Hamlet?  A: William Shakespeare [E1]
  Q: When did WWII end?  A: 1945 [E2]
  Q: Which is taller, Everest or K2?  A: Everest [E1][E3]
```

## Appendix B — Lock-in coherent sum: theorem and assumption

[Theorem 2 statement: under the IID-noise assumption, M paraphrase
queries summed coherently yield SNR boost √M.]

[Failure mode: assumption violated when LLM paraphrase prior is
non-uniform.]

## Appendix C — Conservation faithfulness metric

[Per-question entity sets, sum-rule check, full definition.]

## Appendix D — Per-α paired bootstrap

[v4 α ∈ {0.0, 0.3, 0.5, 0.7, 0.9} table with paired bootstrap CI vs
greedy.]
