# Nomenclature: quark/hadron, not atom (2026-05-03)

## Rationale

User correction (as a particle physicist): "atom" is wrong physics
because atoms are composite (nucleus + electrons), not fundamental.
The truly indivisible content unit at our pipeline's finest scale is
the single-claim sentence — that's a *quark*, not an atom.

This gives us a *better* (not just renamed) framing because each
physics concept actually maps to a pipeline mechanism we already
have:

## Mapping (paper-level terminology) — REVISED 2026-05-03 to start at hadron

User refinement: starting at quarks claims more granularity than the
pipeline actually probes. We don't decompose sentences below sentence
level. Hadrons as the fundamental working unit matches what we do.

| Old name | New name | Why this is correct physics |
|---|---|---|
| atom (single-claim sentence) | **hadron** | irreducible at our working scale; has internal structure (type+content) |
| claim_type (WHO/WHEN/WHERE/WHAT_NUM/WHAT_OBJ) | **flavor quantum number** | discrete labels distinguishing hadron species |
| chunk (~3 sentences) | **nucleus** | bound state of ~3 hadrons (matches actual ratio: 3.25 sentences/chunk) |
| document (~5–10 chunks) | molecular complex | larger structure |
| corpus | bulk matter | — |
| query | external probe particle | scatters off the corpus |
| similarity score | scattering amplitude | direct analog |

## Pipeline stages — physics-correct terms

| Old name | New name |
|---|---|
| atomic decomposition (analysis) | **quark-level decomposition** or QCD-style analysis |
| atom retrieval | **deep inelastic scattering** (probe quarks at high resolution) |
| atom→chunk presentation (ATCP) | **hadronization with context** (probe quark; observe hadron) |
| selection | **hadron formation** (which quarks bind into evidence) |
| multi-scale retrieval (RG) | **running coupling** / scale-dependent QCD |
| token-overlap citation filter | **confinement check** (no free quarks) |
| MaxEnt selection | **partition function with chemical potentials** |
| prompt-ordering replicas | **multi-event averaging** (suppresses detector position bias) |

## Concepts that gain clarity

### Confinement
In QCD, quarks cannot exist in isolation — they bind into hadrons. In
RAG: a single sentence retrieved without surrounding context has poor
predictive value; the LLM needs the hadronic (chunk-level) context.
This is *exactly* the ATCP method (retrieve on quarks, present on
hadrons) — and now it has a real physics name and motivation.

### Asymptotic freedom
At high resolution (high momentum transfer), QCD coupling weakens —
quarks behave as nearly free particles. Maps to: at fine query
specificity (precise entity match), individual sentence relevance
dominates. At low resolution (topic-level), quarks are confined into
hadrons — coarse-scale chunk similarity dominates. The multi-scale RG
retrieval is exploiting this *running coupling*.

### Hadronization
In a collision, quark fragments bind into observable hadrons by
QCD constraints. In RAG selection: chosen sentences must form a
coherent evidence set (the LLM perceives them together as a
"hadron" of evidence). Submodular set-cover with claim-typed facets
is one hadronization scheme; MaxEnt with Lagrange multipliers is
another. Different hadronization schemes give different observed
final states.

### Deep inelastic scattering (DIS)
Probe particles (queries) at high enough energy that they resolve
quark structure within hadrons. RAG analog: precise queries that
target specific facts within larger documents. Multi-scale retrieval
mimics the DIS process: at coarse scale you see hadrons (docs/chunks);
at fine scale you resolve quarks (sentences).

## Implementation strategy

- **Paper / change_log / docs**: use the new terminology immediately.
- **Code identifiers**: keep `atoms`, `atom_id`, `atom_embs.npy` as-is
  for now. The 6 jobs currently queued all reference these names;
  renaming mid-flight would break the queue. Refactor once the
  experimental campaign settles.
- **Public-facing artifacts** (paper, blog post, talk slides): use
  quark/hadron from day one.

## Title proposal v3

*"Quark-level decomposition of retrieval-augmented generation:
a QCD-inspired framework for multi-hop reasoning"*

Subsections:
1. **Quark identification** (sentence-claim deblending — what we called D04)
2. **Hadronization schemes** (selection — comparison of submodular,
   MaxEnt, etc.)
3. **Confinement and citation** (you can't have a free quark in your
   answer — citations must be hadronically grounded)
4. **Running coupling** (multi-scale retrieval with kernel flow)
5. **Multi-event averaging** (prompt-ordering replicas for B4)

This is a Nature MI title I'd actually believe.

## Falsification of the analogy itself

If running our methods at "quark scale" never beats running at "hadron
scale" (chunks), then the analogy is decorative — the smaller scale
isn't physically distinct in our pipeline. So far the data goes the
*other* way: quark-only retrieval underperforms hadron-only, but the
combined (probe-as-quark, present-as-hadron, ATCP) is predicted to
beat both. We have not yet measured this directly — it's the next
unrun experiment. ATCP becomes *the headline test* of the analogy.
