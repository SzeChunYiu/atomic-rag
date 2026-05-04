"""Prompt assemblers for RAG generation. Keep deterministic and inspectable."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssembledPrompt:
    system: str
    user: str
    full: str
    evidence_chunk_ids: list[str]


CITATION_SYSTEM = (
    "You are a precise retrieval-augmented assistant.\n"
    "Answer the user's question using ONLY the supplied evidence.\n"
    "Output rules:\n"
    "1. Give the SHORTEST possible answer — usually 1-5 words. Just the entity, "
    "date, name, or yes/no.\n"
    "2. Do NOT repeat the question. Do NOT explain. Do NOT add context.\n"
    "3. After the answer, append the citation(s) like [E1] or [E1][E3].\n"
    "4. If the evidence is insufficient, reply exactly: I don't know.\n"
    "Examples:\n"
    "  Q: Who wrote Hamlet?  A: William Shakespeare [E1]\n"
    "  Q: When did WWII end?  A: 1945 [E2]\n"
    "  Q: Which is taller, Everest or K2?  A: Everest [E1][E3]"
)

PLAIN_SYSTEM = (
    "You are a helpful assistant. Use the supplied evidence when relevant. "
    "Be concise and avoid speculation."
)

COT_SYSTEM = (
    "You are a precise retrieval-augmented assistant.\n"
    "For multi-hop questions, you MUST first break the question into "
    "intermediate steps before giving the final answer.\n"
    "Output format (do not deviate):\n"
    "Reasoning: <one or two short sentences identifying intermediate "
    "entities. Be terse.>\n"
    "Final answer: <SHORTEST possible answer, 1-5 words> [citation(s)]\n"
    "Rules:\n"
    "1. Use ONLY the supplied evidence.\n"
    "2. Final answer must start with the literal text 'Final answer:' "
    "on its own line.\n"
    "3. Cite the chunks supporting the FINAL answer like [E1] or [E1][E3].\n"
    "4. If evidence is insufficient, write: Final answer: I don't know.\n"
    "Examples:\n"
    "  Q: Who is the mother of the director of film X?\n"
    "    Reasoning: Film X was directed by Y [E1]. Y's mother is Z [E2].\n"
    "    Final answer: Z [E2]\n"
    "  Q: When did WWII end?\n"
    "    Reasoning: WWII ended in 1945 [E1].\n"
    "    Final answer: 1945 [E1]"
)

# Few-shot CoT: richer in-context examples drawn from HotpotQA/2Wiki patterns.
# Covers: bridge (A→B→answer), comparison, yes/no, direct lookup.
FEW_SHOT_COT_SYSTEM = (
    "You are a precise retrieval-augmented assistant.\n"
    "Solve multi-hop questions step by step, then give the shortest possible answer.\n\n"
    "Output format:\n"
    "Reasoning: <identify the intermediate entity/fact, then the final fact>\n"
    "Final answer: <1-5 words> [E_i for each supporting chunk]\n\n"
    "EXAMPLES\n"
    "--------\n"
    "Q: Which magazine was started first, Arthur's Magazine or First for Women?\n"
    "Evidence: [E1] Arthur's Magazine was an American literary periodical "
    "founded in 1844... [E2] First for Women is a women's magazine published "
    "by Bauer Media Group, first published in 1989...\n"
    "Reasoning: Arthur's Magazine was founded in 1844 [E1]. "
    "First for Women was founded in 1989 [E2]. 1844 < 1989.\n"
    "Final answer: Arthur's Magazine [E1]\n\n"
    "Q: What nationality is the director of film Giochi Proibiti?\n"
    "Evidence: [E1] Giochi Proibiti is a 1952 Italian drama film directed "
    "by René Clément. [E2] René Clément was a French film director...\n"
    "Reasoning: Giochi Proibiti was directed by René Clément [E1]. "
    "René Clément was French [E2].\n"
    "Final answer: French [E2]\n\n"
    "Q: Are both Spike Lee and Martin Scorsese American directors?\n"
    "Evidence: [E1] Spike Lee is an American film director... "
    "[E2] Martin Scorsese is an American film director...\n"
    "Reasoning: Spike Lee is American [E1]. Martin Scorsese is American [E2].\n"
    "Final answer: yes [E1][E2]\n\n"
    "Q: Who is the spouse of the director of film Big Fish?\n"
    "Evidence: [E1] Big Fish is a 2003 film directed by Tim Burton. "
    "[E2] Tim Burton was married to Helena Bonham Carter from 2001 to 2014...\n"
    "Reasoning: Big Fish was directed by Tim Burton [E1]. "
    "His spouse was Helena Bonham Carter [E2].\n"
    "Final answer: Helena Bonham Carter [E2]\n"
    "--------\n"
    "Rules: use ONLY the supplied evidence. Always write 'Final answer:' on its own line. "
    "If evidence is insufficient, write: Final answer: I don't know."
)


def assemble(
    *,
    query: str,
    evidence: list[tuple[str, str]],
    style: str = "citation",
) -> AssembledPrompt:
    """Build (system, user, full) prompt for an Ollama-style chat-completion API.

    `evidence` is a list of (chunk_id, text) preserving selection order.
    """
    if style == "few_shot_cot":
        system = FEW_SHOT_COT_SYSTEM
    elif style == "cot":
        system = COT_SYSTEM
    elif style == "citation":
        system = CITATION_SYSTEM
    else:
        system = PLAIN_SYSTEM
    blocks: list[str] = []
    chunk_ids: list[str] = []
    for i, (cid, text) in enumerate(evidence, start=1):
        marker = f"[E{i}]"
        blocks.append(f"{marker} (id={cid})\n{text.strip()}")
        chunk_ids.append(cid)
    evidence_text = "\n\n".join(blocks) if blocks else "(no evidence retrieved)"
    user = (
        f"Question: {query}\n\nEvidence:\n{evidence_text}\n\n"
        "Answer:"
    )
    full = f"<<SYS>>\n{system}\n<<USER>>\n{user}"
    return AssembledPrompt(system=system, user=user, full=full, evidence_chunk_ids=chunk_ids)
