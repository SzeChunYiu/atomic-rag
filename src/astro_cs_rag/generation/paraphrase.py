"""LLM-driven paraphrase generation for lock-in retrieval.

We deliberately use simple deterministic prompts so paraphrases are
reproducible across runs given a fixed seed and model digest.
"""

from __future__ import annotations

import re

from astro_cs_rag.config.schema import GeneratorSettings
from astro_cs_rag.generation.generator import build_generator


_PARAPHRASE_SYSTEM = (
    "You rewrite a question into surface-form variants while preserving "
    "EXACTLY its information content. Constraints:\n"
    "1. Each output must still be a question that asks the SAME thing.\n"
    "2. NEVER include the answer or any candidate answer.\n"
    "3. NEVER add new entities, dates, places, or facts not in the original.\n"
    "4. Vary syntax (active/passive, word order, synonyms) only.\n"
    "Output ONE paraphrase per line. No numbering. No commentary."
)


def generate_paraphrases(
    query: str,
    *,
    n: int,
    settings: GeneratorSettings,
) -> list[str]:
    """Return n paraphrases (deduplicated, including the original)."""
    if n <= 1:
        return [query]
    gen = build_generator(settings)
    prompt = (
        f"Original question: {query}\n\n"
        f"Write exactly {n - 1} alternative phrasings of THIS question. "
        f"Each must end with a '?' and must NOT reveal or guess the answer. "
        f"One per line:"
    )
    ans = gen.answer(query_id="paraphrase_synth", query_text=prompt, evidence=[])
    lines = [line.strip(" -*0123456789.\t") for line in ans.answer_text.splitlines()]
    candidates = [re.sub(r"\s+", " ", line).strip() for line in lines if line.strip()]
    seen: set[str] = {query}
    out: list[str] = [query]
    for c in candidates:
        if c not in seen and len(c) > 5:
            seen.add(c)
            out.append(c)
        if len(out) >= n:
            break
    return out
