"""High-level Generator interface — Ollama-backed and stub variants."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Protocol

from astro_cs_rag.atoms.schemas import GeneratedAnswer
from astro_cs_rag.config.schema import GeneratorSettings
from astro_cs_rag.generation.ollama_client import OllamaClient, OllamaResponse, now_seconds
from astro_cs_rag.generation.prompts import AssembledPrompt, assemble


class Generator(Protocol):
    provider: str
    model: str

    def answer(
        self,
        *,
        query_id: str,
        query_text: str,
        evidence: list[tuple[str, str]],
    ) -> GeneratedAnswer: ...


@dataclass
class StubGenerator:
    """Deterministic answerer for CI / smoke tests.

    The stub now concatenates the first sentence of EACH evidence chunk and
    cites EVERY chunk it consumed. This makes selector differences visible
    in citation_accuracy and answer_f1 — the original implementation cited
    only the top-1 chunk, so selectors that picked the same top-1 (most
    pairs of greedy / anti-kT v2) gave identical metrics regardless of
    what they picked beyond chunk 1.
    """

    provider: str = "stub"
    model: str = "stub"

    def answer(
        self,
        *,
        query_id: str,
        query_text: str,
        evidence: list[tuple[str, str]],
    ) -> GeneratedAnswer:
        if not evidence:
            return GeneratedAnswer(
                query_id=query_id,
                answer_text="I don't know.",
                cited_chunk_ids=[],
                selected_chunk_ids=[],
                provider=self.provider,
                model=self.model,
            )
        sentence_pat = re.compile(r"(?<=[.!?])\s+")
        parts: list[str] = []
        cited: list[str] = []
        prompt_tokens = len(query_text.split())
        for i, (cid, text) in enumerate(evidence, start=1):
            if not text:
                continue
            first_sentence = sentence_pat.split(text.strip())[0]
            if not first_sentence:
                continue
            parts.append(f"{first_sentence} [E{i}]")
            cited.append(cid)
            prompt_tokens += len(text.split())
        answer_text = " ".join(parts) if parts else "I don't know."
        return GeneratedAnswer(
            query_id=query_id,
            answer_text=answer_text,
            cited_chunk_ids=cited,
            selected_chunk_ids=[c for c, _ in evidence],
            prompt_tokens_estimate=prompt_tokens,
            completion_tokens_estimate=len(answer_text.split()),
            provider=self.provider,
            model=self.model,
        )


_CITATION_RE = re.compile(r"\[E(\d+)\]")


@dataclass
class OllamaGenerator:
    settings: GeneratorSettings
    client: OllamaClient
    provider: str = "ollama"

    @property
    def model(self) -> str:
        return self.settings.model_name

    def answer(
        self,
        *,
        query_id: str,
        query_text: str,
        evidence: list[tuple[str, str]],
    ) -> GeneratedAnswer:
        prompt: AssembledPrompt = assemble(
            query=query_text, evidence=evidence, style=self.settings.prompt_style
        )
        t0 = now_seconds()
        resp: OllamaResponse = self.client.generate(
            model=self.settings.model_name,
            prompt=prompt.user,
            system=prompt.system,
            temperature=self.settings.temperature,
            seed=self.settings.seed,
            num_predict=self.settings.max_tokens,
        )
        elapsed = time.perf_counter() - t0
        cited = _resolve_citations(resp.text, prompt.evidence_chunk_ids)
        return GeneratedAnswer(
            query_id=query_id,
            answer_text=resp.text.strip(),
            cited_chunk_ids=cited,
            selected_chunk_ids=prompt.evidence_chunk_ids,
            prompt_tokens_estimate=resp.prompt_eval_count,
            completion_tokens_estimate=resp.eval_count,
            latency_seconds=float(elapsed),
            provider=self.provider,
            model=resp.model,
            metadata={
                "total_duration_ns": resp.total_duration_ns,
                "temperature": self.settings.temperature,
                "seed": self.settings.seed,
            },
        )


def _resolve_citations(text: str, evidence_chunk_ids: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for m in _CITATION_RE.finditer(text):
        i = int(m.group(1)) - 1
        if 0 <= i < len(evidence_chunk_ids):
            cid = evidence_chunk_ids[i]
            if cid not in seen:
                out.append(cid)
                seen.add(cid)
    return out


def build_generator(settings: GeneratorSettings) -> Generator:
    if settings.provider == "stub" or not settings.enabled:
        return StubGenerator()
    if settings.provider == "ollama":
        return OllamaGenerator(
            settings=settings,
            client=OllamaClient(base_url=settings.base_url, timeout_s=settings.timeout_s),
        )
    if settings.provider == "transformers":
        from astro_cs_rag.generation.transformers_generator import TransformersGenerator
        return TransformersGenerator(settings=settings)
    msg = f"unknown generator provider: {settings.provider}"
    raise ValueError(msg)
