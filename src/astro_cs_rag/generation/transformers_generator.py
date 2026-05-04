"""HuggingFace transformers-backed generator.

Loaded lazily — only imports torch/transformers when first used. Reuses
the same prompt schema (`prompts.assemble`) and `GeneratedAnswer` output
shape as the Ollama path so downstream evaluation is identical.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from astro_cs_rag.atoms.schemas import GeneratedAnswer
from astro_cs_rag.config.schema import GeneratorSettings
from astro_cs_rag.generation.prompts import AssembledPrompt, assemble


@dataclass
class TransformersGenerator:
    settings: GeneratorSettings
    provider: str = "transformers"
    _pipe: Any = field(default=None, init=False, repr=False)
    _tokenizer: Any = field(default=None, init=False, repr=False)

    @property
    def _hf_model(self) -> str:
        # hf_model_id takes priority; fall back to model_name so YAML only needs one field.
        return self.settings.hf_model_id or self.settings.model_name

    @property
    def model(self) -> str:
        return self._hf_model

    def _load(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map[self.settings.hf_dtype]
        device_map = "auto" if self.settings.hf_device == "auto" else None
        device = self.settings.hf_device

        tok = AutoTokenizer.from_pretrained(self._hf_model)
        if tok.pad_token_id is None:
            tok.pad_token_id = tok.eos_token_id
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        if device_map is not None:
            kwargs["device_map"] = device_map
        model = AutoModelForCausalLM.from_pretrained(self._hf_model, **kwargs)
        if device_map is None and device != "auto":
            model = model.to(device)
        model.train(False)
        self._tokenizer = tok
        self._pipe = model

    def _format_chat(self, prompt: AssembledPrompt) -> str:
        msgs = [{"role": "system", "content": prompt.system}, {"role": "user", "content": prompt.user}]
        try:
            return self._tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"{prompt.system}\n\n{prompt.user}\n"

    def answer(
        self,
        *,
        query_id: str,
        query_text: str,
        evidence: list[tuple[str, str]],
    ) -> GeneratedAnswer:
        import torch

        self._load()
        prompt = assemble(query=query_text, evidence=evidence, style=self.settings.prompt_style)
        text = self._format_chat(prompt)
        enc = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=8192)
        input_ids = enc["input_ids"].to(self._pipe.device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(self._pipe.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.settings.max_tokens),
            "do_sample": self.settings.temperature > 0.0,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if self.settings.temperature > 0.0:
            gen_kwargs["temperature"] = float(self.settings.temperature)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._pipe.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)
        elapsed = time.perf_counter() - t0
        new_tokens = out[0, input_ids.shape[1]:]
        gen_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        cited = _resolve_citations(gen_text, prompt.evidence_chunk_ids)
        return GeneratedAnswer(
            query_id=query_id,
            answer_text=gen_text,
            cited_chunk_ids=cited,
            selected_chunk_ids=prompt.evidence_chunk_ids,
            prompt_tokens_estimate=int(input_ids.shape[1]),
            completion_tokens_estimate=int(new_tokens.shape[0]),
            latency_seconds=float(elapsed),
            provider=self.provider,
            model=self._hf_model,
            metadata={"temperature": self.settings.temperature, "seed": self.settings.seed},
        )


_CITATION_RE = re.compile(r"\[E(\d+)\]")


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
return out
