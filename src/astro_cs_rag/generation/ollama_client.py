"""Minimal Ollama HTTP client with deterministic settings logged in artifacts.

We deliberately avoid the `ollama` Python SDK: it adds an extra wrapper
and pins behavior we cannot easily inspect. The HTTP API is small and stable.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OllamaResponse:
    text: str
    prompt_eval_count: int
    eval_count: int
    total_duration_ns: int
    model: str
    raw: dict[str, Any]


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float = 0.0,
        seed: int = 0,
        num_predict: int = 512,
        system: str | None = None,
    ) -> OllamaResponse:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "seed": int(seed),
                "num_predict": int(num_predict),
            },
        }
        if system is not None:
            payload["system"] = system

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.URLError as exc:  # pragma: no cover - environment dependent
            msg = f"Ollama request failed against {self.base_url}: {exc}"
            raise RuntimeError(msg) from exc

        raw = json.loads(body)
        return OllamaResponse(
            text=str(raw.get("response", "")),
            prompt_eval_count=int(raw.get("prompt_eval_count", 0) or 0),
            eval_count=int(raw.get("eval_count", 0) or 0),
            total_duration_ns=int(raw.get("total_duration", 0) or 0),
            model=str(raw.get("model", model)),
            raw=raw,
        )

    def show(self, model: str) -> dict[str, Any]:
        """Return Ollama's model card — used to record digest in run manifest."""
        req = urllib.request.Request(
            f"{self.base_url}/api/show",
            data=json.dumps({"name": model}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:  # pragma: no cover - environment dependent
            return {"model": model, "error": str(exc)}


def health_check(base_url: str = "http://localhost:11434", timeout_s: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout_s) as resp:
            resp.read()
            return True
    except Exception:  # pragma: no cover - environment dependent
        return False


def now_seconds() -> float:
    return time.perf_counter()
