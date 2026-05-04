"""Pydantic contracts for YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PathsConfig(BaseModel):
    corpus_path: Path | None = None
    queries_path: Path | None = None
    gold_path: Path | None = None
    output_dir: Path = Field(default_factory=lambda: Path("runs"))


class RetrieverSettings(BaseModel):
    candidate_top_n: int = 50
    bm25_tokenizer: Literal["whitespace"] = "whitespace"
    mode: Literal[
        "fusion_rrf",       # BM25 + dense RRF (default — original P0 path).
        "bm25",
        "dense",
        "late_interaction", # multi-vector MaxSim (BGE-M3 / colbert-style).
        "hierarchical",     # RAPTOR-style collapsed-tree retrieval.
        "splade",           # learned-sparse SPLADE.
        "lockin",           # P4: lock-in coherent paraphrase aggregation.
    ] = "fusion_rrf"
    lockin_n_paraphrases: int = 4
    lockin_use_fixed_pattern_phase: bool = True
    lockin_paraphrase_model: str = "llama3.1:8b-instruct-q4_K_M"
    # Optional pre-cached paraphrases file (one JSON per line: {"query_id", "paraphrases"}).
    # When set, lockin_scores reads paraphrases from disk instead of calling an LLM at runtime.
    lockin_paraphrase_cache_path: Path | None = None


class EmbeddingSettings(BaseModel):
    model_name: str = "BAAI/bge-m3"
    batch_size: int = 32
    use_hash_embedder: bool = False
    cache_dir: Path | None = None


class SideIndexSettings(BaseModel):
    """Build optional retrieval modes alongside BM25+dense."""

    late_interaction: bool = False
    late_interaction_model: str = "BAAI/bge-m3"
    late_interaction_use_hash: bool = False
    hierarchical: bool = False
    hierarchical_branching: int = 6
    hierarchical_max_levels: int = 4
    splade: bool = False
    splade_model: str = "naver/splade-v3"
    splade_use_hash: bool = False


class RerankerSettings(BaseModel):
    enabled: bool = False
    model_name: str = "BAAI/bge-reranker-v2-m3"
    top_n_in: int = 50
    top_n_out: int = 20
    batch_size: int = 32


class GeneratorSettings(BaseModel):
    enabled: bool = False
    provider: Literal["ollama", "stub", "transformers"] = "stub"
    model_name: str = "llama3.1:8b-instruct-q4_K_M"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    max_tokens: int = 512
    seed: int = 0
    timeout_s: float = 120.0
    prompt_style: Literal["citation", "plain", "cot", "few_shot_cot"] = "citation"
    # transformers provider extras: HF model id, dtype, device.
    # When None, falls back to model_name (allows YAML to set model_name once for all providers).
    hf_model_id: str | None = None
    hf_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    hf_device: Literal["auto", "cuda", "cpu"] = "auto"


class DetectorSettings(BaseModel):
    window: int = 10
    threshold: float = 0.0
    background_mode: Literal["tail", "global", "aperture"] = "tail"
    length_normalize_snr: bool = False
    aperture_radius_in: float = 0.10
    aperture_radius_out: float = 0.50


class SelectorSettings(BaseModel):
    token_budget: int = 512
    mode: Literal["greedy", "anti_kt", "mmr", "clean_rag"] = "greedy"
    # CLEAN-RAG residual-aware selection (radio-astronomy CLEAN deconvolution).
    clean_rag_gain: float = 0.7
    clean_rag_residual_floor: float = 0.05
    clean_rag_max_iters: int = 20
    anti_kt_R: float = 1.0
    anti_kt_n_jets: int = -1  # -1 = pack across all jets in relevance order until budget
    # v4 score-gated partner pull-in (only active when n_jets == -2).
    # Pull a jet partner only if score(partner) >= max(score(primary)*alpha, median floor).
    # alpha=0.0 reproduces v3 (all partners). alpha=1.0 = only equal-or-better partners.
    partner_score_gate_alpha: float = 0.0
    partner_use_median_floor: bool = False
    mmr_lambda: float = 0.7


class MetricSettings(BaseModel):
    ks: list[int] = Field(default_factory=lambda: [1, 5, 10])


class IndexConfig(BaseModel):
    dataset: str = "default"
    seed: int = 42
    paths: PathsConfig
    chunk_size: int = 512
    chunk_overlap: int = 64
    retriever: RetrieverSettings = Field(default_factory=RetrieverSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    side_indices: SideIndexSettings = Field(default_factory=SideIndexSettings)

    @model_validator(mode="after")
    def _require_corpus(self) -> IndexConfig:
        if self.paths.corpus_path is None:
            msg = "index config requires paths.corpus_path"
            raise ValueError(msg)
        return self


class RetrieveConfig(BaseModel):
    dataset: str = "default"
    seed: int = 42
    paths: PathsConfig
    index_dir: Path
    retriever: RetrieverSettings = Field(default_factory=RetrieverSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)


class DetectConfig(BaseModel):
    dataset: str = "default"
    seed: int = 42
    run_dir: Path
    index_dir: Path | None = None
    detector: DetectorSettings = Field(default_factory=DetectorSettings)


class SelectConfig(BaseModel):
    dataset: str = "default"
    seed: int = 42
    run_dir: Path
    index_dir: Path
    queries_path: Path | None = None
    selector: SelectorSettings = Field(default_factory=SelectorSettings)


class EvaluateConfig(BaseModel):
    dataset: str = "default"
    seed: int = 42
    paths: PathsConfig
    run_dir: Path
    index_dir: Path
    metrics: MetricSettings = Field(default_factory=MetricSettings)
    score_answers: bool = True

    @model_validator(mode="after")
    def _require_queries(self) -> EvaluateConfig:
        if self.paths.queries_path is None:
            msg = "evaluate requires paths.queries_path"
            raise ValueError(msg)
        return self


class BenchmarkConfig(BaseModel):
    dataset: str = "default"
    seed: int = 42
    paths: PathsConfig
    index_dir: Path | None = None
    chunk_size: int = 512
    chunk_overlap: int = 64
    retriever: RetrieverSettings = Field(default_factory=RetrieverSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    side_indices: SideIndexSettings = Field(default_factory=SideIndexSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    detector: DetectorSettings = Field(default_factory=DetectorSettings)
    selector: SelectorSettings = Field(default_factory=SelectorSettings)
    generator: GeneratorSettings = Field(default_factory=GeneratorSettings)
    metrics: MetricSettings = Field(default_factory=MetricSettings)

    @model_validator(mode="after")
    def _require_paths(self) -> BenchmarkConfig:
        if self.paths.corpus_path is None or self.paths.queries_path is None:
            msg = "benchmark requires paths.corpus_path and paths.queries_path"
            raise ValueError(msg)
        return self


class RerankConfig(BaseModel):
    dataset: str = "default"
    seed: int = 42
    run_dir: Path
    index_dir: Path
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)


class GenerateConfig(BaseModel):
    dataset: str = "default"
    seed: int = 42
    run_dir: Path
    index_dir: Path
    queries_path: Path
    generator: GeneratorSettings = Field(default_factory=GeneratorSettings)


class SanityConfig(BaseModel):
    run_dir: Path


class ErrorAnalysisConfig(BaseModel):
    """Post-hoc error mining for a completed run."""

    dataset: str = "default"
    run_dir: Path
    index_dir: Path
    paths: PathsConfig
    baseline_run_dir: Path | None = None

    @model_validator(mode="after")
    def _require_queries(self) -> ErrorAnalysisConfig:
        if self.paths.queries_path is None:
            msg = "error analysis requires paths.queries_path"
            raise ValueError(msg)
        return self


class AblationVariant(BaseModel):
    name: str
    overrides: dict[str, object] = Field(default_factory=dict)


class AblationConfig(BaseModel):
    """Grid of YAML overrides applied to a base benchmark config."""

    base_config_path: Path
    output_dir: Path = Field(default_factory=lambda: Path("runs/ablations"))
    variants: list[AblationVariant] = Field(default_factory=list)
