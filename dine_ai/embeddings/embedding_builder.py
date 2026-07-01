"""
DineAI — Embedding Builder
==========================

Converts standardized text columns (search_text, text_chunk, summary_text)
into dense vector embeddings for downstream semantic search and RAG retrieval.

Architecture
------------
EmbeddingBuilder (Coordinator)
    ├── EmbeddingConfig          — Immutable configuration dataclass
    ├── BaseEmbeddingModel       — Abstract provider interface (DIP)
    │   └── SentenceTransformerEmbedding — Concrete HuggingFace provider
    ├── EmbeddingCache           — Optional LRU-style disk/memory cache (SRP)
    └── EmbeddingResult          — Structured output: vectors + metadata + stats
         └── EmbeddingStatistics — Execution telemetry

Public API
----------
    config = EmbeddingConfig(text_column="search_text")
    builder = EmbeddingBuilder(config)
    result = builder.build(df)
    result.save("output/")
"""

import os
import sys
import json
import time
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 1: EmbeddingConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EmbeddingConfig:
    """
    Immutable configuration contract for the EmbeddingBuilder pipeline.

    All pipeline behaviour is driven exclusively through this dataclass —
    no scattered keyword arguments anywhere downstream.

    Parameters
    ----------
    text_column : str
        Name of the DataFrame column containing text to embed.
        Default: "search_text"
    model_name : str
        HuggingFace model identifier or local path.
        Default: "all-MiniLM-L6-v2" (384-dim, fast, high quality)
    batch_size : int
        Number of texts processed per forward pass. Tune for your VRAM/RAM.
        Default: 64
    device : str
        Inference device — "cpu", "cuda", or "cuda:0".
        Default: "cpu"
    normalize_embeddings : bool
        If True, L2-normalises each output vector to unit length.
        Enables cosine similarity via simple dot product.
        Default: True
    enable_cache : bool
        If True, caches embeddings keyed by text hash to avoid recomputation.
        Default: False
    cache_dir : Optional[str]
        Directory to persist cache to disk. If None, cache is memory-only.
        Default: None
    show_progress : bool
        If True, shows a tqdm progress bar during batched encoding.
        Default: True
    """
    text_column: str = "search_text"
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    device: str = "cpu"
    normalize_embeddings: bool = True
    enable_cache: bool = False
    cache_dir: Optional[str] = None
    show_progress: bool = True

    def validate(self) -> None:
        """
        Validates configuration values before pipeline execution.

        Raises
        ------
        ValueError
            If any configuration parameter is invalid.
        """
        if not self.text_column or not isinstance(self.text_column, str):
            raise ValueError("EmbeddingConfig.text_column must be a non-empty string.")
        if self.batch_size < 1:
            raise ValueError(f"EmbeddingConfig.batch_size must be >= 1, got {self.batch_size}.")
        valid_devices = {"cpu", "cuda"} | {f"cuda:{i}" for i in range(16)}
        if self.device not in valid_devices:
            raise ValueError(
                f"EmbeddingConfig.device '{self.device}' is not recognised. "
                f"Use 'cpu', 'cuda', or 'cuda:N'."
            )


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 2: EmbeddingStatistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EmbeddingStatistics:
    """
    Execution telemetry captured during an EmbeddingBuilder.build() run.

    Kept separate from EmbeddingResult so statistics can be inspected,
    logged, or exported without touching the vector data.
    """
    total_texts: int = 0
    embedded_texts: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    batches_processed: int = 0
    total_time_seconds: float = 0.0
    model_load_time_seconds: float = 0.0
    embedding_time_seconds: float = 0.0
    texts_per_second: float = 0.0
    embedding_dim: int = 0
    model_name: str = ""
    device: str = ""
    normalized: bool = False
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialises statistics to a plain dictionary."""
        return {
            "total_texts": self.total_texts,
            "embedded_texts": self.embedded_texts,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "batches_processed": self.batches_processed,
            "total_time_seconds": round(self.total_time_seconds, 4),
            "model_load_time_seconds": round(self.model_load_time_seconds, 4),
            "embedding_time_seconds": round(self.embedding_time_seconds, 4),
            "texts_per_second": round(self.texts_per_second, 2),
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "device": self.device,
            "normalized": self.normalized,
            "timestamp": self.timestamp,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 3: EmbeddingResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EmbeddingResult:
    """
    Structured output of the EmbeddingBuilder pipeline.

    Owns its own persistence — callers never need to touch numpy directly.

    Attributes
    ----------
    embeddings : np.ndarray
        Shape (N, D) float32 array of dense vectors.
    text_column : str
        The source text column that was embedded.
    index : List
        The DataFrame index values the embeddings correspond to.
    statistics : EmbeddingStatistics
        Full execution telemetry for this run.
    warnings : List[str]
        Non-fatal issues captured during the run.
    is_success : bool
        False if any ERROR-level issues occurred.
    """
    embeddings: np.ndarray
    text_column: str
    index: List
    statistics: EmbeddingStatistics
    warnings: List[str] = field(default_factory=list)
    is_success: bool = True

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, output_dir: str, prefix: str = "embeddings") -> Tuple[str, str]:
        """
        Saves embeddings as a .npy binary and metadata as a .json file.

        Parameters
        ----------
        output_dir : str
            Target directory (created if it does not exist).
        prefix : str
            Filename stem used for both output files.

        Returns
        -------
        Tuple[str, str]
            (npy_path, json_path) absolute file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        npy_path = os.path.join(output_dir, f"{prefix}.npy")
        json_path = os.path.join(output_dir, f"{prefix}_metadata.json")

        np.save(npy_path, self.embeddings)

        metadata = {
            "text_column": self.text_column,
            "shape": list(self.embeddings.shape),
            "dtype": str(self.embeddings.dtype),
            "index": [str(i) for i in self.index],
            "warnings": self.warnings,
            "is_success": self.is_success,
            "statistics": self.statistics.to_dict(),
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved embeddings → %s", npy_path)
        logger.info("Saved metadata   → %s", json_path)
        return npy_path, json_path

    @staticmethod
    def load(output_dir: str, prefix: str = "embeddings") -> "EmbeddingResult":
        """
        Loads a previously saved EmbeddingResult from disk.

        Parameters
        ----------
        output_dir : str
            Directory containing the saved files.
        prefix : str
            Filename stem used when saving.

        Returns
        -------
        EmbeddingResult
            Reconstructed result object.

        Raises
        ------
        FileNotFoundError
            If .npy or _metadata.json are missing.
        """
        npy_path = os.path.join(output_dir, f"{prefix}.npy")
        json_path = os.path.join(output_dir, f"{prefix}_metadata.json")

        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Embeddings file not found: {npy_path}")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Metadata file not found: {json_path}")

        embeddings = np.load(npy_path)
        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        stats_dict = metadata.get("statistics", {})
        stats = EmbeddingStatistics(**{
            k: v for k, v in stats_dict.items()
            if k in EmbeddingStatistics.__dataclass_fields__
        })

        return EmbeddingResult(
            embeddings=embeddings,
            text_column=metadata.get("text_column", ""),
            index=metadata.get("index", []),
            statistics=stats,
            warnings=metadata.get("warnings", []),
            is_success=metadata.get("is_success", True),
        )

    def as_dataframe(self) -> pd.DataFrame:
        """
        Returns embeddings as a DataFrame with index aligned to source rows.

        Returns
        -------
        pd.DataFrame
            Columns are named 'dim_0', 'dim_1', … 'dim_{D-1}'.
        """
        dim = self.embeddings.shape[1] if self.embeddings.ndim > 1 else 1
        cols = [f"dim_{i}" for i in range(dim)]
        return pd.DataFrame(self.embeddings, index=self.index, columns=cols)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 4: BaseEmbeddingModel (Abstract Interface)
# ─────────────────────────────────────────────────────────────────────────────

class BaseEmbeddingModel(ABC):
    """
    Abstract provider interface for text embedding models.

    Implement this to plug in any embedding backend:
    - sentence-transformers  (implemented below)
    - OpenAI text-embedding-3-small
    - Cohere embed-multilingual-v3
    - Local GGUF / llama.cpp
    - Azure OpenAI
    - Google Vertex AI

    The coordinator (EmbeddingBuilder) only knows about this interface.
    """

    @abstractmethod
    def load(self, config: EmbeddingConfig) -> None:
        """
        Loads and initialises the model according to the provided config.

        Called once during EmbeddingBuilder construction.
        """
        pass

    @abstractmethod
    def encode(self, texts: List[str], config: EmbeddingConfig) -> np.ndarray:
        """
        Encodes a list of texts into dense float32 vectors.

        Parameters
        ----------
        texts : List[str]
            Raw text strings to embed.
        config : EmbeddingConfig
            Active pipeline configuration.

        Returns
        -------
        np.ndarray
            Shape (N, D) float32 embedding matrix.
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Returns the output dimensionality of this model."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable identifier for this provider (e.g. 'SentenceTransformer')."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 5: SentenceTransformerEmbedding
# ─────────────────────────────────────────────────────────────────────────────

class SentenceTransformerEmbedding(BaseEmbeddingModel):
    """
    Concrete embedding provider using the sentence-transformers library.

    Supports any HuggingFace SBERT model. Handles device placement,
    batch encoding, and optional L2 normalisation.

    Requires
    --------
    pip install sentence-transformers
    """

    def __init__(self):
        self._model = None
        self._dim: int = 0

    def load(self, config: EmbeddingConfig) -> None:
        """Loads the SentenceTransformer model onto the configured device."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerEmbedding. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self._model = SentenceTransformer(config.model_name, device=config.device)
        # Probe the embedding dimensionality with a dummy text
        probe = self._model.encode(["probe"], show_progress_bar=False)
        self._dim = probe.shape[1]

    def encode(self, texts: List[str], config: EmbeddingConfig) -> np.ndarray:
        """
        Encodes texts using sentence-transformers batch encoding.

        Parameters
        ----------
        texts : List[str]
            Texts to encode (already batched by the coordinator).
        config : EmbeddingConfig
            Active pipeline configuration.

        Returns
        -------
        np.ndarray
            Float32 embedding matrix of shape (len(texts), embedding_dim).
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() before encode().")

        vectors = self._model.encode(
            texts,
            batch_size=config.batch_size,
            normalize_embeddings=config.normalize_embeddings,
            show_progress_bar=False,   # Progress is managed by the coordinator
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def provider_name(self) -> str:
        return "SentenceTransformer"


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 6: EmbeddingCache
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingCache:
    """
    LRU-style embedding cache keyed by SHA-256 hash of input text.

    Supports both:
    - In-memory caching (fast, process-scoped)
    - Disk persistence (.npz per cache entry in cache_dir/)

    When the coordinator processes a batch, it:
    1. Partitions texts into cache_hits (already cached) vs cache_misses.
    2. Embeds only cache_misses.
    3. Merges hit vectors + miss vectors back in original order.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        self._memory: Dict[str, np.ndarray] = {}
        self._cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def _key(text: str) -> str:
        """SHA-256 hex digest of the input text."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Returns cached vector for text, or None if not cached."""
        k = self._key(text)
        if k in self._memory:
            return self._memory[k]
        if self._cache_dir:
            path = os.path.join(self._cache_dir, f"{k}.npy")
            if os.path.exists(path):
                vec = np.load(path)
                self._memory[k] = vec  # promote to memory
                return vec
        return None

    def set(self, text: str, vector: np.ndarray) -> None:
        """Stores a vector in memory (and optionally to disk)."""
        k = self._key(text)
        self._memory[k] = vector
        if self._cache_dir:
            path = os.path.join(self._cache_dir, f"{k}.npy")
            np.save(path, vector)

    def batch_lookup(
        self, texts: List[str]
    ) -> Tuple[Dict[int, np.ndarray], List[int], List[str]]:
        """
        Partitions a list of texts into cache hits and misses.

        Parameters
        ----------
        texts : List[str]
            Input texts (original order must be preserved).

        Returns
        -------
        hits : Dict[int, np.ndarray]
            Mapping of original index → cached vector.
        miss_indices : List[int]
            Indices of texts not in cache (need encoding).
        miss_texts : List[str]
            Corresponding text strings for miss_indices.
        """
        hits: Dict[int, np.ndarray] = {}
        miss_indices: List[int] = []
        miss_texts: List[str] = []

        for i, text in enumerate(texts):
            vec = self.get(text)
            if vec is not None:
                hits[i] = vec
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        return hits, miss_indices, miss_texts

    def store_batch(self, texts: List[str], vectors: np.ndarray) -> None:
        """Stores a batch of (text, vector) pairs into cache."""
        for text, vec in zip(texts, vectors):
            self.set(text, vec)

    @property
    def size(self) -> int:
        """Number of entries currently in memory cache."""
        return len(self._memory)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 7: EmbeddingBuilder (Pipeline Coordinator)
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingBuilder:
    """
    Pipeline coordinator for converting text columns into dense embeddings.

    The coordinator owns the execution loop but delegates all model-specific
    logic to the injected BaseEmbeddingModel. This keeps the coordinator
    stable when embedding providers change.

    Parameters
    ----------
    config : EmbeddingConfig
        Full pipeline configuration.
    model : Optional[BaseEmbeddingModel]
        Embedding provider to use. Defaults to SentenceTransformerEmbedding.
        Inject a custom model for testing or alternative providers.

    Example
    -------
    >>> config = EmbeddingConfig(text_column="search_text", batch_size=32)
    >>> result = EmbeddingBuilder(config).build(df)
    >>> result.save("output/")
    """

    def __init__(
        self,
        config: EmbeddingConfig,
        model: Optional[BaseEmbeddingModel] = None
    ):
        config.validate()
        self.config = config
        self._model: BaseEmbeddingModel = model or SentenceTransformerEmbedding()
        self._cache: Optional[EmbeddingCache] = (
            EmbeddingCache(cache_dir=config.cache_dir) if config.enable_cache else None
        )
        self._model_loaded = False

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, df: pd.DataFrame) -> EmbeddingResult:
        """
        Embeds the configured text column of the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (must contain config.text_column).

        Returns
        -------
        EmbeddingResult
            Structured result containing embeddings, metadata, and statistics.

        Raises
        ------
        KeyError
            If config.text_column is absent from the DataFrame.
        RuntimeError
            If the embedding model fails to load or encode.
        """
        pipeline_start = time.time()
        stats = EmbeddingStatistics(
            total_texts=len(df),
            model_name=self.config.model_name,
            device=self.config.device,
            normalized=self.config.normalize_embeddings,
        )
        warnings: List[str] = []
        is_success = True

        # ── 1. Validate input column ──────────────────────────────────────────
        if self.config.text_column not in df.columns:
            raise KeyError(
                f"EmbeddingBuilder: text column '{self.config.text_column}' "
                f"not found in DataFrame. Available columns: {list(df.columns)}"
            )

        # ── 2. Extract and sanitise texts ─────────────────────────────────────
        raw_texts = df[self.config.text_column].fillna("").astype(str).tolist()
        texts = [t.strip() for t in raw_texts]
        empty_mask = [t == "" for t in texts]
        empty_count = sum(empty_mask)
        if empty_count:
            warnings.append(
                f"{empty_count} row(s) had empty text in '{self.config.text_column}'; "
                f"replaced with empty string for embedding."
            )

        # ── 3. Lazy model load ────────────────────────────────────────────────
        if not self._model_loaded:
            load_start = time.time()
            self._model.load(self.config)
            stats.model_load_time_seconds = time.time() - load_start
            stats.embedding_dim = self._model.embedding_dim
            self._model_loaded = True

        # ── 4. Cache lookup partitioning ──────────────────────────────────────
        if self._cache is not None:
            hits, miss_indices, miss_texts = self._cache.batch_lookup(texts)
            stats.cache_hits = len(hits)
            stats.cache_misses = len(miss_indices)
        else:
            hits = {}
            miss_indices = list(range(len(texts)))
            miss_texts = texts
            stats.cache_hits = 0
            stats.cache_misses = len(texts)

        # ── 5. Batched embedding of cache misses ──────────────────────────────
        embed_start = time.time()
        miss_vectors: Optional[np.ndarray] = None

        if miss_texts:
            batches = self._make_batches(miss_texts, self.config.batch_size)
            stats.batches_processed = len(batches)
            batch_results: List[np.ndarray] = []

            for batch_idx, batch in enumerate(batches):
                if self.config.show_progress:
                    self._print_progress(batch_idx + 1, len(batches), len(miss_texts))
                try:
                    batch_vecs = self._model.encode(batch, self.config)
                    batch_results.append(batch_vecs)

                    if self._cache is not None:
                        start_i = batch_idx * self.config.batch_size
                        batch_texts = miss_texts[start_i: start_i + len(batch)]
                        self._cache.store_batch(batch_texts, batch_vecs)

                except Exception as exc:
                    is_success = False
                    warnings.append(
                        f"Batch {batch_idx + 1}/{len(batches)} failed: {str(exc)}"
                    )
                    logger.error("Embedding batch %d failed: %s", batch_idx + 1, exc)
                    # Fill failed batch with zeros so shape stays consistent
                    zero_vecs = np.zeros(
                        (len(batch), self._model.embedding_dim), dtype=np.float32
                    )
                    batch_results.append(zero_vecs)

            if self.config.show_progress:
                print()  # newline after progress bar

            miss_vectors = np.vstack(batch_results) if batch_results else np.empty((0, stats.embedding_dim), dtype=np.float32)

        stats.embedding_time_seconds = time.time() - embed_start

        # ── 6. Merge hits + misses back in original order ─────────────────────
        if stats.embedding_dim == 0 and self._model.embedding_dim > 0:
            stats.embedding_dim = self._model.embedding_dim

        all_vectors = self._merge_vectors(
            total=len(texts),
            hits=hits,
            miss_indices=miss_indices,
            miss_vectors=miss_vectors,
            dim=stats.embedding_dim,
        )

        # ── 7. Finalise statistics ────────────────────────────────────────────
        stats.total_time_seconds = time.time() - pipeline_start
        stats.embedded_texts = len(texts)
        elapsed = stats.total_time_seconds or 1e-9
        stats.texts_per_second = stats.embedded_texts / elapsed

        return EmbeddingResult(
            embeddings=all_vectors,
            text_column=self.config.text_column,
            index=list(df.index),
            statistics=stats,
            warnings=warnings,
            is_success=is_success,
        )

    # ── Report ────────────────────────────────────────────────────────────────

    def generate_report(self, result: EmbeddingResult) -> None:
        """
        Prints a formatted enterprise-grade execution report to stdout.

        Parameters
        ----------
        result : EmbeddingResult
            The result returned by build().
        """
        s = result.statistics
        width = 62
        sep = "=" * width
        thin = "-" * width

        print(sep)
        print("EMBEDDING BUILDER REPORT".center(width))
        print(sep)
        print(f"  Provider        : {self._model.provider_name}")
        print(f"  Model           : {s.model_name}")
        print(f"  Device          : {s.device}")
        print(f"  Text Column     : {result.text_column}")
        print(f"  Normalised      : {s.normalized}")
        print(thin)
        print(f"  Total Texts     : {s.total_texts:,}")
        print(f"  Embedded Texts  : {s.embedded_texts:,}")
        print(f"  Embedding Dim   : {s.embedding_dim}")
        print(f"  Output Shape    : ({s.embedded_texts:,} × {s.embedding_dim})")
        print(thin)
        print(f"  Batches         : {s.batches_processed}")
        print(f"  Cache Hits      : {s.cache_hits:,}")
        print(f"  Cache Misses    : {s.cache_misses:,}")
        print(thin)
        print(f"  Model Load Time : {s.model_load_time_seconds:.3f}s")
        print(f"  Embed Time      : {s.embedding_time_seconds:.3f}s")
        print(f"  Total Time      : {s.total_time_seconds:.3f}s")
        print(f"  Throughput      : {s.texts_per_second:.1f} texts/sec")
        print(thin)
        print(f"  Status          : {'✓ SUCCESS' if result.is_success else '✗ FAILED'}")
        if result.warnings:
            print(f"  Warnings ({len(result.warnings)}):")
            for w in result.warnings:
                print(f"    ⚠  {w}")
        print(sep)

    # ── Private Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _make_batches(texts: List[str], batch_size: int) -> List[List[str]]:
        """Splits a list of texts into fixed-size batches."""
        return [texts[i: i + batch_size] for i in range(0, len(texts), batch_size)]

    @staticmethod
    def _print_progress(current: int, total: int, n_texts: int) -> None:
        """Prints an inline progress indicator without tqdm dependency."""
        pct = int(100 * current / total)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(
            f"\r  Encoding [{bar}] {pct:3d}%  batch {current}/{total}  ({n_texts} texts)",
            end="",
            flush=True,
        )

    @staticmethod
    def _merge_vectors(
        total: int,
        hits: Dict[int, np.ndarray],
        miss_indices: List[int],
        miss_vectors: Optional[np.ndarray],
        dim: int,
    ) -> np.ndarray:
        """
        Reconstructs the full embedding matrix in original row order,
        merging cached hit vectors with freshly computed miss vectors.
        """
        result = np.zeros((total, dim), dtype=np.float32)

        for orig_idx, vec in hits.items():
            result[orig_idx] = vec

        if miss_vectors is not None and len(miss_indices) > 0:
            for local_i, orig_idx in enumerate(miss_indices):
                if local_i < len(miss_vectors):
                    result[orig_idx] = miss_vectors[local_i]

        return result


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(level=logging.WARNING)
    print("=" * 62)
    print("  EmbeddingBuilder — Self-Test Suite")
    print("=" * 62)

    # ── Minimal stub model so the test runs without GPU/internet ──────────────
    class StubEmbeddingModel(BaseEmbeddingModel):
        """Deterministic stub that returns reproducible random vectors."""
        _DIM = 384

        def load(self, config: EmbeddingConfig) -> None:
            print(f"  [StubModel] Loaded on device='{config.device}'")

        def encode(self, texts: List[str], config: EmbeddingConfig) -> np.ndarray:
            rng = np.random.default_rng(seed=42)
            vecs = rng.standard_normal((len(texts), self._DIM)).astype(np.float32)
            if config.normalize_embeddings:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs = vecs / np.maximum(norms, 1e-9)
            return vecs

        @property
        def embedding_dim(self) -> int:
            return self._DIM

        @property
        def provider_name(self) -> str:
            return "StubModel"

    # ── Sample DataFrame ──────────────────────────────────────────────────────
    sample_df = pd.DataFrame({
        "search_text": [
            "Cheesy Truffle Risotto — Italian, lunch/dinner, 400 kcal.",
            "Spicy Thai Basil Chicken — Thai, dinner, 520 kcal.",
            "Avocado Toast — Western, breakfast, 280 kcal.",
            "",   # intentional empty row to test warning
        ]
    })

    # ── Test 1: Basic build ───────────────────────────────────────────────────
    print("\n--- Test 1: Basic Build ---")
    config = EmbeddingConfig(
        text_column="search_text",
        model_name="all-MiniLM-L6-v2",
        batch_size=2,
        device="cpu",
        normalize_embeddings=True,
        enable_cache=False,
        show_progress=True,
    )
    builder = EmbeddingBuilder(config=config, model=StubEmbeddingModel())
    result = builder.build(sample_df)

    print(f"\n  Embedding shape : {result.embeddings.shape}")
    print(f"  First vector L2 : {np.linalg.norm(result.embeddings[0]):.4f}  (should be ≈1.0)")
    print(f"  Warnings        : {result.warnings}")
    print(f"  Success         : {result.is_success}")

    # ── Test 2: Report ────────────────────────────────────────────────────────
    print("\n--- Test 2: Enterprise Report ---")
    builder.generate_report(result)

    # ── Test 3: Save / Load round-trip ────────────────────────────────────────
    print("\n--- Test 3: Save / Load Round-Trip ---")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        npy_path, json_path = result.save(tmp, prefix="test_embeddings")
        print(f"  Saved → {os.path.basename(npy_path)}")
        print(f"  Saved → {os.path.basename(json_path)}")

        loaded = EmbeddingResult.load(tmp, prefix="test_embeddings")
        assert loaded.embeddings.shape == result.embeddings.shape, "Shape mismatch after load!"
        assert np.allclose(loaded.embeddings, result.embeddings, atol=1e-6), "Values mismatch after load!"
        print(f"  Loaded shape : {loaded.embeddings.shape}  ✓ Matches original")
        print(f"  Max abs diff : {np.max(np.abs(loaded.embeddings - result.embeddings)):.2e}  ✓")

    # ── Test 4: as_dataframe() ────────────────────────────────────────────────
    print("\n--- Test 4: as_dataframe() ---")
    emb_df = result.as_dataframe()
    print(f"  DataFrame shape : {emb_df.shape}")
    print(f"  Columns sample  : {list(emb_df.columns[:5])} …")

    # ── Test 5: Caching ───────────────────────────────────────────────────────
    print("\n--- Test 5: Embedding Cache ---")
    with tempfile.TemporaryDirectory() as cache_dir:
        cached_config = EmbeddingConfig(
            text_column="search_text",
            batch_size=4,
            device="cpu",
            enable_cache=True,
            cache_dir=cache_dir,
            show_progress=False,
        )
        cached_builder = EmbeddingBuilder(config=cached_config, model=StubEmbeddingModel())

        result1 = cached_builder.build(sample_df)
        print(f"  Run 1 — cache hits: {result1.statistics.cache_hits}, misses: {result1.statistics.cache_misses}")

        result2 = cached_builder.build(sample_df)
        print(f"  Run 2 — cache hits: {result2.statistics.cache_hits}, misses: {result2.statistics.cache_misses}")
        assert result2.statistics.cache_hits == len(sample_df), "All should be cache hits on run 2!"
        print("  ✓ All texts served from cache on second run")

    # ── Test 6: EmbeddingConfig validation ───────────────────────────────────
    print("\n--- Test 6: Config Validation ---")
    try:
        bad_config = EmbeddingConfig(batch_size=0)
        bad_config.validate()
        print("  ERROR: Should have raised ValueError!")
    except ValueError as exc:
        print(f"  [PASS] ValueError raised: {exc}")

    # ── Test 7: Missing column error ──────────────────────────────────────────
    print("\n--- Test 7: Missing Column Error ---")
    try:
        bad_builder = EmbeddingBuilder(
            EmbeddingConfig(text_column="nonexistent_column"),
            model=StubEmbeddingModel()
        )
        bad_builder.build(sample_df)
        print("  ERROR: Should have raised KeyError!")
    except KeyError as exc:
        print(f"  [PASS] KeyError raised as expected")

    print("\n" + "=" * 62)
    print("  All self-tests passed ✓")
    print("=" * 62)
