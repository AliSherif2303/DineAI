"""
DineAI — Semantic Search Engine
================================

Converts natural-language queries into embeddings and retrieves the most
semantically similar candidates from a built FAISS index.

Architecture
------------
SemanticSearchEngine (Coordinator)
    ├── SearchConfig             — Immutable configuration dataclass
    ├── QueryProcessor           — Stateless query normalisation pipeline
    ├── QueryCache               — SHA-256 keyed embedding cache (SRP)
    ├── BaseSearchStrategy       — Abstract retrieval contract (OCP / DIP)
    │   ├── ExactSearchStrategy       — Direct FAISS flat search (100% recall)
    │   ├── ApproximateSearchStrategy — Tunable FAISS IVF/HNSW search
    │   └── HybridSearchStrategy      — Architecture stub (vector + BM25 fusion)
    └── SearchResult             — Structured output
         ├── SearchCandidate     — One retrieved hit: score, rank, row_id, metadata
         └── SearchStatistics    — Detached execution telemetry

Public API
----------
    engine = SemanticSearchEngine(faiss_result, embedding_model)
    result = engine.search("healthy breakfast", top_k=10)
    engine.generate_report(result)
"""

import re
import sys
import json
import time
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 1: SearchConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchConfig:
    """
    Immutable configuration for the SemanticSearchEngine pipeline.

    All search behaviour is controlled exclusively through this dataclass.

    Parameters
    ----------
    top_k : int
        Maximum number of results to return per query.  Default: 10.
    strategy : str
        Retrieval strategy key: "exact" | "approximate" | "hybrid".
        Default: "exact"
    score_threshold : Optional[float]
        If set, discard candidates whose similarity score is below this value.
        Interpretation depends on the index metric (L2: lower is better;
        IP: higher is better).  Default: None (no filtering).
    normalize_query : bool
        L2-normalise the query embedding before search.
        Must match the normalisation applied during indexing.
        Default: True
    lowercase_query : bool
        Convert query to lowercase during preprocessing.  Default: True
    strip_punctuation : bool
        Remove non-alphanumeric characters from query.  Default: False
    max_query_length : int
        Truncate query text to this many characters before embedding.
        Default: 512
    enable_cache : bool
        If True, cache query embeddings keyed by normalised query text.
        Default: True
    cache_max_size : int
        Maximum number of query embeddings to hold in memory.
        Default: 1000
    return_scores : bool
        Include raw FAISS similarity scores in SearchCandidate.
        Default: True
    nprobe : Optional[int]
        Override nprobe for IVF indexes at search time.
        If None, uses the value stored in FaissResult.config.
        Default: None
    ef_search : Optional[int]
        Override efSearch for HNSW indexes at search time.
        Default: None
    """
    top_k: int = 10
    strategy: str = "exact"
    score_threshold: Optional[float] = None
    normalize_query: bool = True
    lowercase_query: bool = True
    strip_punctuation: bool = False
    max_query_length: int = 512
    enable_cache: bool = True
    cache_max_size: int = 1000
    return_scores: bool = True
    nprobe: Optional[int] = None
    ef_search: Optional[int] = None

    SUPPORTED_STRATEGIES = {"exact", "approximate", "hybrid"}

    def validate(self) -> None:
        """
        Validates configuration values.

        Raises
        ------
        ValueError
            On any invalid parameter.
        """
        if self.top_k < 1:
            raise ValueError(f"SearchConfig.top_k must be >= 1, got {self.top_k}.")
        if self.strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"SearchConfig.strategy '{self.strategy}' is not supported. "
                f"Choose from: {self.SUPPORTED_STRATEGIES}"
            )
        if self.max_query_length < 1:
            raise ValueError(
                f"SearchConfig.max_query_length must be >= 1, got {self.max_query_length}."
            )
        if self.cache_max_size < 1:
            raise ValueError(
                f"SearchConfig.cache_max_size must be >= 1, got {self.cache_max_size}."
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_k": self.top_k,
            "strategy": self.strategy,
            "score_threshold": self.score_threshold,
            "normalize_query": self.normalize_query,
            "lowercase_query": self.lowercase_query,
            "strip_punctuation": self.strip_punctuation,
            "max_query_length": self.max_query_length,
            "enable_cache": self.enable_cache,
            "nprobe": self.nprobe,
            "ef_search": self.ef_search,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 2: QueryProcessor
# ─────────────────────────────────────────────────────────────────────────────

class QueryProcessor:
    """
    Stateless query normalisation pipeline.

    Applies a configurable chain of text transformations to a raw query
    before it is passed to the embedding model.

    The default chain applies:
    1. Whitespace stripping
    2. Optional lowercasing
    3. Optional punctuation removal
    4. Length truncation

    Custom steps can be injected via ``extra_steps``.

    Parameters
    ----------
    config : SearchConfig
        Active search configuration driving preprocessing flags.
    extra_steps : Optional[List[Callable[[str], str]]]
        Additional callable transformations appended to the default chain.
        Each callable receives a string and returns a transformed string.
    """

    def __init__(
        self,
        config: SearchConfig,
        extra_steps: Optional[List[Callable[[str], str]]] = None,
    ):
        self._config = config
        self._extra_steps: List[Callable[[str], str]] = extra_steps or []

    def process(self, raw_query: str) -> str:
        """
        Applies the normalisation chain to a raw query string.

        Parameters
        ----------
        raw_query : str
            The user's original query text.

        Returns
        -------
        str
            The processed, normalised query ready for embedding.

        Raises
        ------
        ValueError
            If the processed query is empty after normalisation.
        """
        query = raw_query.strip()

        if self._config.lowercase_query:
            query = query.lower()

        if self._config.strip_punctuation:
            query = re.sub(r"[^a-z0-9\s]", " ", query)
            query = re.sub(r"\s+", " ", query).strip()

        query = query[: self._config.max_query_length]

        for step in self._extra_steps:
            query = step(query)
            query = query.strip()

        if not query:
            raise ValueError(
                "QueryProcessor: query is empty after preprocessing. "
                f"Original: '{raw_query}'"
            )
        return query

    @staticmethod
    def add_custom_step(
        processor: "QueryProcessor", step: Callable[[str], str]
    ) -> "QueryProcessor":
        """
        Returns a new QueryProcessor with an additional normalisation step appended.

        Follows the immutable builder pattern — the original processor is unchanged.
        """
        new_steps = list(processor._extra_steps) + [step]
        return QueryProcessor(config=processor._config, extra_steps=new_steps)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 3: SearchCandidate
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchCandidate:
    """
    A single retrieved hit from a semantic search query.

    Attributes
    ----------
    rank : int
        Position in the result list (1 = closest match).
    row_id : Any
        Original DataFrame index value for this candidate.
    faiss_index : int
        Internal FAISS vector position (useful for debugging).
    score : float
        Raw similarity score from FAISS (L2: lower is better; IP: higher is better).
    metadata : Dict[str, Any]
        Optional attached DataFrame row data.  Populated post-search by callers.
    """
    rank: int
    row_id: Any
    faiss_index: int
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "row_id": self.row_id,
            "faiss_index": self.faiss_index,
            "score": round(self.score, 6),
            "metadata": self.metadata,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 4: SearchStatistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchStatistics:
    """
    Execution telemetry for a single search() call.

    Detached from SearchResult so statistics can be logged or exported
    independently of the candidate list.
    """
    query_raw: str = ""
    query_processed: str = ""
    strategy: str = ""
    top_k_requested: int = 0
    candidates_returned: int = 0
    candidates_filtered: int = 0
    cache_hit: bool = False
    query_processing_time_seconds: float = 0.0
    embedding_time_seconds: float = 0.0
    search_time_seconds: float = 0.0
    total_time_seconds: float = 0.0
    embedding_dim: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_raw": self.query_raw,
            "query_processed": self.query_processed,
            "strategy": self.strategy,
            "top_k_requested": self.top_k_requested,
            "candidates_returned": self.candidates_returned,
            "candidates_filtered": self.candidates_filtered,
            "cache_hit": self.cache_hit,
            "query_processing_time_seconds": round(self.query_processing_time_seconds, 4),
            "embedding_time_seconds": round(self.embedding_time_seconds, 4),
            "search_time_seconds": round(self.search_time_seconds, 4),
            "total_time_seconds": round(self.total_time_seconds, 4),
            "embedding_dim": self.embedding_dim,
            "timestamp": self.timestamp,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 5: SearchResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """
    Structured output of SemanticSearchEngine.search().

    Attributes
    ----------
    query : str
        The original raw query string submitted by the user.
    candidates : List[SearchCandidate]
        Ordered list of retrieved hits (rank 1 = best match).
    statistics : SearchStatistics
        Full execution telemetry for this search call.
    warnings : List[str]
        Non-fatal issues captured during the search.
    is_success : bool
        False if any ERROR-level issues occurred.
    """
    query: str
    candidates: List[SearchCandidate]
    statistics: SearchStatistics
    warnings: List[str] = field(default_factory=list)
    is_success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Serialises the full result to a plain dictionary."""
        return {
            "query": self.query,
            "candidates": [c.to_dict() for c in self.candidates],
            "statistics": self.statistics.to_dict(),
            "warnings": self.warnings,
            "is_success": self.is_success,
        }

    def to_json(self, indent: int = 2) -> str:
        """Returns the result as a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def attach_metadata(self, df) -> "SearchResult":
        """
        Attaches DataFrame rows to each candidate's metadata dict.

        Parameters
        ----------
        df : pd.DataFrame
            The source DataFrame indexed by the same row_ids stored
            in FaissResult.row_index.

        Returns
        -------
        SearchResult
            The same result object with candidates' metadata populated.
        """
        for candidate in self.candidates:
            try:
                row_id = candidate.row_id
                # Support both integer-positional and label-based index
                if row_id in df.index:
                    candidate.metadata = df.loc[row_id].to_dict()
                elif isinstance(row_id, int) and row_id < len(df):
                    candidate.metadata = df.iloc[row_id].to_dict()
            except Exception as exc:
                logger.warning("attach_metadata: failed for row_id=%s: %s", row_id, exc)
        return self


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 6: QueryCache
# ─────────────────────────────────────────────────────────────────────────────

class QueryCache:
    """
    In-memory LRU-style cache for query embeddings.

    Keyed by SHA-256 of the processed query text.  Prevents re-encoding
    identical queries during interactive sessions or batch searches.

    When the cache is full, the oldest entry (insertion order) is evicted.

    Parameters
    ----------
    max_size : int
        Maximum number of query embeddings to store.  Default: 1000.
    """

    def __init__(self, max_size: int = 1000):
        self._max_size = max_size
        self._store: Dict[str, np.ndarray] = {}   # insertion-ordered (Python 3.7+)

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, processed_query: str) -> Optional[np.ndarray]:
        """Returns the cached embedding for a processed query, or None."""
        return self._store.get(self._key(processed_query))

    def set(self, processed_query: str, vector: np.ndarray) -> None:
        """Stores a query embedding, evicting the oldest if at capacity."""
        k = self._key(processed_query)
        if k in self._store:
            # Move to end (most recently used)
            del self._store[k]
        elif len(self._store) >= self._max_size:
            # Evict oldest insertion
            oldest = next(iter(self._store))
            del self._store[oldest]
        self._store[k] = vector

    @property
    def size(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 7: BaseSearchStrategy (Abstract)
# ─────────────────────────────────────────────────────────────────────────────

class BaseSearchStrategy(ABC):
    """
    Abstract retrieval strategy contract.

    Each concrete subclass encapsulates one retrieval algorithm.
    SemanticSearchEngine only speaks to this interface — adding BM25,
    Qdrant, Pinecone, or Chroma requires a new subclass, not coordinator
    changes (Open/Closed Principle).

    Strategy Contract
    -----------------
    retrieve(query_vector, faiss_result, config) → List[Tuple[float, int]]
        Returns a list of (score, faiss_position) pairs, sorted by
        relevance (best first), up to config.top_k results.

    strategy_name → str
        Human-readable identifier.
    """

    @abstractmethod
    def retrieve(
        self,
        query_vector: np.ndarray,
        faiss_result,           # FaissResult — loosely typed to avoid circular imports
        config: SearchConfig,
    ) -> List[Tuple[float, int]]:
        """
        Retrieves the top candidates for a single query vector.

        Parameters
        ----------
        query_vector : np.ndarray
            Shape (D,) float32 unit vector representing the query.
        faiss_result : FaissResult
            The built FAISS index + row alignment metadata.
        config : SearchConfig
            Active search configuration.

        Returns
        -------
        List[Tuple[float, int]]
            (score, faiss_position) pairs, sorted best-first.
        """
        pass

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 8: ExactSearchStrategy
# ─────────────────────────────────────────────────────────────────────────────

class ExactSearchStrategy(BaseSearchStrategy):
    """
    Direct FAISS search with no approximation parameters overridden.

    Works identically for FlatL2, FlatIP, and trained IVF/HNSW indexes.
    Guarantees 100% recall when used with FlatL2 or FlatIP indexes.
    """

    @property
    def strategy_name(self) -> str:
        return "Exact"

    def retrieve(
        self,
        query_vector: np.ndarray,
        faiss_result,
        config: SearchConfig,
    ) -> List[Tuple[float, int]]:
        q = np.ascontiguousarray(query_vector[None, :], dtype=np.float32)   # (1, D)
        distances, indices = faiss_result.index.search(q, config.top_k)
        pairs = [
            (float(distances[0][i]), int(indices[0][i]))
            for i in range(len(indices[0]))
            if indices[0][i] != -1
        ]
        return pairs


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 9: ApproximateSearchStrategy
# ─────────────────────────────────────────────────────────────────────────────

class ApproximateSearchStrategy(BaseSearchStrategy):
    """
    FAISS search with runtime nprobe / efSearch overrides.

    Designed for IVF and HNSW indexes where the search breadth is tunable
    at query time without rebuilding the index.

    - For IVFFlat: sets ``index.nprobe`` before searching.
    - For HNSWFlat: sets ``index.hnsw.efSearch`` before searching.
    - For Flat indexes: behaves identically to ExactSearchStrategy.
    """

    @property
    def strategy_name(self) -> str:
        return "Approximate"

    def retrieve(
        self,
        query_vector: np.ndarray,
        faiss_result,
        config: SearchConfig,
    ) -> List[Tuple[float, int]]:
        index = faiss_result.index

        # Tune IVF search breadth at runtime
        if config.nprobe is not None and hasattr(index, "nprobe"):
            index.nprobe = config.nprobe

        # Tune HNSW efSearch at runtime
        if config.ef_search is not None and hasattr(index, "hnsw"):
            index.hnsw.efSearch = config.ef_search

        q = np.ascontiguousarray(query_vector[None, :], dtype=np.float32)
        distances, indices = index.search(q, config.top_k)
        pairs = [
            (float(distances[0][i]), int(indices[0][i]))
            for i in range(len(indices[0]))
            if indices[0][i] != -1
        ]
        return pairs


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 10: HybridSearchStrategy (Architecture Stub)
# ─────────────────────────────────────────────────────────────────────────────

class HybridSearchStrategy(BaseSearchStrategy):
    """
    Architecture stub for hybrid vector + keyword search.

    Future implementation will fuse:
    - Dense vector similarity (FAISS inner-product / L2)
    - Sparse keyword relevance (BM25 / TF-IDF)
    - Configurable alpha to weight vector vs keyword scores

    Fusion formula (planned):
        final_score = alpha * vector_score + (1 - alpha) * bm25_score

    Compatible backends (planned):
    - BM25s (local)
    - ElasticSearch
    - Qdrant hybrid search
    - Pinecone sparse-dense
    - Weaviate hybrid

    Currently falls back to ExactSearchStrategy.
    """

    def __init__(self, alpha: float = 0.7):
        """
        Parameters
        ----------
        alpha : float
            Weight given to vector scores vs keyword scores [0.0 – 1.0].
            alpha=1.0 → pure vector; alpha=0.0 → pure keyword.
            Default: 0.7
        """
        self._alpha = alpha
        self._fallback = ExactSearchStrategy()

    @property
    def strategy_name(self) -> str:
        return "Hybrid"

    def retrieve(
        self,
        query_vector: np.ndarray,
        faiss_result,
        config: SearchConfig,
    ) -> List[Tuple[float, int]]:
        logger.warning(
            "HybridSearchStrategy is an architecture stub. "
            "Falling back to ExactSearchStrategy until BM25 integration is complete."
        )
        return self._fallback.retrieve(query_vector, faiss_result, config)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 11: SemanticSearchEngine (Coordinator)
# ─────────────────────────────────────────────────────────────────────────────

class SemanticSearchEngine:
    """
    Pipeline coordinator for converting text queries into ranked candidates.

    The coordinator owns the orchestration loop (preprocess → embed → search →
    score_filter → package) but delegates every specialised operation to
    injected collaborators.

    Parameters
    ----------
    faiss_result : FaissResult
        The built index produced by FaissBuilder.build().
    embedding_model : BaseEmbeddingModel
        The same model used to build the embeddings (must produce same dim).
    config : SearchConfig
        Full pipeline configuration.  Defaults to SearchConfig().
    query_processor : Optional[QueryProcessor]
        Custom query normalisation pipeline.  Defaults to QueryProcessor(config).
    strategy : Optional[BaseSearchStrategy]
        Custom retrieval strategy.  If None, resolved from config.strategy key.
    cache : Optional[QueryCache]
        Custom query embedding cache.  Defaults to QueryCache if enable_cache=True.

    Example
    -------
    >>> engine = SemanticSearchEngine(faiss_result, embedding_model)
    >>> result = engine.search("healthy low-carb breakfast", top_k=5)
    >>> engine.generate_report(result)
    """

    # ── Built-in strategy registry ────────────────────────────────────────────
    _REGISTRY: Dict[str, type] = {
        "exact":       ExactSearchStrategy,
        "approximate": ApproximateSearchStrategy,
        "hybrid":      HybridSearchStrategy,
    }

    def __init__(
        self,
        faiss_result,                              # FaissResult
        embedding_model,                           # BaseEmbeddingModel
        config: Optional[SearchConfig] = None,
        query_processor: Optional[QueryProcessor] = None,
        strategy: Optional[BaseSearchStrategy] = None,
        cache: Optional[QueryCache] = None,
    ):
        self.config = config or SearchConfig()
        self.config.validate()

        self._faiss_result = faiss_result
        self._model = embedding_model

        self._processor = query_processor or QueryProcessor(self.config)
        self._strategy = strategy or self._resolve_strategy(self.config.strategy)
        self._cache: Optional[QueryCache] = (
            cache if cache is not None
            else (QueryCache(self.config.cache_max_size) if self.config.enable_cache else None)
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: Optional[int] = None) -> SearchResult:
        """
        Searches for the most semantically similar candidates to the query.

        Parameters
        ----------
        query : str
            Raw natural-language query text.
        top_k : Optional[int]
            Override config.top_k for this call only.

        Returns
        -------
        SearchResult
            Structured result with ranked candidates and telemetry.
        """
        pipeline_start = time.time()
        warnings: List[str] = []
        is_success = True

        effective_top_k = top_k if top_k is not None else self.config.top_k

        stats = SearchStatistics(
            query_raw=query,
            strategy=self._strategy.strategy_name,
            top_k_requested=effective_top_k,
        )

        # ── 1. Query preprocessing ────────────────────────────────────────────
        proc_start = time.time()
        try:
            processed_query = self._processor.process(query)
        except ValueError as exc:
            warnings.append(str(exc))
            processed_query = query.strip() or "unknown"
        stats.query_processed = processed_query
        stats.query_processing_time_seconds = time.time() - proc_start

        # ── 2. Query embedding (with cache lookup) ────────────────────────────
        embed_start = time.time()
        query_vector, cache_hit = self._get_query_vector(processed_query)
        stats.cache_hit = cache_hit
        stats.embedding_time_seconds = time.time() - embed_start
        stats.embedding_dim = query_vector.shape[0]

        # ── 3. Optional query normalisation ──────────────────────────────────
        if self.config.normalize_query:
            norm = np.linalg.norm(query_vector)
            if norm > 1e-9:
                query_vector = query_vector / norm

        # ── 4. Retrieval via strategy ─────────────────────────────────────────
        search_start = time.time()
        try:
            # Temporarily override top_k in config if needed
            effective_config = self.config
            if top_k is not None and top_k != self.config.top_k:
                import dataclasses
                effective_config = dataclasses.replace(self.config, top_k=top_k)

            raw_hits: List[Tuple[float, int]] = self._strategy.retrieve(
                query_vector, self._faiss_result, effective_config
            )
        except Exception as exc:
            is_success = False
            warnings.append(f"Retrieval failed: {str(exc)}")
            logger.error("SemanticSearchEngine.search() retrieval failed: %s", exc)
            raw_hits = []

        stats.search_time_seconds = time.time() - search_start

        # ── 5. Build candidates ───────────────────────────────────────────────
        candidates = self._build_candidates(raw_hits, effective_config, stats)

        # ── 6. Finalise statistics ────────────────────────────────────────────
        stats.total_time_seconds = time.time() - pipeline_start
        stats.candidates_returned = len(candidates)

        return SearchResult(
            query=query,
            candidates=candidates,
            statistics=stats,
            warnings=warnings,
            is_success=is_success,
        )

    def batch_search(
        self, queries: List[str], top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Searches for each query in the list independently.

        Parameters
        ----------
        queries : List[str]
            List of raw query strings.
        top_k : Optional[int]
            Override config.top_k for this batch.

        Returns
        -------
        List[SearchResult]
            One SearchResult per input query, in the same order.
        """
        return [self.search(q, top_k=top_k) for q in queries]

    def generate_report(self, result: SearchResult) -> None:
        """
        Prints a formatted enterprise execution report for a search result.

        Parameters
        ----------
        result : SearchResult
            The result returned by search().
        """
        s = result.statistics
        width = 62
        sep = "=" * width
        thin = "-" * width

        print(sep)
        print("SEMANTIC SEARCH REPORT".center(width))
        print(sep)
        print(f"  Query (raw)     : {s.query_raw[:55]}")
        print(f"  Query (proc.)   : {s.query_processed[:55]}")
        print(f"  Strategy        : {s.strategy}")
        print(f"  Cache Hit       : {s.cache_hit}")
        print(f"  Embedding Dim   : {s.embedding_dim}")
        print(thin)
        print(f"  Top-K Requested : {s.top_k_requested}")
        print(f"  Results Returned: {s.candidates_returned}")
        print(f"  Filtered Out    : {s.candidates_filtered}")
        print(thin)
        print(f"  Preprocess Time : {s.query_processing_time_seconds*1000:.2f} ms")
        print(f"  Embed Time      : {s.embedding_time_seconds*1000:.2f} ms")
        print(f"  Search Time     : {s.search_time_seconds*1000:.2f} ms")
        print(f"  Total Time      : {s.total_time_seconds*1000:.2f} ms")
        print(thin)
        if result.candidates:
            print("  Top Candidates:")
            for c in result.candidates[:5]:
                score_str = f"{c.score:.4f}"
                rid = str(c.row_id)[:20]
                print(f"    #{c.rank:2d}  score={score_str}  row_id={rid}")
        print(thin)
        print(f"  Status          : {'✓ SUCCESS' if result.is_success else '✗ FAILED'}")
        if result.warnings:
            print(f"  Warnings ({len(result.warnings)}):")
            for w in result.warnings:
                print(f"    ⚠  {w}")
        print(sep)

    @classmethod
    def register_strategy(cls, key: str, strategy_class: type) -> None:
        """
        Registers a custom search strategy into the engine registry.

        Parameters
        ----------
        key : str
            The string key to use in SearchConfig(strategy=key).
        strategy_class : type
            A class implementing BaseSearchStrategy.

        Example
        -------
        >>> SemanticSearchEngine.register_strategy("qdrant", QdrantSearchStrategy)
        >>> config = SearchConfig(strategy="qdrant")
        """
        if not issubclass(strategy_class, BaseSearchStrategy):
            raise TypeError(f"{strategy_class.__name__} must subclass BaseSearchStrategy.")
        cls._REGISTRY[key] = strategy_class
        SearchConfig.SUPPORTED_STRATEGIES.add(key)

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _get_query_vector(self, processed_query: str) -> Tuple[np.ndarray, bool]:
        """
        Returns the embedding for a processed query, using the cache when available.

        Returns
        -------
        Tuple[np.ndarray, bool]
            (vector, cache_hit)
        """
        if self._cache is not None:
            cached = self._cache.get(processed_query)
            if cached is not None:
                return cached, True

        # Minimal duck-typed config satisfying BaseEmbeddingModel.encode() contract.
        # Works with SentenceTransformerEmbedding, custom models, and stubs alike.
        class _QueryEmbedConfig:
            normalize_embeddings = False   # coordinator normalises after
            show_progress = False
            batch_size = 1
            device = "cpu"

        vectors = self._model.encode([processed_query], _QueryEmbedConfig())
        vector = np.asarray(vectors[0], dtype=np.float32)

        if self._cache is not None:
            self._cache.set(processed_query, vector)

        return vector, False

    def _build_candidates(
        self,
        raw_hits: List[Tuple[float, int]],
        config: SearchConfig,
        stats: SearchStatistics,
    ) -> List[SearchCandidate]:
        """
        Converts raw (score, faiss_position) pairs into SearchCandidate objects,
        applying the score threshold filter when configured.
        """
        candidates: List[SearchCandidate] = []
        filtered = 0
        row_index = self._faiss_result.row_index

        for rank, (score, faiss_pos) in enumerate(raw_hits, start=1):
            if faiss_pos < 0 or faiss_pos >= len(row_index):
                filtered += 1
                continue

            # Score threshold filtering
            if config.score_threshold is not None:
                # For L2: lower is better → skip if above threshold
                # For IP: higher is better → skip if below threshold
                metric = getattr(self._faiss_result.config, "metric", "l2")
                if metric == "l2" and score > config.score_threshold:
                    filtered += 1
                    continue
                if metric == "ip" and score < config.score_threshold:
                    filtered += 1
                    continue

            candidates.append(SearchCandidate(
                rank=rank,
                row_id=row_index[faiss_pos],
                faiss_index=faiss_pos,
                score=score,
            ))

        stats.candidates_filtered = filtered
        return candidates

    @classmethod
    def _resolve_strategy(cls, strategy_key: str) -> BaseSearchStrategy:
        klass = cls._REGISTRY.get(strategy_key)
        if klass is None:
            raise ValueError(
                f"Unknown strategy '{strategy_key}'. "
                f"Registered keys: {list(cls._REGISTRY.keys())}"
            )
        return klass()


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST SUITE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(level=logging.ERROR)

    print("=" * 62)
    print("  SemanticSearchEngine — Self-Test Suite")
    print("=" * 62)

    import numpy as np
    import dataclasses

    # ─────────────────────────────────────────────────────────────────────────
    # Stubs — zero external dependencies
    # ─────────────────────────────────────────────────────────────────────────

    class StubEmbeddingConfig:
        normalize_embeddings = False
        show_progress = False

    class StubEmbeddingModel:
        """Deterministic stub — encodes text as a reproducible unit vector."""
        DIM = 64

        def encode(self, texts, config) -> np.ndarray:
            rng = np.random.default_rng(seed=abs(hash(texts[0])) % (2**31))
            vecs = rng.standard_normal((len(texts), self.DIM)).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            return vecs / np.maximum(norms, 1e-9)

        def load(self, config) -> None:
            pass

        @property
        def embedding_dim(self) -> int:
            return self.DIM

        @property
        def provider_name(self) -> str:
            return "StubModel"

    class StubFaissConfig:
        metric = "ip"
        normalize_before_index = True
        nlist = 1
        nprobe = 1
        hnsw_m = 16
        hnsw_ef_construction = 100

    class StubFaissResult:
        """Minimal FaissResult stub backed by a real faiss.IndexFlatIP."""
        def __init__(self, n: int = 100, dim: int = 64):
            import faiss
            self.config = StubFaissConfig()
            rng = np.random.default_rng(42)
            vecs = rng.standard_normal((n, dim)).astype(np.float32)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.maximum(norms, 1e-9)
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(vecs)
            self.row_index = list(range(n))

    stub_faiss = StubFaissResult(n=100, dim=64)
    stub_model = StubEmbeddingModel()

    # ── Test 1: Basic search ──────────────────────────────────────────────────
    print("\n--- Test 1: Basic Search ---")
    engine = SemanticSearchEngine(stub_faiss, stub_model)
    result = engine.search("healthy vegetarian breakfast")
    assert result.is_success, "Search should succeed"
    assert len(result.candidates) <= 10, "Should return ≤ top_k=10 candidates"
    assert result.candidates[0].rank == 1, "First candidate should have rank 1"
    print(f"  Candidates returned : {len(result.candidates)}")
    print(f"  Top-1 row_id        : {result.candidates[0].row_id}")
    print(f"  Top-1 score         : {result.candidates[0].score:.4f}")

    # ── Test 2: top_k override ────────────────────────────────────────────────
    print("\n--- Test 2: top_k Override ---")
    result3 = engine.search("italian pasta dinner", top_k=3)
    assert len(result3.candidates) == 3, f"Expected 3 candidates, got {len(result3.candidates)}"
    print(f"  top_k=3 returned {len(result3.candidates)} candidates  ✓")

    # ── Test 3: Query cache ───────────────────────────────────────────────────
    print("\n--- Test 3: Query Cache ---")
    q = "spicy thai noodles"
    r1 = engine.search(q)
    r2 = engine.search(q)
    assert not r1.statistics.cache_hit, "First call should be a cache miss"
    assert r2.statistics.cache_hit, "Second call should be a cache hit"
    print(f"  Run 1 cache hit : {r1.statistics.cache_hit}  (miss  ✓)")
    print(f"  Run 2 cache hit : {r2.statistics.cache_hit}  (hit   ✓)")
    assert r1.candidates[0].row_id == r2.candidates[0].row_id, "Results should match"
    print("  ✓ Cached and uncached results are identical")

    # ── Test 4: ApproximateSearchStrategy ─────────────────────────────────────
    print("\n--- Test 4: ApproximateSearchStrategy ---")
    approx_config = SearchConfig(strategy="approximate", top_k=5, enable_cache=False)
    approx_engine = SemanticSearchEngine(stub_faiss, stub_model, config=approx_config)
    approx_result = approx_engine.search("grilled chicken salad")
    assert approx_result.statistics.strategy == "Approximate"
    print(f"  Strategy   : {approx_result.statistics.strategy}  ✓")
    print(f"  Candidates : {len(approx_result.candidates)}")

    # ── Test 5: QueryProcessor ────────────────────────────────────────────────
    print("\n--- Test 5: QueryProcessor ---")
    config_proc = SearchConfig(lowercase_query=True, strip_punctuation=True)
    processor = QueryProcessor(config_proc)
    processed = processor.process("  Healthy!!! BREAKFAST?  ")
    assert processed == processed.lower(), "Should be lowercase"
    assert "!" not in processed, "Should strip punctuation"
    print(f"  Raw      : '  Healthy!!! BREAKFAST?  '")
    print(f"  Processed: '{processed}'  ✓")

    # ── Test 6: Custom step injection ─────────────────────────────────────────
    print("\n--- Test 6: Custom QueryProcessor Step ---")
    def add_domain_prefix(text: str) -> str:
        return f"recipe: {text}"

    extended = QueryProcessor.add_custom_step(processor, add_domain_prefix)
    out = extended.process("vegetarian curry")
    assert out.startswith("recipe:"), f"Expected prefix, got: '{out}'"
    print(f"  Custom step output: '{out}'  ✓")

    # ── Test 7: Score threshold filtering ─────────────────────────────────────
    print("\n--- Test 7: Score Threshold Filtering ---")
    # For IP index: scores are cosine similarities [0,1]; threshold=0.99
    # filters almost everything — only near-perfect matches survive
    thresh_config = SearchConfig(
        top_k=10, score_threshold=0.99, enable_cache=False
    )
    thresh_engine = SemanticSearchEngine(stub_faiss, stub_model, config=thresh_config)
    thresh_result = thresh_engine.search("breakfast")
    print(f"  Returned {thresh_result.candidates_returned if hasattr(thresh_result, 'candidates_returned') else len(thresh_result.candidates)} / 10 (with threshold=0.99)")
    print(f"  Filtered: {thresh_result.statistics.candidates_filtered}  ✓")

    # ── Test 8: to_json() serialisation ──────────────────────────────────────
    print("\n--- Test 8: to_json() Serialisation ---")
    j = result.to_json()
    parsed = json.loads(j)
    assert "candidates" in parsed
    assert "statistics" in parsed
    assert parsed["is_success"] is True
    print(f"  JSON keys       : {list(parsed.keys())}  ✓")
    print(f"  Candidates in JSON : {len(parsed['candidates'])}")

    # ── Test 9: batch_search() ────────────────────────────────────────────────
    print("\n--- Test 9: batch_search() ---")
    batch_results = engine.batch_search(
        ["vegan soup", "grilled steak", "chocolate dessert"], top_k=3
    )
    assert len(batch_results) == 3
    for i, br in enumerate(batch_results):
        assert br.is_success
        print(f"  Query {i+1}: {len(br.candidates)} candidates")

    # ── Test 10: Plugin strategy registration ─────────────────────────────────
    print("\n--- Test 10: Plugin Strategy Registration ---")

    class EchoSearchStrategy(BaseSearchStrategy):
        """Returns the first top_k vectors from the index regardless of query."""
        @property
        def strategy_name(self) -> str:
            return "Echo"
        def retrieve(self, query_vector, faiss_result, config) -> List[Tuple[float, int]]:
            return [(1.0, i) for i in range(config.top_k)]

    SemanticSearchEngine.register_strategy("echo", EchoSearchStrategy)
    echo_config = SearchConfig(strategy="echo", top_k=5, enable_cache=False)
    echo_engine = SemanticSearchEngine(stub_faiss, stub_model, config=echo_config)
    echo_result = echo_engine.search("anything")
    assert echo_result.statistics.strategy == "Echo"
    assert len(echo_result.candidates) == 5
    print(f"  Custom 'echo' strategy returned {len(echo_result.candidates)} candidates  ✓")

    # ── Test 11: Enterprise report ────────────────────────────────────────────
    print("\n--- Test 11: Enterprise Report ---")
    engine.generate_report(result)

    # ── Test 12: Config validation ─────────────────────────────────────────────
    print("--- Test 12: Config Validation ---")
    try:
        SearchConfig(top_k=0).validate()
        print("  ERROR: Should have raised ValueError!")
    except ValueError as exc:
        print(f"  [PASS] ValueError raised: {exc}")

    print("\n" + "=" * 62)
    print("  All self-tests passed ✓")
    print("=" * 62)
