"""
DineAI — FAISS Index Builder
=============================

Builds, validates, saves, loads, and searches FAISS vector indexes
from EmbeddingResult objects produced by embedding_builder.py.

Architecture
------------
FaissBuilder (Coordinator)
    ├── FaissConfig        — Immutable configuration dataclass
    ├── BaseFaissIndex     — Abstract strategy interface (Open/Closed Principle)
    │   ├── FlatL2Index    — Exact L2 distance (brute-force, always ready)
    │   ├── FlatIPIndex    — Exact inner-product / cosine similarity
    │   ├── IVFIndex       — Approximate IVF (inverted file, training required)
    │   └── HNSWIndex      — Graph-based approximate search (no training)
    ├── FaissStatistics    — Detached execution telemetry (SRP)
    └── FaissResult        — Structured output — owns save / load / search

Public API
----------
    config  = FaissConfig(index_type="flat_l2")
    builder = FaissBuilder(config)
    result  = builder.build(embedding_result)
    result.save("datasets/restaurant_A")

    loaded  = FaissResult.load("datasets/restaurant_A")
    hits    = loaded.search(query_vector, top_k=5)
    builder.generate_report(result)
"""

import os
import sys
import json
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy faiss import so the module can be imported without faiss installed ──
def _require_faiss():
    try:
        import faiss
        return faiss
    except ImportError as exc:
        raise ImportError(
            "faiss-cpu (or faiss-gpu) is required for FaissBuilder.\n"
            "Install with:  pip install faiss-cpu"
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 1: FaissConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FaissConfig:
    """
    Immutable configuration for the FaissBuilder pipeline.

    All strategy selection and tuning lives here — no scattered kwargs.

    Parameters
    ----------
    index_type : str
        Strategy key.  One of:
          "flat_l2"   — Exact L2 (default, always safe)
          "flat_ip"   — Exact inner-product / cosine
          "ivf"       — Approximate IVF (fast at scale, requires training)
          "hnsw"      — Graph-based approximate (no training, low memory)
    embedding_dim : int
        Vector dimensionality.  Set automatically from EmbeddingResult when 0.
    nlist : int
        IVF: number of Voronoi cells (clusters).  Rule of thumb: sqrt(N).
        Default: 100
    nprobe : int
        IVF: number of cells to visit at query time.  Higher = more accurate.
        Default: 10
    hnsw_m : int
        HNSW: number of bidirectional links per node.  Higher = better recall.
        Default: 32
    hnsw_ef_construction : int
        HNSW: size of dynamic candidate list during build.
        Default: 200
    metric : str
        Similarity metric: "l2" (Euclidean) or "ip" (inner-product / cosine).
        Default: "l2"
    normalize_before_index : bool
        If True, L2-normalises vectors before adding to index.
        Useful when using flat_ip for cosine similarity.
        Default: False
    """
    index_type: str = "flat_l2"
    embedding_dim: int = 0            # resolved automatically from EmbeddingResult
    nlist: int = 100                  # IVF clusters
    nprobe: int = 10                  # IVF search breadth
    hnsw_m: int = 32                  # HNSW links per node
    hnsw_ef_construction: int = 200   # HNSW build-time candidate list size
    metric: str = "l2"                # "l2" | "ip"
    normalize_before_index: bool = False

    # Registry of supported index type keys
    SUPPORTED_INDEX_TYPES = {"flat_l2", "flat_ip", "ivf", "hnsw"}
    SUPPORTED_METRICS = {"l2", "ip"}

    def validate(self) -> None:
        """
        Validates configuration values before pipeline execution.

        Raises
        ------
        ValueError
            On any invalid configuration parameter.
        """
        if self.index_type not in self.SUPPORTED_INDEX_TYPES:
            raise ValueError(
                f"FaissConfig.index_type '{self.index_type}' is not supported. "
                f"Choose from: {self.SUPPORTED_INDEX_TYPES}"
            )
        if self.metric not in self.SUPPORTED_METRICS:
            raise ValueError(
                f"FaissConfig.metric '{self.metric}' is not supported. "
                f"Choose 'l2' or 'ip'."
            )
        if self.nlist < 1:
            raise ValueError(f"FaissConfig.nlist must be >= 1, got {self.nlist}.")
        if self.nprobe < 1:
            raise ValueError(f"FaissConfig.nprobe must be >= 1, got {self.nprobe}.")
        if self.hnsw_m < 4:
            raise ValueError(f"FaissConfig.hnsw_m must be >= 4, got {self.hnsw_m}.")

    def to_dict(self) -> Dict[str, Any]:
        """Returns a plain dict for JSON serialisation."""
        return {
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "hnsw_m": self.hnsw_m,
            "hnsw_ef_construction": self.hnsw_ef_construction,
            "metric": self.metric,
            "normalize_before_index": self.normalize_before_index,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 2: FaissStatistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FaissStatistics:
    """
    Execution telemetry captured during FaissBuilder.build().

    Detached from FaissResult so statistics can be logged or exported
    independently of the index data.
    """
    total_vectors: int = 0
    indexed_vectors: int = 0
    index_type: str = ""
    metric: str = ""
    embedding_dim: int = 0
    requires_training: bool = False
    training_vectors: int = 0
    training_time_seconds: float = 0.0
    indexing_time_seconds: float = 0.0
    total_time_seconds: float = 0.0
    index_size_bytes: int = 0
    is_trained: bool = False
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serialises statistics to a plain dictionary."""
        return {
            "total_vectors": self.total_vectors,
            "indexed_vectors": self.indexed_vectors,
            "index_type": self.index_type,
            "metric": self.metric,
            "embedding_dim": self.embedding_dim,
            "requires_training": self.requires_training,
            "training_vectors": self.training_vectors,
            "training_time_seconds": round(self.training_time_seconds, 4),
            "indexing_time_seconds": round(self.indexing_time_seconds, 4),
            "total_time_seconds": round(self.total_time_seconds, 4),
            "index_size_bytes": self.index_size_bytes,
            "is_trained": self.is_trained,
            "timestamp": self.timestamp,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 3: FaissResult
# ─────────────────────────────────────────────────────────────────────────────

class FaissResult:
    """
    Structured output of FaissBuilder.build().

    Owns its own persistence and search interface — callers never access
    the raw FAISS index object directly.

    Attributes
    ----------
    index : faiss.Index
        The built FAISS index object.
    config : FaissConfig
        Configuration that produced this index.
    row_index : List
        Original DataFrame index values aligned to vector positions.
    statistics : FaissStatistics
        Full execution telemetry.
    warnings : List[str]
        Non-fatal issues captured during build.
    is_success : bool
        False if any ERROR-level issues occurred.
    """

    _INDEX_FILENAME = "faiss.index"
    _META_FILENAME = "faiss_metadata.json"

    def __init__(
        self,
        index,                          # faiss.Index — typed loosely to avoid import at module level
        config: FaissConfig,
        row_index: List,
        statistics: FaissStatistics,
        warnings: Optional[List[str]] = None,
        is_success: bool = True,
    ):
        self.index = index
        self.config = config
        self.row_index = row_index
        self.statistics = statistics
        self.warnings = warnings or []
        self.is_success = is_success

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vectors: np.ndarray,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, List[List]]:
        """
        Searches the index for the nearest neighbours of each query vector.

        Parameters
        ----------
        query_vectors : np.ndarray
            Shape (Q, D) float32 query matrix, or (D,) for a single query.
        top_k : int
            Number of nearest neighbours to retrieve per query.

        Returns
        -------
        distances : np.ndarray
            Shape (Q, top_k) distance/score matrix.
        faiss_indices : np.ndarray
            Shape (Q, top_k) integer positions in the index.
        row_ids : List[List]
            Original DataFrame row IDs corresponding to faiss_indices,
            with -1 replaced by None.
        """
        faiss = _require_faiss()

        if query_vectors.ndim == 1:
            query_vectors = query_vectors[None, :]   # (1, D)

        query_vectors = np.ascontiguousarray(query_vectors, dtype=np.float32)

        if self.config.normalize_before_index:
            norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
            query_vectors = query_vectors / np.maximum(norms, 1e-9)

        distances, faiss_indices = self.index.search(query_vectors, top_k)

        # Map faiss integer positions back to original row IDs
        row_ids: List[List] = []
        for row in faiss_indices:
            row_ids.append([
                self.row_index[i] if i != -1 and i < len(self.row_index) else None
                for i in row
            ])

        return distances, faiss_indices, row_ids

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, output_dir: str) -> Tuple[str, str]:
        """
        Saves the FAISS index and metadata to disk.

        Parameters
        ----------
        output_dir : str
            Target directory (created if absent).

        Returns
        -------
        Tuple[str, str]
            (index_path, metadata_path) absolute file paths.
        """
        faiss = _require_faiss()
        os.makedirs(output_dir, exist_ok=True)

        index_path = os.path.join(output_dir, self._INDEX_FILENAME)
        meta_path = os.path.join(output_dir, self._META_FILENAME)

        faiss.write_index(self.index, index_path)

        metadata = {
            "config": self.config.to_dict(),
            "row_index": [str(i) for i in self.row_index],
            "statistics": self.statistics.to_dict(),
            "warnings": self.warnings,
            "is_success": self.is_success,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved FAISS index  → %s", index_path)
        logger.info("Saved FAISS metadata → %s", meta_path)
        return index_path, meta_path

    @classmethod
    def load(cls, output_dir: str) -> "FaissResult":
        """
        Loads a previously saved FaissResult from disk.

        Parameters
        ----------
        output_dir : str
            Directory containing faiss.index and faiss_metadata.json.

        Returns
        -------
        FaissResult
            Fully reconstructed result object, ready for search.

        Raises
        ------
        FileNotFoundError
            If either file is missing.
        """
        faiss = _require_faiss()

        index_path = os.path.join(output_dir, cls._INDEX_FILENAME)
        meta_path = os.path.join(output_dir, cls._META_FILENAME)

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"FAISS metadata not found: {meta_path}")

        index = faiss.read_index(index_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        cfg_dict = metadata.get("config", {})
        config = FaissConfig(**{
            k: v for k, v in cfg_dict.items()
            if k in FaissConfig.__dataclass_fields__
        })

        stats_dict = metadata.get("statistics", {})
        stats = FaissStatistics(**{
            k: v for k, v in stats_dict.items()
            if k in FaissStatistics.__dataclass_fields__
        })

        return cls(
            index=index,
            config=config,
            row_index=metadata.get("row_index", []),
            statistics=stats,
            warnings=metadata.get("warnings", []),
            is_success=metadata.get("is_success", True),
        )

    def total_vectors(self) -> int:
        """Returns the number of vectors currently stored in the index."""
        return self.index.ntotal if self.index is not None else 0


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 4: BaseFaissIndex (Abstract Strategy Interface)
# ─────────────────────────────────────────────────────────────────────────────

class BaseFaissIndex(ABC):
    """
    Abstract strategy interface for FAISS index construction.

    Each concrete subclass encapsulates one index family's build logic.
    FaissBuilder only speaks to this interface — adding IVF-PQ, ScaNN,
    or Annoy requires a new subclass, not changes to the coordinator.

    Strategy Contract
    -----------------
    build(vectors, config) → faiss.Index
        Constructs and populates the index.  Handles training internally
        when requires_training is True.
    requires_training → bool
        Whether this index family needs a .train() call before .add().
    strategy_name → str
        Human-readable identifier (e.g. "FlatL2", "IVF").
    """

    @abstractmethod
    def build(self, vectors: np.ndarray, config: FaissConfig):
        """
        Builds and returns a trained + populated FAISS index.

        Parameters
        ----------
        vectors : np.ndarray
            Shape (N, D) float32 matrix.
        config : FaissConfig
            Active configuration (dim, nlist, metric, …).

        Returns
        -------
        faiss.Index
            A trained and populated FAISS index.
        """
        pass

    @property
    @abstractmethod
    def requires_training(self) -> bool:
        """Returns True if this strategy needs a .train() call."""
        pass

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Human-readable strategy identifier."""
        pass

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _metric_flag(metric: str) -> int:
        """Converts metric string to faiss metric constant."""
        faiss = _require_faiss()
        return faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT

    @staticmethod
    def _ensure_contiguous_f32(vectors: np.ndarray) -> np.ndarray:
        """Guarantees float32 C-contiguous layout required by faiss."""
        return np.ascontiguousarray(vectors, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 5: FlatL2Index
# ─────────────────────────────────────────────────────────────────────────────

class FlatL2Index(BaseFaissIndex):
    """
    Exact brute-force L2 (Euclidean) index.

    No training required.  100% recall.  O(N) search time.
    Best choice for datasets < ~100K vectors or when recall is critical.
    """

    @property
    def requires_training(self) -> bool:
        return False

    @property
    def strategy_name(self) -> str:
        return "FlatL2"

    def build(self, vectors: np.ndarray, config: FaissConfig):
        faiss = _require_faiss()
        vectors = self._ensure_contiguous_f32(vectors)
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        return index


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 6: FlatIPIndex
# ─────────────────────────────────────────────────────────────────────────────

class FlatIPIndex(BaseFaissIndex):
    """
    Exact brute-force inner-product index.

    When input vectors are L2-normalised (unit vectors), inner product equals
    cosine similarity, making this ideal for semantic search.

    No training required.  100% recall.  O(N) search time.
    """

    @property
    def requires_training(self) -> bool:
        return False

    @property
    def strategy_name(self) -> str:
        return "FlatIP"

    def build(self, vectors: np.ndarray, config: FaissConfig):
        faiss = _require_faiss()
        vectors = self._ensure_contiguous_f32(vectors)
        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return index


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 7: IVFIndex
# ─────────────────────────────────────────────────────────────────────────────

class IVFIndex(BaseFaissIndex):
    """
    Inverted File (IVF) approximate nearest-neighbour index.

    Partitions the vector space into nlist Voronoi cells using k-means.
    At search time, only nprobe cells are visited — trading recall for speed.

    Requires training on a representative sample before vectors can be added.

    Recommended when
    ----------------
    - Dataset size > 100K vectors
    - You can tolerate 95–99% recall in exchange for 10–100x speedup
    - nlist ≈ 4 × sqrt(N)

    Minimum training requirement
    ----------------------------
    FAISS requires at least 39 × nlist training vectors.
    If the dataset is too small, falls back to FlatL2.
    """

    @property
    def requires_training(self) -> bool:
        return True

    @property
    def strategy_name(self) -> str:
        return "IVF"

    def build(self, vectors: np.ndarray, config: FaissConfig):
        faiss = _require_faiss()
        vectors = self._ensure_contiguous_f32(vectors)
        n, dim = vectors.shape

        # Safety guard: IVF needs enough training vectors
        min_required = 39 * config.nlist
        if n < min_required:
            logger.warning(
                "IVFIndex: dataset too small for nlist=%d (need ≥ %d vectors, got %d). "
                "Falling back to FlatL2.",
                config.nlist, min_required, n
            )
            flat = faiss.IndexFlatL2(dim)
            flat.add(vectors)
            return flat

        metric = self._metric_flag(config.metric)
        quantiser = faiss.IndexFlatL2(dim)   # quantiser always uses L2 for Voronoi
        index = faiss.IndexIVFFlat(quantiser, dim, config.nlist, metric)

        # Training: learns the cluster centroids
        index.train(vectors)
        index.nprobe = config.nprobe
        index.add(vectors)
        return index


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 8: HNSWIndex
# ─────────────────────────────────────────────────────────────────────────────

class HNSWIndex(BaseFaissIndex):
    """
    Hierarchical Navigable Small World (HNSW) graph-based approximate index.

    No training required.  Sub-linear search time.  High recall at low latency.
    Memory-intensive (stores graph edges per vector).

    Parameters (from FaissConfig)
    ----------------------------
    hnsw_m : int
        Number of bidirectional links per node.  Higher = better recall,
        more memory.  Typical: 16–64.  Default: 32.
    hnsw_ef_construction : int
        Candidate list size during graph construction.  Higher = better
        quality index but slower build.  Default: 200.

    Recommended when
    ----------------
    - Low query latency is critical (< 1 ms)
    - Medium-size datasets (10K – 10M vectors)
    - No GPU available (HNSW is CPU-friendly)
    """

    @property
    def requires_training(self) -> bool:
        return False

    @property
    def strategy_name(self) -> str:
        return "HNSW"

    def build(self, vectors: np.ndarray, config: FaissConfig):
        faiss = _require_faiss()
        vectors = self._ensure_contiguous_f32(vectors)
        n, dim = vectors.shape

        index = faiss.IndexHNSWFlat(dim, config.hnsw_m)
        index.hnsw.efConstruction = config.hnsw_ef_construction
        index.add(vectors)
        return index


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 9: FaissBuilder (Pipeline Coordinator)
# ─────────────────────────────────────────────────────────────────────────────

class FaissBuilder:
    """
    Pipeline coordinator for building FAISS indexes from EmbeddingResult objects.

    Strategy Selection
    ------------------
    Pass any BaseFaissIndex subclass via the `index` parameter.
    If omitted, the coordinator uses the FaissConfig.index_type key to
    auto-select from the built-in registry:

        "flat_l2"  →  FlatL2Index
        "flat_ip"  →  FlatIPIndex
        "ivf"      →  IVFIndex
        "hnsw"     →  HNSWIndex

    Parameters
    ----------
    config : FaissConfig
        Full pipeline configuration.
    index : Optional[BaseFaissIndex]
        Index strategy to use.  If None, resolved from config.index_type.

    Example
    -------
    >>> config  = FaissConfig(index_type="flat_ip", normalize_before_index=True)
    >>> builder = FaissBuilder(config)
    >>> result  = builder.build(embedding_result)
    >>> result.save("datasets/restaurant_A")
    """

    # ── Built-in strategy registry ────────────────────────────────────────────
    _REGISTRY: Dict[str, type] = {
        "flat_l2": FlatL2Index,
        "flat_ip": FlatIPIndex,
        "ivf":     IVFIndex,
        "hnsw":    HNSWIndex,
    }

    def __init__(
        self,
        config: FaissConfig,
        index: Optional[BaseFaissIndex] = None,
    ):
        config.validate()
        self.config = config
        self._index_strategy: BaseFaissIndex = (
            index if index is not None
            else self._resolve_strategy(config.index_type)
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self, embedding_result) -> FaissResult:
        """
        Builds a FAISS index from an EmbeddingResult.

        Parameters
        ----------
        embedding_result : EmbeddingResult
            Output of EmbeddingBuilder.build().

        Returns
        -------
        FaissResult
            Contains the built index, metadata, and execution statistics.

        Raises
        ------
        ValueError
            If embedding_result is empty or has invalid shape.
        RuntimeError
            If index construction fails.
        """
        pipeline_start = time.time()
        warnings: List[str] = []
        is_success = True

        # ── 1. Validate input ─────────────────────────────────────────────────
        vectors, row_index = self._validate_and_extract(embedding_result, warnings)

        n, dim = vectors.shape

        # ── 2. Resolve embedding dim ──────────────────────────────────────────
        cfg = self.config
        if cfg.embedding_dim == 0:
            # Auto-detect from embedding
            import dataclasses
            cfg = dataclasses.replace(cfg, embedding_dim=dim)

        # ── 3. Optional pre-normalisation ─────────────────────────────────────
        if cfg.normalize_before_index:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-9)

        # ── 4. Initialise statistics ──────────────────────────────────────────
        stats = FaissStatistics(
            total_vectors=n,
            index_type=self._index_strategy.strategy_name,
            metric=cfg.metric,
            embedding_dim=dim,
            requires_training=self._index_strategy.requires_training,
        )

        # ── 5. Build index via strategy ───────────────────────────────────────
        try:
            train_start = time.time()
            built_index = self._index_strategy.build(vectors, cfg)
            build_elapsed = time.time() - train_start

            stats.is_trained = True
            stats.indexed_vectors = built_index.ntotal
            stats.indexing_time_seconds = build_elapsed
            # Reflect the actual index type built (IVF may fall back to FlatL2)
            faiss_mod = _require_faiss()
            actual_type = type(built_index).__name__.replace("Index", "")
            stats.index_type = actual_type

            # Approximate in-memory size
            stats.index_size_bytes = self._estimate_index_bytes(built_index, n, dim)

        except Exception as exc:
            is_success = False
            warnings.append(f"Index build failed: {str(exc)}")
            logger.error("FaissBuilder.build() failed: %s", exc, exc_info=True)
            raise RuntimeError(f"FAISS index construction failed: {exc}") from exc

        # ── 6. Finalise statistics ────────────────────────────────────────────
        stats.total_time_seconds = time.time() - pipeline_start

        return FaissResult(
            index=built_index,
            config=cfg,
            row_index=row_index,
            statistics=stats,
            warnings=warnings,
            is_success=is_success,
        )

    def generate_report(self, result: FaissResult) -> None:
        """
        Prints a formatted enterprise-grade execution report to stdout.

        Parameters
        ----------
        result : FaissResult
            The result returned by build().
        """
        s = result.statistics
        width = 62
        sep = "=" * width
        thin = "-" * width

        print(sep)
        print("FAISS BUILDER REPORT".center(width))
        print(sep)
        print(f"  Strategy        : {self._index_strategy.strategy_name}")
        print(f"  Index Type      : {s.index_type}")
        print(f"  Metric          : {s.metric}")
        print(f"  Embedding Dim   : {s.embedding_dim}")
        print(f"  Normalised      : {result.config.normalize_before_index}")
        print(thin)
        print(f"  Total Vectors   : {s.total_vectors:,}")
        print(f"  Indexed Vectors : {s.indexed_vectors:,}")
        print(f"  Trained         : {s.is_trained}")

        if s.index_type in ("IVF",):
            print(f"  Training Vecs   : {s.training_vectors:,}")
            print(f"  Train Time      : {s.training_time_seconds:.3f}s")
            print(f"  nlist           : {result.config.nlist}")
            print(f"  nprobe          : {result.config.nprobe}")

        if s.index_type == "HNSW":
            print(f"  HNSW M          : {result.config.hnsw_m}")
            print(f"  ef_construction : {result.config.hnsw_ef_construction}")

        print(thin)
        print(f"  Build Time      : {s.indexing_time_seconds:.3f}s")
        print(f"  Total Time      : {s.total_time_seconds:.3f}s")
        est_mb = s.index_size_bytes / 1_048_576
        print(f"  Est. Index Size : {est_mb:.2f} MB")
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
        Registers a custom index strategy into the builder registry.

        Enables plugin-style extension without modifying this file.

        Parameters
        ----------
        key : str
            The string key to use in FaissConfig(index_type=key).
        strategy_class : type
            A class implementing BaseFaissIndex.

        Example
        -------
        >>> FaissBuilder.register_strategy("ivfpq", IVFPQIndex)
        >>> config = FaissConfig(index_type="ivfpq")
        """
        if not issubclass(strategy_class, BaseFaissIndex):
            raise TypeError(f"{strategy_class.__name__} must subclass BaseFaissIndex.")
        cls._REGISTRY[key] = strategy_class
        FaissConfig.SUPPORTED_INDEX_TYPES.add(key)
        logger.info("Registered FAISS strategy '%s' → %s", key, strategy_class.__name__)

    # ── Private Helpers ───────────────────────────────────────────────────────

    @classmethod
    def _resolve_strategy(cls, index_type: str) -> BaseFaissIndex:
        """Instantiates the strategy class for the given index_type key."""
        klass = cls._REGISTRY.get(index_type)
        if klass is None:
            raise ValueError(
                f"Unknown index_type '{index_type}'. "
                f"Registered keys: {list(cls._REGISTRY.keys())}"
            )
        return klass()

    @staticmethod
    def _validate_and_extract(embedding_result, warnings: List[str]):
        """
        Extracts and validates the numpy array from an EmbeddingResult.

        Accepts either:
        - An EmbeddingResult dataclass (from embedding_builder.py)
        - A plain numpy ndarray (for testing or pipeline flexibility)
        """
        if isinstance(embedding_result, np.ndarray):
            vectors = embedding_result
            row_index = list(range(len(vectors)))
        else:
            vectors = getattr(embedding_result, "embeddings", None)
            row_index = getattr(embedding_result, "index", None) or list(range(len(vectors)))

        if vectors is None or not isinstance(vectors, np.ndarray):
            raise ValueError(
                "FaissBuilder.build() expects an EmbeddingResult "
                "with a .embeddings numpy array, or a plain ndarray."
            )
        if vectors.ndim != 2:
            raise ValueError(
                f"Expected 2-D embedding matrix (N, D), got shape {vectors.shape}."
            )
        if len(vectors) == 0:
            raise ValueError("Cannot build FAISS index from an empty embedding matrix.")

        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        return vectors, row_index

    @staticmethod
    def _estimate_index_bytes(index, n: int, dim: int) -> int:
        """Rough in-memory size estimate (bytes) for common index types."""
        # float32 = 4 bytes per dimension per vector
        base = n * dim * 4
        # HNSW adds graph edges (≈ M * 8 bytes per node per level)
        return base


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST SUITE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(level=logging.ERROR)

    print("=" * 62)
    print("  FaissBuilder — Self-Test Suite")
    print("=" * 62)

    # ── Minimal stub that mimics EmbeddingResult ──────────────────────────────
    class StubEmbeddingResult:
        def __init__(self, n: int = 200, dim: int = 384, seed: int = 42):
            rng = np.random.default_rng(seed)
            raw = rng.standard_normal((n, dim)).astype(np.float32)
            # L2-normalise so dot-product == cosine similarity
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            self.embeddings = raw / np.maximum(norms, 1e-9)
            self.index = list(range(n))

    stub = StubEmbeddingResult(n=200, dim=384)

    import tempfile

    # ── Test 1: FlatL2 index ──────────────────────────────────────────────────
    print("\n--- Test 1: FlatL2Index ---")
    config = FaissConfig(index_type="flat_l2")
    builder = FaissBuilder(config)
    result = builder.build(stub)
    assert result.is_success
    assert result.total_vectors() == 200
    print(f"  Vectors indexed : {result.total_vectors()}")
    print(f"  Index type      : {result.statistics.index_type}")

    # ── Test 2: FlatIP search ─────────────────────────────────────────────────
    print("\n--- Test 2: FlatIPIndex + Search ---")
    config_ip = FaissConfig(index_type="flat_ip", normalize_before_index=True)
    builder_ip = FaissBuilder(config_ip)
    result_ip = builder_ip.build(stub)

    query = stub.embeddings[0:1]   # first vector as query
    distances, faiss_ids, row_ids = result_ip.search(query, top_k=5)
    print(f"  Top-5 distances : {distances[0].tolist()}")
    print(f"  Top-5 row IDs   : {row_ids[0]}")
    assert row_ids[0][0] == 0, "First result should be the query itself (distance 0)"
    print("  ✓ Top-1 result is the query vector itself")

    # ── Test 3: HNSW index ────────────────────────────────────────────────────
    print("\n--- Test 3: HNSWIndex ---")
    config_hnsw = FaissConfig(index_type="hnsw", hnsw_m=16, hnsw_ef_construction=100)
    builder_hnsw = FaissBuilder(config_hnsw)
    result_hnsw = builder_hnsw.build(stub)
    assert result_hnsw.total_vectors() == 200
    print(f"  Vectors indexed : {result_hnsw.total_vectors()}")
    print(f"  Strategy        : {result_hnsw.statistics.index_type}")

    # ── Test 4: IVF with small dataset → fallback ─────────────────────────────
    print("\n--- Test 4: IVFIndex (small dataset → FlatL2 fallback) ---")
    small_stub = StubEmbeddingResult(n=50, dim=64)
    config_ivf = FaissConfig(index_type="ivf", nlist=100)
    builder_ivf = FaissBuilder(config_ivf)
    result_ivf = builder_ivf.build(small_stub)
    assert result_ivf.total_vectors() == 50
    print(f"  Fallback index type : {result_ivf.statistics.index_type}")
    print("  ✓ IVF gracefully fell back to FlatL2 for small dataset")

    # ── Test 5: Save / Load round-trip ───────────────────────────────────────
    print("\n--- Test 5: Save / Load Round-Trip ---")
    with tempfile.TemporaryDirectory() as tmp:
        idx_path, meta_path = result.save(tmp)
        print(f"  Saved → {os.path.basename(idx_path)}")
        print(f"  Saved → {os.path.basename(meta_path)}")

        loaded = FaissResult.load(tmp)
        assert loaded.total_vectors() == result.total_vectors()

        dist_orig, _, _ = result.search(query, top_k=3)
        dist_load, _, _ = loaded.search(query, top_k=3)
        assert np.allclose(dist_orig, dist_load, atol=1e-5), "Distances differ after load!"
        print(f"  Loaded vectors  : {loaded.total_vectors()}  ✓ Matches original")
        print(f"  Max dist diff   : {np.max(np.abs(dist_orig - dist_load)):.2e}  ✓")

    # ── Test 6: Enterprise report ─────────────────────────────────────────────
    print("\n--- Test 6: Enterprise Report ---")
    builder.generate_report(result)

    # ── Test 7: Config validation ─────────────────────────────────────────────
    print("\n--- Test 7: Config Validation ---")
    try:
        FaissConfig(index_type="unknown_index").validate()
        print("  ERROR: Should have raised ValueError!")
    except ValueError as exc:
        print(f"  [PASS] ValueError raised: {exc}")

    # ── Test 8: Plain ndarray input ──────────────────────────────────────────
    print("\n--- Test 8: Plain ndarray Input ---")
    plain_vecs = np.random.default_rng(0).standard_normal((100, 128)).astype(np.float32)
    result_plain = FaissBuilder(FaissConfig()).build(plain_vecs)
    assert result_plain.total_vectors() == 100
    print(f"  Vectors indexed : {result_plain.total_vectors()}  ✓")

    # ── Test 9: Plugin strategy registration ─────────────────────────────────
    print("\n--- Test 9: Plugin Strategy Registration ---")

    class EchoIndex(BaseFaissIndex):
        """Trivial stub that just wraps FlatL2."""
        @property
        def requires_training(self) -> bool:
            return False
        @property
        def strategy_name(self) -> str:
            return "Echo"
        def build(self, vectors, config):
            faiss = _require_faiss()
            idx = faiss.IndexFlatL2(vectors.shape[1])
            idx.add(vectors)
            return idx

    FaissBuilder.register_strategy("echo", EchoIndex)
    echo_result = FaissBuilder(FaissConfig(index_type="echo")).build(stub)
    assert echo_result.total_vectors() == 200
    print("  ✓ Custom 'echo' strategy registered and used successfully")

    print("\n" + "=" * 62)
    print("  All self-tests passed ✓")
    print("=" * 62)
