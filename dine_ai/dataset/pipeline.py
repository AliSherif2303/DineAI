"""
DineAI — Dataset Pipeline
==========================
Stateless adapter orchestrator and EmbeddingProvider abstraction layer.

This module contains two distinct areas of responsibility:

1. EmbeddingProvider Abstraction
   ─────────────────────────────
   ``BaseEmbeddingProvider`` is a Strategy interface that wraps the
   existing ``EmbeddingBuilder`` pipeline.  ``DatasetPipeline`` depends
   *only* on this interface — it never directly instantiates any specific
   embedding backend.

   Concrete providers ship with this module:
       SentenceTransformerProvider   — HuggingFace SBERT models
       MockEmbeddingProvider         — Deterministic random vectors (testing / offline)
       BGEProvider                   — BAAI BGE family
       E5Provider                    — Microsoft E5 family

2. DatasetPipeline
   ─────────────────
   Orchestrates the six existing DineAI adapters in sequence and saves
   all four artifacts to disk.  It contains ZERO business logic —
   it only routes DataFrames and results between existing modules.

   Three build modes:
       build()                 — Full pipeline (CSV → all four artifacts)
       build_embeddings_only() — Partial: reload recipes.pkl, re-embed, re-index
       build_faiss_only()      — Partial: reload embeddings.npy, re-index only

Architecture
────────────
DatasetPipeline
    │
    ├─ SchemaMapper.fit_transform(df)
    ├─ DataFrameValidator.validate(df)
    ├─ FeatureEngineer.transform(df)
    ├─ TextBuilder.transform(df)
    ├─ BaseEmbeddingProvider.build_embeddings(df)   ← Strategy
    ├─ FaissBuilder.build(embedding_result)
    ├─ pickle.dump(df)                              → recipes.pkl
    ├─ EmbeddingResult.save()                       → embeddings.npy
    ├─ FaissResult.save()                           → faiss.index
    └─ DatasetMetadata.save()                       → metadata.json
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from dine_ai.adapters.schema_mapper import SchemaMapper
from dine_ai.adapters.validator import DataFrameValidator, ValidationResult
from dine_ai.adapters.feature_engineering import FeatureEngineer
from dine_ai.adapters.text_builder import TextBuilder
from dine_ai.embeddings.embedding_builder import (
    BaseEmbeddingModel,
    EmbeddingBuilder,
    EmbeddingConfig,
    EmbeddingResult,
)
from dine_ai.embeddings.faiss_builder import FaissBuilder, FaissConfig, FaissResult
from dine_ai.dataset.metadata import (
    DatasetMetadata,
    FRAMEWORK_VERSION,
    FRAMEWORK_BUILD_NUMBER,
    DATASET_VERSION_DEFAULT,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_RECIPES_FILENAME: str = "recipes.pkl"
_DEFAULT_TEXT_COLUMN: str = "search_text"


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class DatasetBuildError(Exception):
    """
    Raised when ``DatasetPipeline`` cannot complete a build step.

    Wraps underlying adapter exceptions with context about which pipeline
    stage failed.
    """


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: EMBEDDING PROVIDER ABSTRACTION
# ─────────────────────────────────────────────────────────────────────────────

class BaseEmbeddingProvider(ABC):
    """
    Strategy interface for embedding backends used by ``DatasetPipeline``.

    ``DatasetPipeline`` depends exclusively on this interface — never on
    specific model classes.  Adding a new embedding backend requires only
    a new ``BaseEmbeddingProvider`` subclass; the pipeline code is unchanged.

    Implementing a custom provider
    ─────────────────────────────
    >>> class MyCustomProvider(BaseEmbeddingProvider):
    ...     def build_embeddings(self, df: pd.DataFrame) -> EmbeddingResult:
    ...         config = EmbeddingConfig(text_column="search_text")
    ...         builder = EmbeddingBuilder(config, model=MyModel())
    ...         return builder.build(df)
    ...
    ...     @property
    ...     def provider_name(self) -> str: return "MyCustom"
    ...     @property
    ...     def model_version(self) -> str: return "1.0"
    ...     @property
    ...     def embedding_config(self) -> EmbeddingConfig:
    ...         return EmbeddingConfig(text_column="search_text")
    """

    @abstractmethod
    def build_embeddings(self, df: pd.DataFrame) -> EmbeddingResult:
        """
        Embeds the ``search_text`` column of ``df`` into dense vectors.

        Parameters
        ----------
        df : pd.DataFrame
            Enriched DataFrame containing at minimum a ``search_text`` column.

        Returns
        -------
        EmbeddingResult
            Structured result containing the embedding matrix, index, and
            execution statistics.
        """
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable identifier (e.g. ``'SentenceTransformerProvider'``)."""
        ...

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Version string for the underlying embedding model or library."""
        ...

    @property
    @abstractmethod
    def embedding_config(self) -> EmbeddingConfig:
        """
        The ``EmbeddingConfig`` this provider uses internally.

        Exposed so ``DatasetManager`` can compute the ``embedding_configuration_hash``
        for change-detection without knowing the provider's implementation.
        """
        ...

    def config_dict_for_hash(self) -> Dict[str, Any]:
        """
        Returns a JSON-serialisable dict of the fields that affect embedding
        *output* (not pipeline speed or device placement).

        Used by ``DatasetMetadata.compute_config_hash()`` to detect when
        a configuration change requires a partial rebuild.

        The default implementation extracts ``model_name`` and
        ``normalize_embeddings`` from ``self.embedding_config``.
        Override if your provider has additional output-affecting settings.
        """
        cfg = self.embedding_config
        return {
            "provider": self.provider_name,
            "model_name": cfg.model_name,
            "normalize_embeddings": cfg.normalize_embeddings,
        }


# ── Private mock model (used by MockEmbeddingProvider) ──────────────────────

class _MockEmbeddingModel(BaseEmbeddingModel):
    """
    Minimal deterministic embedding model for offline testing.

    Generates pseudo-random vectors seeded with a hash of the input text so
    that repeated calls with the same text produce identical vectors.  The
    dimensionality is fixed at 384 to match popular SBERT models.
    """

    DIM: int = 384

    def load(self, config: EmbeddingConfig) -> None:
        pass  # No-op; nothing to load

    def encode(self, texts: List[str], config: EmbeddingConfig) -> np.ndarray:
        import hashlib
        import re
        vectors = np.zeros((len(texts), self.DIM), dtype=np.float32)
        for i, text in enumerate(texts):
            words = re.findall(r'[a-zA-Z0-9_]+', text.lower())
            v = np.zeros(self.DIM, dtype=np.float32)
            for word in words:
                for k in range(8):
                    word_hash = hashlib.sha256(f"{word}_{k}".encode("utf-8")).digest()
                    idx = int.from_bytes(word_hash[:4], "little") % self.DIM
                    v[idx] += 1.0
            
            text_seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")
            rng = np.random.default_rng(text_seed)
            v += rng.random(self.DIM).astype(np.float32) * 0.05
            
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            vectors[i] = v
        return vectors

    @property
    def embedding_dim(self) -> int:
        return self.DIM

    @property
    def provider_name(self) -> str:
        return "MockEmbeddingModel"


# ─────────────────────────────────────────────────────────────────────────────
# CONCRETE PROVIDERS
# ─────────────────────────────────────────────────────────────────────────────

class MockEmbeddingProvider(BaseEmbeddingProvider):
    """
    Embedding provider backed by a deterministic mock model.

    Intended for unit tests, CI pipelines, and offline development where
    real embedding models are unavailable or would be too slow.

    Parameters
    ----------
    text_column : str
        DataFrame column to embed.  Default: ``"search_text"``.
    """

    def __init__(self, text_column: str = _DEFAULT_TEXT_COLUMN) -> None:
        self._config = EmbeddingConfig(
            text_column=text_column,
            model_name="mock-384-v1",
            show_progress=False,
        )
        self._model = _MockEmbeddingModel()
        self._builder = EmbeddingBuilder(config=self._config, model=self._model)

    def build_embeddings(self, df: pd.DataFrame) -> EmbeddingResult:
        return self._builder.build(df)

    @property
    def provider_name(self) -> str:
        return "MockEmbeddingProvider"

    @property
    def model_version(self) -> str:
        return "mock-1.0"

    @property
    def embedding_config(self) -> EmbeddingConfig:
        return self._config

    def config_dict_for_hash(self) -> Dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model_name": self._config.model_name,
            "normalize_embeddings": self._config.normalize_embeddings,
        }


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """
    Embedding provider backed by the ``sentence-transformers`` library.

    Supports any HuggingFace SBERT model.  Automatically falls back to
    ``MockEmbeddingProvider`` if ``sentence-transformers`` is not installed,
    so the dataset pipeline never crashes in offline environments.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.  Default: ``"all-MiniLM-L6-v2"``.
    device : str
        Inference device (``"cpu"``, ``"cuda"``, ``"cuda:0"``).
        Default: ``"cpu"``.
    normalize_embeddings : bool
        L2-normalise output vectors.  Default: ``True``.
    text_column : str
        DataFrame column to embed.  Default: ``"search_text"``.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        text_column: str = _DEFAULT_TEXT_COLUMN,
    ) -> None:
        self._model_name = model_name
        self._config = EmbeddingConfig(
            text_column=text_column,
            model_name=model_name,
            device=device,
            normalize_embeddings=normalize_embeddings,
            show_progress=False,
        )
        self._resolved_model: Optional[BaseEmbeddingModel] = None
        self._resolved_version: str = "unknown"

    def _resolve_model(self) -> BaseEmbeddingModel:
        """Lazy-loads the real model; falls back to mock if unavailable."""
        if self._resolved_model is not None:
            return self._resolved_model
        try:
            from dine_ai.embeddings.embedding_builder import SentenceTransformerEmbedding
            import importlib.metadata as im
            try:
                self._resolved_version = im.version("sentence-transformers")
            except Exception:
                self._resolved_version = "unknown"
            self._resolved_model = SentenceTransformerEmbedding()
            logger.info(
                "SentenceTransformerProvider: Using model '%s' (sentence-transformers %s)",
                self._model_name, self._resolved_version,
            )
        except ImportError:
            logger.warning(
                "SentenceTransformerProvider: 'sentence-transformers' not installed. "
                "Falling back to MockEmbeddingModel."
            )
            self._resolved_model = _MockEmbeddingModel()
            self._resolved_version = "mock-fallback"
        return self._resolved_model

    def build_embeddings(self, df: pd.DataFrame) -> EmbeddingResult:
        model = self._resolve_model()
        builder = EmbeddingBuilder(config=self._config, model=model)
        return builder.build(df)

    @property
    def provider_name(self) -> str:
        return "SentenceTransformerProvider"

    @property
    def model_version(self) -> str:
        self._resolve_model()   # ensure version is populated
        return self._resolved_version

    @property
    def embedding_config(self) -> EmbeddingConfig:
        return self._config


class BGEProvider(BaseEmbeddingProvider):
    """
    Embedding provider using the BAAI BGE model family via sentence-transformers.

    BGE (BAAI General Embedding) models are high-quality multilingual
    embedding models optimised for semantic search and retrieval.

    Parameters
    ----------
    model_name : str
        BGE model identifier.  Default: ``"BAAI/bge-base-en-v1.5"``.
    device : str
        Inference device.  Default: ``"cpu"``.
    text_column : str
        DataFrame column to embed.  Default: ``"search_text"``.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
        text_column: str = _DEFAULT_TEXT_COLUMN,
    ) -> None:
        self._delegate = SentenceTransformerProvider(
            model_name=model_name,
            device=device,
            normalize_embeddings=True,   # BGE requires normalisation for cosine
            text_column=text_column,
        )

    def build_embeddings(self, df: pd.DataFrame) -> EmbeddingResult:
        return self._delegate.build_embeddings(df)

    @property
    def provider_name(self) -> str:
        return "BGEProvider"

    @property
    def model_version(self) -> str:
        return self._delegate.model_version

    @property
    def embedding_config(self) -> EmbeddingConfig:
        return self._delegate.embedding_config

    def config_dict_for_hash(self) -> Dict[str, Any]:
        d = self._delegate.config_dict_for_hash()
        d["provider"] = self.provider_name
        return d


class E5Provider(BaseEmbeddingProvider):
    """
    Embedding provider using the Microsoft E5 model family via sentence-transformers.

    E5 (Embeddings from bidirectional Encoder representations) models are
    state-of-the-art text embedding models tuned for retrieval tasks.

    Parameters
    ----------
    model_name : str
        E5 model identifier.  Default: ``"intfloat/e5-base-v2"``.
    device : str
        Inference device.  Default: ``"cpu"``.
    text_column : str
        DataFrame column to embed.  Default: ``"search_text"``.
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        device: str = "cpu",
        text_column: str = _DEFAULT_TEXT_COLUMN,
    ) -> None:
        self._delegate = SentenceTransformerProvider(
            model_name=model_name,
            device=device,
            normalize_embeddings=True,
            text_column=text_column,
        )

    def build_embeddings(self, df: pd.DataFrame) -> EmbeddingResult:
        return self._delegate.build_embeddings(df)

    @property
    def provider_name(self) -> str:
        return "E5Provider"

    @property
    def model_version(self) -> str:
        return self._delegate.model_version

    @property
    def embedding_config(self) -> EmbeddingConfig:
        return self._delegate.embedding_config

    def config_dict_for_hash(self) -> Dict[str, Any]:
        d = self._delegate.config_dict_for_hash()
        d["provider"] = self.provider_name
        return d


# ─────────────────────────────────────────────────────────────────────────────
# DTO: PipelineArtifacts
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineArtifacts:
    """
    Immutable result DTO produced by a ``DatasetPipeline`` build run.

    Carries every intermediate and final result so callers can inspect
    any stage's output without re-running the pipeline.

    Attributes
    ----------
    restaurant_name : str
        Name of the restaurant this build belongs to.
    output_dir : str
        Absolute path to the directory where artifacts were saved.
    schema_mapping : Dict
        Column-mapping result from SchemaMapper.
    validation_result : ValidationResult
        Accumulated result from DataFrameValidator.
    enriched_df : pd.DataFrame
        DataFrame after schema mapping, validation, feature engineering,
        and text building.  This is the DataFrame pickled as ``recipes.pkl``.
    embedding_result : EmbeddingResult
        Result from the embedding provider.
    faiss_result : FaissResult
        Result from FaissBuilder.
    metadata : DatasetMetadata
        The provenance record saved alongside the artifacts.
    build_time_seconds : float
        Wall-clock time for the entire build run.
    warnings : List[str]
        Non-fatal warnings accumulated from all pipeline stages.
    """

    restaurant_name: str
    output_dir: str
    schema_mapping: Dict = field(default_factory=dict)
    validation_result: Optional[ValidationResult] = None
    enriched_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    embedding_result: Optional[EmbeddingResult] = None
    faiss_result: Optional[FaissResult] = None
    metadata: Optional[DatasetMetadata] = None
    build_time_seconds: float = 0.0
    warnings: List[str] = field(default_factory=list)
    capability_report: Any = None


# ─────────────────────────────────────────────────────────────────────────────
# CLASS: DatasetPipeline
# ─────────────────────────────────────────────────────────────────────────────

class DatasetPipeline:
    """
    Stateless orchestrator that sequences the six DineAI adapters to produce
    a fully built restaurant dataset from a CSV file.

    The pipeline contains ZERO business logic.  It only routes DataFrames
    and result objects between existing, independently-tested adapters.

    Dependency Injection
    --------------------
    All adapters are accepted via constructor parameters for testability and
    extensibility.  Concrete defaults are provided for production use.

    Build Modes
    -----------
    ``build(csv_path, output_dir, restaurant_name)``
        Full 8-step pipeline.  Required when ``recipes.csv`` changes or
        artifacts are missing.

    ``build_embeddings_only(output_dir, restaurant_name)``
        Partial rebuild.  Loads the existing ``recipes.pkl`` and re-runs
        only the embedding and FAISS steps.  Used when the embedding model
        or configuration changes.

    ``build_faiss_only(output_dir, restaurant_name)``
        Minimal rebuild.  Loads the existing ``embeddings.npy`` and
        re-runs only the FAISS build step.

    Parameters
    ----------
    schema_mapper : Optional[SchemaMapper]
        Column normaliser and alias resolver.
    validator : Optional[DataFrameValidator]
        DataFrame quality validator.
    feature_engineer : Optional[FeatureEngineer]
        Feature enrichment coordinator.
    text_builder : Optional[TextBuilder]
        Text chunk generator for embedding and LLM contexts.
    embedding_provider : Optional[BaseEmbeddingProvider]
        Embedding strategy (defaults to ``SentenceTransformerProvider``
        with automatic fallback to ``MockEmbeddingProvider``).
    faiss_builder : Optional[FaissBuilder]
        FAISS index builder.
    """

    def __init__(
        self,
        schema_mapper: Optional[SchemaMapper] = None,
        validator: Optional[DataFrameValidator] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
        text_builder: Optional[TextBuilder] = None,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        faiss_builder: Optional[FaissBuilder] = None,
        capability_detector: Optional[Any] = None,
    ) -> None:
        self._schema_mapper = schema_mapper or SchemaMapper()
        self._validator = validator or DataFrameValidator()
        self._feature_engineer = feature_engineer or FeatureEngineer()
        self._text_builder = text_builder or TextBuilder()
        self._embedding_provider = embedding_provider or SentenceTransformerProvider()
        self._faiss_builder = faiss_builder or FaissBuilder(
            FaissConfig(index_type="flat_ip", normalize_before_index=True)
        )
        from dine_ai.capabilities.detector import CapabilityDetector
        self._capability_detector = capability_detector or CapabilityDetector()

    @property
    def embedding_provider(self) -> BaseEmbeddingProvider:
        """Exposes the active embedding provider for hash computation."""
        return self._embedding_provider

    # ──────────────────────────────────────────────────────────────────────────
    # Public Build API
    # ──────────────────────────────────────────────────────────────────────────

    def build(
        self,
        csv_path: str,
        output_dir: str,
        restaurant_name: str,
    ) -> PipelineArtifacts:
        """
        Executes the full 8-step dataset build pipeline.

        Steps
        -----
        1. Read CSV
        2. SchemaMapper.fit_transform()
        3. DataFrameValidator.validate()
        4. FeatureEngineer.transform()
        5. TextBuilder.transform()
        6. EmbeddingProvider.build_embeddings()
        7. FaissBuilder.build()
        8. Save all four artifacts + print build report

        Parameters
        ----------
        csv_path : str
            Path to the source ``recipes.csv``.
        output_dir : str
            Target directory for all generated artifacts.
        restaurant_name : str
            Logical restaurant identifier (stored in metadata).

        Returns
        -------
        PipelineArtifacts
            Complete build result including all intermediate outputs.

        Raises
        ------
        DatasetBuildError
            If any pipeline stage fails.
        FileNotFoundError
            If ``csv_path`` does not exist.
        """
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"DatasetPipeline.build: CSV not found at '{csv_path}'"
            )

        t_start = time.perf_counter()
        warnings: List[str] = []
        os.makedirs(output_dir, exist_ok=True)

        logger.info("DatasetPipeline: Starting full build for '%s'", restaurant_name)

        # ── Step 1: Read CSV ──────────────────────────────────────────────────
        try:
            df = pd.read_csv(csv_path)
            logger.info("DatasetPipeline [1/8]: Read CSV — %d rows, %d columns",
                        len(df), len(df.columns))
        except Exception as exc:
            raise DatasetBuildError(f"[Step 1] Failed to read CSV '{csv_path}': {exc}") from exc

        # ── Step 2: Schema Mapping ────────────────────────────────────────────
        try:
            df, schema_result = self._schema_mapper.fit_transform(df)
            logger.info("DatasetPipeline [2/8]: Schema mapped — %d columns matched",
                        len(schema_result.get("matched", [])))
        except Exception as exc:
            raise DatasetBuildError(f"[Step 2] SchemaMapper failed: {exc}") from exc

        # ── Step 3: Validation ────────────────────────────────────────────────
        try:
            val_result = self._validator.validate(df)
            if not val_result.is_valid:
                error_msgs = [e.message for e in val_result.errors]
                raise DatasetBuildError(
                    f"[Step 3] Validation failed with {len(val_result.errors)} error(s): "
                    f"{error_msgs[:3]}"
                )
            warnings.extend(w.message for w in val_result.warnings)
            logger.info("DatasetPipeline [3/8]: Validation passed (%d warning(s))",
                        len(val_result.warnings))
        except DatasetBuildError:
            raise
        except Exception as exc:
            raise DatasetBuildError(f"[Step 3] DataFrameValidator failed: {exc}") from exc

        # ── Step 4: Feature Engineering ───────────────────────────────────────
        try:
            df = self._feature_engineer.transform(df)
            logger.info("DatasetPipeline [4/8]: Feature engineering complete — %d columns",
                        len(df.columns))
        except Exception as exc:
            raise DatasetBuildError(f"[Step 4] FeatureEngineer failed: {exc}") from exc

        # ── Step 5: Text Building ─────────────────────────────────────────────
        try:
            df = self._text_builder.transform(df)
            logger.info("DatasetPipeline [5/8]: Text building complete")
        except Exception as exc:
            raise DatasetBuildError(f"[Step 5] TextBuilder failed: {exc}") from exc

        # ── Step 5b: Capability Detection ─────────────────────────────────────
        try:
            capability_report = self._capability_detector.detect(df, restaurant_name)
            capability_report.print_report()
            logger.info("DatasetPipeline [5b/8]: Capability detection complete — Score: %.1f%%",
                        capability_report.capability_score)
        except Exception as exc:
            raise DatasetBuildError(f"[Step 5b] CapabilityDetector failed: {exc}") from exc

        # ── Step 6: Embedding ─────────────────────────────────────────────────
        embedding_result = self._embed(df, step_label="[Step 6]")

        # ── Step 7: FAISS Index ───────────────────────────────────────────────
        faiss_result = self._build_faiss(embedding_result, step_label="[Step 7]")

        # ── Step 8: Save artifacts ────────────────────────────────────────────
        metadata = self._save_all(
            df=df,
            embedding_result=embedding_result,
            faiss_result=faiss_result,
            output_dir=output_dir,
            restaurant_name=restaurant_name,
            csv_path=csv_path,
            capability_report=capability_report,
        )

        elapsed = time.perf_counter() - t_start
        artifacts = PipelineArtifacts(
            restaurant_name=restaurant_name,
            output_dir=output_dir,
            schema_mapping=schema_result,
            validation_result=val_result,
            enriched_df=df,
            embedding_result=embedding_result,
            faiss_result=faiss_result,
            metadata=metadata,
            build_time_seconds=elapsed,
            warnings=warnings,
            capability_report=capability_report,
        )

        self._print_build_report(artifacts, csv_path)
        logger.info("DatasetPipeline: Full build completed in %.2fs", elapsed)
        return artifacts

    def build_embeddings_only(
        self,
        output_dir: str,
        restaurant_name: str,
    ) -> PipelineArtifacts:
        """
        Partial rebuild: loads ``recipes.pkl`` and re-runs embedding + FAISS.

        Use when the embedding provider or configuration changes, making the
        existing ``embeddings.npy`` and ``faiss.index`` stale.

        Parameters
        ----------
        output_dir : str
            Directory containing the existing ``recipes.pkl`` and where
            new ``embeddings.npy``, ``faiss.index``, and ``metadata.json``
            will be written.
        restaurant_name : str
            Logical restaurant identifier.

        Returns
        -------
        PipelineArtifacts

        Raises
        ------
        DatasetBuildError
            If ``recipes.pkl`` cannot be loaded or any downstream step fails.
        """
        import pickle as _pk

        t_start = time.perf_counter()
        logger.info("DatasetPipeline: Starting partial rebuild (embeddings+FAISS) for '%s'",
                    restaurant_name)

        pkl_path = os.path.join(output_dir, "recipes.pkl")
        if not os.path.isfile(pkl_path):
            raise DatasetBuildError(
                f"DatasetPipeline.build_embeddings_only: "
                f"'recipes.pkl' not found in '{output_dir}'"
            )
        try:
            with open(pkl_path, "rb") as fh:
                df = _pk.load(fh)
            if isinstance(df, list):
                df = pd.DataFrame(df)
        except Exception as exc:
            raise DatasetBuildError(
                f"DatasetPipeline.build_embeddings_only: Cannot load 'recipes.pkl': {exc}"
            ) from exc

        try:
            capability_report = self._capability_detector.detect(df, restaurant_name)
        except Exception as exc:
            capability_report = None

        embedding_result = self._embed(df, step_label="[Partial-Emb]")
        faiss_result = self._build_faiss(embedding_result, step_label="[Partial-FAISS]")
        metadata = self._save_all(
            df=df,
            embedding_result=embedding_result,
            faiss_result=faiss_result,
            output_dir=output_dir,
            restaurant_name=restaurant_name,
            csv_path=None,            # CSV path unknown in partial rebuild
            capability_report=capability_report,
        )
        elapsed = time.perf_counter() - t_start

        artifacts = PipelineArtifacts(
            restaurant_name=restaurant_name,
            output_dir=output_dir,
            enriched_df=df,
            embedding_result=embedding_result,
            faiss_result=faiss_result,
            metadata=metadata,
            build_time_seconds=elapsed,
            capability_report=capability_report,
        )
        self._print_build_report(artifacts, csv_path="(existing recipes.pkl)", partial=True)
        logger.info("DatasetPipeline: Partial rebuild (emb+FAISS) completed in %.2fs", elapsed)
        return artifacts

    def build_faiss_only(
        self,
        output_dir: str,
        restaurant_name: str,
    ) -> PipelineArtifacts:
        """
        Minimal rebuild: loads existing ``embeddings.npy`` and re-builds FAISS.

        Use when only the FAISS index is missing or when FAISS configuration
        changes without affecting the embedding vectors.

        Parameters
        ----------
        output_dir : str
            Directory containing the existing ``embeddings.npy``.
        restaurant_name : str
            Logical restaurant identifier.

        Returns
        -------
        PipelineArtifacts

        Raises
        ------
        DatasetBuildError
            If ``embeddings.npy`` cannot be loaded or FAISS build fails.
        """
        t_start = time.perf_counter()
        logger.info("DatasetPipeline: Starting minimal rebuild (FAISS only) for '%s'",
                    restaurant_name)

        try:
            embedding_result = EmbeddingResult.load(output_dir)
        except Exception as exc:
            raise DatasetBuildError(
                f"DatasetPipeline.build_faiss_only: Cannot load 'embeddings.npy': {exc}"
            ) from exc

        # Try loading existing metadata to preserve capability_report and available_columns
        try:
            old_meta = DatasetMetadata.load(output_dir)
            capability_report_dict = old_meta.capability_report
            available_columns = old_meta.available_columns
        except Exception:
            capability_report_dict = {}
            available_columns = []

        # Save FAISS + refresh metadata
        faiss_result.save(output_dir)
        metadata = self._build_metadata(
            df_rows=embedding_result.embeddings.shape[0],
            df_cols=0,
            embedding_result=embedding_result,
            faiss_result=faiss_result,
            output_dir=output_dir,
            restaurant_name=restaurant_name,
            csv_path=None,
            capability_report_dict=capability_report_dict,
            available_columns=available_columns,
        )
        metadata.save(output_dir)
        elapsed = time.perf_counter() - t_start

        artifacts = PipelineArtifacts(
            restaurant_name=restaurant_name,
            output_dir=output_dir,
            embedding_result=embedding_result,
            faiss_result=faiss_result,
            metadata=metadata,
            build_time_seconds=elapsed,
        )
        self._print_build_report(artifacts, csv_path="(existing embeddings.npy)", partial=True)
        logger.info("DatasetPipeline: FAISS-only rebuild completed in %.2fs", elapsed)
        return artifacts

    # ──────────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _embed(self, df: pd.DataFrame, step_label: str) -> EmbeddingResult:
        """Runs the embedding provider and returns EmbeddingResult."""
        try:
            result = self._embedding_provider.build_embeddings(df)
            logger.info("DatasetPipeline %s: Embedding complete — shape %s",
                        step_label, result.embeddings.shape)
            return result
        except Exception as exc:
            raise DatasetBuildError(f"{step_label} Embedding failed: {exc}") from exc

    def _build_faiss(self, embedding_result: EmbeddingResult, step_label: str) -> FaissResult:
        """Runs FaissBuilder and returns FaissResult."""
        try:
            result = self._faiss_builder.build(embedding_result)
            logger.info("DatasetPipeline %s: FAISS built — %d vectors",
                        step_label, result.statistics.total_vectors)
            return result
        except Exception as exc:
            raise DatasetBuildError(f"{step_label} FaissBuilder failed: {exc}") from exc

    def _save_all(
        self,
        df: pd.DataFrame,
        embedding_result: EmbeddingResult,
        faiss_result: FaissResult,
        output_dir: str,
        restaurant_name: str,
        csv_path: Optional[str],
        capability_report: Optional[Any] = None,
    ) -> DatasetMetadata:
        """Saves recipes.pkl, embeddings.npy, faiss.index, and metadata.json."""
        os.makedirs(output_dir, exist_ok=True)

        # recipes.pkl — enriched DataFrame
        pkl_path = os.path.join(output_dir, _RECIPES_FILENAME)
        with open(pkl_path, "wb") as fh:
            pickle.dump(df, fh)
        logger.info("DatasetPipeline [8a/8]: Saved recipes.pkl  →  %s", pkl_path)

        # embeddings.npy + embeddings_metadata.json
        embedding_result.save(output_dir, prefix="embeddings")
        logger.info("DatasetPipeline [8b/8]: Saved embeddings.npy")

        # faiss.index + faiss_metadata.json
        faiss_result.save(output_dir)
        logger.info("DatasetPipeline [8c/8]: Saved faiss.index")

        # metadata.json
        metadata = self._build_metadata(
            df_rows=len(df),
            df_cols=len(df.columns),
            embedding_result=embedding_result,
            faiss_result=faiss_result,
            output_dir=output_dir,
            restaurant_name=restaurant_name,
            csv_path=csv_path,
            capability_report_dict=capability_report.to_dict() if capability_report else {},
            available_columns=df.columns.tolist() if df is not None else [],
        )
        metadata.save(output_dir)
        logger.info("DatasetPipeline [8d/8]: Saved metadata.json")
        return metadata

    def _build_metadata(
        self,
        df_rows: int,
        df_cols: int,
        embedding_result: EmbeddingResult,
        faiss_result: FaissResult,
        output_dir: str,
        restaurant_name: str,
        csv_path: Optional[str],
        capability_report_dict: Optional[Dict[str, Any]] = None,
        available_columns: Optional[List[str]] = None,
    ) -> DatasetMetadata:
        """Constructs and populates a DatasetMetadata instance."""
        provider = self._embedding_provider
        config_hash = DatasetMetadata.compute_config_hash(provider.config_dict_for_hash())
        artifacts_hash = DatasetMetadata.compute_artifacts_hash(output_dir)
        csv_hash = ""
        if csv_path and os.path.isfile(csv_path):
            csv_hash = DatasetMetadata.compute_csv_hash(csv_path)

        dim = embedding_result.embeddings.shape[1] if embedding_result.embeddings.ndim > 1 else 0

        return DatasetMetadata(
            restaurant_name=restaurant_name,
            dataset_version=DATASET_VERSION_DEFAULT,
            framework_version=FRAMEWORK_VERSION,
            framework_build_number=FRAMEWORK_BUILD_NUMBER,
            rows=df_rows,
            columns=df_cols,
            embedding_dim=dim,
            embedding_model=provider.embedding_config.model_name,
            embedding_model_version=provider.model_version,
            embedding_configuration_hash=config_hash,
            faiss_type=faiss_result.statistics.index_type,
            build_timestamp=datetime.now(timezone.utc).isoformat(),
            csv_hash=csv_hash,
            artifacts_hash=artifacts_hash,
            capability_report=capability_report_dict or {},
            available_columns=available_columns or [],
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Enterprise Report
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _print_build_report(
        artifacts: PipelineArtifacts,
        csv_path: str,
        partial: bool = False,
    ) -> None:
        """Renders and prints the enterprise-style DATASET BUILD REPORT."""
        meta = artifacts.metadata
        if meta is None:
            return
        width = 80
        sep = "=" * width
        thin = "-" * width
        mode = "PARTIAL REBUILD" if partial else "FULL BUILD"

        emb_dim = meta.embedding_dim
        emb_hash_short = meta.embedding_configuration_hash[:12] + "..." if meta.embedding_configuration_hash else "N/A"
        art_hash_short = meta.artifacts_hash[:12] + "..." if meta.artifacts_hash else "N/A"
        csv_hash_short = meta.csv_hash[:12] + "..." if meta.csv_hash else "N/A"

        features_generated = 0
        if artifacts.enriched_df is not None and not artifacts.enriched_df.empty:
            orig_cols = len(artifacts.schema_mapping) if artifacts.schema_mapping else 0
            features_generated = max(0, len(artifacts.enriched_df.columns) - orig_cols)

        print(sep)
        print(f"DINEAI DATASET BUILD REPORT  [{mode}]".center(width))
        print(sep)
        print(f" Restaurant         : {meta.restaurant_name}")
        print(f" CSV Path           : {csv_path}")
        print(f" Rows               : {meta.rows}  |  Columns : {meta.columns}"
              f"  |  Features Generated : {features_generated}")
        print(thin)
        print(f" Embedding Provider : {artifacts.metadata.embedding_model}")
        emb_provider_name = ""
        if artifacts.embedding_result:
            emb_provider_name = artifacts.embedding_result.statistics.model_name
        print(f" Embedding Model    : {emb_provider_name or meta.embedding_model}")
        print(f" Embedding Model Ver: {meta.embedding_model_version}")
        print(f" Embedding Dim      : {emb_dim}")
        print(f" Emb Config Hash    : {emb_hash_short}")
        print(f" FAISS Type         : {meta.faiss_type}")
        print(thin)
        print(f" Artifacts Saved    : recipes.pkl, embeddings.npy, faiss.index, metadata.json")
        print(f" Artifacts Hash     : {art_hash_short}")
        print(f" CSV Hash           : {csv_hash_short}")
        print(f" Python Version     : {meta.python_version.splitlines()[0]}")
        print(f" Framework Version  : {meta.framework_version}")
        print(f" Framework Build    : {meta.framework_build_number}")
        print(f" Total Runtime      : {artifacts.build_time_seconds:.4f}s")
        print(thin)
        print(f" Status             : SUCCESS")
        print(sep)
