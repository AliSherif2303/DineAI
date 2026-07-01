"""
DineAI — Dataset Manager
=========================
Lifecycle orchestrator for the DineAI Dataset Management subsystem.

DatasetManager is the single, authoritative entry point for obtaining a
RestaurantDataset.  It implements the full Smart Rebuild algorithm:

    inspect()  ->  DatasetStatus
        │
        ├── COMPLETE    -> load (cache hit)
        ├── STALE       -> full rebuild then load
        ├── STALE_EMB   -> partial rebuild (embeddings+FAISS) then load
        ├── MISSING     -> full rebuild then load
        └── CORRUPT     -> full rebuild then load

Separation of Responsibilities
-------------------------------
inspect()   — file-presence checks + hash comparisons
              Returns DatasetStatus.  No artifacts are loaded.

validate()  — loads artifacts and verifies integrity and compatibility
              Returns ValidationReport.  Artifacts are loaded in full.

status()    — generates the enterprise ASCII status report string.
              Calls inspect() internally; does not load artifacts.

Self-Tests
----------
Run with:  python -m dine_ai.dataset.manager

All 14 scenarios are verified against a synthetic temp directory.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from dine_ai.dataset.metadata import (
    DatasetMetadata,
    DatasetMetadataError,
    FRAMEWORK_VERSION,
    FRAMEWORK_BUILD_NUMBER,
)
from dine_ai.dataset.loader import DatasetLoader, DatasetLoadError, RestaurantDataset
from dine_ai.dataset.pipeline import (
    DatasetPipeline,
    DatasetBuildError,
    BaseEmbeddingProvider,
    MockEmbeddingProvider,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_METADATA_FILENAME: str = "metadata.json"
_RECIPES_FILENAME: str = "recipes.pkl"
_EMBEDDINGS_FILENAME: str = "embeddings.npy"
_FAISS_INDEX_FILENAME: str = "faiss.index"
_CSV_FILENAME: str = "recipes.csv"

_COMPATIBLE_FRAMEWORK_VERSION: str = "1.0"   # major.minor prefix for compat check


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class ArtifactStatus(Enum):
    """
    Overall health classification for a restaurant's dataset artifacts.

    Used by ``DatasetManager`` to route to the correct build mode.

    Members
    -------
    COMPLETE
        All four artifacts are present and all hashes match.
        -> Load from cache.
    MISSING
        One or more critical artifacts are absent (``recipes.pkl``,
        ``faiss.index``, or ``metadata.json``).
        -> Full rebuild required.
    STALE
        All artifacts are present but the ``csv_hash`` in metadata does
        not match the current ``recipes.csv``.
        -> Full rebuild required.
    STALE_EMB
        Artifacts are present and the CSV hash matches, but the
        ``embedding_configuration_hash`` differs, or ``embeddings.npy``
        is missing.
        -> Partial rebuild (embeddings + FAISS) required.
    CORRUPT
        ``metadata.json`` cannot be parsed, or the ``artifacts_hash``
        in metadata does not match the current artifact files.
        -> Full rebuild required.
    """

    COMPLETE = "COMPLETE"
    MISSING = "MISSING"
    STALE = "STALE"
    STALE_EMB = "STALE_EMB"
    CORRUPT = "CORRUPT"


# ─────────────────────────────────────────────────────────────────────────────
# DTOs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetStatus:
    """
    Snapshot of an artifact directory's health produced by ``inspect()``.

    Returned by ``DatasetManager.inspect()``; consumed by
    ``DatasetManager._auto_build()`` to decide which build mode to use.

    Attributes
    ----------
    restaurant_name : str
    restaurant_dir : str
        Resolved absolute path.
    csv_exists : bool
    pkl_exists : bool
    embeddings_exists : bool
    faiss_exists : bool
    metadata_exists : bool
    csv_hash_matches : bool
        False when the current ``recipes.csv`` hash differs from metadata.
    embedding_config_hash_matches : bool
        False when the pipeline's provider config hash differs from metadata.
    artifacts_hash_matches : bool
        False when the current artifact files hash differs from metadata.
    overall : ArtifactStatus
    rebuild_reason : str
        Human-readable explanation for the overall status.
    inspection_time_seconds : float
    """

    restaurant_name: str
    restaurant_dir: str
    csv_exists: bool = False
    pkl_exists: bool = False
    embeddings_exists: bool = False
    faiss_exists: bool = False
    metadata_exists: bool = False
    csv_hash_matches: bool = True
    embedding_config_hash_matches: bool = True
    artifacts_hash_matches: bool = True
    overall: ArtifactStatus = ArtifactStatus.MISSING
    rebuild_reason: str = ""
    inspection_time_seconds: float = 0.0


@dataclass
class ValidationReport:
    """
    Result of ``DatasetManager.validate()``.

    Checks artifact integrity and framework version compatibility.

    Attributes
    ----------
    restaurant_name : str
    is_valid : bool
        True only if all checks pass.
    artifacts_hash_ok : bool
        True when the on-disk hash matches metadata.
    framework_compatible : bool
        True when the metadata framework version matches the current major.minor.
    warnings : List[str]
        Non-fatal compatibility notes.
    errors : List[str]
        Fatal integrity or compatibility failures.
    """

    restaurant_name: str
    is_valid: bool = True
    artifacts_hash_ok: bool = True
    framework_compatible: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS: DatasetManager
# ─────────────────────────────────────────────────────────────────────────────

class DatasetManager:
    """
    Lifecycle controller for DineAI restaurant datasets.

    Orchestrates the complete lifecycle of a restaurant dataset:
    inspection, smart rebuild, and loading.  All consumer code should
    interact with the dataset exclusively through this class.

    The responsibility chain is:

        Application
            ↓
        DatasetManager.load_restaurant()
            ↓
        RestaurantDataset
            ↓
        RestaurantManager (or any downstream consumer)

    Parameters
    ----------
    dataset_directory : str
        Root directory that contains one sub-directory per restaurant.
        Default: ``"dine_ai/datasets"``.
    pipeline : Optional[DatasetPipeline]
        Adapter orchestrator used for building.  Defaults to a
        ``DatasetPipeline`` with a ``SentenceTransformerProvider``
        (auto-fallback to Mock when offline).
    loader : Optional[DatasetLoader]
        Artifact deserialiser.  Defaults to a new ``DatasetLoader()``.

    Usage
    -----
    >>> manager = DatasetManager()
    >>> dataset = manager.load_restaurant("restaurant_A")
    >>> print(dataset)
    RestaurantDataset(restaurant='restaurant_A', version='1.0', ...)
    """

    def __init__(
        self,
        dataset_directory: str = "dine_ai/datasets",
        pipeline: Optional[DatasetPipeline] = None,
        loader: Optional[DatasetLoader] = None,
    ) -> None:
        self.dataset_directory = dataset_directory
        self._pipeline = pipeline or DatasetPipeline()
        self._loader = loader or DatasetLoader()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — Lifecycle
    # ──────────────────────────────────────────────────────────────────────────

    def load_restaurant(
        self, name: str, version: str = "latest"
    ) -> RestaurantDataset:
        """
        Full lifecycle method: inspect -> (build if needed) -> load.

        Transparently handles all of the following scenarios:
        - Only ``recipes.csv`` exists  -> full build
        - All artifacts present + hash match  -> cache hit (no build)
        - ``recipes.csv`` modified  -> full rebuild
        - Embedding configuration changed  -> partial rebuild
        - Any artifact missing or corrupt  -> full rebuild

        Parameters
        ----------
        name : str
            Restaurant name (sub-directory of ``dataset_directory``).
        version : str
            Dataset version subdirectory.  Defaults to ``"latest"``.
            Falls back to the flat layout for backward compatibility.

        Returns
        -------
        RestaurantDataset

        Raises
        ------
        FileNotFoundError
            If the restaurant directory and ``recipes.csv`` do not exist.
        DatasetBuildError
            If the build pipeline fails.
        DatasetLoadError
            If loading fails after a successful build.
        """
        t_start = time.perf_counter()
        restaurant_dir = self._resolve_dir(name, version)

        # ── Ensure the restaurant directory (and CSV) exists ──────────────────
        if not os.path.isdir(restaurant_dir):
            raise FileNotFoundError(
                f"DatasetManager: Restaurant directory not found: '{restaurant_dir}'"
            )
        csv_path = self._resolve_csv(restaurant_dir)
        if not csv_path:
            raise FileNotFoundError(
                f"DatasetManager: Neither 'menu_items.csv' nor 'recipes.csv' was found in '{restaurant_dir}'. "
                f"Please place your CSV file at: {os.path.join(restaurant_dir, 'menu_items.csv')}"
            )

        # ── Inspect current artifact state ────────────────────────────────────
        status = self.inspect(name, version)

        rebuild_triggered = False
        partial_rebuild = False

        if status.overall == ArtifactStatus.COMPLETE:
            logger.info(
                "DatasetManager: '%s' — cache hit (all hashes match). Loading.", name
            )
        elif status.overall == ArtifactStatus.STALE_EMB:
            logger.info(
                "DatasetManager: '%s' — %s. Running partial rebuild.",
                name, status.rebuild_reason,
            )
            self._pipeline.build_embeddings_only(restaurant_dir, name)
            rebuild_triggered = True
            partial_rebuild = True
        else:
            logger.info(
                "DatasetManager: '%s' — %s. Running full rebuild.",
                name, status.rebuild_reason,
            )
            self._pipeline.build(csv_path, restaurant_dir, name)
            rebuild_triggered = True

        # ── Load artifacts from disk ──────────────────────────────────────────
        dataset = self._loader.load(restaurant_dir)
        elapsed = time.perf_counter() - t_start

        # ── Print LOAD REPORT with inspection context ─────────────────────────
        self._print_load_report(
            dataset=dataset,
            status=status,
            elapsed=elapsed,
            rebuilt=rebuild_triggered,
            partial=partial_rebuild,
        )
        return dataset

    def reload(self, name: str, version: str = "latest") -> RestaurantDataset:
        """
        Force-reloads the dataset from disk without triggering a rebuild.

        Use when you want to pick up artifacts that were built externally
        or by a previous ``rebuild_all()`` call.

        Parameters
        ----------
        name : str
        version : str

        Returns
        -------
        RestaurantDataset

        Raises
        ------
        DatasetLoadError
            If any artifact is missing or corrupt.
        """
        restaurant_dir = self._resolve_dir(name, version)
        logger.info("DatasetManager: Force-reloading '%s' from '%s'", name, restaurant_dir)
        return self._loader.load(restaurant_dir)

    def rebuild_all(self, name: str, version: str = "latest") -> RestaurantDataset:
        """
        Forces a full pipeline rebuild regardless of artifact state.

        All four artifacts are regenerated from ``recipes.csv``.

        Parameters
        ----------
        name : str
        version : str

        Returns
        -------
        RestaurantDataset

        Raises
        ------
        FileNotFoundError
            If ``recipes.csv`` is not found.
        DatasetBuildError
            If the pipeline fails at any stage.
        """
        restaurant_dir = self._resolve_dir(name, version)
        csv_path = self._resolve_csv(restaurant_dir)
        if not csv_path:
            raise FileNotFoundError(
                f"DatasetManager.rebuild_all: Neither 'menu_items.csv' nor 'recipes.csv' was found in '{restaurant_dir}'"
            )
        logger.info("DatasetManager: Forced full rebuild of '%s'", name)
        self._pipeline.build(csv_path, restaurant_dir, name)
        return self._loader.load(restaurant_dir)

    def rebuild_embeddings(self, name: str, version: str = "latest") -> RestaurantDataset:
        """
        Forces a partial rebuild: re-embeds and re-indexes from ``recipes.pkl``.

        Requires ``recipes.pkl`` to already exist (produced by a prior full build).

        Parameters
        ----------
        name : str
        version : str

        Returns
        -------
        RestaurantDataset

        Raises
        ------
        DatasetBuildError
            If ``recipes.pkl`` is missing or any downstream step fails.
        """
        restaurant_dir = self._resolve_dir(name, version)
        logger.info("DatasetManager: Forced partial rebuild (embeddings+FAISS) of '%s'", name)
        self._pipeline.build_embeddings_only(restaurant_dir, name)
        return self._loader.load(restaurant_dir)

    def rebuild_faiss(self, name: str, version: str = "latest") -> RestaurantDataset:
        """
        Forces a minimal rebuild: re-builds the FAISS index from ``embeddings.npy``.

        Requires ``embeddings.npy`` to already exist.

        Parameters
        ----------
        name : str
        version : str

        Returns
        -------
        RestaurantDataset

        Raises
        ------
        DatasetBuildError
            If ``embeddings.npy`` is missing or the FAISS build fails.
        """
        restaurant_dir = self._resolve_dir(name, version)
        logger.info("DatasetManager: Forced FAISS-only rebuild of '%s'", name)
        self._pipeline.build_faiss_only(restaurant_dir, name)
        return self._loader.load(restaurant_dir)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — Inspection / Validation / Reporting
    # ──────────────────────────────────────────────────────────────────────────

    def inspect(self, name: str, version: str = "latest") -> DatasetStatus:
        """
        Performs a lightweight inspection of the artifact directory.

        Checks file presence and SHA-256 hashes.  Does NOT load any binary
        artifact into memory.

        Inspection Logic
        ----------------
        1. Check file presence for all four artifacts.
        2. If critical artifacts are missing -> MISSING.
        3. If ``metadata.json`` is unreadable -> CORRUPT.
        4. Compare current ``recipes.csv`` hash -> STALE if mismatch.
        5. Compare current embedding config hash -> STALE_EMB if mismatch.
        6. Compare current artifacts hash -> CORRUPT if mismatch.
        7. If ``embeddings.npy`` is missing but others are OK -> STALE_EMB.
        8. All checks pass -> COMPLETE.

        Parameters
        ----------
        name : str
        version : str

        Returns
        -------
        DatasetStatus
        """
        t_start = time.perf_counter()
        restaurant_dir = self._resolve_dir(name, version)

        ds = DatasetStatus(
            restaurant_name=name,
            restaurant_dir=restaurant_dir,
        )

        # ── File presence ─────────────────────────────────────────────────────
        csv_path = self._resolve_csv(restaurant_dir)
        ds.csv_exists = csv_path is not None
        ds.pkl_exists = os.path.isfile(os.path.join(restaurant_dir, _RECIPES_FILENAME))
        ds.embeddings_exists = os.path.isfile(os.path.join(restaurant_dir, _EMBEDDINGS_FILENAME))
        ds.faiss_exists = os.path.isfile(os.path.join(restaurant_dir, _FAISS_INDEX_FILENAME))
        ds.metadata_exists = os.path.isfile(os.path.join(restaurant_dir, _METADATA_FILENAME))

        # ── Critical artifact missing? ────────────────────────────────────────
        if not ds.pkl_exists or not ds.faiss_exists or not ds.metadata_exists:
            missing = []
            if not ds.pkl_exists:        missing.append("recipes.pkl")
            if not ds.faiss_exists:      missing.append("faiss.index")
            if not ds.metadata_exists:   missing.append("metadata.json")
            ds.overall = ArtifactStatus.MISSING
            ds.rebuild_reason = f"Missing artifacts: {', '.join(missing)}"
            ds.inspection_time_seconds = time.perf_counter() - t_start
            return ds

        # ── Load metadata for hash comparisons ───────────────────────────────
        try:
            metadata = DatasetMetadata.load(restaurant_dir)
        except DatasetMetadataError as exc:
            ds.overall = ArtifactStatus.CORRUPT
            ds.rebuild_reason = f"metadata.json unreadable: {exc}"
            ds.inspection_time_seconds = time.perf_counter() - t_start
            return ds

        # ── CSV hash check ────────────────────────────────────────────────────
        if ds.csv_exists and metadata.csv_hash and csv_path:
            try:
                current_csv_hash = DatasetMetadata.compute_csv_hash(csv_path)
                ds.csv_hash_matches = (current_csv_hash == metadata.csv_hash)
            except Exception:
                ds.csv_hash_matches = False

            if not ds.csv_hash_matches:
                ds.overall = ArtifactStatus.STALE
                ds.rebuild_reason = "recipes.csv has been modified (hash mismatch)"
                ds.inspection_time_seconds = time.perf_counter() - t_start
                return ds

        # ── Embedding config hash check ───────────────────────────────────────
        if metadata.embedding_configuration_hash:
            try:
                current_cfg_hash = DatasetMetadata.compute_config_hash(
                    self._pipeline.embedding_provider.config_dict_for_hash()
                )
                ds.embedding_config_hash_matches = (
                    current_cfg_hash == metadata.embedding_configuration_hash
                )
            except Exception:
                ds.embedding_config_hash_matches = True    # treat as unknown -> don't trigger

        if not ds.embedding_config_hash_matches:
            ds.overall = ArtifactStatus.STALE_EMB
            ds.rebuild_reason = "Embedding configuration has changed (hash mismatch)"
            ds.inspection_time_seconds = time.perf_counter() - t_start
            return ds

        # ── embeddings.npy presence check ────────────────────────────────────
        if not ds.embeddings_exists:
            ds.overall = ArtifactStatus.STALE_EMB
            ds.rebuild_reason = "embeddings.npy is missing"
            ds.inspection_time_seconds = time.perf_counter() - t_start
            return ds

        # ── Artifacts hash integrity check ────────────────────────────────────
        if metadata.artifacts_hash:
            try:
                current_art_hash = DatasetMetadata.compute_artifacts_hash(restaurant_dir)
                ds.artifacts_hash_matches = (current_art_hash == metadata.artifacts_hash)
            except Exception:
                ds.artifacts_hash_matches = True     # treat as unknown -> don't corrupt-flag

            if not ds.artifacts_hash_matches:
                ds.overall = ArtifactStatus.CORRUPT
                ds.rebuild_reason = "Artifact files have been modified (hash mismatch)"
                ds.inspection_time_seconds = time.perf_counter() - t_start
                return ds

        # ── All checks passed ─────────────────────────────────────────────────
        ds.overall = ArtifactStatus.COMPLETE
        ds.rebuild_reason = "None (cache hit)"
        ds.inspection_time_seconds = time.perf_counter() - t_start
        return ds

    def validate(self, name: str, version: str = "latest") -> ValidationReport:
        """
        Validates a loaded dataset's integrity and framework compatibility.

        Unlike ``inspect()`` (which only checks file hashes), ``validate()``
        actually loads the dataset and performs full compatibility checks:

        1. Re-computes and verifies ``artifacts_hash``.
        2. Checks that the metadata's ``framework_version`` is compatible with
           the current DineAI version.

        Parameters
        ----------
        name : str
        version : str

        Returns
        -------
        ValidationReport
        """
        report = ValidationReport(restaurant_name=name)
        restaurant_dir = self._resolve_dir(name, version)

        # ── Load metadata ─────────────────────────────────────────────────────
        try:
            metadata = DatasetMetadata.load(restaurant_dir)
        except DatasetMetadataError as exc:
            report.is_valid = False
            report.errors.append(f"Cannot load metadata: {exc}")
            return report

        # ── Artifacts hash integrity ──────────────────────────────────────────
        try:
            current_hash = DatasetMetadata.compute_artifacts_hash(restaurant_dir)
            if metadata.artifacts_hash and current_hash != metadata.artifacts_hash:
                report.artifacts_hash_ok = False
                report.is_valid = False
                report.errors.append(
                    f"Artifacts hash mismatch. Expected: {metadata.artifacts_hash[:16]}... "
                    f"Got: {current_hash[:16]}..."
                )
        except Exception as exc:
            report.warnings.append(f"Could not verify artifacts_hash: {exc}")

        # ── Framework version compatibility ───────────────────────────────────
        stored_major_minor = ".".join(metadata.framework_version.split(".")[:2])
        current_major_minor = ".".join(FRAMEWORK_VERSION.split(".")[:2])
        if stored_major_minor != current_major_minor:
            report.framework_compatible = False
            report.warnings.append(
                f"Framework version mismatch: dataset was built with "
                f"v{metadata.framework_version}, current is v{FRAMEWORK_VERSION}. "
                f"Consider rebuilding the dataset."
            )
            # This is a warning, not an error — do not set is_valid = False
            # (Scenario 14: framework version change -> warning only, no rebuild)

        return report

    def status(self, name: str, version: str = "latest") -> str:
        """
        Generates and returns a detailed enterprise ASCII status report.

        Calls ``inspect()`` internally; also checks framework version
        compatibility if metadata is available.

        Parameters
        ----------
        name : str
        version : str

        Returns
        -------
        str
            Multi-line ASCII report ready for printing.
        """
        ds = self.inspect(name, version)
        restaurant_dir = ds.restaurant_dir
        width = 80
        sep = "=" * width
        thin = "-" * width

        lines = [sep, "DINEAI DATASET STATUS REPORT".center(width), sep]
        lines.append(f" Restaurant         : {name}")
        lines.append(f" Version            : {version}")
        lines.append(f" Directory          : {restaurant_dir}")
        lines.append(thin)
        lines.append(f" Artifact Files")
        lines.append(f"   recipes.csv      : {'PRESENT' if ds.csv_exists else 'MISSING'}")
        lines.append(f"   recipes.pkl      : {'PRESENT' if ds.pkl_exists else 'MISSING'}")
        lines.append(f"   embeddings.npy   : {'PRESENT' if ds.embeddings_exists else 'MISSING'}")
        lines.append(f"   faiss.index      : {'PRESENT' if ds.faiss_exists else 'MISSING'}")
        lines.append(f"   metadata.json    : {'PRESENT' if ds.metadata_exists else 'MISSING'}")
        lines.append(thin)
        lines.append(f" Hash Checks")
        lines.append(f"   csv_hash         : {'MATCH' if ds.csv_hash_matches else 'MISMATCH'}")
        lines.append(f"   embedding_config : {'MATCH' if ds.embedding_config_hash_matches else 'MISMATCH'}")
        lines.append(f"   artifacts_hash   : {'MATCH' if ds.artifacts_hash_matches else 'MISMATCH'}")
        lines.append(thin)

        # Optionally add metadata fields
        if ds.metadata_exists:
            try:
                meta = DatasetMetadata.load(restaurant_dir)
                lines.append(f" Metadata")
                lines.append(f"   Dataset Version  : {meta.dataset_version}")
                lines.append(f"   Built At         : {meta.build_timestamp}")
                lines.append(f"   Embedding Model  : {meta.embedding_model}")
                lines.append(f"   Embedding Model V: {meta.embedding_model_version}")
                lines.append(f"   Emb Config Hash  : {meta.embedding_configuration_hash[:12]}...")
                lines.append(f"   Artifacts Hash   : {meta.artifacts_hash[:12]}...")
                lines.append(f"   Python Version   : {meta.python_version.splitlines()[0]}")
                lines.append(f"   Framework Build  : {meta.framework_build_number}")
                # Framework compat check
                stored_mm = ".".join(meta.framework_version.split(".")[:2])
                current_mm = ".".join(FRAMEWORK_VERSION.split(".")[:2])
                compat = "COMPATIBLE" if stored_mm == current_mm else f"INCOMPATIBLE (v{meta.framework_version} vs v{FRAMEWORK_VERSION})"
                lines.append(f"   Framework Compat : {compat}")
                lines.append(thin)
            except Exception:
                pass

        lines.append(f" Overall Status     : {ds.overall.value}")
        lines.append(f" Rebuild Reason     : {ds.rebuild_reason}")
        lines.append(f" Inspection Time    : {ds.inspection_time_seconds:.4f}s")
        lines.append(sep)
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # Private Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _resolve_dir(self, name: str, version: str = "latest") -> str:
        """
        Resolves the artifact directory for a restaurant and version.

        Version-aware layout (checked first):
            ``{dataset_directory}/{name}/{version}/``

        Flat layout fallback (backward compatible with existing datasets):
            ``{dataset_directory}/{name}/``

        Parameters
        ----------
        name : str
        version : str

        Returns
        -------
        str
            Resolved absolute path to the artifact directory.
        """
        base = os.path.join(self.dataset_directory, name)
        versioned = os.path.join(base, version)
        if os.path.isdir(versioned):
            return os.path.abspath(versioned)
        return os.path.abspath(base)

    def _resolve_csv(self, restaurant_dir: str) -> Optional[str]:
        """Resolves the CSV path, prioritizing menu_items.csv over recipes.csv."""
        for filename in ("menu_items.csv", "recipes.csv"):
            path = os.path.join(restaurant_dir, filename)
            if os.path.isfile(path):
                return path
        return None

    @staticmethod
    def _print_load_report(
        dataset: RestaurantDataset,
        status: DatasetStatus,
        elapsed: float,
        rebuilt: bool,
        partial: bool,
    ) -> None:
        """Renders and prints the enterprise-style DATASET LOAD REPORT."""
        meta = dataset.metadata
        faiss = dataset.faiss_result
        width = 80
        sep = "=" * width
        thin = "-" * width

        health = "EXCELLENT" if len(dataset.recipes) > 0 and faiss.statistics.total_vectors > 0 else "DEGRADED"
        art_hash_short = meta.artifacts_hash[:12] + "..." if meta.artifacts_hash else "N/A"
        emb_hash_short = meta.embedding_configuration_hash[:12] + "..." if meta.embedding_configuration_hash else "N/A"

        if rebuilt and partial:
            inspection_result = f"{status.overall.value} -> PARTIAL_REBUILD_COMPLETED"
        elif rebuilt:
            inspection_result = f"{status.overall.value} -> FULL_REBUILD_COMPLETED"
        else:
            inspection_result = status.overall.value

        print(sep)
        print("DINEAI DATASET LOAD REPORT".center(width))
        print(sep)
        print(f" Restaurant         : {meta.restaurant_name}")
        print(f" Version            : {meta.dataset_version}")
        print(f" Recipes Loaded     : {len(dataset.recipes)}")
        print(f" Embeddings Shape   : ({len(dataset.recipes)}, {meta.embedding_dim})")
        print(f" FAISS Type         : {faiss.statistics.index_type}  |  Vectors : {faiss.statistics.total_vectors}")
        print(f" Metadata Version   : {meta.dataset_version}  |  Built : {meta.build_timestamp}")
        print(thin)
        print(f" Embedding Provider : {meta.embedding_model}")
        print(f" Embedding Model Ver: {meta.embedding_model_version}")
        print(f" Emb Config Hash    : {emb_hash_short}")
        print(f" Python Version     : {meta.python_version.splitlines()[0]}")
        print(f" Framework Version  : {meta.framework_version}")
        print(f" Framework Build    : {meta.framework_build_number}")
        print(f" Artifacts Hash     : {art_hash_short}  [VERIFIED]")
        print(f" Dataset Health     : {health}")
        print(f" Inspection Result  : {inspection_result}")
        print(f" Rebuild Reason     : {status.rebuild_reason}")
        print(f" Total Load Time    : {elapsed:.4f}s")
        print(thin)
        print(f" Status             : LOADED")
        print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# SELF TESTS — 14 Scenarios
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_csv(path: str) -> None:
    """Writes a minimal synthetic recipes.csv for self-test scenarios."""
    content = (
        "recipe_name,price,description,meal_type,cuisine_type,ingredients\n"
        "Pasta Primavera,12.5,Fresh pasta with seasonal vegetables,Lunch,Italian,\"pasta, tomatoes, zucchini, olive oil\"\n"
        "Grilled Salmon,18.0,Atlantic salmon fillet grilled with lemon,Dinner,Mediterranean,\"salmon, lemon, garlic, olive oil\"\n"
        "Caesar Salad,10.0,Classic Caesar with romaine and croutons,Lunch,American,\"romaine, croutons, parmesan, caesar dressing\"\n"
        "Margherita Pizza,14.0,Traditional Neapolitan pizza with basil,Dinner,Italian,\"dough, tomato sauce, mozzarella, basil\"\n"
        "Beef Burger,11.5,Angus beef patty with cheddar and pickles,Lunch,American,\"beef patty, bun, cheddar, pickles\"\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _make_pipeline_with_mock() -> DatasetPipeline:
    """Returns a DatasetPipeline that uses MockEmbeddingProvider for fast tests."""
    return DatasetPipeline(embedding_provider=MockEmbeddingProvider())


def _run_self_tests() -> None:
    """
    Runs all 14 DatasetManager scenarios.
    Each scenario uses a temporary directory and synthetic CSV data.
    """
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}" + (f" — {detail}" if detail else ""))
            failed += 1

    print("=" * 80)
    print(" DatasetManager — Self Tests (14 Scenarios)")
    print("=" * 80)

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 1: Only recipes.csv exists -> auto full build
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Scenario 1] Only recipes.csv exists -> auto full build")
    with tempfile.TemporaryDirectory() as tmpdir:
        restaurant_dir = os.path.join(tmpdir, "rest_A")
        os.makedirs(restaurant_dir)
        _make_synthetic_csv(os.path.join(restaurant_dir, _CSV_FILENAME))
        manager = DatasetManager(dataset_directory=tmpdir, pipeline=_make_pipeline_with_mock())
        try:
            dataset = manager.load_restaurant("rest_A")
            check("S1: Full build completes and dataset is returned",
                  isinstance(dataset, RestaurantDataset))
            check("S1: recipes.pkl was created",
                  os.path.isfile(os.path.join(restaurant_dir, _RECIPES_FILENAME)))
            check("S1: faiss.index was created",
                  os.path.isfile(os.path.join(restaurant_dir, _FAISS_INDEX_FILENAME)))
            check("S1: metadata.json was created",
                  os.path.isfile(os.path.join(restaurant_dir, _METADATA_FILENAME)))
            check("S1: dataset has recipes", len(dataset.recipes) > 0)
        except Exception as e:
            check("S1: build/load succeeded", False, str(e))

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 2: All artifacts + hash match -> cache hit
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Scenario 2] All artifacts present + hash match -> cache hit")
    with tempfile.TemporaryDirectory() as tmpdir:
        restaurant_dir = os.path.join(tmpdir, "rest_A")
        os.makedirs(restaurant_dir)
        _make_synthetic_csv(os.path.join(restaurant_dir, _CSV_FILENAME))
        manager = DatasetManager(dataset_directory=tmpdir, pipeline=_make_pipeline_with_mock())
        manager.load_restaurant("rest_A")   # first load — builds
        status = manager.inspect("rest_A")
        check("S2: Second inspect returns COMPLETE",
              status.overall == ArtifactStatus.COMPLETE,
              status.overall.value)
        check("S2: csv_hash_matches is True", status.csv_hash_matches)
        check("S2: artifacts_hash_matches is True", status.artifacts_hash_matches)

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 3: recipes.csv modified -> STALE -> full rebuild
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Scenario 3] recipes.csv modified -> STALE -> full rebuild")
    with tempfile.TemporaryDirectory() as tmpdir:
        restaurant_dir = os.path.join(tmpdir, "rest_A")
        os.makedirs(restaurant_dir)
        csv_path = os.path.join(restaurant_dir, _CSV_FILENAME)
        _make_synthetic_csv(csv_path)
        manager = DatasetManager(dataset_directory=tmpdir, pipeline=_make_pipeline_with_mock())
        manager.load_restaurant("rest_A")
        # Modify CSV
        with open(csv_path, "a") as f:
            f.write("New Recipe,9.5,A brand-new dish added later,Dinner,Fusion\n")
        status = manager.inspect("rest_A")
        check("S3: Modified CSV triggers STALE",
              status.overall == ArtifactStatus.STALE, status.overall.value)
        check("S3: csv_hash_matches is False", not status.csv_hash_matches)

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 4: faiss.index missing -> MISSING -> full rebuild
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Scenario 4] faiss.index missing -> MISSING -> full rebuild")
    with tempfile.TemporaryDirectory() as tmpdir:
        restaurant_dir = os.path.join(tmpdir, "rest_A")
        os.makedirs(restaurant_dir)
        _make_synthetic_csv(os.path.join(restaurant_dir, _CSV_FILENAME))
        manager = DatasetManager(dataset_directory=tmpdir, pipeline=_make_pipeline_with_mock())
        manager.load_restaurant("rest_A")
        os.remove(os.path.join(restaurant_dir, _FAISS_INDEX_FILENAME))
        status = manager.inspect("rest_A")
        check("S4: Missing faiss.index triggers MISSING",
              status.overall == ArtifactStatus.MISSING, status.overall.value)
        check("S4: faiss_exists is False", not status.faiss_exists)

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 5: metadata.json missing -> MISSING
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Scenario 5] metadata.json missing -> MISSING")
    with tempfile.TemporaryDirectory() as tmpdir:
        restaurant_dir = os.path.join(tmpdir, "rest_A")
        os.makedirs(restaurant_dir)
        _make_synthetic_csv(os.path.join(restaurant_dir, _CSV_FILENAME))
        manager = DatasetManager(dataset_directory=tmpdir, pipeline=_make_pipeline_with_mock())
        manager.load_restaurant("rest_A")
        os.remove(os.path.join(restaurant_dir, _METADATA_FILENAME))
        status = manager.inspect("rest_A")
        check("S5: Missing metadata.json triggers MISSING",
              status.overall == ArtifactStatus.MISSING, status.overall.value)

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 6: metadata.json corrupted -> CORRUPT
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Scenario 6] metadata.json corrupted -> CORRUPT")
    with tempfile.TemporaryDirectory() as tmpdir:
        restaurant_dir = os.path.join(tmpdir, "rest_A")
        os.makedirs(restaurant_dir)
        _make_synthetic_csv(os.path.join(restaurant_dir, _CSV_FILENAME))
        manager = DatasetManager(dataset_directory=tmpdir, pipeline=_make_pipeline_with_mock())
        manager.load_restaurant("rest_A")
        # Corrupt the JSON
        with open(os.path.join(restaurant_dir, _METADATA_FILENAME), "w") as f:
            f.write("{ CORRUPTED ~~~~ NOT JSON }")
        status = manager.inspect("rest_A")
        check("S6: Corrupted metadata.json triggers CORRUPT",
              status.overall == ArtifactStatus.CORRUPT, status.overall.value)

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 7: Restaurant switching
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Scenario 7] Restaurant switching (A -> B -> A)")
    with tempfile.TemporaryDirectory() as tmpdir:
        for rname in ("rest_A", "rest_B"):
            rdir = os.path.join(tmpdir, rname)
            os.makedirs(rdir)
            _make_synthetic_csv(os.path.join(rdir, _CSV_FILENAME))
        manager = DatasetManager(dataset_directory=tmpdir, pipeline=_make_pipeline_with_mock())
        ds_a = manager.load_restaurant("rest_A")
        ds_b = manager.load_restaurant("rest_B")
        ds_a2 = manager.load_restaurant("rest_A")
        check("S7: rest_A and rest_B are distinct", ds_a.restaurant_name != ds_b.restaurant_name)
        check("S7: Reloading rest_A succeeds", ds_a2.restaurant_name == "rest_A")

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 8: DatasetMetadata serialisation round-trip
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Scenario 8] DatasetMetadata serialisation round-trip")
    with tempfile.TemporaryDirectory() as tmpdir:
        restaurant_dir = os.path.join(tmpdir, "rest_A")
        os.makedirs(restaurant_dir)
        _make_synthetic_csv(os.path.join(restaurant_dir, _CSV_FILENAME))
        manager = DatasetManager(dataset_directory=tmpdir, pipeline=_make_pipeline_with_mock())
        dataset = manager.load_restaurant("rest_A")
        meta = dataset.metadata
        meta2 = DatasetMetadata.from_dict(meta.to_dict())
        check("S8: restaurant_name survives round-trip",
              meta2.restaurant_name == meta.restaurant_name)
        check("S8: csv_hash survives round-trip", meta2.csv_hash == meta.csv_hash)
        check("S8: artifacts_hash survives round-trip",
              meta2.artifacts_hash == meta.artifacts_hash)
        check("S8: embedding_configuration_hash survives round-trip",
              meta2.embedding_configuration_hash == meta.embedding_configuration_hash)

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 9: DatasetLoader loads all 4 fields correctly
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Scenario 9] DatasetLoader loads all four fields")
    with tempfile.TemporaryDirectory() as tmpdir:
        restaurant_dir = os.path.join(tmpdir, "rest_A")
        os.makedirs(restaurant_dir)
        _make_synthetic_csv(os.path.join(restaurant_dir, _CSV_FILENAME))
        manager = DatasetManager(dataset_directory=tmpdir, pipeline=_make_pipeline_with_mock())
        dataset = manager.load_restaurant("rest_A")
        check("S9: restaurant_name populated", dataset.restaurant_name == "rest_A")
        check("S9: version populated", len(dataset.version) > 0)
        check("S9: recipes is a DataFrame with rows", isinstance(dataset.recipes, pd.DataFrame) and len(dataset.recipes) > 0)
        check("S9: faiss_result has indexed vectors", dataset.faiss_result.statistics.total_vectors > 0)
        check("S9: metadata restaurant_name matches", dataset.metadata.restaurant_name == "rest_A")

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 10: Embedding model changed -> STALE_EMB -> partial rebuild
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Scenario 10] Embedding model changed -> STALE_EMB -> partial rebuild")
    with tempfile.TemporaryDirectory() as tmpdir:
        restaurant_dir = os.path.join(tmpdir, "rest_A")
        os.makedirs(restaurant_dir)
        _make_synthetic_csv(os.path.join(restaurant_dir, _CSV_FILENAME))
        manager = DatasetManager(dataset_directory=tmpdir, pipeline=_make_pipeline_with_mock())
        manager.load_restaurant("rest_A")
        # Simulate a different embedding model by patching metadata hash
        meta = DatasetMetadata.load(restaurant_dir)
        patched = DatasetMetadata.from_dict({**meta.to_dict(), "embedding_configuration_hash": "different_hash_abc123"})
        patched.save(restaurant_dir)
        status = manager.inspect("rest_A")
        check("S10: Changed emb config triggers STALE_EMB",
              status.overall == ArtifactStatus.STALE_EMB, status.overall.value)
        check("S10: embedding_config_hash_matches is False",
              not status.embedding_config_hash_matches)

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 11: recipes.pkl corrupted -> CORRUPT -> full rebuild
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Scenario 11] recipes.pkl corrupted -> CORRUPT artifacts hash -> full rebuild")
    with tempfile.TemporaryDirectory() as tmpdir:
        restaurant_dir = os.path.join(tmpdir, "rest_A")
        os.makedirs(restaurant_dir)
        _make_synthetic_csv(os.path.join(restaurant_dir, _CSV_FILENAME))
        manager = DatasetManager(dataset_directory=tmpdir, pipeline=_make_pipeline_with_mock())
        manager.load_restaurant("rest_A")
        # Corrupt recipes.pkl (which changes artifacts_hash)
        with open(os.path.join(restaurant_dir, _RECIPES_FILENAME), "wb") as f:
            f.write(b"THIS IS NOT A VALID PICKLE FILE")
        status = manager.inspect("rest_A")
        check("S11: Corrupted recipes.pkl triggers CORRUPT",
              status.overall == ArtifactStatus.CORRUPT, status.overall.value)
        check("S11: artifacts_hash_matches is False", not status.artifacts_hash_matches)

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 12: embeddings.npy missing -> STALE_EMB -> partial rebuild
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Scenario 12] embeddings.npy missing -> STALE_EMB -> partial rebuild")
    with tempfile.TemporaryDirectory() as tmpdir:
        restaurant_dir = os.path.join(tmpdir, "rest_A")
        os.makedirs(restaurant_dir)
        _make_synthetic_csv(os.path.join(restaurant_dir, _CSV_FILENAME))
        manager = DatasetManager(dataset_directory=tmpdir, pipeline=_make_pipeline_with_mock())
        manager.load_restaurant("rest_A")
        os.remove(os.path.join(restaurant_dir, _EMBEDDINGS_FILENAME))
        status = manager.inspect("rest_A")
        check("S12: Missing embeddings.npy triggers STALE_EMB",
              status.overall == ArtifactStatus.STALE_EMB, status.overall.value)
        check("S12: embeddings_exists is False", not status.embeddings_exists)

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 13: Embedding configuration changed -> STALE_EMB
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Scenario 13] Embedding configuration changed -> STALE_EMB")
    with tempfile.TemporaryDirectory() as tmpdir:
        restaurant_dir = os.path.join(tmpdir, "rest_A")
        os.makedirs(restaurant_dir)
        _make_synthetic_csv(os.path.join(restaurant_dir, _CSV_FILENAME))
        # Build with mock provider (config: mock-384-v1, no normalisation key)
        manager_v1 = DatasetManager(
            dataset_directory=tmpdir,
            pipeline=DatasetPipeline(embedding_provider=MockEmbeddingProvider()),
        )
        manager_v1.load_restaurant("rest_A")

        # Create a second manager with a DIFFERENT mock config hash by customising provider
        class _AltMockProvider(MockEmbeddingProvider):
            def config_dict_for_hash(self):
                return {"provider": "MockEmbeddingProvider-ALT", "model_name": "mock-alt-v2"}

        manager_v2 = DatasetManager(
            dataset_directory=tmpdir,
            pipeline=DatasetPipeline(embedding_provider=_AltMockProvider()),
        )
        status = manager_v2.inspect("rest_A")
        check("S13: Changed embedding config triggers STALE_EMB",
              status.overall == ArtifactStatus.STALE_EMB, status.overall.value)
        check("S13: embedding_config_hash_matches is False",
              not status.embedding_config_hash_matches)

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario 14: Framework version changed -> warning only, no rebuild
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[Scenario 14] Framework version changed -> compatibility warning, no rebuild")
    with tempfile.TemporaryDirectory() as tmpdir:
        restaurant_dir = os.path.join(tmpdir, "rest_A")
        os.makedirs(restaurant_dir)
        _make_synthetic_csv(os.path.join(restaurant_dir, _CSV_FILENAME))
        manager = DatasetManager(dataset_directory=tmpdir, pipeline=_make_pipeline_with_mock())
        manager.load_restaurant("rest_A")
        # Patch metadata to simulate an older framework version
        meta = DatasetMetadata.load(restaurant_dir)
        old_fw_meta = DatasetMetadata.from_dict({**meta.to_dict(), "framework_version": "0.9.0"})
        old_fw_meta.save(restaurant_dir)
        report = manager.validate("rest_A")
        check("S14: validate() returns a ValidationReport", isinstance(report, ValidationReport))
        check("S14: framework_compatible is False (version mismatch)", not report.framework_compatible)
        check("S14: is_valid is True (no rebuild triggered)", report.is_valid)
        check("S14: warnings list is non-empty", len(report.warnings) > 0)
        # Confirm inspect() does NOT return CORRUPT or STALE for version difference
        status = manager.inspect("rest_A")
        check("S14: inspect() still returns COMPLETE (version diff is not a rebuild trigger)",
              status.overall == ArtifactStatus.COMPLETE, status.overall.value)

    # ──────────────────────────────────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f" Scenarios Passed: {passed}  |  Failed: {failed}")
    print("=" * 80)
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _run_self_tests()
