"""
DineAI — Dataset Metadata
==========================
Rich provenance record for a restaurant dataset build.

Responsibilities
----------------
• Store the full build environment snapshot (what, when, how, who)
• Compute SHA-256 hashes for change-detection and integrity verification
• Serialise / deserialise to JSON  (metadata.json)

This module has NO side-effects beyond its own JSON file.
It never reads CSV data, runs adapters, or loads binary artifacts.

Change Detection Keys
---------------------
csv_hash                     ← recipes.csv modified?      → full rebuild
embedding_configuration_hash ← embedding config changed?  → partial rebuild
artifacts_hash               ← artifact corruption?       → full rebuild
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

FRAMEWORK_VERSION: str = "1.0.0"
FRAMEWORK_BUILD_NUMBER: str = "20260701.1"
DATASET_VERSION_DEFAULT: str = "1.0"

_METADATA_FILENAME: str = "metadata.json"
_RECIPES_FILENAME: str = "recipes.pkl"
_EMBEDDINGS_FILENAME: str = "embeddings.npy"
_FAISS_INDEX_FILENAME: str = "faiss.index"

_HASH_CHUNK_SIZE: int = 65_536   # 64 KB per read; balances RAM vs syscall count


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class DatasetMetadataError(Exception):
    """
    Raised when DatasetMetadata cannot be loaded, parsed, or verified.

    Callers (typically DatasetManager) should treat this exception as a
    definitive signal to trigger a full dataset rebuild.
    """


# ─────────────────────────────────────────────────────────────────────────────
# CLASS: DatasetMetadata
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatasetMetadata:
    """
    Rich, immutable provenance record for a single DineAI restaurant dataset.

    Every field captures a snapshot of the build environment and configuration
    so that DatasetManager can determine precisely what kind of rebuild — if any
    — is required when the application starts.

    Change-Detection Design
    -----------------------
    Three hash fields serve as the integrity and staleness sentinels:

    ``csv_hash``
        SHA-256 of the source ``recipes.csv``.
        A mismatch triggers a **full rebuild** (all stages).

    ``embedding_configuration_hash``
        SHA-256 of the JSON-serialised embedding configuration.
        A mismatch triggers a **partial rebuild** (embeddings + FAISS only).

    ``artifacts_hash``
        SHA-256 over the concatenated bytes of the three primary binary
        artifacts: ``recipes.pkl``, ``embeddings.npy``, ``faiss.index``.
        A mismatch indicates **artifact corruption** → full rebuild.

    Parameters
    ----------
    restaurant_name : str
        Logical identifier of the restaurant (e.g. ``"restaurant_A"``).
    dataset_version : str
        Semantic version of this dataset record (e.g. ``"1.0"``).
    framework_version : str
        DineAI framework version that produced this dataset.
    framework_build_number : str
        Build number in the format ``"YYYYMMDD.N"``.
    python_version : str
        ``sys.version`` snapshot from the build environment.
    rows : int
        Number of recipes in the enriched DataFrame.
    columns : int
        Number of DataFrame columns after feature engineering.
    embedding_dim : int
        Dimensionality of the produced embedding vectors (e.g. 384).
    embedding_model : str
        Name/identifier of the embedding model (e.g. ``"all-MiniLM-L6-v2"``).
    embedding_model_version : str
        Version string reported by the embedding provider.
    embedding_configuration_hash : str
        SHA-256 of the provider's configuration dictionary (change-detection).
    faiss_type : str
        FAISS index type string (e.g. ``"FlatIP"``).
    build_timestamp : str
        ISO-8601 UTC timestamp of when the build completed.
    csv_hash : str
        SHA-256 of the source ``recipes.csv``.
    artifacts_hash : str
        SHA-256 of the combined binary content of the three primary artifacts.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    restaurant_name: str
    dataset_version: str = DATASET_VERSION_DEFAULT

    # ── Framework provenance ──────────────────────────────────────────────────
    framework_version: str = FRAMEWORK_VERSION
    framework_build_number: str = FRAMEWORK_BUILD_NUMBER
    python_version: str = field(default_factory=lambda: sys.version)

    # ── Dataset shape ─────────────────────────────────────────────────────────
    rows: int = 0
    columns: int = 0

    # ── Embedding configuration ───────────────────────────────────────────────
    embedding_dim: int = 0
    embedding_model: str = ""
    embedding_model_version: str = ""
    embedding_configuration_hash: str = ""

    # ── Index configuration ───────────────────────────────────────────────────
    faiss_type: str = ""

    # ── Timestamps ────────────────────────────────────────────────────────────
    build_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── Integrity hashes ──────────────────────────────────────────────────────
    csv_hash: str = ""
    artifacts_hash: str = ""

    # ── Restaurant Capabilities ───────────────────────────────────────────────
    capability_report: Dict[str, Any] = field(default_factory=dict)
    available_columns: List[str] = field(default_factory=list)

    # ──────────────────────────────────────────────────────────────────────────
    # Static Hash Helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_csv_hash(csv_path: str) -> str:
        """
        Computes the SHA-256 hex digest of a CSV file.

        Used as the **full-rebuild change-detection key**.  Any modification
        to ``recipes.csv`` (including a row addition or column rename) will
        produce a different digest, triggering an automatic rebuild.

        Parameters
        ----------
        csv_path : str
            Absolute or relative path to the ``recipes.csv`` file.

        Returns
        -------
        str
            64-character lowercase hex digest.

        Raises
        ------
        FileNotFoundError
            If no file exists at ``csv_path``.
        """
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(
                f"DatasetMetadata.compute_csv_hash: File not found: '{csv_path}'"
            )
        sha = hashlib.sha256()
        with open(csv_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(_HASH_CHUNK_SIZE), b""):
                sha.update(chunk)
        return sha.hexdigest()

    @staticmethod
    def compute_artifacts_hash(output_dir: str) -> str:
        """
        Computes a single SHA-256 hash representing the combined binary
        content of the three primary artifact files:

            ``recipes.pkl``  +  ``embeddings.npy``  +  ``faiss.index``

        The filename of each artifact is mixed into the hash so that
        a substituted file (e.g. ``faiss.index`` replaced with an empty file
        named ``faiss.index``) is detected as corrupt.

        Missing files are silently skipped; the returned hash is still
        deterministic given the same set of present files.

        Parameters
        ----------
        output_dir : str
            Directory containing the artifact files.

        Returns
        -------
        str
            64-character lowercase hex digest.
        """
        sha = hashlib.sha256()
        for name in (_RECIPES_FILENAME, _EMBEDDINGS_FILENAME, _FAISS_INDEX_FILENAME):
            path = os.path.join(output_dir, name)
            if os.path.isfile(path):
                sha.update(name.encode("utf-8"))        # mix in filename
                with open(path, "rb") as fh:
                    for chunk in iter(lambda: fh.read(_HASH_CHUNK_SIZE), b""):
                        sha.update(chunk)
        return sha.hexdigest()

    @staticmethod
    def compute_config_hash(config_dict: Dict[str, Any]) -> str:
        """
        Computes the SHA-256 hash of a JSON-serialisable configuration
        dictionary.

        Used as the **partial-rebuild change-detection key** for embedding
        configuration changes.  Any change to fields that affect the output
        vectors (e.g. ``model_name``, ``normalize_embeddings``) will produce
        a different digest.

        Parameters
        ----------
        config_dict : Dict[str, Any]
            JSON-serialisable dictionary representing the configuration
            (typically a subset of ``EmbeddingConfig`` fields).

        Returns
        -------
        str
            64-character lowercase hex digest.
        """
        serialised = json.dumps(config_dict, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

    # ──────────────────────────────────────────────────────────────────────────
    # Serialisation
    # ──────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialises the metadata record to a plain, JSON-serialisable dictionary.

        The dictionary is the canonical format stored in ``metadata.json``.

        Returns
        -------
        Dict[str, Any]
            All metadata fields as a flat key-value mapping.
        """
        return {
            "restaurant_name": self.restaurant_name,
            "dataset_version": self.dataset_version,
            "framework_version": self.framework_version,
            "framework_build_number": self.framework_build_number,
            "python_version": self.python_version,
            "rows": self.rows,
            "columns": self.columns,
            "embedding_dim": self.embedding_dim,
            "embedding_model": self.embedding_model,
            "embedding_model_version": self.embedding_model_version,
            "embedding_configuration_hash": self.embedding_configuration_hash,
            "faiss_type": self.faiss_type,
            "build_timestamp": self.build_timestamp,
            "csv_hash": self.csv_hash,
            "artifacts_hash": self.artifacts_hash,
            "capability_report": self.capability_report,
            "available_columns": self.available_columns,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatasetMetadata":
        """
        Deserialises a dictionary (produced by ``to_dict()``) back into a
        ``DatasetMetadata`` instance.

        Parameters
        ----------
        d : Dict[str, Any]
            Dictionary produced by ``to_dict()`` or loaded from ``metadata.json``.

        Returns
        -------
        DatasetMetadata

        Raises
        ------
        DatasetMetadataError
            If any required field is missing from the dictionary.
        """
        required = {
            "restaurant_name", "rows", "columns",
            "embedding_dim", "embedding_model",
            "faiss_type", "build_timestamp",
            "csv_hash", "artifacts_hash",
        }
        missing = required - d.keys()
        if missing:
            raise DatasetMetadataError(
                f"DatasetMetadata.from_dict: Missing required fields: "
                f"{sorted(missing)}"
            )
        return cls(
            restaurant_name=d["restaurant_name"],
            dataset_version=d.get("dataset_version", DATASET_VERSION_DEFAULT),
            framework_version=d.get("framework_version", FRAMEWORK_VERSION),
            framework_build_number=d.get("framework_build_number", FRAMEWORK_BUILD_NUMBER),
            python_version=d.get("python_version", ""),
            rows=int(d["rows"]),
            columns=int(d["columns"]),
            embedding_dim=int(d["embedding_dim"]),
            embedding_model=d["embedding_model"],
            embedding_model_version=d.get("embedding_model_version", ""),
            embedding_configuration_hash=d.get("embedding_configuration_hash", ""),
            faiss_type=d["faiss_type"],
            build_timestamp=d["build_timestamp"],
            csv_hash=d["csv_hash"],
            artifacts_hash=d["artifacts_hash"],
            capability_report=d.get("capability_report", {}),
            available_columns=d.get("available_columns", []),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, output_dir: str) -> str:
        """
        Writes the metadata record to ``metadata.json`` inside ``output_dir``.

        The target directory is created if it does not already exist.

        Parameters
        ----------
        output_dir : str
            Target directory path.

        Returns
        -------
        str
            Absolute path to the written ``metadata.json`` file.
        """
        os.makedirs(output_dir, exist_ok=True)
        meta_path = os.path.join(output_dir, _METADATA_FILENAME)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        logger.info("DatasetMetadata: Saved  →  %s", meta_path)
        return meta_path

    @classmethod
    def load(cls, output_dir: str) -> "DatasetMetadata":
        """
        Loads and deserialises ``metadata.json`` from the given directory.

        Parameters
        ----------
        output_dir : str
            Directory that contains a ``metadata.json`` file.

        Returns
        -------
        DatasetMetadata

        Raises
        ------
        DatasetMetadataError
            If the file is missing, cannot be read, contains invalid JSON,
            or is missing required fields.
        """
        meta_path = os.path.join(output_dir, _METADATA_FILENAME)
        if not os.path.isfile(meta_path):
            raise DatasetMetadataError(
                f"DatasetMetadata.load: 'metadata.json' not found in '{output_dir}'"
            )
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            raise DatasetMetadataError(
                f"DatasetMetadata.load: Failed to parse 'metadata.json': {exc}"
            ) from exc
        return cls.from_dict(data)


# ─────────────────────────────────────────────────────────────────────────────
# SELF TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _run_self_tests() -> None:
    """Lightweight self-tests for DatasetMetadata (no pytest required)."""
    import tempfile

    passed = 0
    failed = 0

    def check(name: str, condition: bool) -> None:
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}")
            failed += 1

    print("=" * 70)
    print(" DatasetMetadata — Self Tests")
    print("=" * 70)

    # ── Test 1: Construction with defaults ────────────────────────────────────
    meta = DatasetMetadata(
        restaurant_name="test_A",
        rows=100, columns=20,
        embedding_dim=384,
        embedding_model="all-MiniLM-L6-v2",
        embedding_model_version="2.2.2",
        embedding_configuration_hash="abc123",
        faiss_type="FlatIP",
        csv_hash="deadbeef",
        artifacts_hash="cafebabe",
    )
    check("Construction sets restaurant_name correctly", meta.restaurant_name == "test_A")
    check("Framework version is injected from constant", meta.framework_version == FRAMEWORK_VERSION)
    check("Python version is captured", len(meta.python_version) > 0)
    check("Build timestamp is non-empty", len(meta.build_timestamp) > 0)

    # ── Test 2: to_dict / from_dict round-trip ────────────────────────────────
    d = meta.to_dict()
    meta2 = DatasetMetadata.from_dict(d)
    check("to_dict / from_dict round-trip: restaurant_name", meta2.restaurant_name == meta.restaurant_name)
    check("to_dict / from_dict round-trip: rows", meta2.rows == meta.rows)
    check("to_dict / from_dict round-trip: csv_hash", meta2.csv_hash == meta.csv_hash)
    check("to_dict / from_dict round-trip: artifacts_hash", meta2.artifacts_hash == meta.artifacts_hash)
    check("to_dict / from_dict round-trip: embedding_model", meta2.embedding_model == meta.embedding_model)

    # ── Test 3: from_dict raises on missing required key ─────────────────────
    bad = {k: v for k, v in d.items() if k != "csv_hash"}
    try:
        DatasetMetadata.from_dict(bad)
        check("from_dict raises DatasetMetadataError on missing key", False)
    except DatasetMetadataError:
        check("from_dict raises DatasetMetadataError on missing key", True)

    # ── Test 4: save / load round-trip ───────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        path = meta.save(tmpdir)
        check("save returns a valid file path", os.path.isfile(path))
        loaded = DatasetMetadata.load(tmpdir)
        check("save / load round-trip: restaurant_name", loaded.restaurant_name == meta.restaurant_name)
        check("save / load round-trip: artifacts_hash", loaded.artifacts_hash == meta.artifacts_hash)
        check("save / load round-trip: embedding_configuration_hash",
              loaded.embedding_configuration_hash == meta.embedding_configuration_hash)

    # ── Test 5: load raises on missing file ──────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            DatasetMetadata.load(tmpdir)
            check("load raises DatasetMetadataError on missing file", False)
        except DatasetMetadataError:
            check("load raises DatasetMetadataError on missing file", True)

    # ── Test 6: load raises on corrupted JSON ─────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, _METADATA_FILENAME), "w") as f:
            f.write("{ NOT: VALID JSON ~~~")
        try:
            DatasetMetadata.load(tmpdir)
            check("load raises DatasetMetadataError on corrupted JSON", False)
        except DatasetMetadataError:
            check("load raises DatasetMetadataError on corrupted JSON", True)

    # ── Test 7: compute_csv_hash is deterministic ─────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "recipes.csv")
        with open(csv_path, "w") as f:
            f.write("recipe_name,price\nPasta,12.5\n")
        h1 = DatasetMetadata.compute_csv_hash(csv_path)
        h2 = DatasetMetadata.compute_csv_hash(csv_path)
        check("compute_csv_hash is deterministic", h1 == h2)
        check("compute_csv_hash produces 64-char hex digest", len(h1) == 64)

    # ── Test 8: compute_csv_hash detects modifications ─────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "recipes.csv")
        with open(csv_path, "w") as f:
            f.write("recipe_name,price\nPasta,12.5\n")
        h_before = DatasetMetadata.compute_csv_hash(csv_path)
        with open(csv_path, "a") as f:
            f.write("Salmon,18.0\n")
        h_after = DatasetMetadata.compute_csv_hash(csv_path)
        check("compute_csv_hash detects CSV modification", h_before != h_after)

    # ── Test 9: compute_config_hash is deterministic ──────────────────────────
    cfg = {"model_name": "all-MiniLM-L6-v2", "normalize_embeddings": True}
    h1 = DatasetMetadata.compute_config_hash(cfg)
    h2 = DatasetMetadata.compute_config_hash(cfg)
    check("compute_config_hash is deterministic", h1 == h2)
    check("compute_config_hash produces 64-char hex digest", len(h1) == 64)

    # ── Test 10: compute_config_hash detects changes ──────────────────────────
    cfg2 = {"model_name": "BAAI/bge-base-en-v1.5", "normalize_embeddings": True}
    check("compute_config_hash detects model change",
          DatasetMetadata.compute_config_hash(cfg) != DatasetMetadata.compute_config_hash(cfg2))

    # ── Test 11: compute_csv_hash raises on missing file ─────────────────────
    try:
        DatasetMetadata.compute_csv_hash("/nonexistent/path/recipes.csv")
        check("compute_csv_hash raises FileNotFoundError on missing file", False)
    except FileNotFoundError:
        check("compute_csv_hash raises FileNotFoundError on missing file", True)

    print("-" * 70)
    print(f" Passed: {passed}  |  Failed: {failed}")
    print("=" * 70)
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    _run_self_tests()
