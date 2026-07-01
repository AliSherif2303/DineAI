"""
DineAI — Dataset Loader
========================
Stateless loader that reads the four pre-built artifact files from disk
and assembles them into a RestaurantDataset data container.

Responsibilities
----------------
• Load  recipes.pkl         → pd.DataFrame
• Load  faiss.index         → FaissResult  (via FaissResult.load)
• Load  metadata.json       → DatasetMetadata
• Assemble and return       → RestaurantDataset

This module does NOT:
• Read CSV files
• Run any adapter (SchemaMapper, FeatureEngineer, etc.)
• Write any file to disk
• Contain any business logic

The DatasetLoader is the *only* class permitted to deserialise
the four dataset artifacts.  All other components that need the
dataset must receive a RestaurantDataset object.
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass

import pandas as pd

from dine_ai.dataset.metadata import DatasetMetadata, DatasetMetadataError
from dine_ai.embeddings.faiss_builder import FaissResult

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_RECIPES_FILENAME: str = "recipes.pkl"


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTIONS
# ─────────────────────────────────────────────────────────────────────────────

class DatasetLoadError(Exception):
    """
    Raised when DatasetLoader cannot load one or more required artifact files.

    Callers (typically DatasetManager) should treat this exception as a
    definitive signal to trigger a dataset rebuild.
    """


# ─────────────────────────────────────────────────────────────────────────────
# CLASS: RestaurantDataset
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RestaurantDataset:
    """
    Immutable data container for a fully loaded restaurant dataset.

    This object is the sole output of ``DatasetLoader`` and the canonical
    input for any downstream consumer (e.g. ``RestaurantManager``).
    It carries no behaviour — only data.

    Attributes
    ----------
    restaurant_name : str
        Logical identifier of the restaurant (e.g. ``"restaurant_A"``).
    version : str
        Dataset version string (e.g. ``"1.0"``).
    recipes : pd.DataFrame
        The fully enriched recipe DataFrame loaded from ``recipes.pkl``.
        Contains all schema-mapped, validated, feature-engineered, and
        text-built columns (``search_text``, ``text_chunk``, ``summary_text``).
    faiss_result : FaissResult
        Loaded and ready-to-search FAISS index.
    metadata : DatasetMetadata
        Full provenance record describing this build.
    """

    restaurant_name: str
    version: str
    recipes: pd.DataFrame
    faiss_result: FaissResult
    metadata: DatasetMetadata

    def __repr__(self) -> str:
        n_vectors = getattr(
            getattr(self.faiss_result, "statistics", None), "total_vectors", "?"
        )
        return (
            f"RestaurantDataset("
            f"restaurant='{self.restaurant_name}', "
            f"version='{self.version}', "
            f"recipes={len(self.recipes)} rows, "
            f"faiss_vectors={n_vectors}"
            f")"
        )

    @property
    def recipe_count(self) -> int:
        """Number of recipes loaded from ``recipes.pkl``."""
        return len(self.recipes)

    @property
    def embedding_dim(self) -> int:
        """Embedding dimensionality as stored in metadata."""
        return self.metadata.embedding_dim

    @property
    def capabilities(self) -> Dict[str, Any]:
        """Pluggable capabilities detected for this restaurant dataset."""
        return self.metadata.capability_report

    @property
    def available_columns(self) -> List[str]:
        """Canonical columns available in this restaurant dataset."""
        return self.metadata.available_columns


# ─────────────────────────────────────────────────────────────────────────────
# CLASS: DatasetLoader
# ─────────────────────────────────────────────────────────────────────────────

class DatasetLoader:
    """
    Stateless loader for pre-built DineAI restaurant dataset artifacts.

    Loads the four artifact files produced by ``DatasetPipeline``:

    ============  ===========================
    Artifact      Destination
    ============  ===========================
    recipes.pkl   → ``RestaurantDataset.recipes``
    faiss.index   → ``RestaurantDataset.faiss_result``
    metadata.json → ``RestaurantDataset.metadata``
    ============  ===========================

    Note: ``embeddings.npy`` is an intermediate build artifact. It is saved
    to disk for partial-rebuild support but is not loaded into the live
    application — the FAISS index already encodes the vectors.

    Usage
    -----
    >>> loader = DatasetLoader()
    >>> dataset = loader.load("dine_ai/datasets/restaurant_A")
    >>> print(dataset)
    RestaurantDataset(restaurant='restaurant_A', version='1.0', ...)
    """

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def load(self, restaurant_dir: str) -> RestaurantDataset:
        """
        Loads all artifact files from ``restaurant_dir`` and returns a
        fully populated ``RestaurantDataset``.

        Parameters
        ----------
        restaurant_dir : str
            Directory containing ``recipes.pkl``, ``faiss.index``,
            ``faiss_metadata.json``, and ``metadata.json``.

        Returns
        -------
        RestaurantDataset
            Fully populated data container ready for downstream consumers.

        Raises
        ------
        DatasetLoadError
            If any required artifact is absent, unreadable, or has an
            incompatible format.
        """
        import time
        t_start = time.perf_counter()

        restaurant_dir = os.path.abspath(restaurant_dir)
        logger.info("DatasetLoader: Loading artifacts from '%s'", restaurant_dir)

        # ── 1. Load metadata (needed for RestaurantDataset fields) ────────────
        metadata = self._load_metadata(restaurant_dir)

        # ── 2. Load recipes DataFrame ─────────────────────────────────────────
        recipes_df = self._load_recipes(restaurant_dir)

        # ── 3. Load FAISS index ───────────────────────────────────────────────
        faiss_result = self._load_faiss(restaurant_dir)

        elapsed = time.perf_counter() - t_start

        dataset = RestaurantDataset(
            restaurant_name=metadata.restaurant_name,
            version=metadata.dataset_version,
            recipes=recipes_df,
            faiss_result=faiss_result,
            metadata=metadata,
        )

        logger.info(
            "DatasetLoader: Loaded '%s' — %d recipes, %d vectors (%.3fs)",
            metadata.restaurant_name,
            len(recipes_df),
            faiss_result.statistics.total_vectors,
            elapsed,
        )
        return dataset

    # ──────────────────────────────────────────────────────────────────────────
    # Private Artifact Loaders
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_metadata(restaurant_dir: str) -> DatasetMetadata:
        """
        Loads and deserialises ``metadata.json``.

        Raises
        ------
        DatasetLoadError
            On missing or corrupt ``metadata.json``.
        """
        try:
            return DatasetMetadata.load(restaurant_dir)
        except DatasetMetadataError as exc:
            raise DatasetLoadError(
                f"DatasetLoader: Cannot load metadata from '{restaurant_dir}': {exc}"
            ) from exc

    @staticmethod
    def _load_recipes(restaurant_dir: str) -> pd.DataFrame:
        """
        Deserialises ``recipes.pkl`` into a ``pd.DataFrame``.

        Accepts both a serialised ``pd.DataFrame`` (new format produced by
        ``DatasetPipeline``) and a ``List[Dict]`` (legacy format for backward
        compatibility with the existing ``RestaurantManager`` test datasets).

        Raises
        ------
        DatasetLoadError
            On missing file, unpicklable content, or unsupported format.
        """
        pkl_path = os.path.join(restaurant_dir, _RECIPES_FILENAME)
        if not os.path.isfile(pkl_path):
            raise DatasetLoadError(
                f"DatasetLoader: 'recipes.pkl' not found in '{restaurant_dir}'"
            )
        try:
            with open(pkl_path, "rb") as fh:
                data = pickle.load(fh)
        except Exception as exc:
            raise DatasetLoadError(
                f"DatasetLoader: Cannot deserialise 'recipes.pkl': {exc}"
            ) from exc

        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, list):
            # Backward-compatible list-of-dicts format
            return pd.DataFrame(data)
        raise DatasetLoadError(
            f"DatasetLoader: Unexpected 'recipes.pkl' content type: "
            f"{type(data).__name__}.  Expected pd.DataFrame or List[Dict]."
        )

    @staticmethod
    def _load_faiss(restaurant_dir: str) -> FaissResult:
        """
        Loads the FAISS index and its companion ``faiss_metadata.json``.

        Raises
        ------
        DatasetLoadError
            On any FAISS loading failure.
        """
        try:
            return FaissResult.load(restaurant_dir)
        except Exception as exc:
            raise DatasetLoadError(
                f"DatasetLoader: Cannot load FAISS index from '{restaurant_dir}': {exc}"
            ) from exc
