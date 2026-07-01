"""
dine_ai.dataset — Dataset Management Package
=============================================

Public API for the DineAI Dataset Management subsystem.

Entry Point
-----------
``DatasetManager`` is the single authoritative entry point for obtaining a
``RestaurantDataset``.  All other classes are exposed for type annotations,
dependency injection, and extensibility.

Typical Usage
-------------
>>> from dine_ai.dataset import DatasetManager
>>> manager = DatasetManager(dataset_directory="dine_ai/datasets")
>>> dataset = manager.load_restaurant("restaurant_A")
>>> print(dataset)
RestaurantDataset(restaurant='restaurant_A', version='1.0', ...)

Custom Embedding Provider
-------------------------
>>> from dine_ai.dataset import DatasetManager, DatasetPipeline, BGEProvider
>>> pipeline = DatasetPipeline(embedding_provider=BGEProvider())
>>> manager = DatasetManager(pipeline=pipeline)
>>> dataset = manager.load_restaurant("restaurant_A")

Manual Status Check
-------------------
>>> manager = DatasetManager()
>>> print(manager.status("restaurant_A"))
"""

from dine_ai.dataset.metadata import (
    DatasetMetadata,
    DatasetMetadataError,
    FRAMEWORK_VERSION,
    FRAMEWORK_BUILD_NUMBER,
    DATASET_VERSION_DEFAULT,
)

from dine_ai.dataset.loader import (
    RestaurantDataset,
    DatasetLoader,
    DatasetLoadError,
)

from dine_ai.dataset.pipeline import (
    BaseEmbeddingProvider,
    SentenceTransformerProvider,
    MockEmbeddingProvider,
    BGEProvider,
    E5Provider,
    PipelineArtifacts,
    DatasetPipeline,
    DatasetBuildError,
)

from dine_ai.dataset.manager import (
    ArtifactStatus,
    DatasetStatus,
    ValidationReport,
    DatasetManager,
)

from dine_ai.capabilities.detector import CapabilityDetector

__all__ = [
    # Metadata
    "DatasetMetadata",
    "DatasetMetadataError",
    "FRAMEWORK_VERSION",
    "FRAMEWORK_BUILD_NUMBER",
    "DATASET_VERSION_DEFAULT",
    # Loader
    "RestaurantDataset",
    "DatasetLoader",
    "DatasetLoadError",
    # Pipeline
    "BaseEmbeddingProvider",
    "SentenceTransformerProvider",
    "MockEmbeddingProvider",
    "BGEProvider",
    "E5Provider",
    "PipelineArtifacts",
    "DatasetPipeline",
    "DatasetBuildError",
    # Manager
    "ArtifactStatus",
    "DatasetStatus",
    "ValidationReport",
    "DatasetManager",
    # Capabilities
    "CapabilityDetector",
]
