"""
dine_ai.capabilities — Capability Registry & Detection Package
==============================================================

Public API for the DineAI Capability Management subsystem.
"""

from dine_ai.capabilities.base import (
    ColumnRequirement,
    CapabilityStatus,
    CapabilityResult,
    BaseCapability,
)

from dine_ai.capabilities.registry import (
    CapabilityRegistry,
    get_default_registry,
)

from dine_ai.capabilities.detector import (
    RestaurantCapabilityReport,
    CapabilityDetector,
)

__all__ = [
    "ColumnRequirement",
    "CapabilityStatus",
    "CapabilityResult",
    "BaseCapability",
    "CapabilityRegistry",
    "get_default_registry",
    "RestaurantCapabilityReport",
    "CapabilityDetector",
]
