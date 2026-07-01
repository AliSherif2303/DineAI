"""
DineAI — Capability Registry
=============================
Plugin registry for BaseCapability instances.

The CapabilityRegistry is the central store for all capabilities.
It provides detection against a live DataFrame and reports on which
capabilities are enabled or disabled.

Usage
-----
>>> from dine_ai.capabilities.registry import get_default_registry
>>> registry = get_default_registry()
>>> results = registry.get_enabled(df)

Extending the registry
----------------------
Register a new capability at runtime (e.g. in your application config):

>>> from dine_ai.capabilities.registry import get_default_registry
>>> from myapp.capabilities import DeliveryTimeCapability
>>> get_default_registry().register(DeliveryTimeCapability())
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from dine_ai.capabilities.base import BaseCapability, CapabilityResult, CapabilityStatus

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS: CapabilityRegistry
# ─────────────────────────────────────────────────────────────────────────────

class CapabilityRegistry:
    """
    Plugin registry for BaseCapability instances.

    Capabilities are stored by name (last writer wins on duplicates).
    Detection is stateless — the same registry instance can be used
    across multiple restaurants with different DataFrames.

    Parameters
    ----------
    None

    Methods
    -------
    register(capability)        — Add a single capability.
    register_many(capabilities) — Add multiple capabilities at once.
    unregister(name)            — Remove a capability by name.
    get_all()                   — Return all registered capabilities.
    detect_all(df)              — Detect all capabilities against a DataFrame.
    get_enabled(df)             — Return only ENABLED results.
    get_disabled(df)            — Return DISABLED + PARTIAL results.
    enabled_names(df)           — Set of enabled capability names.
    """

    def __init__(self) -> None:
        self._capabilities: Dict[str, BaseCapability] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, capability: BaseCapability) -> None:
        """Register a single capability (overwrites on duplicate name)."""
        if capability.name in self._capabilities:
            logger.debug(
                "CapabilityRegistry: Overwriting capability '%s'", capability.name
            )
        self._capabilities[capability.name] = capability

    def register_many(self, capabilities: List[BaseCapability]) -> None:
        """Register multiple capabilities at once."""
        for cap in capabilities:
            self.register(cap)

    def unregister(self, name: str) -> None:
        """Remove a capability by name. No-op if not found."""
        self._capabilities.pop(name, None)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def get_all(self) -> List[BaseCapability]:
        """Returns all registered capabilities (in registration order)."""
        return list(self._capabilities.values())

    def get_by_name(self, name: str) -> Optional[BaseCapability]:
        """Returns a capability by name, or None if not found."""
        return self._capabilities.get(name)

    def get_by_category(self, category: str) -> List[BaseCapability]:
        """Returns all capabilities belonging to a given category."""
        return [c for c in self._capabilities.values() if c.category == category]

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect_all(self, df: pd.DataFrame) -> List[CapabilityResult]:
        """
        Detects all registered capabilities against the given DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The enriched dataset DataFrame.

        Returns
        -------
        List[CapabilityResult]
        """
        return [cap.detect(df) for cap in self._capabilities.values()]

    def get_enabled(self, df: pd.DataFrame) -> List[CapabilityResult]:
        """Returns only ENABLED capability results."""
        return [
            r for r in self.detect_all(df)
            if r.status == CapabilityStatus.ENABLED
        ]

    def get_disabled(self, df: pd.DataFrame) -> List[CapabilityResult]:
        """Returns DISABLED and PARTIAL capability results."""
        return [
            r for r in self.detect_all(df)
            if r.status != CapabilityStatus.ENABLED
        ]

    def enabled_names(self, df: pd.DataFrame) -> frozenset:
        """Returns a frozenset of enabled capability names for fast lookup."""
        return frozenset(r.name for r in self.get_enabled(df))

    # ── Introspection ─────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._capabilities)

    def __repr__(self) -> str:
        return f"CapabilityRegistry({len(self._capabilities)} capabilities registered)"


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULT REGISTRY — lazy singleton
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_REGISTRY: Optional[CapabilityRegistry] = None


def get_default_registry() -> CapabilityRegistry:
    """
    Returns the global default CapabilityRegistry (singleton).

    Pre-populated with all built-in capabilities from definitions.py.
    Thread-safe for read operations; registration should happen at startup.

    Returns
    -------
    CapabilityRegistry
    """
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        from dine_ai.capabilities.definitions import ALL_CAPABILITIES
        _DEFAULT_REGISTRY = CapabilityRegistry()
        _DEFAULT_REGISTRY.register_many(ALL_CAPABILITIES)
        logger.debug(
            "CapabilityRegistry: Default registry initialized with %d capabilities.",
            len(_DEFAULT_REGISTRY),
        )
    return _DEFAULT_REGISTRY
