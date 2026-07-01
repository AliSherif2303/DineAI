"""
DineAI — Capability Base
=========================
Core abstractions for the plugin-based Capability Registry.

A Capability represents a specific recommendation feature that can be
activated when the required data columns are present in a restaurant dataset.

Design Principles
-----------------
• Open / Closed Principle: new capabilities are added by subclassing BaseCapability
  — no existing code needs to change.
• Single Responsibility: each capability knows only its own requirements.
• Strategy Pattern: each capability implements detect() independently.
• Plugin-ready: capabilities self-register via definitions.py + registry.py.

Example: adding a new capability
---------------------------------
>>> class DeliveryTimeCapability(BaseCapability):
...     @property
...     def name(self) -> str: return "Delivery Time"
...     @property
...     def required_columns(self) -> List[str]: return ["delivery_time_minutes"]
...     @property
...     def description(self) -> str: return "Enables delivery time filtering"
...     @property
...     def category(self) -> str: return "Logistics"
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class ColumnRequirement(Enum):
    """
    Classification tier for dataset schema columns.

    Used by the validator to decide how to handle missing columns.

    Members
    -------
    MANDATORY
        Missing → validation ERROR. Framework cannot operate.
    RECOMMENDED
        Missing → capability DISABLED + warning. Framework continues.
    OPTIONAL
        Missing → warning only. No capability impact.
    """
    MANDATORY   = "mandatory"
    RECOMMENDED = "recommended"
    OPTIONAL    = "optional"


class CapabilityStatus(Enum):
    """
    Health status of a detected capability.

    Members
    -------
    ENABLED
        All required columns are present and (if applicable) required values exist.
    DISABLED
        One or more required columns are absent.
    PARTIAL
        Columns are present but required values (e.g. item_type='drink') are absent.
    """
    ENABLED  = "enabled"
    DISABLED = "disabled"
    PARTIAL  = "partial"


# ─────────────────────────────────────────────────────────────────────────────
# DTO: CapabilityResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CapabilityResult:
    """
    Result of detecting a single capability against a DataFrame.

    Produced by BaseCapability.detect() and consumed by CapabilityRegistry,
    CapabilityDetector, and RestaurantCapabilityReport.

    Attributes
    ----------
    name : str
        Human-readable capability name (e.g. 'Budget Recommendation').
    status : CapabilityStatus
        ENABLED / DISABLED / PARTIAL.
    reason : str
        Human-readable explanation for the status.
    available_columns : List[str]
        Subset of required_columns that ARE present in the DataFrame.
    missing_required : List[str]
        Subset of required_columns that are ABSENT.
    missing_optional : List[str]
        Subset of optional_columns that are ABSENT.
    category : str
        Grouping category for the capability report (e.g. 'Nutrition', 'Pricing').
    """
    name: str
    status: CapabilityStatus
    reason: str = ""
    available_columns: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    missing_optional: List[str] = field(default_factory=list)
    category: str = "General"

    @property
    def is_enabled(self) -> bool:
        """True if the capability is fully enabled."""
        return self.status == CapabilityStatus.ENABLED

    def to_dict(self) -> Dict[str, Any]:
        """Serialises this result to a JSON-safe dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "reason": self.reason,
            "available_columns": self.available_columns,
            "missing_required": self.missing_required,
            "missing_optional": self.missing_optional,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CapabilityResult":
        """Deserialises from a dictionary (e.g. loaded from metadata.json)."""
        return cls(
            name=d.get("name", ""),
            status=CapabilityStatus(d.get("status", "disabled")),
            reason=d.get("reason", ""),
            available_columns=d.get("available_columns", []),
            missing_required=d.get("missing_required", []),
            missing_optional=d.get("missing_optional", []),
            category=d.get("category", "General"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# ABSTRACT BASE: BaseCapability
# ─────────────────────────────────────────────────────────────────────────────

class BaseCapability(ABC):
    """
    Strategy interface for a single detectable recommendation capability.

    Each subclass declares:
    • which columns it requires (required_columns)
    • which columns enhance it  (optional_columns)
    • which values must appear  (required_values)
    • what it enables          (description)
    • how it is grouped        (category)

    detect(df) evaluates availability and returns a CapabilityResult.

    Implementing a custom capability
    --------------------------------
    Subclass BaseCapability and add it to ALL_CAPABILITIES in definitions.py:

    >>> class DeliveryTimeCapability(BaseCapability):
    ...     @property
    ...     def name(self) -> str: return "Delivery Time Filtering"
    ...     @property
    ...     def required_columns(self) -> List[str]: return ["delivery_time_minutes"]
    ...     @property
    ...     def description(self) -> str: return "Filter by estimated delivery time"
    ...     @property
    ...     def category(self) -> str: return "Logistics"
    """

    # ── Abstract interface ────────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable capability name (e.g. 'Budget Recommendation')."""
        ...

    @property
    @abstractmethod
    def required_columns(self) -> List[str]:
        """
        Columns that MUST be present for this capability to be ENABLED.

        If any are missing → CapabilityStatus.DISABLED.
        """
        ...

    @property
    def optional_columns(self) -> List[str]:
        """Columns that enhance this capability but are not strictly required."""
        return []

    @property
    def required_values(self) -> Dict[str, List[str]]:
        """
        Specifies that at least one row must have a specific value in a column.

        If defined and no matching rows exist → CapabilityStatus.PARTIAL.

        Example
        -------
        {"item_type": ["drink", "coffee", "smoothie"]}
        → At least one row in the dataset must have item_type in that list.
        """
        return {}

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of what this capability enables."""
        ...

    @property
    def category(self) -> str:
        """
        Grouping category for the Restaurant Capability Report.

        Suggested values: "Nutrition", "Pricing", "Dietary",
        "Item Type", "Rating & Popularity", "Media", "Logistics".
        """
        return "General"

    # ── Detection Logic ───────────────────────────────────────────────────────

    def detect(self, df: pd.DataFrame) -> CapabilityResult:
        """
        Evaluates this capability against the provided DataFrame.

        Detection Logic (in order)
        --------------------------
        1. Check required_columns presence → DISABLED if any missing.
        2. Check required_values presence → PARTIAL if no rows match.
        3. All checks pass → ENABLED.

        Parameters
        ----------
        df : pd.DataFrame
            The enriched dataset DataFrame (post feature engineering).

        Returns
        -------
        CapabilityResult
        """
        df_cols = set(df.columns.tolist())
        req = self.required_columns
        opt = self.optional_columns

        present   = [c for c in req if c in df_cols]
        missing   = [c for c in req if c not in df_cols]
        miss_opt  = [c for c in opt if c not in df_cols]

        # ── Step 1: required columns ──────────────────────────────────────────
        if missing:
            return CapabilityResult(
                name=self.name,
                status=CapabilityStatus.DISABLED,
                reason=f"Missing required columns: {', '.join(missing)}",
                available_columns=present,
                missing_required=missing,
                missing_optional=miss_opt,
                category=self.category,
            )

        # ── Step 2: required values ───────────────────────────────────────────
        if self.required_values:
            for col, expected in self.required_values.items():
                if col not in df_cols:
                    continue
                actual_set = set(
                    df[col].dropna().astype(str).str.lower().unique().tolist()
                )
                expected_lower = [v.lower() for v in expected]
                if not any(v in actual_set for v in expected_lower):
                    return CapabilityResult(
                        name=self.name,
                        status=CapabilityStatus.PARTIAL,
                        reason=(
                            f"Column '{col}' is present but none of the expected values "
                            f"({expected}) appear in the dataset."
                        ),
                        available_columns=present,
                        missing_required=[],
                        missing_optional=miss_opt,
                        category=self.category,
                    )

        # ── Step 3: all checks passed ─────────────────────────────────────────
        return CapabilityResult(
            name=self.name,
            status=CapabilityStatus.ENABLED,
            reason="All required columns and values are present.",
            available_columns=present,
            missing_required=[],
            missing_optional=miss_opt,
            category=self.category,
        )
