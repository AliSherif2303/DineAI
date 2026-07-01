"""
DineAI — Capability Detector & Report
======================================
Automated feature and capability detection for restaurant datasets.

Runs during the dataset build pipeline to inspect the enriched pandas DataFrame,
evaluate all registered BaseCapability rules, and compile a structured and
human-readable capability report.

Design Principles
-----------------
• Pluggable: uses CapabilityRegistry to fetch capabilities dynamically.
• Serializable: report can be converted to/from JSON-compatible dictionaries.
• Informative: prints an enterprise-grade ASCII report with overall capability score.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Set

import pandas as pd

from dine_ai.capabilities.base import CapabilityResult, CapabilityStatus
from dine_ai.capabilities.registry import CapabilityRegistry, get_default_registry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DTO: RestaurantCapabilityReport
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RestaurantCapabilityReport:
    """
    Serializable container representing the capability profile of a restaurant.

    Stored inside metadata.json and printed during dataset build.

    Attributes
    ----------
    restaurant_name : str
    detected_item_types: List[str]
        List of unique non-null item_types found in the dataset.
    enabled_capabilities: List[CapabilityResult]
        Capabilities that were successfully activated.
    disabled_capabilities: List[CapabilityResult]
        Capabilities that could not be activated.
    capability_score: float
        The ratio of enabled capabilities to total registered capabilities (0-100%).
    available_columns: List[str]
        List of all columns currently present in the dataset.
    missing_recommended_columns: List[str]
        Subset of RECOMMENDED canonical columns that are missing.
    missing_mandatory_columns: List[str]
        Subset of MANDATORY canonical columns that are missing.
    warnings: List[str]
        General warnings/notices about capability limitations.
    build_timestamp: str
        ISO timestamp of the detection run.
    """
    restaurant_name: str
    detected_item_types: List[str] = field(default_factory=list)
    enabled_capabilities: List[CapabilityResult] = field(default_factory=list)
    disabled_capabilities: List[CapabilityResult] = field(default_factory=list)
    capability_score: float = 0.0
    available_columns: List[str] = field(default_factory=list)
    missing_recommended_columns: List[str] = field(default_factory=list)
    missing_mandatory_columns: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    build_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the report to a JSON-compatible dictionary."""
        return {
            "restaurant_name": self.restaurant_name,
            "detected_item_types": self.detected_item_types,
            "enabled_capabilities": [c.to_dict() for c in self.enabled_capabilities],
            "disabled_capabilities": [c.to_dict() for c in self.disabled_capabilities],
            "capability_score": round(self.capability_score, 2),
            "available_columns": self.available_columns,
            "missing_recommended_columns": self.missing_recommended_columns,
            "missing_mandatory_columns": self.missing_mandatory_columns,
            "warnings": self.warnings,
            "build_timestamp": self.build_timestamp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> RestaurantCapabilityReport:
        """Deserializes the report from a dictionary."""
        enabled = [CapabilityResult.from_dict(c) for c in d.get("enabled_capabilities", [])]
        disabled = [CapabilityResult.from_dict(c) for c in d.get("disabled_capabilities", [])]
        return cls(
            restaurant_name=d.get("restaurant_name", ""),
            detected_item_types=d.get("detected_item_types", []),
            enabled_capabilities=enabled,
            disabled_capabilities=disabled,
            capability_score=d.get("capability_score", 0.0),
            available_columns=d.get("available_columns", []),
            missing_recommended_columns=d.get("missing_recommended_columns", []),
            missing_mandatory_columns=d.get("missing_mandatory_columns", []),
            warnings=d.get("warnings", []),
            build_timestamp=d.get("build_timestamp", ""),
        )

    def print_report(self) -> None:
        """Prints an enterprise-style ASCII Restaurant Capability Report."""
        width = 60
        double_sep = "=" * width
        single_sep = "-" * width

        print()
        print(double_sep)
        print("RESTAURANT CAPABILITY REPORT".center(width))
        print(double_sep)
        print(f" Restaurant       : {self.restaurant_name}")
        print(f" Build Timestamp  : {self.build_timestamp}")
        print(single_sep)

        # ── Item Types ────────────────────────────────────────────────────────
        print(" Detected Menu Items")
        if self.detected_item_types:
            for item in sorted(self.detected_item_types):
                # Format to title case
                item_title = item.replace("_", " ").title()
                print(f"  [OK] {item_title}")
        else:
            print("  [--] No items classified (using defaults)")

        print(single_sep)

        # ── Group capabilities by category for clean printing ────────────────
        print(" Available Recommendation Features")
        
        # Categorized lists
        all_caps = self.enabled_capabilities + self.disabled_capabilities
        by_category: Dict[str, List[CapabilityResult]] = {}
        for c in all_caps:
            by_category.setdefault(c.category, []).append(c)

        for cat, results in sorted(by_category.items()):
            # Print category header if we want, or just print them flat with indicators
            for r in sorted(results, key=lambda x: x.name):
                status_indicator = "[OK]" if r.status == CapabilityStatus.ENABLED else "[--]"
                print(f"  {status_indicator} {r.name:<30} ({cat})")

        print(single_sep)
        print(f" Overall Capability Score     {self.capability_score:.1f}%")
        print(double_sep)
        print()


# ─────────────────────────────────────────────────────────────────────────────
# CLASS: CapabilityDetector
# ─────────────────────────────────────────────────────────────────────────────

class CapabilityDetector:
    """
    Scans restaurant DataFrames to detect features, columns, and active capabilities.

    Integrates with CapabilityRegistry.

    Parameters
    ----------
    registry : Optional[CapabilityRegistry]
        Pluggable registry containing capabilities rules.
        Defaults to the system DEFAULT_REGISTRY.
    """

    def __init__(self, registry: Optional[CapabilityRegistry] = None) -> None:
        self.registry = registry or get_default_registry()

    def detect(self, df: pd.DataFrame, restaurant_name: str) -> RestaurantCapabilityReport:
        """
        Scans the DataFrame and produces a detailed capability report.

        Parameters
        ----------
        df : pd.DataFrame
            The pandas DataFrame after columns mapping and feature engineering.
        restaurant_name : str

        Returns
        -------
        RestaurantCapabilityReport
        """
        # Ensure we have clean columns list
        available_columns = df.columns.tolist()
        cols_set = set(available_columns)

        # Detect item types
        detected_item_types: List[str] = []
        if "item_type" in cols_set:
            detected_item_types = df["item_type"].dropna().unique().tolist()
            detected_item_types = [str(x) for x in detected_item_types]

        # Scan registry capabilities
        enabled_capabilities: List[CapabilityResult] = []
        disabled_capabilities: List[CapabilityResult] = []

        results = self.registry.detect_all(df)
        for r in results:
            if r.status == CapabilityStatus.ENABLED:
                enabled_capabilities.append(r)
            else:
                disabled_capabilities.append(r)

        # Compute capability score
        total_caps = len(results)
        capability_score = (len(enabled_capabilities) / total_caps * 100.0) if total_caps > 0 else 0.0

        # Define 3-tier checks for validation warnings
        # Mandatory, recommended, optional are classified inside validator, but we can summarize here.
        from dine_ai.adapters.validator import MANDATORY_COLUMNS, RECOMMENDED_COLUMNS
        
        missing_mandatory = [c for c in MANDATORY_COLUMNS if c not in cols_set]
        missing_recommended = [c for c in RECOMMENDED_COLUMNS if c not in cols_set]

        # Construct warnings
        warnings: List[str] = []
        for mr in missing_recommended:
            # Generate descriptive warning
            cap_disabled_str = ""
            if mr == "price":
                cap_disabled_str = "Budget Filtering disabled."
            elif mr == "meal_type":
                cap_disabled_str = "Meal Type Filtering disabled."
            elif mr == "ingredients":
                cap_disabled_str = "Ingredient Search disabled."
            elif mr == "description":
                cap_disabled_str = "Text Summaries limited."
            
            warnings.append(
                f"Recommended column '{mr}' is missing — {cap_disabled_str}"
            )

        return RestaurantCapabilityReport(
            restaurant_name=restaurant_name,
            detected_item_types=detected_item_types,
            enabled_capabilities=enabled_capabilities,
            disabled_capabilities=disabled_capabilities,
            capability_score=capability_score,
            available_columns=available_columns,
            missing_recommended_columns=missing_recommended,
            missing_mandatory_columns=missing_mandatory,
            warnings=warnings,
        )


# ─────────────────────────────────────────────────────────────────────────────
# SELF TESTS
# ─────────────────────────────────────────────────────────────────────────────

def _run_self_tests() -> None:
    """Verifies basic capability detection behavior."""
    print("=" * 80)
    print(" CapabilityDetector — Self-Tests")
    print("=" * 80)

    # Make dummy DataFrame
    data = {
        "recipe_name": ["Burger", "Pizza", "Salad", "Soda"],
        "price": [10.0, 12.0, 8.0, 2.0],
        "item_type": ["main_course", "main_course", "salad", "drink"],
        "diet_labels": ["keto", "none", "vegan", "none"]
    }
    df = pd.DataFrame(data)

    detector = CapabilityDetector()
    report = detector.detect(df, "Test Restaurant")
    report.print_report()

    # Validate some expectations
    assert len(report.detected_item_types) == 3, f"Expected 3 item types, got {len(report.detected_item_types)}"
    
    # Check if budget recommendation is enabled
    budget_enabled = any(c.name == "Budget Filtering" for c in report.enabled_capabilities)
    assert budget_enabled, "Budget Filtering should be enabled"

    print("CapabilityDetector self-tests passed successfully!")


if __name__ == "__main__":
    # Standard dummy check
    import pandas as pd
    
    # We need to temporarily mock validator if we run it as a standalone script
    # and validator isn't loaded correctly, but Python path resolution will handle it.
    try:
        from dine_ai.adapters.validator import MANDATORY_COLUMNS
    except ImportError:
        # Fallback for standalone run if dependencies aren't set
        import sys
        import os
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    
    _run_self_tests()
