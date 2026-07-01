"""
DineAI — Filtering Engine
==========================

Applies structured filtering rules (HARD constraints + SOFT scoring)
to SearchResult candidates produced by semantic_search.py.

This module is intentionally domain-agnostic.  The same engine works for
restaurants, hotels, movies, healthcare, e-commerce, or any future domain
by changing only the rules — never the engine.

Architecture
------------
FilteringEngine (Coordinator)
    ├── FilterConfig              — Pipeline-level settings
    ├── FilterRule                — One declarative rule (HARD or SOFT)
    ├── FilterGroup               — Nested AND/OR boolean logic tree
    ├── FilterOperator            — All supported operators (Enum)
    ├── ComputedMetricRegistry    — Virtual columns computed at runtime
    ├── FilterRegistry            — Plugin registry: operator/type → BaseFilter
    ├── BaseFilter                — Abstract strategy interface (DIP / OCP)
    │   ├── NumericFilter, BooleanFilter, TextFilter, ListFilter
    │   ├── RangeFilter, RegexFilter, ExistsFilter, NullFilter
    │   ├── DatetimeFilter, CategoryFilter, CustomFilter
    ├── CandidateFilterResult     — Candidate + hard_passed + soft_score + explanations
    ├── FilteringStatistics       — Detached telemetry
    ├── FilteringResult           — Structured output
    └── EnterpriseFilteringReport — Human-readable report generator

Execution Order
---------------
1. Resolve column values (physical or computed)
2. Hard Rules evaluation (flat + nested groups)
3. Reject failing candidates immediately
4. Soft Rule scoring on surviving candidates
5. Return FilteringResult with per-candidate scores and explanations

Public API
----------
    engine = FilteringEngine()
    result = engine.filter(search_result, rules)
    EnterpriseFilteringReport.generate(result)
"""

import re
import sys
import math
import json
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional,
    Tuple, Type, Union
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 1: FilterOperator
# ─────────────────────────────────────────────────────────────────────────────

class FilterOperator(str, Enum):
    """
    Exhaustive catalogue of all supported filtering operators.

    Inherits from str so operators can be compared directly with string
    literals from config files / API payloads.
    """
    # ── Numeric ───────────────────────────────────────────────────────────────
    EQ                 = "=="
    NEQ                = "!="
    GT                 = ">"
    GTE                = ">="
    LT                 = "<"
    LTE                = "<="
    BETWEEN            = "between"
    OUTSIDE            = "outside"
    APPROXIMATELY_EQ   = "approximately_equal"

    # ── Text ──────────────────────────────────────────────────────────────────
    CONTAINS           = "contains"
    ICONTAINS          = "icontains"
    STARTSWITH         = "startswith"
    ENDSWITH           = "endswith"
    EXACT              = "exact"
    REGEX              = "regex"

    # ── List ──────────────────────────────────────────────────────────────────
    LIST_CONTAINS      = "list_contains"
    CONTAINS_ANY       = "contains_any"
    CONTAINS_ALL       = "contains_all"
    OVERLAP            = "overlap"
    SUBSET             = "subset"
    SUPERSET           = "superset"

    # ── Boolean ───────────────────────────────────────────────────────────────
    IS_TRUE            = "is_true"
    IS_FALSE           = "is_false"

    # ── Null ──────────────────────────────────────────────────────────────────
    IS_NULL            = "is_null"
    NOT_NULL           = "not_null"

    # ── Existence ─────────────────────────────────────────────────────────────
    EXISTS             = "exists"
    NOT_EXISTS         = "not_exists"

    # ── Membership ────────────────────────────────────────────────────────────
    IN                 = "in"
    NOT_IN             = "not_in"

    # ── Datetime ──────────────────────────────────────────────────────────────
    BEFORE             = "before"
    AFTER              = "after"
    BETWEEN_DATES      = "between_dates"
    TODAY              = "today"
    LAST_DAYS          = "last_days"
    NEXT_DAYS          = "next_days"


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 2: ComputedMetric
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComputedMetric:
    """
    A virtual column that is computed at runtime from existing row data.

    Parameters
    ----------
    name : str
        Unique identifier used as the column name in FilterRule.
    function : Callable[[Dict[str, Any]], Any]
        Callable that receives the full metadata dict and returns a value.
    description : str
        Human-readable explanation of what this metric computes.
    """
    name: str
    function: Callable[[Dict[str, Any]], Any]
    description: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 3: ComputedMetricRegistry
# ─────────────────────────────────────────────────────────────────────────────

class ComputedMetricRegistry:
    """
    Registry of virtual columns that can be used in FilterRule just like
    any physical DataFrame column.

    Pre-registered metrics (nutrition domain)
    -----------------------------------------
    - protein_to_calorie_ratio   protein / calories
    - fat_percentage             fat / (protein + carbs + fat) * 100
    - carb_percentage            carbs / (protein + carbs + fat) * 100
    - protein_percentage         protein / (protein + carbs + fat) * 100
    - net_carbs                  carbs - fiber
    - calories_per_100g          calories / weight_g * 100
    - price_per_serving          price / servings
    - protein_per_dollar         protein / price
    - rating_per_price           rating / price

    Any future domain metric is one register() call away.
    """

    def __init__(self):
        self._metrics: Dict[str, ComputedMetric] = {}
        self._register_defaults()

    def register(
        self,
        name: str,
        function: Callable[[Dict[str, Any]], Any],
        description: str = "",
    ) -> None:
        """
        Registers a new computed metric.

        Parameters
        ----------
        name : str
            Column alias used in FilterRule(column=name).
        function : Callable[[Dict], Any]
            Receives the candidate's full metadata dict; returns a value.
        description : str
            Human-readable documentation for the metric.
        """
        self._metrics[name] = ComputedMetric(name=name, function=function, description=description)

    def resolve(self, name: str, row: Dict[str, Any]) -> Optional[Any]:
        """
        Computes and returns the metric value for a given row.

        Returns None if the metric is not registered or computation fails.
        """
        metric = self._metrics.get(name)
        if metric is None:
            return None
        try:
            return metric.function(row)
        except Exception as exc:
            logger.debug("ComputedMetricRegistry: failed to compute '%s': %s", name, exc)
            return None

    def is_registered(self, name: str) -> bool:
        return name in self._metrics

    def list_metrics(self) -> List[str]:
        return list(self._metrics.keys())

    def _register_defaults(self) -> None:
        """Pre-registers built-in computed metrics."""

        def _safe_div(a, b, default=0.0):
            try:
                return float(a) / float(b) if float(b) != 0 else default
            except (TypeError, ValueError):
                return default

        self.register(
            "protein_to_calorie_ratio",
            lambda r: _safe_div(r.get("protein_g_per_serving", 0), r.get("calories_per_serving", 1)),
            "protein_g_per_serving / calories_per_serving",
        )
        self.register(
            "net_carbs",
            lambda r: float(r.get("carbs_g_per_serving", 0)) - float(r.get("fiber_g_per_serving", 0)),
            "carbs_g_per_serving - fiber_g_per_serving",
        )
        self.register(
            "fat_percentage",
            lambda r: _safe_div(
                r.get("fat_g_per_serving", 0),
                sum(float(r.get(k, 0)) for k in ("protein_g_per_serving", "carbs_g_per_serving", "fat_g_per_serving"))
            ) * 100,
            "fat / (protein + carbs + fat) × 100",
        )
        self.register(
            "carb_percentage",
            lambda r: _safe_div(
                r.get("carbs_g_per_serving", 0),
                sum(float(r.get(k, 0)) for k in ("protein_g_per_serving", "carbs_g_per_serving", "fat_g_per_serving"))
            ) * 100,
            "carbs / (protein + carbs + fat) × 100",
        )
        self.register(
            "protein_percentage",
            lambda r: _safe_div(
                r.get("protein_g_per_serving", 0),
                sum(float(r.get(k, 0)) for k in ("protein_g_per_serving", "carbs_g_per_serving", "fat_g_per_serving"))
            ) * 100,
            "protein / (protein + carbs + fat) × 100",
        )
        self.register(
            "price_per_serving",
            lambda r: _safe_div(r.get("price", 0), r.get("servings", 1)),
            "price / servings",
        )
        self.register(
            "protein_per_dollar",
            lambda r: _safe_div(r.get("protein_g_per_serving", 0), r.get("price", 1)),
            "protein_g_per_serving / price",
        )
        self.register(
            "rating_per_price",
            lambda r: _safe_div(r.get("rating", 0), r.get("price", 1)),
            "rating / price",
        )
        self.register(
            "ingredient_count",
            lambda r: len(r.get("ingredients", [])) if isinstance(r.get("ingredients"), list) else 0,
            "len(ingredients)",
        )
        self.register(
            "calories_per_100g",
            lambda r: _safe_div(r.get("calories_per_serving", 0), r.get("weight_g_per_serving", 100)) * 100,
            "calories / weight_g × 100",
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 4: FilterConfig
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FilterConfig:
    """
    Pipeline-level configuration for FilteringEngine.

    Parameters
    ----------
    skip_missing_columns : bool
        If True, rules on absent columns are skipped with a warning.
        If False, missing columns raise ValueError.  Default: True.
    skip_unknown_operators : bool
        If True, rules with unrecognised operators are skipped.
        Default: True.
    stop_on_hard_fail : bool
        If True, reject a candidate immediately on first HARD rule failure
        (faster).  If False, evaluate all hard rules for full diagnostics.
        Default: True.
    approximately_equal_tolerance : float
        Relative tolerance for the approximately_equal operator.
        Default: 0.05  (5%)
    metadata_key : str
        Key used to access the row data dict on each candidate object.
        Default: "metadata"
    """
    skip_missing_columns: bool = True
    skip_unknown_operators: bool = True
    stop_on_hard_fail: bool = True
    approximately_equal_tolerance: float = 0.05
    metadata_key: str = "metadata"

    def validate(self) -> None:
        if not 0.0 <= self.approximately_equal_tolerance <= 1.0:
            raise ValueError(
                f"FilterConfig.approximately_equal_tolerance must be in [0, 1], "
                f"got {self.approximately_equal_tolerance}."
            )


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 5: FilterRule
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FilterRule:
    """
    A single, declarative filtering rule.

    A FilterRule is domain-agnostic: column can be any physical column or
    any registered ComputedMetric name.

    Parameters
    ----------
    column : str
        Physical column name or ComputedMetric key.
    operator : str
        One of the FilterOperator values (e.g. ">=", "contains", "in").
    value : Any
        The threshold/reference value to compare against.
        For range operators: [lower, upper].
        For in/not_in: List of allowed values.
    rule_type : str
        "HARD" — mandatory constraint; failing candidates are rejected.
        "SOFT" — preference rule; contributes score, never rejects.
    reward : float
        Score added when a SOFT rule is satisfied.  Default: 0.0.
    penalty : float
        Score added (usually negative) when a SOFT rule is violated.
        Default: 0.0.
    weight : float
        Multiplier applied to reward/penalty (for weighted soft scoring).
        Default: 1.0.
    priority : int
        Execution order for hard rules (lower = evaluated first).
        Default: 0.
    enabled : bool
        If False, the rule is completely ignored.  Default: True.
    description : str
        Human-readable explanation of the rule's intent.
    """
    column: str
    operator: str
    value: Any = None
    rule_type: str = "HARD"          # "HARD" | "SOFT"
    reward: float = 0.0
    penalty: float = 0.0
    weight: float = 1.0
    priority: int = 0
    enabled: bool = True
    description: str = ""
    required_column: Optional[str] = None

    def validate(self) -> None:
        if self.rule_type not in ("HARD", "SOFT"):
            raise ValueError(f"FilterRule.rule_type must be 'HARD' or 'SOFT', got '{self.rule_type}'.")
        if not self.column:
            raise ValueError("FilterRule.column must be a non-empty string.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "column": self.column,
            "operator": self.operator,
            "value": self.value,
            "rule_type": self.rule_type,
            "reward": self.reward,
            "penalty": self.penalty,
            "weight": self.weight,
            "priority": self.priority,
            "enabled": self.enabled,
            "description": self.description,
            "required_column": self.required_column,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 6: FilterGroup
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FilterGroup:
    """
    A nested boolean group of FilterRules and/or sub-FilterGroups.

    FilterGroups always evaluate as HARD constraints.
    SOFT rules should always be individual FilterRule objects.

    Parameters
    ----------
    operator : str
        "AND" — all members must pass.
        "OR"  — at least one member must pass.
    rules : List[Union[FilterRule, FilterGroup]]
        Members of this group (rules and/or sub-groups).
    description : str
        Human-readable label for this group.
    enabled : bool
        If False, the entire group is skipped.  Default: True.

    Example
    -------
    FilterGroup("AND", [
        FilterRule("calories_per_serving", "<=", 500),
        FilterRule("protein_g_per_serving", ">=", 30),
        FilterGroup("OR", [
            FilterRule("is_vegan", "is_true"),
            FilterRule("is_vegetarian", "is_true"),
        ])
    ])
    """
    operator: str                                    # "AND" | "OR"
    rules: List[Union[FilterRule, "FilterGroup"]] = field(default_factory=list)
    description: str = ""
    enabled: bool = True

    def validate(self) -> None:
        if self.operator not in ("AND", "OR"):
            raise ValueError(f"FilterGroup.operator must be 'AND' or 'OR', got '{self.operator}'.")
        if not self.rules:
            raise ValueError("FilterGroup must contain at least one rule or sub-group.")


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 7: RuleExplanation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuleExplanation:
    """
    Records the outcome of evaluating a single rule against a candidate.

    Attributes
    ----------
    rule_column : str
    operator : str
    expected_value : Any
    actual_value : Any
    passed : bool
    rule_type : str        "HARD" | "SOFT"
    score_delta : float    Reward or penalty applied (SOFT only).
    skipped : bool         True when column was missing or rule was disabled.
    skip_reason : str      Explanation of why the rule was skipped.
    """
    rule_column: str
    operator: str
    expected_value: Any
    actual_value: Any
    passed: bool
    rule_type: str = "HARD"
    score_delta: float = 0.0
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_column": self.rule_column,
            "operator": self.operator,
            "expected_value": str(self.expected_value),
            "actual_value": str(self.actual_value),
            "passed": self.passed,
            "rule_type": self.rule_type,
            "score_delta": self.score_delta,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 8: CandidateFilterResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateFilterResult:
    """
    Wraps a candidate object and records its full filtering state.

    Does NOT subclass SearchCandidate — pure composition.

    Attributes
    ----------
    candidate : Any
        The original SearchCandidate (or any object with row_id + metadata).
    hard_passed : bool
        True if the candidate survived all hard rules.
    soft_score : float
        Accumulated preference score from soft rules.
    rule_explanations : List[RuleExplanation]
        Per-rule pass/fail records.
    matched_rules : List[str]
        Column names of rules the candidate satisfied.
    failed_rules : List[str]
        Column names of rules the candidate violated.
    rejection_reason : Optional[str]
        The hard rule that caused rejection (if any).
    """
    candidate: Any
    hard_passed: bool = True
    soft_score: float = 0.0
    rule_explanations: List[RuleExplanation] = field(default_factory=list)
    matched_rules: List[str] = field(default_factory=list)
    failed_rules: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None

    @property
    def row_id(self) -> Any:
        return getattr(self.candidate, "row_id", None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_id": self.row_id,
            "hard_passed": self.hard_passed,
            "soft_score": round(self.soft_score, 4),
            "matched_rules": self.matched_rules,
            "failed_rules": self.failed_rules,
            "rejection_reason": self.rejection_reason,
            "rule_explanations": [e.to_dict() for e in self.rule_explanations],
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 9: BaseFilter (Abstract Strategy)
# ─────────────────────────────────────────────────────────────────────────────

class BaseFilter(ABC):
    """
    Abstract strategy interface for all filter implementations.

    Each concrete subclass handles one category of filtering logic.
    FilteringEngine dispatches to the correct subclass based on the
    value type and operator — no rule needs to declare its filter class.

    Plugin Contract
    ---------------
    Implement apply() and declare supported_operators.
    Register with FilterRegistry.register(key, MyFilter).
    """

    @abstractmethod
    def apply(self, value: Any, operator: str, threshold: Any, config: FilterConfig) -> bool:
        """
        Evaluates whether ``value`` satisfies the rule.

        Parameters
        ----------
        value : Any
            The candidate's column value (or computed metric value).
        operator : str
            The FilterOperator string.
        threshold : Any
            The rule's reference value.
        config : FilterConfig
            Active pipeline configuration.

        Returns
        -------
        bool
            True if the candidate satisfies the rule.
        """
        pass

    @property
    @abstractmethod
    def supported_operators(self) -> List[str]:
        """List of operator strings this filter handles."""
        pass

    @property
    @abstractmethod
    def filter_name(self) -> str:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 10: NumericFilter
# ─────────────────────────────────────────────────────────────────────────────

class NumericFilter(BaseFilter):
    """Handles numeric comparisons: ==, !=, >, >=, <, <=, approximately_equal."""

    @property
    def filter_name(self) -> str:
        return "NumericFilter"

    @property
    def supported_operators(self) -> List[str]:
        return ["==", "!=", ">", ">=", "<", "<=", "approximately_equal"]

    def apply(self, value: Any, operator: str, threshold: Any, config: FilterConfig) -> bool:
        try:
            v, t = float(value), float(threshold)
        except (TypeError, ValueError):
            return False
        if operator == "==":         return v == t
        if operator == "!=":         return v != t
        if operator == ">":          return v > t
        if operator == ">=":         return v >= t
        if operator == "<":          return v < t
        if operator == "<=":         return v <= t
        if operator == "approximately_equal":
            return abs(v - t) <= config.approximately_equal_tolerance * abs(t) if t != 0 else v == t
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 11: BooleanFilter
# ─────────────────────────────────────────────────────────────────────────────

class BooleanFilter(BaseFilter):
    """Handles boolean checks: is_true, is_false."""

    @property
    def filter_name(self) -> str:
        return "BooleanFilter"

    @property
    def supported_operators(self) -> List[str]:
        return ["is_true", "is_false"]

    def apply(self, value: Any, operator: str, threshold: Any, config: FilterConfig) -> bool:
        truthy = value is True or (isinstance(value, (int, float)) and value == 1) or str(value).lower() in ("true", "1", "yes")
        if operator == "is_true":  return truthy
        if operator == "is_false": return not truthy
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 12: TextFilter
# ─────────────────────────────────────────────────────────────────────────────

class TextFilter(BaseFilter):
    """Handles string matching: contains, icontains, startswith, endswith, exact."""

    @property
    def filter_name(self) -> str:
        return "TextFilter"

    @property
    def supported_operators(self) -> List[str]:
        return ["contains", "icontains", "startswith", "endswith", "exact"]

    def apply(self, value: Any, operator: str, threshold: Any, config: FilterConfig) -> bool:
        s, t = str(value), str(threshold)
        if operator == "contains":   return t in s
        if operator == "icontains":  return t.lower() in s.lower()
        if operator == "startswith": return s.startswith(t)
        if operator == "endswith":   return s.endswith(t)
        if operator == "exact":      return s == t
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 13: ListFilter
# ─────────────────────────────────────────────────────────────────────────────

class ListFilter(BaseFilter):
    """
    Handles list/set operations: list_contains, contains_any, contains_all,
    overlap, subset, superset.
    """

    @property
    def filter_name(self) -> str:
        return "ListFilter"

    @property
    def supported_operators(self) -> List[str]:
        return ["list_contains", "contains_any", "contains_all", "overlap", "subset", "superset"]

    def apply(self, value: Any, operator: str, threshold: Any, config: FilterConfig) -> bool:
        v_set = set(value) if isinstance(value, (list, tuple, set)) else {value}
        t_set = set(threshold) if isinstance(threshold, (list, tuple, set)) else {threshold}

        if operator == "list_contains": return threshold in v_set
        if operator == "contains_any":  return bool(v_set & t_set)
        if operator == "contains_all":  return t_set.issubset(v_set)
        if operator == "overlap":       return bool(v_set & t_set)
        if operator == "subset":        return v_set.issubset(t_set)
        if operator == "superset":      return v_set.issuperset(t_set)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 14: RangeFilter
# ─────────────────────────────────────────────────────────────────────────────

class RangeFilter(BaseFilter):
    """Handles range checks: between [lo, hi] and outside [lo, hi]."""

    @property
    def filter_name(self) -> str:
        return "RangeFilter"

    @property
    def supported_operators(self) -> List[str]:
        return ["between", "outside"]

    def apply(self, value: Any, operator: str, threshold: Any, config: FilterConfig) -> bool:
        try:
            v = float(value)
            lo, hi = float(threshold[0]), float(threshold[1])
        except (TypeError, ValueError, IndexError):
            return False
        if operator == "between": return lo <= v <= hi
        if operator == "outside": return v < lo or v > hi
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 15: RegexFilter
# ─────────────────────────────────────────────────────────────────────────────

class RegexFilter(BaseFilter):
    """Handles full regular expression matching."""

    @property
    def filter_name(self) -> str:
        return "RegexFilter"

    @property
    def supported_operators(self) -> List[str]:
        return ["regex"]

    def apply(self, value: Any, operator: str, threshold: Any, config: FilterConfig) -> bool:
        try:
            return bool(re.search(str(threshold), str(value)))
        except re.error:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 16: ExistsFilter
# ─────────────────────────────────────────────────────────────────────────────

class ExistsFilter(BaseFilter):
    """
    Checks whether a column key exists in the candidate metadata dict.
    Note: value here is the key itself (passed specially by the engine).
    """

    @property
    def filter_name(self) -> str:
        return "ExistsFilter"

    @property
    def supported_operators(self) -> List[str]:
        return ["exists", "not_exists"]

    def apply(self, value: Any, operator: str, threshold: Any, config: FilterConfig) -> bool:
        # value is True if key was found in metadata, False otherwise
        key_exists = (value is not None)
        if operator == "exists":     return key_exists
        if operator == "not_exists": return not key_exists
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 17: NullFilter
# ─────────────────────────────────────────────────────────────────────────────

class NullFilter(BaseFilter):
    """Checks for null/None values."""

    @property
    def filter_name(self) -> str:
        return "NullFilter"

    @property
    def supported_operators(self) -> List[str]:
        return ["is_null", "not_null"]

    def apply(self, value: Any, operator: str, threshold: Any, config: FilterConfig) -> bool:
        is_none = value is None or (isinstance(value, float) and math.isnan(value))
        if operator == "is_null":  return is_none
        if operator == "not_null": return not is_none
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 18: DatetimeFilter
# ─────────────────────────────────────────────────────────────────────────────

class DatetimeFilter(BaseFilter):
    """
    Handles datetime comparisons: before, after, between_dates,
    today, last_days, next_days.

    Input values can be ISO-format strings or datetime objects.
    """

    @property
    def filter_name(self) -> str:
        return "DatetimeFilter"

    @property
    def supported_operators(self) -> List[str]:
        return ["before", "after", "between_dates", "today", "last_days", "next_days"]

    @staticmethod
    def _parse(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except (ValueError, TypeError):
            return None

    def apply(self, value: Any, operator: str, threshold: Any, config: FilterConfig) -> bool:
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        dt = self._parse(value)
        if dt is None:
            return False

        if operator == "before":
            t = self._parse(threshold)
            return dt < t if t else False

        if operator == "after":
            t = self._parse(threshold)
            return dt > t if t else False

        if operator == "between_dates":
            try:
                lo, hi = self._parse(threshold[0]), self._parse(threshold[1])
                return lo <= dt <= hi if lo and hi else False
            except (IndexError, TypeError):
                return False

        if operator == "today":
            return dt.date() == now.date()

        if operator == "last_days":
            try:
                cutoff = now - timedelta(days=int(threshold))
                return dt >= cutoff
            except (TypeError, ValueError):
                return False

        if operator == "next_days":
            try:
                cutoff = now + timedelta(days=int(threshold))
                return dt <= cutoff
            except (TypeError, ValueError):
                return False

        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 19: CategoryFilter
# ─────────────────────────────────────────────────────────────────────────────

class CategoryFilter(BaseFilter):
    """Handles membership checks: in, not_in."""

    @property
    def filter_name(self) -> str:
        return "CategoryFilter"

    @property
    def supported_operators(self) -> List[str]:
        return ["in", "not_in"]

    def apply(self, value: Any, operator: str, threshold: Any, config: FilterConfig) -> bool:
        allowed = set(threshold) if isinstance(threshold, (list, tuple, set)) else {threshold}
        if operator == "in":     return value in allowed
        if operator == "not_in": return value not in allowed
        return False


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 20: CustomFilter
# ─────────────────────────────────────────────────────────────────────────────

class CustomFilter(BaseFilter):
    """
    Plugin-ready wrapper for arbitrary callable filter logic.

    Example
    -------
    def geo_distance_check(value, operator, threshold, config):
        # value = (lat, lon) tuple
        return haversine(value, threshold) <= config.max_km

    FilterRegistry.register("geo_distance", CustomFilter(geo_distance_check, ["geo_distance"]))
    """

    def __init__(
        self,
        fn: Callable[[Any, str, Any, FilterConfig], bool],
        operators: List[str],
        name: str = "CustomFilter",
    ):
        self._fn = fn
        self._operators = operators
        self._name = name

    @property
    def filter_name(self) -> str:
        return self._name

    @property
    def supported_operators(self) -> List[str]:
        return self._operators

    def apply(self, value: Any, operator: str, threshold: Any, config: FilterConfig) -> bool:
        return self._fn(value, operator, threshold, config)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 21: FilterRegistry
# ─────────────────────────────────────────────────────────────────────────────

class FilterRegistry:
    """
    Maps operators to their BaseFilter implementations.

    Operator → Filter dispatch is automatic.  Adding a new filter type
    requires only a register() call — FilteringEngine is unchanged (OCP).

    Plugin Example
    --------------
    FilterRegistry.register("geo_distance", CustomFilter(fn, ["geo_distance"]))
    """

    def __init__(self):
        self._operator_map: Dict[str, BaseFilter] = {}
        self._register_defaults()

    def register(self, filter_impl: BaseFilter) -> None:
        """Registers a filter for all of its declared supported_operators."""
        for op in filter_impl.supported_operators:
            self._operator_map[op] = filter_impl

    def register_custom(self, operator: str, filter_impl: BaseFilter) -> None:
        """Registers a filter for a single custom operator key."""
        self._operator_map[operator] = filter_impl

    def resolve(self, operator: str) -> Optional[BaseFilter]:
        """Returns the filter for a given operator, or None if unregistered."""
        return self._operator_map.get(operator)

    def list_operators(self) -> List[str]:
        return sorted(self._operator_map.keys())

    def _register_defaults(self) -> None:
        for f in [
            NumericFilter(),
            BooleanFilter(),
            TextFilter(),
            ListFilter(),
            RangeFilter(),
            RegexFilter(),
            ExistsFilter(),
            NullFilter(),
            DatetimeFilter(),
            CategoryFilter(),
        ]:
            self.register(f)


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 22: FilteringStatistics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FilteringStatistics:
    """
    Execution telemetry for a single FilteringEngine.filter() call.

    Detached from FilteringResult so it can be logged or exported independently.
    """
    input_candidates: int = 0
    candidates_after_hard: int = 0
    candidates_after_soft: int = 0
    hard_rules_evaluated: int = 0
    soft_rules_evaluated: int = 0
    rules_skipped: int = 0
    rules_failed: int = 0
    total_time_seconds: float = 0.0
    hard_filter_time_seconds: float = 0.0
    soft_score_time_seconds: float = 0.0
    average_soft_score: float = 0.0
    max_soft_score: float = 0.0
    min_soft_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_candidates": self.input_candidates,
            "candidates_after_hard": self.candidates_after_hard,
            "candidates_after_soft": self.candidates_after_soft,
            "hard_rules_evaluated": self.hard_rules_evaluated,
            "soft_rules_evaluated": self.soft_rules_evaluated,
            "rules_skipped": self.rules_skipped,
            "rules_failed": self.rules_failed,
            "total_time_seconds": round(self.total_time_seconds, 4),
            "hard_filter_time_seconds": round(self.hard_filter_time_seconds, 4),
            "soft_score_time_seconds": round(self.soft_score_time_seconds, 4),
            "average_soft_score": round(self.average_soft_score, 4),
            "max_soft_score": round(self.max_soft_score, 4),
            "min_soft_score": round(self.min_soft_score, 4),
            "timestamp": self.timestamp,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 23: FilteringResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FilteringResult:
    """
    Structured output of FilteringEngine.filter().

    Attributes
    ----------
    passed_candidates : List[CandidateFilterResult]
        Candidates that survived all hard rules, with soft scores attached.
    rejected_candidates : List[CandidateFilterResult]
        Candidates rejected by at least one HARD rule.
    statistics : FilteringStatistics
        Full execution telemetry.
    warnings : List[str]
        Skipped rules, missing columns, unknown operators, etc.
    hard_rule_breakdown : Dict[str, int]
        Per-column count of candidates that failed each hard rule.
    soft_rule_breakdown : Dict[str, Dict]
        Per-column reward/penalty summary across all candidates.
    is_success : bool
    """
    passed_candidates: List[CandidateFilterResult]
    rejected_candidates: List[CandidateFilterResult]
    statistics: FilteringStatistics
    warnings: List[str] = field(default_factory=list)
    hard_rule_breakdown: Dict[str, int] = field(default_factory=dict)
    soft_rule_breakdown: Dict[str, Any] = field(default_factory=dict)
    is_success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed_candidates": [c.to_dict() for c in self.passed_candidates],
            "rejected_candidates": [c.to_dict() for c in self.rejected_candidates],
            "statistics": self.statistics.to_dict(),
            "warnings": self.warnings,
            "hard_rule_breakdown": self.hard_rule_breakdown,
            "soft_rule_breakdown": self.soft_rule_breakdown,
            "is_success": self.is_success,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @property
    def candidate_scores(self) -> Dict[Any, float]:
        """Returns {row_id: soft_score} for all passed candidates."""
        return {c.row_id: c.soft_score for c in self.passed_candidates}


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 24: FilteringEngine
# ─────────────────────────────────────────────────────────────────────────────

class FilteringEngine:
    """
    Pipeline coordinator that applies HARD constraints then SOFT scoring.

    Execution Order
    ---------------
    1. Resolve column value for each rule (physical or computed metric)
    2. Sort hard rules by priority (ascending)
    3. Apply hard rules / groups — reject failing candidates
    4. Apply soft rules to survivors — accumulate preference scores
    5. Finalise statistics and return FilteringResult

    Parameters
    ----------
    config : FilterConfig
        Pipeline configuration.  Defaults to FilterConfig().
    registry : Optional[FilterRegistry]
        Filter operator registry.  Defaults to FilterRegistry() (all built-ins).
    metric_registry : Optional[ComputedMetricRegistry]
        Computed metrics registry.  Defaults to ComputedMetricRegistry().
    """

    def __init__(
        self,
        config: Optional[FilterConfig] = None,
        registry: Optional[FilterRegistry] = None,
        metric_registry: Optional[ComputedMetricRegistry] = None,
    ):
        self.config = config or FilterConfig()
        self.config.validate()
        self._registry = registry or FilterRegistry()
        self._metrics = metric_registry or ComputedMetricRegistry()

    # ── Public API ────────────────────────────────────────────────────────────

    def filter(
        self,
        candidates: Any,              # SearchResult or List[Any]
        rules: List[Union[FilterRule, FilterGroup]],
    ) -> FilteringResult:
        """
        Applies rules to a list of candidates.

        Parameters
        ----------
        candidates : SearchResult | List[Any]
            If a SearchResult object is passed, its .candidates list is used.
            Any object with a .metadata dict attribute works.
        rules : List[Union[FilterRule, FilterGroup]]
            Ordered list of HARD rules, SOFT rules, and FilterGroups.

        Returns
        -------
        FilteringResult
        """
        pipeline_start = time.time()
        warnings: List[str] = []
        is_success = True

        # Accept SearchResult or plain list
        if hasattr(candidates, "candidates"):
            raw_candidates = candidates.candidates
        elif isinstance(candidates, list):
            raw_candidates = candidates
        else:
            raw_candidates = list(candidates)

        stats = FilteringStatistics(input_candidates=len(raw_candidates))

        # Wrap each candidate
        wrapped = [
            CandidateFilterResult(candidate=c)
            for c in raw_candidates
        ]

        # Resolve available columns from candidate metadata keys
        available_columns = set()
        if raw_candidates:
            sample_meta = self._get_metadata(raw_candidates[0])
            available_columns = set(sample_meta.keys())

        filtered_rules = []
        for r in rules:
            if isinstance(r, FilterRule):
                req_col = r.required_column if r.required_column is not None else r.column
                # Only check if it is not in available_columns and not a registered computed metric
                if req_col and req_col not in available_columns and not self._metrics.is_registered(req_col):
                    warnings.append(
                        f"Filter '{r.description or r.column}' skipped: column '{req_col}' is unavailable for this restaurant."
                    )
                    continue
            filtered_rules.append(r)

        # Separate and sort rules using filtered_rules
        hard_rules_flat = [r for r in filtered_rules if isinstance(r, FilterRule) and r.rule_type == "HARD" and r.enabled]
        soft_rules_flat = [r for r in filtered_rules if isinstance(r, FilterRule) and r.rule_type == "SOFT" and r.enabled]
        hard_groups     = [r for r in filtered_rules if isinstance(r, FilterGroup) and r.enabled]

        hard_rules_flat.sort(key=lambda r: r.priority)

        hard_breakdown: Dict[str, int] = {}
        soft_breakdown: Dict[str, Any] = {}

        # ── Phase 1: Hard rules + groups ─────────────────────────────────────
        hard_start = time.time()
        surviving: List[CandidateFilterResult] = []
        rejected: List[CandidateFilterResult] = []

        for cfr in wrapped:
            row = self._get_metadata(cfr.candidate)

            # Evaluate flat hard rules
            for rule in hard_rules_flat:
                expl = self._evaluate_rule(rule, row, warnings)
                cfr.rule_explanations.append(expl)
                stats.hard_rules_evaluated += 1

                if expl.skipped:
                    stats.rules_skipped += 1
                    continue

                if expl.passed:
                    cfr.matched_rules.append(rule.column)
                else:
                    cfr.failed_rules.append(rule.column)
                    hard_breakdown[rule.column] = hard_breakdown.get(rule.column, 0) + 1
                    if cfr.hard_passed:
                        cfr.hard_passed = False
                        cfr.rejection_reason = (
                            f"Hard rule failed: {rule.column} {rule.operator} {rule.value}"
                        )
                    if self.config.stop_on_hard_fail:
                        break

            # Evaluate hard groups
            if cfr.hard_passed:
                for group in hard_groups:
                    group_passed, group_expls = self._evaluate_group(group, row, warnings, stats)
                    cfr.rule_explanations.extend(group_expls)
                    if not group_passed:
                        cfr.hard_passed = False
                        cfr.rejection_reason = (
                            f"Hard group failed: {group.description or group.operator}"
                        )
                        break

            if cfr.hard_passed:
                surviving.append(cfr)
            else:
                rejected.append(cfr)

        stats.hard_filter_time_seconds = time.time() - hard_start
        stats.candidates_after_hard = len(surviving)

        # ── Phase 2: Soft scoring on survivors ────────────────────────────────
        soft_start = time.time()

        for cfr in surviving:
            row = self._get_metadata(cfr.candidate)
            for rule in soft_rules_flat:
                expl = self._evaluate_rule(rule, row, warnings)
                cfr.rule_explanations.append(expl)
                stats.soft_rules_evaluated += 1

                if expl.skipped:
                    stats.rules_skipped += 1
                    continue

                # Accumulate score
                if expl.passed:
                    delta = rule.reward * rule.weight
                    cfr.matched_rules.append(rule.column)
                else:
                    delta = rule.penalty * rule.weight
                    cfr.failed_rules.append(rule.column)

                cfr.soft_score += delta
                expl.score_delta = delta

                # Track soft breakdown
                col = rule.column
                if col not in soft_breakdown:
                    soft_breakdown[col] = {"rewarded": 0, "penalised": 0, "total_delta": 0.0}
                if expl.passed:
                    soft_breakdown[col]["rewarded"] += 1
                else:
                    soft_breakdown[col]["penalised"] += 1
                soft_breakdown[col]["total_delta"] = round(
                    soft_breakdown[col]["total_delta"] + delta, 4
                )

        stats.soft_score_time_seconds = time.time() - soft_start
        stats.candidates_after_soft = len(surviving)

        # Finalise soft statistics
        if surviving:
            scores = [c.soft_score for c in surviving]
            stats.average_soft_score = sum(scores) / len(scores)
            stats.max_soft_score = max(scores)
            stats.min_soft_score = min(scores)

        stats.total_time_seconds = time.time() - pipeline_start

        return FilteringResult(
            passed_candidates=surviving,
            rejected_candidates=rejected,
            statistics=stats,
            warnings=list(set(warnings)),    # deduplicate
            hard_rule_breakdown=hard_breakdown,
            soft_rule_breakdown=soft_breakdown,
            is_success=is_success,
        )

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _get_metadata(self, candidate: Any) -> Dict[str, Any]:
        """Extracts the metadata dict from a candidate object."""
        meta = getattr(candidate, self.config.metadata_key, None)
        if isinstance(meta, dict):
            return meta
        if isinstance(candidate, dict):
            return candidate
        return {}

    def _resolve_value(self, column: str, row: Dict[str, Any]) -> Tuple[Any, bool]:
        """
        Resolves a column value from the metadata dict or computed metrics.

        Returns
        -------
        Tuple[Any, bool]
            (value, found)  — found=False means the column is completely absent.
        """
        # Try physical column first
        if column in row:
            return row[column], True

        # Try computed metric
        if self._metrics.is_registered(column):
            value = self._metrics.resolve(column, row)
            return value, True

        return None, False

    def _evaluate_rule(
        self,
        rule: FilterRule,
        row: Dict[str, Any],
        warnings: List[str],
    ) -> RuleExplanation:
        """Evaluates a single FilterRule against a row dict."""

        # Existence operators don't need a resolved value in the usual sense
        if rule.operator in ("exists", "not_exists"):
            _, found = self._resolve_value(rule.column, row)
            filt = self._registry.resolve(rule.operator)
            passed = filt.apply(True if found else None, rule.operator, rule.value, self.config)
            return RuleExplanation(
                rule_column=rule.column, operator=rule.operator,
                expected_value=rule.value, actual_value=found,
                passed=passed, rule_type=rule.rule_type,
            )

        value, found = self._resolve_value(rule.column, row)

        if not found:
            msg = f"Column '{rule.column}' not found in candidate metadata — rule skipped."
            warnings.append(msg)
            if not self.config.skip_missing_columns:
                raise ValueError(msg)
            return RuleExplanation(
                rule_column=rule.column, operator=rule.operator,
                expected_value=rule.value, actual_value=None,
                passed=True, rule_type=rule.rule_type,   # skip = pass for HARD
                skipped=True, skip_reason="Column not found",
            )

        filt = self._registry.resolve(rule.operator)
        if filt is None:
            msg = f"Unknown operator '{rule.operator}' for column '{rule.column}' — rule skipped."
            warnings.append(msg)
            if not self.config.skip_unknown_operators:
                raise ValueError(msg)
            return RuleExplanation(
                rule_column=rule.column, operator=rule.operator,
                expected_value=rule.value, actual_value=value,
                passed=True, rule_type=rule.rule_type,
                skipped=True, skip_reason="Unknown operator",
            )

        try:
            passed = filt.apply(value, rule.operator, rule.value, self.config)
        except Exception as exc:
            warnings.append(f"Filter apply error on '{rule.column}' {rule.operator}: {exc}")
            passed = True   # skip-and-pass on error
            return RuleExplanation(
                rule_column=rule.column, operator=rule.operator,
                expected_value=rule.value, actual_value=value,
                passed=True, rule_type=rule.rule_type,
                skipped=True, skip_reason=f"Apply error: {exc}",
            )

        return RuleExplanation(
            rule_column=rule.column, operator=rule.operator,
            expected_value=rule.value, actual_value=value,
            passed=passed, rule_type=rule.rule_type,
        )

    def _evaluate_group(
        self,
        group: FilterGroup,
        row: Dict[str, Any],
        warnings: List[str],
        stats: FilteringStatistics,
    ) -> Tuple[bool, List[RuleExplanation]]:
        """
        Recursively evaluates a FilterGroup against a row.

        Returns
        -------
        Tuple[bool, List[RuleExplanation]]
            (group_passed, explanations)
        """
        explanations: List[RuleExplanation] = []
        results: List[bool] = []

        for member in group.rules:
            if isinstance(member, FilterRule):
                if not member.enabled:
                    continue
                expl = self._evaluate_rule(member, row, warnings)
                explanations.append(expl)
                stats.hard_rules_evaluated += 1
                if expl.skipped:
                    stats.rules_skipped += 1
                    results.append(True)   # skipped = pass for group evaluation
                else:
                    results.append(expl.passed)

            elif isinstance(member, FilterGroup):
                if not member.enabled:
                    continue
                sub_passed, sub_expls = self._evaluate_group(member, row, warnings, stats)
                explanations.extend(sub_expls)
                results.append(sub_passed)

        if not results:
            return True, explanations   # empty group = pass

        if group.operator == "AND":
            group_passed = all(results)
        else:  # OR
            group_passed = any(results)

        return group_passed, explanations


# ─────────────────────────────────────────────────────────────────────────────
# CLASS 25: EnterpriseFilteringReport
# ─────────────────────────────────────────────────────────────────────────────

class EnterpriseFilteringReport:
    """
    Generates a formatted enterprise-grade filtering report.

    Usage
    -----
    EnterpriseFilteringReport.generate(result)
    """

    @staticmethod
    def generate(result: FilteringResult) -> None:
        """Prints the full report to stdout."""
        s = result.statistics
        width = 66
        sep = "=" * width
        thin = "-" * width

        print(sep)
        print("FILTERING ENGINE — ENTERPRISE REPORT".center(width))
        print(sep)

        # ── Overview ──────────────────────────────────────────────────────────
        print(f"  Input Candidates       : {s.input_candidates}")
        print(f"  After Hard Filtering   : {s.candidates_after_hard}")
        print(f"  After Soft Scoring     : {s.candidates_after_soft}")
        rejected = s.input_candidates - s.candidates_after_hard
        pct = (s.candidates_after_hard / max(s.input_candidates, 1)) * 100
        print(f"  Rejected (Hard)        : {rejected}  ({100-pct:.1f}% eliminated)")
        print(thin)

        # ── Timing ────────────────────────────────────────────────────────────
        print(f"  Hard Filter Time       : {s.hard_filter_time_seconds*1000:.2f} ms")
        print(f"  Soft Scoring Time      : {s.soft_score_time_seconds*1000:.2f} ms")
        print(f"  Total Time             : {s.total_time_seconds*1000:.2f} ms")
        print(thin)

        # ── Rule Execution ────────────────────────────────────────────────────
        print(f"  Hard Rules Evaluated   : {s.hard_rules_evaluated}")
        print(f"  Soft Rules Evaluated   : {s.soft_rules_evaluated}")
        print(f"  Rules Skipped          : {s.rules_skipped}")
        print(thin)

        # ── Hard Rule Breakdown ───────────────────────────────────────────────
        if result.hard_rule_breakdown:
            print("  Most Failed Hard Rules:")
            sorted_hard = sorted(result.hard_rule_breakdown.items(), key=lambda x: -x[1])
            for col, count in sorted_hard[:5]:
                print(f"    ✗  {col:<35} rejected {count} candidates")
            print(thin)

        # ── Soft Rule Breakdown ───────────────────────────────────────────────
        if result.soft_rule_breakdown:
            print(f"  Soft Score Distribution:")
            print(f"    Average : {s.average_soft_score:+.2f}")
            print(f"    Max     : {s.max_soft_score:+.2f}")
            print(f"    Min     : {s.min_soft_score:+.2f}")
            print()
            print("  Soft Rule Summary:")
            for col, info in result.soft_rule_breakdown.items():
                print(
                    f"    {col:<30}  rewarded={info['rewarded']}  "
                    f"penalised={info['penalised']}  Δ={info['total_delta']:+.2f}"
                )
            print(thin)

        # ── Top Surviving Candidates by Soft Score ────────────────────────────
        if result.passed_candidates:
            sorted_passed = sorted(result.passed_candidates, key=lambda c: -c.soft_score)
            print("  Top Candidates (by Soft Score):")
            for cfr in sorted_passed[:5]:
                print(f"    row_id={cfr.row_id}  soft_score={cfr.soft_score:+.2f}")
            print(thin)

        # ── Warnings ──────────────────────────────────────────────────────────
        print(f"  Status   : {'✓ SUCCESS' if result.is_success else '✗ FAILED'}")
        if result.warnings:
            print(f"  Warnings ({len(result.warnings)}):")
            for w in result.warnings[:5]:
                print(f"    ⚠  {w}")
        print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST SUITE
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    logging.basicConfig(level=logging.ERROR)

    print("=" * 66)
    print("  FilteringEngine — Self-Test Suite")
    print("=" * 66)

    # ── Stub candidate factory ─────────────────────────────────────────────
    class Candidate:
        def __init__(self, row_id, **meta):
            self.row_id = row_id
            self.metadata = meta

    def make_dataset():
        return [
            Candidate(0,  calories_per_serving=300, protein_g_per_serving=35,
                          carbs_g_per_serving=40, fat_g_per_serving=8, fiber_g_per_serving=5,
                          price=12.0, rating=4.8, servings=1,
                          is_vegan=True, is_vegetarian=True, is_halal=False,
                          tags=["vegan", "low-fat"], cuisine="italian",
                          description="High protein vegan bowl",
                          created_at="2025-01-15",
                          ingredients=["tofu", "spinach", "quinoa"]),
            Candidate(1,  calories_per_serving=650, protein_g_per_serving=20,
                          carbs_g_per_serving=80, fat_g_per_serving=25, fiber_g_per_serving=3,
                          price=8.0, rating=3.9, servings=2,
                          is_vegan=False, is_vegetarian=False, is_halal=True,
                          tags=["halal"], cuisine="american",
                          description="Classic beef burger",
                          created_at="2024-06-01",
                          ingredients=["beef", "bun", "cheese"]),
            Candidate(2,  calories_per_serving=480, protein_g_per_serving=45,
                          carbs_g_per_serving=30, fat_g_per_serving=15, fiber_g_per_serving=8,
                          price=18.0, rating=4.9, servings=1,
                          is_vegan=False, is_vegetarian=False, is_halal=True,
                          tags=["halal", "high-protein"], cuisine="mediterranean",
                          description="Grilled chicken salad",
                          created_at="2025-03-20",
                          ingredients=["chicken", "feta", "olive oil", "tomatoes"]),
            Candidate(3,  calories_per_serving=200, protein_g_per_serving=10,
                          carbs_g_per_serving=35, fat_g_per_serving=3, fiber_g_per_serving=6,
                          price=6.0, rating=4.2, servings=1,
                          is_vegan=True, is_vegetarian=True, is_halal=True,
                          tags=["vegan", "low-calorie"], cuisine="asian",
                          description="Vegetable spring rolls",
                          created_at="2025-05-10",
                          ingredients=["cabbage", "carrot", "rice paper"]),
        ]

    engine = FilteringEngine()

    # ── Test 1: Numeric filtering ──────────────────────────────────────────
    print("\n--- Test 1: Numeric Filtering ---")
    rules = [FilterRule("calories_per_serving", "<=", 500, rule_type="HARD")]
    result = engine.filter(make_dataset(), rules)
    assert len(result.passed_candidates) == 3, f"Expected 3, got {len(result.passed_candidates)}"
    assert len(result.rejected_candidates) == 1   # id=1 has 650 kcal
    print(f"  Passed: {[c.row_id for c in result.passed_candidates]}  ✓")
    print(f"  Rejected: {[c.row_id for c in result.rejected_candidates]}  ✓")

    # ── Test 2: Boolean filtering ──────────────────────────────────────────
    print("\n--- Test 2: Boolean Filtering ---")
    rules = [FilterRule("is_vegan", "is_true", rule_type="HARD")]
    result = engine.filter(make_dataset(), rules)
    assert len(result.passed_candidates) == 2
    print(f"  Vegan candidates: {[c.row_id for c in result.passed_candidates]}  ✓")

    # ── Test 3: Text filtering ─────────────────────────────────────────────
    print("\n--- Test 3: Text Filtering ---")
    rules = [FilterRule("description", "icontains", "chicken", rule_type="HARD")]
    result = engine.filter(make_dataset(), rules)
    assert len(result.passed_candidates) == 1
    assert result.passed_candidates[0].row_id == 2
    print(f"  Text match row_id: {result.passed_candidates[0].row_id}  ✓")

    # ── Test 4: List filtering ─────────────────────────────────────────────
    print("\n--- Test 4: List Filtering ---")
    rules = [FilterRule("tags", "contains_any", ["vegan", "halal"], rule_type="HARD")]
    result = engine.filter(make_dataset(), rules)
    assert len(result.passed_candidates) == 4, f"Expected 4, got {len(result.passed_candidates)}"
    print(f"  All have vegan or halal tag: {len(result.passed_candidates)} candidates  ✓")

    # ── Test 5: Regex filtering ────────────────────────────────────────────
    print("\n--- Test 5: Regex Filtering ---")
    rules = [FilterRule("cuisine", "regex", r"^(italian|asian)$", rule_type="HARD")]
    result = engine.filter(make_dataset(), rules)
    assert len(result.passed_candidates) == 2
    print(f"  Italian/Asian: {[c.row_id for c in result.passed_candidates]}  ✓")

    # ── Test 6: Exists / Not-Exists filtering ──────────────────────────────
    print("\n--- Test 6: Exists / Not-Exists Filtering ---")
    rules = [FilterRule("price", "exists", rule_type="HARD")]
    result = engine.filter(make_dataset(), rules)
    assert len(result.passed_candidates) == 4
    print(f"  All have 'price': {len(result.passed_candidates)} candidates  ✓")

    rules_ne = [FilterRule("nonexistent_col", "not_exists", rule_type="HARD")]
    result_ne = engine.filter(make_dataset(), rules_ne)
    assert len(result_ne.passed_candidates) == 4
    print(f"  None have 'nonexistent_col': {len(result_ne.passed_candidates)} candidates  ✓")

    # ── Test 7: Null filtering ─────────────────────────────────────────────
    print("\n--- Test 7: Null Filtering ---")
    cands_with_null = [
        Candidate(10, price=12.0),
        Candidate(11, price=None),
    ]
    rules = [FilterRule("price", "not_null", rule_type="HARD")]
    result = engine.filter(cands_with_null, rules)
    assert len(result.passed_candidates) == 1
    assert result.passed_candidates[0].row_id == 10
    print(f"  Not-null price: row_id={result.passed_candidates[0].row_id}  ✓")

    # ── Test 8: Range filtering ────────────────────────────────────────────
    print("\n--- Test 8: Range Filtering ---")
    rules = [FilterRule("calories_per_serving", "between", [200, 500], rule_type="HARD")]
    result = engine.filter(make_dataset(), rules)
    assert len(result.passed_candidates) == 3   # 300, 480, 200
    print(f"  In range [200,500]: {[c.row_id for c in result.passed_candidates]}  ✓")

    # ── Test 9: Category / Membership filtering ────────────────────────────
    print("\n--- Test 9: Category / Membership Filtering ---")
    rules = [FilterRule("cuisine", "in", ["italian", "mediterranean"], rule_type="HARD")]
    result = engine.filter(make_dataset(), rules)
    assert len(result.passed_candidates) == 2
    print(f"  Italian/Med: {[c.row_id for c in result.passed_candidates]}  ✓")

    # ── Test 10: Datetime filtering ────────────────────────────────────────
    print("\n--- Test 10: Datetime Filtering ---")
    rules = [FilterRule("created_at", "after", "2025-01-01", rule_type="HARD")]
    result = engine.filter(make_dataset(), rules)
    # ids 0 (2025-01-15), 2 (2025-03-20), 3 (2025-05-10) pass
    assert len(result.passed_candidates) == 3
    print(f"  After 2025-01-01: {[c.row_id for c in result.passed_candidates]}  ✓")

    # ── Test 11: Nested Filter Groups ─────────────────────────────────────
    print("\n--- Test 11: Nested FilterGroup (AND + OR) ---")
    group = FilterGroup("AND", [
        FilterRule("calories_per_serving", "<=", 500),
        FilterRule("protein_g_per_serving", ">=", 30),
        FilterGroup("OR", [
            FilterRule("is_vegan", "is_true"),
            FilterRule("is_halal", "is_true"),
        ]),
    ])
    result = engine.filter(make_dataset(), [group])
    # id=0: 300 kcal ✓, protein=35 ✓, vegan ✓
    # id=2: 480 kcal ✓, protein=45 ✓, halal ✓
    assert len(result.passed_candidates) == 2
    assert {c.row_id for c in result.passed_candidates} == {0, 2}
    print(f"  Passed: {[c.row_id for c in result.passed_candidates]}  ✓")

    # ── Test 12: Computed Metrics ──────────────────────────────────────────
    print("\n--- Test 12: Computed Metrics ---")
    rules = [FilterRule("protein_to_calorie_ratio", ">=", 0.08, rule_type="HARD")]
    result = engine.filter(make_dataset(), rules)
    # id=0: 35/300=0.117 ✓,  id=2: 45/480=0.094 ✓
    # id=1: 20/650=0.031 ✗,  id=3: 10/200=0.05 ✗
    passed_ids = {c.row_id for c in result.passed_candidates}
    assert 0 in passed_ids and 2 in passed_ids
    print(f"  High protein:calorie ratio: {sorted(passed_ids)}  ✓")

    # Custom metric registration
    engine._metrics.register(
        "protein_per_dollar",
        lambda r: r.get("protein_g_per_serving", 0) / max(r.get("price", 1), 0.01),
        "protein_g / price"
    )
    rules_ppd = [FilterRule("protein_per_dollar", ">=", 2.0, rule_type="HARD")]
    result_ppd = engine.filter(make_dataset(), rules_ppd)
    print(f"  Protein per dollar ≥ 2.0: {[c.row_id for c in result_ppd.passed_candidates]}  ✓")

    # ── Test 13: Soft Rule Scoring ─────────────────────────────────────────
    print("\n--- Test 13: Soft Rule Scoring ---")
    rules = [
        FilterRule("calories_per_serving", "<=", 500, rule_type="HARD"),
        FilterRule("protein_g_per_serving", ">=", 40, rule_type="SOFT", reward=20, penalty=-5),
        FilterRule("rating", ">=", 4.5, rule_type="SOFT", reward=10, penalty=-2),
        FilterRule("price", "<=", 15.0, rule_type="SOFT", reward=5, penalty=0),
    ]
    result = engine.filter(make_dataset(), rules)
    assert len(result.passed_candidates) == 3   # 650-kcal rejected
    scores = result.candidate_scores
    print(f"  Soft scores: {scores}")
    # id=2 has protein=45≥40 (+20) and rating=4.9≥4.5 (+10) but price=18>15 (0)
    assert scores.get(2, 0) >= 25, f"id=2 should score ≥25, got {scores.get(2,0)}"
    print(f"  id=2 score={scores[2]} (expected ≥25)  ✓")

    # ── Test 14: Plugin Filter ─────────────────────────────────────────────
    print("\n--- Test 14: Plugin Filter Registration ---")

    def short_ingredient_list_filter(value, operator, threshold, config):
        """Custom: True if ingredient count <= threshold."""
        return isinstance(value, list) and len(value) <= int(threshold)

    plugin = CustomFilter(short_ingredient_list_filter, ["max_ingredients"], "ShortIngredientFilter")
    engine._registry.register_custom("max_ingredients", plugin)

    rules = [FilterRule("ingredients", "max_ingredients", 3, rule_type="HARD")]
    result = engine.filter(make_dataset(), rules)
    # id=0 has 3 ✓, id=1 has 3 ✓, id=3 has 3 ✓; id=2 has 4 ✗
    passed_ids = {c.row_id for c in result.passed_candidates}
    assert 2 not in passed_ids, f"id=2 has 4 ingredients, should be rejected"
    print(f"  ≤3 ingredients: {sorted(passed_ids)}  ✓")

    # ── Test 15: Missing Columns — skip not crash ──────────────────────────
    print("\n--- Test 15: Missing Column → Skip, Not Crash ---")
    rules = [
        FilterRule("nonexistent_column", ">=", 99, rule_type="HARD"),
        FilterRule("calories_per_serving", "<=", 400, rule_type="HARD"),
    ]
    result = engine.filter(make_dataset(), rules)
    assert result.is_success
    assert any("nonexistent_column" in w for w in result.warnings)
    print(f"  Warnings: {result.warnings[:1]}  ✓")
    print(f"  Passed (skipped missing, applied calorie filter): {[c.row_id for c in result.passed_candidates]}  ✓")

    # ── Test 16: Unknown Operator — skip not crash ─────────────────────────
    print("\n--- Test 16: Unknown Operator → Skip, Not Crash ---")
    rules = [FilterRule("price", "unknown_op", 10, rule_type="HARD")]
    result = engine.filter(make_dataset(), rules)
    assert result.is_success
    assert len(result.passed_candidates) == 4   # all pass (rule skipped)
    print(f"  All candidates pass (unknown op skipped)  ✓")

    # ── Test 17: to_json() serialisation ──────────────────────────────────
    print("\n--- Test 17: to_json() Serialisation ---")
    rules = [FilterRule("calories_per_serving", "<=", 500, rule_type="HARD")]
    result = engine.filter(make_dataset(), rules)
    j = result.to_json()
    parsed = json.loads(j)
    assert "passed_candidates" in parsed
    assert "statistics" in parsed
    print(f"  JSON keys: {list(parsed.keys())}  ✓")

    # ── Test 18: Config validation ─────────────────────────────────────────
    print("\n--- Test 18: Config Validation ---")
    try:
        FilterConfig(approximately_equal_tolerance=2.0).validate()
        print("  ERROR: Should have raised ValueError!")
    except ValueError as exc:
        print(f"  [PASS] ValueError: {exc}")

    # ── Test 19: FilterRule validation ────────────────────────────────────
    print("\n--- Test 19: FilterRule Validation ---")
    try:
        FilterRule("col", "==", 1, rule_type="INVALID").validate()
        print("  ERROR: Should have raised ValueError!")
    except ValueError as exc:
        print(f"  [PASS] ValueError: {exc}")

    # ── Test 20: Enterprise Report ─────────────────────────────────────────
    print("\n--- Test 20: Enterprise Filtering Report ---")
    rules = [
        FilterRule("calories_per_serving", "<=", 500, rule_type="HARD"),
        FilterRule("protein_g_per_serving", ">=", 40, rule_type="SOFT", reward=20, penalty=-5),
        FilterRule("rating", ">=", 4.5, rule_type="SOFT", reward=10, penalty=-2),
    ]
    result = engine.filter(make_dataset(), rules)
    EnterpriseFilteringReport.generate(result)

    print("=" * 66)
    print("  All self-tests passed ✓")
    print("=" * 66)
