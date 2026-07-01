"""
ranking.py — DineAI Ranking & Multi-Objective Optimization Framework
===================================================================
Enterprise scoring, normalization, optimization, and ranking engine.
Follows SOLID, Clean Architecture, Strategy, and Dependency Injection patterns.
"""

from __future__ import annotations

import math
import statistics
import logging
import json
import yaml
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

logger = logging.getLogger(__name__)


# ============================================================================
# MOCKS (For standalone execution, fallback compatibility, and tests)
# ============================================================================

class _MockCandidate:
    """Mock Candidate representing the structure returned by vector DB or search."""
    def __init__(self, data: Dict[str, Any], row_id: Any = None):
        self.row_id = row_id or data.get("row_id") or data.get("name")
        self.metadata = data
        self.candidate = data  # Allow dict access compatibility


class FilteringResult:
    """Mock/Fallback FilteringResult mimicking dine_ai.retrieval.filtering.FilteringResult"""
    def __init__(self, passed_candidates: List[Any], rejected_candidates: Optional[List[Any]] = None):
        self.passed_candidates = passed_candidates
        self.rejected_candidates = rejected_candidates or []
        self.statistics = None
        self.warnings = []
        self.is_success = True


# ============================================================================
# 1. ENUMS & NORMALIZATION SYSTEM
# ============================================================================

class NormalizationMethod(str, Enum):
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    IDENTITY = "identity"
    CUSTOM = "custom"


class ScoreNormalizer:
    """
    Handles scaling disparate metrics into comparable ranges (usually 0.0 to 1.0).
    Supports Min-Max, Z-score scaling, Identity scaling, and custom lambdas/callables.
    """
    
    @staticmethod
    def min_max(scores: List[float]) -> List[float]:
        """Scales scores to [0.0, 1.0] range. Handles zero variance safely."""
        if not scores:
            return []
        s_min, s_max = min(scores), max(scores)
        if s_min == s_max:
            # If all values are identical, normalize to 1.0 (or 0.5 if all 0)
            return [1.0 if s_min > 0 else 0.5 for _ in scores]
        return [(s - s_min) / (s_max - s_min) for s in scores]

    @staticmethod
    def z_score(scores: List[float]) -> List[float]:
        """
        Applies standard score normalization. Scales/clips values to approximately [0.0, 1.0].
        Standard normal usually falls between -3 and 3. We shift and scale: (z + 3) / 6.
        """
        if not scores:
            return []
        if len(scores) < 2:
            return [1.0 for _ in scores]
        
        mean = statistics.mean(scores)
        stdev = statistics.stdev(scores)
        if stdev == 0:
            return [1.0 for _ in scores]
            
        normalized = []
        for s in scores:
            z = (s - mean) / stdev
            # Shift and scale to approx 0.0 - 1.0 range, and clip to boundaries
            val = (z + 3.0) / 6.0
            normalized.append(max(0.0, min(1.0, val)))
        return normalized

    @staticmethod
    def identity(scores: List[float]) -> List[float]:
        """Returns the scores unchanged."""
        return scores

    @classmethod
    def apply(
        cls, 
        method: Union[NormalizationMethod, str], 
        scores: List[float], 
        invert: bool = False,
        custom_func: Optional[Callable[[List[float]], List[float]]] = None
    ) -> List[float]:
        """
        Applies chosen normalization method, handles inversion (if invert=True), 
        and supports custom callables.
        """
        if not scores:
            return []
            
        if isinstance(method, Enum):
            method_str = method.value.lower()
        else:
            method_str = str(method).lower()
        
        if method_str == "min_max":
            norm = cls.min_max(scores)
        elif method_str == "z_score":
            norm = cls.z_score(scores)
        elif method_str == "custom" and custom_func is not None:
            norm = custom_func(scores)
        else:
            norm = cls.identity(scores)
            
        if invert:
            norm = [1.0 - s for s in norm]
            
        return norm


# ============================================================================
# 2. CONFIGURATION & DOMAIN MODELS
# ============================================================================

@dataclass
class RankingFactor:
    """Represents a single scoring factor config (e.g., Rating, Price)."""
    name: str
    scorer_key: str
    weight: float = 1.0
    enabled: bool = True
    priority: int = 0
    normalize: NormalizationMethod = NormalizationMethod.MIN_MAX
    invert: bool = False
    config: Dict[str, Any] = field(default_factory=dict)
    required_column: Optional[str] = None

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> RankingFactor:
        """Parses a RankingFactor from a dictionary representation."""
        return cls(
            name=name,
            scorer_key=data.get("scorer_key", name),
            weight=float(data.get("weight", 1.0)),
            enabled=bool(data.get("enabled", True)),
            priority=int(data.get("priority", 0)),
            normalize=NormalizationMethod(data.get("normalize", NormalizationMethod.MIN_MAX)),
            invert=bool(data.get("invert", False)),
            config=data.get("config", {}),
            required_column=data.get("required_column", None)
        )


@dataclass
class RankingConfig:
    """
    Configuration suite for the ranking process. 
    Supports standard scoring factors, multi-objective settings, diversity filters, and logging.
    """
    factors: List[RankingFactor] = field(default_factory=list)
    objectives: List[Objective] = field(default_factory=list)
    objective_optimizer: Optional[str] = "weighted"  # e.g., 'weighted', 'topsis', 'pareto'
    diversity_penalty_enabled: bool = False
    diversity_weight: float = 0.1
    log_level: str = "WARNING"
    max_results: Optional[int] = None

    def validate(self) -> None:
        """Validates configuration parameters."""
        if self.diversity_weight < 0:
            raise ValueError(f"diversity_weight must be non-negative, got {self.diversity_weight}")
        for factor in self.factors:
            if factor.weight < 0:
                raise ValueError(f"Factor {factor.name} weight must be non-negative, got {factor.weight}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RankingConfig:
        """Parses configuration from a dictionary payload."""
        factors_data = data.get("factors", {})
        objectives_data = data.get("objectives", {})
        
        # Load factors
        factors = []
        if isinstance(factors_data, dict):
            for name, f_conf in factors_data.items():
                factors.append(RankingFactor.from_dict(name, f_conf))
        elif isinstance(factors_data, list):
            for f_conf in factors_data:
                factors.append(RankingFactor(
                    name=f_conf["name"],
                    scorer_key=f_conf.get("scorer_key", f_conf["name"]),
                    weight=f_conf.get("weight", 1.0),
                    enabled=f_conf.get("enabled", True),
                    priority=f_conf.get("priority", 0),
                    normalize=NormalizationMethod(f_conf.get("normalize", "min_max")),
                    invert=f_conf.get("invert", False),
                    config=f_conf.get("config", {}),
                    required_column=f_conf.get("required_column", None)
                ))

        # Load objectives (we will define the Objective class parsing logic in the Objective section)
        objectives = []
        if isinstance(objectives_data, dict):
            for name, obj_conf in objectives_data.items():
                objectives.append(Objective.from_dict(name, obj_conf))
        elif isinstance(objectives_data, list):
            for obj_conf in objectives_data:
                objectives.append(Objective.from_dict(obj_conf["name"], obj_conf))

        return cls(
            factors=factors,
            objectives=objectives,
            objective_optimizer=data.get("objective_optimizer", "weighted"),
            diversity_penalty_enabled=bool(data.get("diversity_penalty_enabled", False)),
            diversity_weight=float(data.get("diversity_weight", 0.1)),
            log_level=data.get("log_level", "WARNING"),
            max_results=data.get("max_results")
        )

    @classmethod
    def from_file(cls, filepath: str) -> RankingConfig:
        """Loads configuration from a JSON or YAML file path."""
        with open(filepath, "r", encoding="utf-8") as f:
            if filepath.endswith((".yaml", ".yml")):
                content = yaml.safe_load(f)
            else:
                content = json.load(f)
        return cls.from_dict(content)


@dataclass
class RankingExplanation:
    """
    Detailed explanation of why a candidate was ranked with a certain score.
    Captures raw scores, normalized scores, weighted contributions, penalties, and objective details.
    """
    raw_scores: Dict[str, float] = field(default_factory=dict)
    normalized_scores: Dict[str, float] = field(default_factory=dict)
    weighted_contributions: Dict[str, float] = field(default_factory=dict)
    penalties_applied: Dict[str, float] = field(default_factory=dict)
    bonuses_applied: Dict[str, float] = field(default_factory=dict)
    
    # Multi-Objective Optimization details
    objective_contributions: Dict[str, float] = field(default_factory=dict)
    objective_weights: Dict[str, float] = field(default_factory=dict)
    objective_conflicts: List[str] = field(default_factory=list)
    final_objective_score: float = 0.0
    optimization_strategy_used: str = ""

    def generate_text(self, final_score: float) -> str:
        """Generates a human-readable text explanation of the ranking scores."""
        lines = [f"Final Score: {final_score:.4f}", "-" * 35]
        
        if self.weighted_contributions:
            lines.append("Factor Contributions:")
            for factor, contrib in sorted(self.weighted_contributions.items(), key=lambda x: -x[1]):
                norm = self.normalized_scores.get(factor, 0.0)
                raw = self.raw_scores.get(factor, 0.0)
                lines.append(f"  * {factor:<20} : {contrib:+.4f} [Raw: {raw:.2f}, Norm: {norm:.2f}]")
                
        if self.objective_contributions:
            lines.append("Objective Contributions:")
            for obj, contrib in sorted(self.objective_contributions.items(), key=lambda x: -x[1]):
                w = self.objective_weights.get(obj, 0.0)
                raw = self.raw_scores.get(obj, 0.0)
                lines.append(f"  * {obj:<20} : {contrib:+.4f} [Raw: {raw:.2f}, Weight: {w:.2f}]")
            
            if self.objective_conflicts:
                lines.append("Tradeoff Conflicts Identified:")
                for conflict in self.objective_conflicts:
                    lines.append(f"  ! {conflict}")
                    
            lines.append(f"Final Objective Score   : {self.final_objective_score:.4f}")
            lines.append(f"Optimization Strategy   : {self.optimization_strategy_used}")
            
        if self.penalties_applied:
            lines.append("-" * 35)
            for pen, val in self.penalties_applied.items():
                lines.append(f"  v Penalty ({pen}): -{val:.4f}")
                
        if self.bonuses_applied:
            lines.append("-" * 35)
            for bon, val in self.bonuses_applied.items():
                lines.append(f"  ^ Bonus ({bon}): +{val:.4f}")
                
        return "\n".join(lines)


@dataclass
class RankedCandidate:
    """Composition wrapper containing candidate metadata, final score, and breakdown explanation."""
    candidate: Any
    final_score: float
    explanation: RankingExplanation
    original_index: int

    @property
    def ranking_breakdown(self) -> Dict[str, float]:
        """Provides a detailed dictionary of scoring factor contributions and final score."""
        breakdown = {}
        for k, v in self.explanation.weighted_contributions.items():
            breakdown[k.lower()] = round(v, 4)
        for k, v in self.explanation.objective_contributions.items():
            breakdown[k.lower()] = round(v, 4)
        penalty_sum = sum(self.explanation.penalties_applied.values())
        if penalty_sum > 0 or self.explanation.penalties_applied:
            breakdown["penalty"] = round(-penalty_sum, 4)
        for standard in ["semantic", "price", "rating"]:
            if standard not in breakdown:
                breakdown[standard] = 0.0
        breakdown["final_score"] = round(self.final_score, 4)
        return breakdown


@dataclass
class RankingStatistics:
    """Detached telemetry capturing statistics of the ranking execution pass."""
    total_candidates: int = 0
    top_score: float = 0.0
    avg_score: float = 0.0
    execution_time_ms: float = 0.0
    factor_averages: Dict[str, float] = field(default_factory=dict)
    objective_averages: Dict[str, float] = field(default_factory=dict)


@dataclass
class RankingResult:
    """The complete result of a ranking engine pass."""
    ranked_candidates: List[RankedCandidate] = field(default_factory=list)
    statistics: RankingStatistics = field(default_factory=RankingStatistics)
    enterprise_report: str = ""


# ============================================================================
# Helper Utilities
# ============================================================================

def _get_metadata(candidate: Any) -> Dict[str, Any]:
    """Helper to extract metadata dictionary from a candidate or CandidateFilterResult."""
    obj = candidate
    while hasattr(obj, "candidate"):
        if isinstance(obj, dict):
            break
        obj = getattr(obj, "candidate")
        
    if isinstance(obj, dict):
        meta = obj
    elif hasattr(obj, "metadata") and isinstance(obj.metadata, dict):
        meta = obj.metadata
    elif hasattr(obj, "__dict__"):
        meta = obj.__dict__
    else:
        meta = {}
        
    if meta and "recipe_name" in meta and "name" not in meta:
        meta = dict(meta)
        meta["name"] = meta["recipe_name"]
    return meta


# ============================================================================
# 3. SCORING STRATEGIES (PLUGINS)
# ============================================================================

class BaseScorer(ABC):
    """Abstract base class for all individual metric scorers."""
    @abstractmethod
    def score_batch(self, candidates: List[Any], factor_config: Dict[str, Any]) -> List[float]:
        pass


class SemanticScore(BaseScorer):
    """Extracts semantic similarity score from candidate metadata."""
    def score_batch(self, candidates: List[Any], factor_config: Dict[str, Any]) -> List[float]:
        return [float(_get_metadata(c).get("semantic_similarity", 0.0)) for c in candidates]


class SoftRuleScore(BaseScorer):
    """Extracts the accumulated soft score from CandidateFilterResult."""
    def score_batch(self, candidates: List[Any], factor_config: Dict[str, Any]) -> List[float]:
        return [float(getattr(c, "soft_score", 0.0)) for c in candidates]


class NutritionScore(BaseScorer):
    """Computes nutrition scores based on target calories (closer is better)."""
    def score_batch(self, candidates: List[Any], factor_config: Dict[str, Any]) -> List[float]:
        target_calories = factor_config.get("target_calories", 500)
        scores = []
        for c in candidates:
            meta = _get_metadata(c)
            cal = float(meta.get("calories", meta.get("calories_per_serving", target_calories)))
            # Proximity formula: 1.0 / (1.0 + absolute distance)
            scores.append(1.0 / (1.0 + abs(cal - target_calories)))
        return scores


class RatingScore(BaseScorer):
    """Extracts rating score (e.g. 1.0 to 5.0)."""
    def score_batch(self, candidates: List[Any], factor_config: Dict[str, Any]) -> List[float]:
        return [float(_get_metadata(c).get("rating", 3.0)) for c in candidates]


class PopularityScore(BaseScorer):
    """Extracts order count or general popularity metrics."""
    def score_batch(self, candidates: List[Any], factor_config: Dict[str, Any]) -> List[float]:
        scores = []
        for c in candidates:
            meta = _get_metadata(c)
            val = meta.get("order_count", meta.get("popularity", 0.0))
            scores.append(float(val))
        return scores


class PriceScore(BaseScorer):
    """Extracts item price."""
    def score_batch(self, candidates: List[Any], factor_config: Dict[str, Any]) -> List[float]:
        return [float(_get_metadata(c).get("price", 0.0)) for c in candidates]


class FreshnessScore(BaseScorer):
    """Extracts date-based freshness (lower days_since_added is better)."""
    def score_batch(self, candidates: List[Any], factor_config: Dict[str, Any]) -> List[float]:
        scores = []
        for c in candidates:
            meta = _get_metadata(c)
            scores.append(float(meta.get("days_since_added", meta.get("age_days", 100.0))))
        return scores


class MetadataScore(BaseScorer):
    """Evaluates the completeness of metadata (e.g. key availability)."""
    def score_batch(self, candidates: List[Any], factor_config: Dict[str, Any]) -> List[float]:
        required_keys = factor_config.get("keys", ["image_url", "description", "ingredients"])
        scores = []
        for c in candidates:
            meta = _get_metadata(c)
            hits = sum(1 for k in required_keys if meta.get(k))
            scores.append(hits / len(required_keys) if required_keys else 1.0)
        return scores


class PenaltyScore(BaseScorer):
    """Extracts generic penalty value (e.g. delivery fee)."""
    def score_batch(self, candidates: List[Any], factor_config: Dict[str, Any]) -> List[float]:
        penalty_key = factor_config.get("key", "delivery_fee")
        return [float(_get_metadata(c).get(penalty_key, 0.0)) for c in candidates]


class DiversityScore(BaseScorer):
    """Static category-based uniqueness scoring."""
    def score_batch(self, candidates: List[Any], factor_config: Dict[str, Any]) -> List[float]:
        categories = [_get_metadata(c).get("category", "unknown") for c in candidates]
        counts = {cat: categories.count(cat) for cat in set(categories)}
        return [1.0 / counts[cat] if counts[cat] > 0 else 1.0 for cat in categories]


class CustomScore(BaseScorer):
    """Wraps user-provided scoring function."""
    def __init__(self, func: Callable[[Any], float]):
        self.func = func
        
    def score_batch(self, candidates: List[Any], factor_config: Dict[str, Any]) -> List[float]:
        return [float(self.func(c)) for c in candidates]


# ============================================================================
# 4. REGISTRY (DEPENDENCY INJECTION)
# ============================================================================

class RankingRegistry:
    """Service locator for scoring plugins."""
    _scorers: Dict[str, BaseScorer] = {}

    @classmethod
    def register(cls, name: str, scorer: BaseScorer):
        cls._scorers[name] = scorer

    @classmethod
    def get(cls, name: str) -> BaseScorer:
        if name not in cls._scorers:
            raise ValueError(f"Scorer '{name}' not found in RankingRegistry.")
        return cls._scorers[name]


# Pre-register default scorers
RankingRegistry.register("SemanticScore", SemanticScore())
RankingRegistry.register("SoftRuleScore", SoftRuleScore())
RankingRegistry.register("NutritionScore", NutritionScore())
RankingRegistry.register("RatingScore", RatingScore())
RankingRegistry.register("PopularityScore", PopularityScore())
RankingRegistry.register("PriceScore", PriceScore())
RankingRegistry.register("FreshnessScore", FreshnessScore())
RankingRegistry.register("MetadataScore", MetadataScore())
RankingRegistry.register("PenaltyScore", PenaltyScore())
RankingRegistry.register("DiversityScore", DiversityScore())


# ============================================================================
# 5. RANKING STRATEGIES
# ============================================================================

class BaseRankingStrategy(ABC):
    """Abstract base class for all ranking strategies (Strategy Pattern)."""
    @abstractmethod
    def rank(self, candidates: List[Any], config: RankingConfig) -> List[RankedCandidate]:
        pass


class WeightedRankingStrategy(BaseRankingStrategy):
    """Computes a weighted sum of multiple normalized ranking factors."""
    
    def rank(self, candidates: List[Any], config: RankingConfig) -> List[RankedCandidate]:
        n = len(candidates)
        if n == 0:
            return []

        # Initialize tracking structures
        explanations = [RankingExplanation() for _ in range(n)]
        final_scores = [0.0] * n
        active_factors = sorted([f for f in config.factors if f.enabled], key=lambda f: -f.priority)
        # Filter active factors based on required column presence
        removed_factors = []
        if n > 0:
            from dataclasses import replace
            sample_meta = _get_metadata(candidates[0])
            
            # Auto-resolve missing required_column for standard scorers
            resolved_factors = []
            for f in active_factors:
                req_col = f.required_column
                if req_col is None:
                    if f.scorer_key == "RatingScore":
                        req_col = "rating"
                    elif f.scorer_key == "PriceScore":
                        req_col = "price"
                    elif f.scorer_key == "NutritionScore":
                        req_col = "calories_per_serving"
                    elif f.scorer_key == "PopularityScore":
                        req_col = "popularity"
                
                if req_col is not None:
                    resolved_factors.append(replace(f, required_column=req_col))
                else:
                    resolved_factors.append(f)
            
            # Now filter
            filtered_factors = []
            for f in resolved_factors:
                if f.required_column is not None and f.required_column not in sample_meta:
                    removed_factors.append(f.name)
                else:
                    filtered_factors.append(f)
            
            # Proportional redistribution of weights across remaining factors
            total_weight = sum(f.weight for f in filtered_factors)
            if total_weight > 0:
                active_factors = [replace(f, weight=f.weight / total_weight) for f in filtered_factors]
            else:
                active_factors = filtered_factors

            if removed_factors:
                logger.info("Removed unavailable ranking factors: %s", removed_factors)

        # 1. Compute and Aggregate scores per factor
        for factor in active_factors:
            try:
                scorer = RankingRegistry.get(factor.scorer_key)
            except ValueError as e:
                logger.warning(str(e))
                continue
                
            raw_scores = scorer.score_batch(candidates, factor.config)
            norm_scores = ScoreNormalizer.apply(factor.normalize, raw_scores, factor.invert)
            
            for i in range(n):
                explanations[i].raw_scores[factor.name] = raw_scores[i]
                explanations[i].normalized_scores[factor.name] = norm_scores[i]
                
                contrib = norm_scores[i] * factor.weight
                explanations[i].weighted_contributions[factor.name] = contrib
                final_scores[i] += contrib

        # Build initial candidate list
        ranked = [
            RankedCandidate(
                candidate=candidates[i],
                final_score=final_scores[i],
                explanation=explanations[i],
                original_index=i
            )
            for i in range(n)
        ]
        
        # 2. Sort by score
        ranked.sort(key=lambda x: x.final_score, reverse=True)

        # 3. Apply Sequential Diversity Penalty (MMR-lite) if enabled
        if config.diversity_penalty_enabled:
            ranked = self._apply_diversity(ranked, config.diversity_weight)

        return ranked

    def _apply_diversity(self, ranked: List[RankedCandidate], penalty_weight: float) -> List[RankedCandidate]:
        """Simple diversity penalization based on category repetition (MMR-lite)."""
        if not ranked:
            return ranked
        
        adjusted = []
        seen_categories: Dict[str, int] = {}
        remaining = list(ranked)
        
        while remaining:
            best_idx = 0
            best_score = float('-inf')
            
            # Find the best item in remaining, adjusted for repetition penalties
            for idx, rc in enumerate(remaining):
                cat = _get_metadata(rc.candidate).get("category", "unknown")
                penalty = seen_categories.get(cat, 0) * penalty_weight
                score_adjusted = rc.final_score - penalty
                
                if score_adjusted > best_score:
                    best_score = score_adjusted
                    best_idx = idx
                    
            winner = remaining.pop(best_idx)
            cat = _get_metadata(winner.candidate).get("category", "unknown")
            
            # Record penalty if it was penalized
            times_seen = seen_categories.get(cat, 0)
            if times_seen > 0:
                pen_amount = times_seen * penalty_weight
                winner.final_score -= pen_amount
                winner.explanation.penalties_applied["Diversity_Repetition"] = pen_amount
                
            seen_categories[cat] = times_seen + 1
            adjusted.append(winner)
            
        # Re-sort adjusted list since scores were modified
        adjusted.sort(key=lambda x: x.final_score, reverse=True)
        return adjusted


class LinearRankingStrategy(BaseRankingStrategy):
    """Pure additive ranking without weights (all active weights are forced to 1.0)."""
    def rank(self, candidates: List[Any], config: RankingConfig) -> List[RankedCandidate]:
        config_copy = RankingConfig(
            factors=[RankingFactor(
                name=f.name, scorer_key=f.scorer_key, weight=1.0, enabled=f.enabled,
                priority=f.priority, normalize=f.normalize, invert=f.invert, config=f.config
            ) for f in config.factors],
            diversity_penalty_enabled=config.diversity_penalty_enabled,
            diversity_weight=config.diversity_weight
        )
        return WeightedRankingStrategy().rank(candidates, config_copy)


class MLRankingStrategy(BaseRankingStrategy):
    """Architecture stub for Learning-to-Rank (XGBoost/LightGBM) integration."""
    def rank(self, candidates: List[Any], config: RankingConfig) -> List[RankedCandidate]:
        logger.warning("MLRankingStrategy is a stub. Executing fallback WeightedRankingStrategy.")
        return WeightedRankingStrategy().rank(candidates, config)


class HybridRankingStrategy(BaseRankingStrategy):
    """Combines hard rules (Weighted) with a secondary pass."""
    def rank(self, candidates: List[Any], config: RankingConfig) -> List[RankedCandidate]:
        # Step 1: Base Weighted Rank
        base_ranked = WeightedRankingStrategy().rank(candidates, config)
        # Step 2: Fallback stub
        return base_ranked


class CustomRankingStrategy(BaseRankingStrategy):
    """Executes a fully custom ranking closure."""
    def __init__(self, rank_func: Callable[[List[Any], RankingConfig], List[RankedCandidate]]):
        self.rank_func = rank_func

    def rank(self, candidates: List[Any], config: RankingConfig) -> List[RankedCandidate]:
        return self.rank_func(candidates, config)


# ============================================================================
# 6. MULTI-OBJECTIVE RANKING SYSTEM
# ============================================================================

class BaseObjective(ABC):
    """Abstract base class representing an optimization goal."""
    @abstractmethod
    def evaluate(self, candidate: Any) -> float:
        """Computes the raw objective metric value for the candidate."""
        pass


@dataclass
class Objective(BaseObjective):
    """
    Standard objective definition mapping to a property or a dynamic metric.
    Each objective supports priorities, weights, maximizing/minimizing, and normalization.
    """
    name: str
    weight: float = 1.0
    priority: int = 0
    maximize: bool = True
    minimize: bool = False
    enabled: bool = True
    normalizer: NormalizationMethod = NormalizationMethod.MIN_MAX
    description: str = ""

    def evaluate(self, candidate: Any) -> float:
        """Default evaluation pulls the metadata field with matching name."""
        meta = _get_metadata(candidate)
        val = meta.get(self.name, 0.0)
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> Objective:
        """Parses an Objective definition from a dictionary, determining the correct subclass."""
        obj_type = data.get("type", "numeric").lower()
        kwargs = {
            "name": name,
            "weight": float(data.get("weight", 1.0)),
            "priority": int(data.get("priority", 0)),
            "maximize": bool(data.get("maximize", True)),
            "minimize": bool(data.get("minimize", False)),
            "enabled": bool(data.get("enabled", True)),
            "normalizer": NormalizationMethod(data.get("normalizer", NormalizationMethod.MIN_MAX)),
            "description": data.get("description", "")
        }
        
        # Sync maximize/minimize flags if only one is provided
        if "minimize" in data and "maximize" not in data:
            kwargs["maximize"] = not kwargs["minimize"]
        elif "maximize" in data and "minimize" not in data:
            kwargs["minimize"] = not kwargs["maximize"]

        # Resolve correct Objective subclass
        if obj_type == "categorical":
            return CategoricalObjective(target_value=data.get("target_value"), **kwargs)
        elif obj_type == "boolean":
            return BooleanObjective(target_value=data.get("target_value", True), **kwargs)
        elif obj_type == "computed":
            return ComputedObjective(expression_key=data.get("expression_key", name), **kwargs)
        else:
            return NumericObjective(
                target_value=data.get("target_value"), 
                tolerance=data.get("tolerance"), 
                **kwargs
            )


@dataclass
class NumericObjective(Objective):
    """Evaluates continuous numeric properties. Supports optional targeting (proximity scoring)."""
    target_value: Optional[float] = None
    tolerance: Optional[float] = None

    def evaluate(self, candidate: Any) -> float:
        val = super().evaluate(candidate)
        if self.target_value is not None:
            # Proximity metric: higher score when closer to target_value
            return 1.0 / (1.0 + abs(val - self.target_value))
        return val


@dataclass
class CategoricalObjective(Objective):
    """Evaluates string/discrete categories. Scores 1.0 for a match, 0.0 otherwise."""
    target_value: Any = None

    def evaluate(self, candidate: Any) -> float:
        meta = _get_metadata(candidate)
        val = meta.get(self.name)
        if val == self.target_value:
            return 1.0
        if isinstance(val, (list, tuple, set)) and self.target_value in val:
            return 1.0
        return 0.0


@dataclass
class BooleanObjective(Objective):
    """Evaluates boolean tags. Scores 1.0 if matching target truthiness, 0.0 otherwise."""
    target_value: bool = True

    def evaluate(self, candidate: Any) -> float:
        meta = _get_metadata(candidate)
        val = meta.get(self.name, False)
        truthy = val is True or str(val).lower() in ("true", "1", "yes")
        return 1.0 if truthy == self.target_value else 0.0


@dataclass
class ComputedObjective(Objective):
    """Dynamically computes formulas or targets automatically discoverable via the registry."""
    expression_key: str = ""

    def evaluate(self, candidate: Any) -> float:
        meta = _get_metadata(candidate)
        key = self.expression_key or self.name
        
        # Check global custom registered functions
        if ObjectiveRegistry.has_computed(key):
            try:
                return float(ObjectiveRegistry.get_computed(key)(meta))
            except Exception as e:
                logger.warning(f"Error computing registered metric '{key}': {e}")
                return 0.0
                
        # Built-in fallback evaluation formulas
        def _safe_div(a, b):
            return float(a) / float(b) if float(b) != 0 else 0.0

        if key == "protein_per_dollar":
            p = meta.get("protein", meta.get("protein_g_per_serving", 0.0))
            d = meta.get("price", 1.0)
            return _safe_div(p, d)
            
        elif key == "calories_per_100g":
            c = meta.get("calories", meta.get("calories_per_serving", 0.0))
            w = meta.get("weight_g", meta.get("weight_g_per_serving", 100.0))
            return _safe_div(c, w) * 100.0
            
        elif key == "protein_density":
            p = meta.get("protein", meta.get("protein_g_per_serving", 0.0))
            w = meta.get("weight_g", meta.get("weight_g_per_serving", 1.0))
            return _safe_div(p, w)
            
        elif key == "customer_value_score":
            r = meta.get("rating", 3.0)
            p = meta.get("price", 1.0)
            return _safe_div(r, p)
            
        elif key == "restaurant_margin_score":
            p = meta.get("price", 0.0)
            c = meta.get("cost", 0.0)
            return p - c
            
        elif key == "nutrition_score":
            p = meta.get("protein", meta.get("protein_g_per_serving", 0.0))
            f = meta.get("fiber", meta.get("fiber_g_per_serving", 0.0))
            c = meta.get("calories", meta.get("calories_per_serving", 500.0))
            return _safe_div(p + f, c) * 100.0
            
        return float(meta.get(key, 0.0))


class ObjectiveRegistry:
    """Registry locator for custom objectives and computed metrics."""
    _objectives: Dict[str, Type[Objective]] = {}
    _computed_funcs: Dict[str, Callable[[Dict[str, Any]], float]] = {}

    @classmethod
    def register(cls, name: str, objective_cls: Type[Objective]):
        cls._objectives[name] = objective_cls

    @classmethod
    def get(cls, name: str) -> Type[Objective]:
        return cls._objectives.get(name, Objective)

    @classmethod
    def register_computed(cls, key: str, func: Callable[[Dict[str, Any]], float]):
        cls._computed_funcs[key] = func

    @classmethod
    def has_computed(cls, key: str) -> bool:
        return key in cls._computed_funcs

    @classmethod
    def get_computed(cls, key: str) -> Callable[[Dict[str, Any]], float]:
        return cls._computed_funcs[key]


# Register standard objective sub-types
ObjectiveRegistry.register("numeric", NumericObjective)
ObjectiveRegistry.register("categorical", CategoricalObjective)
ObjectiveRegistry.register("boolean", BooleanObjective)
ObjectiveRegistry.register("computed", ComputedObjective)


@dataclass
class ObjectiveResult:
    """Encapsulates the result of a multi-objective optimization pass for a candidate."""
    final_score: float
    contributions: Dict[str, float]
    conflicts: List[str]
    strategy_name: str


# ============================================================================
# 7. OBJECTIVE OPTIMIZERS
# ============================================================================

def _calculate_conflicts(raw_scores_by_obj: Dict[str, List[float]], objectives: List[Objective]) -> List[str]:
    """
    Computes pairwise Pearson correlation coefficients to identify negative trade-offs
    and conflicts between objectives across the candidate dataset.
    """
    conflicts = []
    obj_names = list(raw_scores_by_obj.keys())
    if not obj_names:
        return []
    n_candidates = len(raw_scores_by_obj[obj_names[0]])
    if n_candidates < 2:
        return []
        
    for i in range(len(obj_names)):
        for j in range(i + 1, len(obj_names)):
            o1, o2 = obj_names[i], obj_names[j]
            config1 = next((o for o in objectives if o.name == o1), None)
            config2 = next((o for o in objectives if o.name == o2), None)
            if not config1 or not config2:
                continue
                
            scores1 = raw_scores_by_obj[o1]
            scores2 = raw_scores_by_obj[o2]
            
            try:
                mean1 = statistics.mean(scores1)
                mean2 = statistics.mean(scores2)
                stdev1 = statistics.stdev(scores1)
                stdev2 = statistics.stdev(scores2)
                
                if stdev1 > 0 and stdev2 > 0:
                    covariance = sum((x - mean1) * (y - mean2) for x, y in zip(scores1, scores2)) / (n_candidates - 1)
                    corr = covariance / (stdev1 * stdev2)
                    
                    # If they should move together but correlate negatively, OR should move oppositely but correlate positively:
                    should_move_together = (config1.maximize == config2.maximize)
                    is_conflict = (should_move_together and corr < -0.2) or (not should_move_together and corr > 0.2)
                    
                    if is_conflict:
                        conflicts.append(f"{o1} vs {o2} (Correlation: {corr:+.2f})")
            except Exception:
                pass
    return conflicts


class BaseObjectiveOptimizer(ABC):
    """Abstract interface for multi-objective optimization algorithms (DIP)."""
    @abstractmethod
    def optimize(self, candidates: List[Any], objectives: List[Objective]) -> List[RankedCandidate]:
        pass


class WeightedObjectiveOptimizer(BaseObjectiveOptimizer):
    """Fully implemented weighted sum multi-objective optimizer with trade-off analysis."""
    
    def optimize(self, candidates: List[Any], objectives: List[Objective]) -> List[RankedCandidate]:
        n = len(candidates)
        if n == 0:
            return []
            
        active_objs = [o for o in objectives if o.enabled]
        if not active_objs:
            # Fallback to zero-scored results
            return [RankedCandidate(
                candidate=c, final_score=0.0, 
                explanation=RankingExplanation(optimization_strategy_used="WeightedObjectiveFallback"),
                original_index=i
            ) for i, c in enumerate(candidates)]
            
        # 1. Gather raw scores
        raw_values: Dict[str, List[float]] = {}
        for obj in active_objs:
            raw_values[obj.name] = [float(obj.evaluate(c)) for c in candidates]
            
        # 2. Analyze trade-off conflicts
        conflicts = _calculate_conflicts(raw_values, active_objs)
        
        # 3. Apply normalization
        norm_values: Dict[str, List[float]] = {}
        for obj in active_objs:
            norm_values[obj.name] = ScoreNormalizer.apply(
                obj.normalizer, raw_values[obj.name], invert=obj.minimize
            )
            
        # 4. Construct candidates with explanations
        ranked = []
        for i in range(n):
            candidate = candidates[i]
            explanation = RankingExplanation(optimization_strategy_used="WeightedObjective")
            final_score = 0.0
            
            for obj in active_objs:
                raw_val = raw_values[obj.name][i]
                norm_val = norm_values[obj.name][i]
                contrib = norm_val * obj.weight
                
                explanation.raw_scores[obj.name] = raw_val
                explanation.normalized_scores[obj.name] = norm_val
                explanation.objective_contributions[obj.name] = contrib
                explanation.objective_weights[obj.name] = obj.weight
                
                final_score += contrib
                
            explanation.final_objective_score = final_score
            explanation.objective_conflicts = conflicts
            
            rc = RankedCandidate(
                candidate=candidate,
                final_score=final_score,
                explanation=explanation,
                original_index=i
            )
            ranked.append(rc)
            
        # Sort candidates descending
        ranked.sort(key=lambda x: x.final_score, reverse=True)
        return ranked


class ParetoOptimizer(BaseObjectiveOptimizer):
    """Architecture stub for Pareto-Frontier multi-objective optimization."""
    def optimize(self, candidates: List[Any], objectives: List[Objective]) -> List[RankedCandidate]:
        logger.warning("ParetoOptimizer is a stub. Executing fallback WeightedObjectiveOptimizer.")
        return WeightedObjectiveOptimizer().optimize(candidates, objectives)


class TOPSISOptimizer(BaseObjectiveOptimizer):
    """Architecture stub for TOPSIS MCDM algorithm."""
    def optimize(self, candidates: List[Any], objectives: List[Objective]) -> List[RankedCandidate]:
        logger.warning("TOPSISOptimizer is a stub. Executing fallback WeightedObjectiveOptimizer.")
        return WeightedObjectiveOptimizer().optimize(candidates, objectives)


class PROMETHEEOptimizer(BaseObjectiveOptimizer):
    """Architecture stub for PROMETHEE MCDM outranking algorithm."""
    def optimize(self, candidates: List[Any], objectives: List[Objective]) -> List[RankedCandidate]:
        logger.warning("PROMETHEEOptimizer is a stub. Executing fallback WeightedObjectiveOptimizer.")
        return WeightedObjectiveOptimizer().optimize(candidates, objectives)


class ELECTREOptimizer(BaseObjectiveOptimizer):
    """Architecture stub for ELECTRE MCDM outranking algorithm."""
    def optimize(self, candidates: List[Any], objectives: List[Objective]) -> List[RankedCandidate]:
        logger.warning("ELECTREOptimizer is a stub. Executing fallback WeightedObjectiveOptimizer.")
        return WeightedObjectiveOptimizer().optimize(candidates, objectives)


class LearningToRankOptimizer(BaseObjectiveOptimizer):
    """Architecture stub for Learning-to-Rank Objective Optimization."""
    def optimize(self, candidates: List[Any], objectives: List[Objective]) -> List[RankedCandidate]:
        logger.warning("LearningToRankOptimizer is a stub. Executing fallback WeightedObjectiveOptimizer.")
        return WeightedObjectiveOptimizer().optimize(candidates, objectives)


class CustomObjectiveOptimizer(BaseObjectiveOptimizer):
    """Wraps user-provided objective optimization logic."""
    def __init__(self, optimize_func: Callable[[List[Any], List[Objective]], List[RankedCandidate]]):
        self.optimize_func = optimize_func

    def optimize(self, candidates: List[Any], objectives: List[Objective]) -> List[RankedCandidate]:
        return self.optimize_func(candidates, objectives)


class ObjectiveOptimizerRegistry:
    """Service locator for registering and retrieving objective optimization plugins."""
    _optimizers: Dict[str, BaseObjectiveOptimizer] = {}

    @classmethod
    def register(cls, name: str, optimizer: BaseObjectiveOptimizer):
        cls._optimizers[name.lower()] = optimizer

    @classmethod
    def get(cls, name: str) -> BaseObjectiveOptimizer:
        name_lower = name.lower()
        if name_lower not in cls._optimizers:
            raise ValueError(f"Objective optimizer '{name}' not found in ObjectiveOptimizerRegistry.")
        return cls._optimizers[name_lower]


# Pre-register default optimizers
ObjectiveOptimizerRegistry.register("weighted", WeightedObjectiveOptimizer())
ObjectiveOptimizerRegistry.register("pareto", ParetoOptimizer())
ObjectiveOptimizerRegistry.register("topsis", TOPSISOptimizer())
ObjectiveOptimizerRegistry.register("promethee", PROMETHEEOptimizer())
ObjectiveOptimizerRegistry.register("electre", ELECTREOptimizer())
ObjectiveOptimizerRegistry.register("learning_to_rank", LearningToRankOptimizer())


# ============================================================================
# 8. ORCHESTRATION & ENGINE
# ============================================================================

class EnterpriseRankingReport:
    """Generates an enterprise-grade report detailing strategy, conflicts, scores, and contributions."""
    
    @staticmethod
    def generate(result: RankingResult, config: RankingConfig) -> str:
        stats = result.statistics
        width = 80
        sep = "=" * width
        thin = "-" * width
        
        is_multi_objective = bool(config.objectives)
        strategy_name = f"Multi-Objective ({config.objective_optimizer})" if is_multi_objective else "Weighted Factor-Based"
        
        lines = [
            sep,
            " DineAI Framework - Enterprise Ranking & Optimization Report".center(width),
            sep,
            f" Optimization Strategy  : {strategy_name}",
            f" Total Candidates Ranked: {stats.total_candidates}",
            f" Total Execution Time   : {stats.execution_time_ms:.2f} ms",
            f" Average Ranking Score  : {stats.avg_score:.4f}",
            f" Highest Ranking Score  : {stats.top_score:.4f}",
            thin,
        ]

        if is_multi_objective:
            lines.append(" ACTIVE OBJECTIVES DEFINITIONS:")
            for obj in config.objectives:
                dir_str = "Maximize" if obj.maximize else "Minimize"
                lines.append(f"  * {obj.name:<25} [Weight: {obj.weight:.2f}, Priority: {obj.priority}, Direction: {dir_str}]")
        else:
            lines.append(" ACTIVE SCORING FACTORS DEFINITIONS:")
            for f in config.factors:
                if f.enabled:
                    lines.append(f"  * {f.name:<25} [Weight: {f.weight:.2f}, Priority: {f.priority}, Key: {f.scorer_key}]")

        # Compile conflict reports
        all_conflicts = set()
        for rc in result.ranked_candidates:
            all_conflicts.update(rc.explanation.objective_conflicts)
            
        if all_conflicts:
            lines.append(thin)
            lines.append(" CONFLICTING OBJECTIVES DETECTED (NEGATIVELY CORRELATED):")
            for conflict in sorted(all_conflicts):
                lines.append(f"  ! {conflict}")

        # Average metrics
        lines.append(thin)
        if is_multi_objective:
            lines.append(" AVERAGE OBJECTIVE RAW METRICS:")
            for obj_name, avg_val in sorted(stats.objective_averages.items()):
                lines.append(f"  * {obj_name:<25} : {avg_val:.2f}")
        else:
            lines.append(" AVERAGE FACTOR RAW METRICS:")
            for f_name, avg_val in sorted(stats.factor_averages.items()):
                lines.append(f"  * {f_name:<25} : {avg_val:.2f}")

        # Top ranked candidates list
        lines.append(thin)
        lines.append(" TOP RANKED CANDIDATES DISTRIBUTION:")
        lines.append(f"  {'Rank':<6} | {'Candidate Name / Row ID':<30} | {'Score':<10} | {'Primary Driver / Breakdown':<26}")
        lines.append("  " + "-" * (width - 4))
        
        for idx, rc in enumerate(result.ranked_candidates[:10]):
            name = _get_metadata(rc.candidate).get("name", f"Item_{rc.original_index}")
            contrib_map = rc.explanation.objective_contributions if is_multi_objective else rc.explanation.weighted_contributions
            top_drivers = sorted(contrib_map.items(), key=lambda x: -x[1])[:2]
            driver_str = ", ".join([f"{k} (+{v:.2f})" for k, v in top_drivers]) if top_drivers else "None"
            
            lines.append(f"  {f'[{idx+1}]':<6} | {str(name)[:30]:<30} | {rc.final_score:<10.4f} | {driver_str:<26}")

        lines.append(sep)
        return "\n".join(lines)


class RankingEngine:
    """The central coordinator governing single strategy execution and multi-objective optimization runs."""
    
    def __init__(self, config: RankingConfig, strategy: Optional[BaseRankingStrategy] = None):
        self.config = config
        self.strategy = strategy or WeightedRankingStrategy()
        logging.basicConfig(level=getattr(logging, self.config.log_level, logging.WARNING))

    def rank(self, filtering_result: FilteringResult) -> RankingResult:
        t0 = time.perf_counter()
        candidates = filtering_result.passed_candidates

        if not candidates:
            empty_stats = RankingStatistics(execution_time_ms=(time.perf_counter() - t0) * 1000)
            return RankingResult(
                ranked_candidates=[],
                statistics=empty_stats,
                enterprise_report="No candidates passed the filtering phase."
            )

        # 1. Check if we optimize via Multi-Objectives
        if self.config.objectives:
            opt_name = self.config.objective_optimizer or "weighted"
            optimizer = ObjectiveOptimizerRegistry.get(opt_name)
            ranked_candidates = optimizer.optimize(candidates, self.config.objectives)
            
            # Post-optimization category diversity check
            if self.config.diversity_penalty_enabled:
                ranked_candidates = WeightedRankingStrategy()._apply_diversity(
                    ranked_candidates, self.config.diversity_weight
                )
        else:
            # Standard single/multi-weighted scoring pass
            ranked_candidates = self.strategy.rank(candidates, self.config)

        # 2. Slice to max results
        if self.config.max_results and len(ranked_candidates) > self.config.max_results:
            ranked_candidates = ranked_candidates[:self.config.max_results]

        # 3. Calculate statistics
        t1 = time.perf_counter()
        execution_time_ms = (t1 - t0) * 1000

        stats = RankingStatistics(
            total_candidates=len(ranked_candidates),
            execution_time_ms=execution_time_ms
        )

        if ranked_candidates:
            stats.top_score = ranked_candidates[0].final_score
            stats.avg_score = statistics.mean([rc.final_score for rc in ranked_candidates])
            
            # Aggregate average metric statistics
            all_factors = set()
            all_objs = set()
            for rc in ranked_candidates:
                all_factors.update(rc.explanation.raw_scores.keys())
                if rc.explanation.objective_contributions:
                    all_objs.update(rc.explanation.objective_contributions.keys())
            
            for f in all_factors:
                scores = [rc.explanation.raw_scores[f] for rc in ranked_candidates if f in rc.explanation.raw_scores]
                if scores:
                    stats.factor_averages[f] = statistics.mean(scores)
                    
            for o in all_objs:
                scores = [rc.explanation.raw_scores[o] for rc in ranked_candidates if o in rc.explanation.raw_scores]
                if scores:
                    stats.objective_averages[o] = statistics.mean(scores)

        result = RankingResult(
            ranked_candidates=ranked_candidates,
            statistics=stats
        )
        result.enterprise_report = EnterpriseRankingReport.generate(result, self.config)
        return result

    def generate_report(self, result: RankingResult) -> str:
        """Convenience public API helper to print/return report."""
        return result.enterprise_report


# ============================================================================
# 9. SELF TESTS
# ============================================================================

def run_self_tests():
    print("=" * 80)
    print(" Running DineAI Ranking & Multi-Objective Optimization Self-Tests".center(80))
    print("=" * 80)

    # 1. Setup Mock Candidates
    mock_data = [
        _MockCandidate({
            "name": "Truffle Burger", 
            "price": 25.0, 
            "rating": 4.8, 
            "category": "Burger", 
            "order_count": 500,
            "calories": 750,
            "protein": 45,
            "fiber": 2,
            "image_url": "http://img.truffle.jpg",
            "description": "Premium truffle double burger",
            "ingredients": ["beef", "truffle", "bun"],
            "days_since_added": 5,
            "delivery_fee": 3.99,
            "semantic_similarity": 0.95
        }),
        _MockCandidate({
            "name": "Basic Burger", 
            "price": 10.0, 
            "rating": 4.1, 
            "category": "Burger", 
            "order_count": 1200,
            "calories": 520,
            "protein": 22,
            "fiber": 1,
            "image_url": "",
            "description": "Standard hamburger",
            "ingredients": ["beef", "bun"],
            "days_since_added": 40,
            "delivery_fee": 1.99,
            "semantic_similarity": 0.85
        }),
        _MockCandidate({
            "name": "Vegan Salad", 
            "price": 12.0, 
            "rating": 4.6, 
            "category": "Salad", 
            "order_count": 300,
            "calories": 320,
            "protein": 12,
            "fiber": 8,
            "image_url": "http://img.salad.jpg",
            "description": "Superfood vegan garden salad",
            "ingredients": ["quinoa", "kale", "avocado"],
            "days_since_added": 1,
            "delivery_fee": 0.0,
            "semantic_similarity": 0.70
        }),
        _MockCandidate({
            "name": "Spicy Wings", 
            "price": 15.0, 
            "rating": 4.4, 
            "category": "Appetizer", 
            "order_count": 900,
            "calories": 610,
            "protein": 30,
            "fiber": 0,
            "image_url": "http://img.wings.jpg",
            "description": "Crispy chicken wings",
            "ingredients": ["chicken", "chili"],
            "days_since_added": 15,
            "delivery_fee": 2.50,
            "semantic_similarity": 0.78
        }),
    ]
    
    filt_res = FilteringResult(passed_candidates=mock_data)

    # 2. Test Score Normalization
    print("Testing Score Normalizer...")
    scores = [10.0, 20.0, 30.0]
    norm_minmax = ScoreNormalizer.apply(NormalizationMethod.MIN_MAX, scores)
    assert norm_minmax == [0.0, 0.5, 1.0], f"Min-Max failed: {norm_minmax}"
    
    norm_invert = ScoreNormalizer.apply(NormalizationMethod.MIN_MAX, scores, invert=True)
    assert norm_invert == [1.0, 0.5, 0.0], f"Inverted Min-Max failed: {norm_invert}"
    
    norm_identity = ScoreNormalizer.apply(NormalizationMethod.IDENTITY, scores)
    assert norm_identity == scores, "Identity normalizer failed"
    
    norm_z = ScoreNormalizer.apply(NormalizationMethod.Z_SCORE, scores)
    assert len(norm_z) == 3 and 0 <= norm_z[0] <= 1, "Z-Score normalizer failed"
    
    norm_custom = ScoreNormalizer.apply(NormalizationMethod.CUSTOM, scores, custom_func=lambda x: [s / 10.0 for s in x])
    assert norm_custom == [1.0, 2.0, 3.0], "Custom normalizer failed"
    
    # 3. Test Configuration Validation
    print("Testing Configuration Validation...")
    invalid_config = RankingConfig(
        factors=[RankingFactor(name="Test", scorer_key="RatingScore", weight=-1.0)]
    )
    try:
        invalid_config.validate()
        assert False, "Should have raised ValueError for negative weight"
    except ValueError:
        pass # Expected

    # 4. Test Registries & Plugins
    print("Testing Scorer Registration...")
    class MyCustomScorer(BaseScorer):
        def score_batch(self, candidates: List[Any], factor_config: Dict[str, Any]) -> List[float]:
            return [len(_get_metadata(c).get("name", "")) for c in candidates]
            
    RankingRegistry.register("MyCustomScore", MyCustomScorer())
    assert isinstance(RankingRegistry.get("MyCustomScore"), MyCustomScorer)

    # 5. Test Factor-Based Weighted Ranking & Changes in ordering
    print("Testing Weighted Factor-Based Ranking...")
    config_pref = RankingConfig(
        factors=[
            RankingFactor(name="Semantic", scorer_key="SemanticScore", weight=0.4),
            RankingFactor(name="Rating", scorer_key="RatingScore", weight=0.3),
            RankingFactor(name="Price", scorer_key="PriceScore", weight=0.3, invert=True) # Low price is good
        ]
    )
    engine_pref = RankingEngine(config_pref)
    result_pref = engine_pref.rank(filt_res)
    
    assert len(result_pref.ranked_candidates) == 4
    
    # Changing weights changes ordering
    print("Testing Weight Changes change order...")
    config_cheap = RankingConfig(
        factors=[
            RankingFactor(name="Price", scorer_key="PriceScore", weight=1.0, invert=True)
        ]
    )
    engine_cheap = RankingEngine(config_cheap)
    result_cheap = engine_cheap.rank(filt_res)
    assert result_cheap.ranked_candidates[0].candidate.metadata["name"] == "Basic Burger", "Cheapest should be first"
    assert result_cheap.ranked_candidates[-1].candidate.metadata["name"] == "Truffle Burger", "Most expensive should be last"

    # 6. Test Custom Strategy
    print("Testing Custom Ranking Strategy...")
    def reverse_strategy(candidates, config):
        return [RankedCandidate(candidate=c, final_score=float(-i), explanation=RankingExplanation(), original_index=i) for i, c in enumerate(candidates)]
    
    config_custom = RankingConfig()
    engine_custom = RankingEngine(config_custom, strategy=CustomRankingStrategy(reverse_strategy))
    result_custom = engine_custom.rank(filt_res)
    assert result_custom.ranked_candidates[0].original_index == 0
    assert result_custom.ranked_candidates[-1].original_index == 3

    # 7. Test Explanation Generation & Telemetry
    print("Testing Explanation Generation...")
    explanation_text = result_pref.ranked_candidates[0].explanation.generate_text(result_pref.ranked_candidates[0].final_score)
    assert "Final Score:" in explanation_text
    assert "Factor Contributions:" in explanation_text

    # 8. Test Diversity Penalty
    print("Testing Diversity Penalty (MMR-lite)...")
    config_div = RankingConfig(
        factors=[
            RankingFactor(name="Rating", scorer_key="RatingScore", weight=1.0)
        ],
        diversity_penalty_enabled=True,
        diversity_weight=2.0
    )
    engine_div = RankingEngine(config_div)
    result_div = engine_div.rank(filt_res)
    # Second item should be Vegan Salad (category Salad) instead of Basic Burger (category Burger) due to category penalty
    assert result_div.ranked_candidates[1].candidate.metadata["category"] == "Salad", "Diversity penalty failed to promote different category"

    # 9. Test Penalty Scorer
    print("Testing Penalty Scorer...")
    config_penalty = RankingConfig(
        factors=[
            RankingFactor(name="DeliveryPenalty", scorer_key="PenaltyScore", weight=1.0, invert=True, config={"key": "delivery_fee"})
        ]
    )
    engine_penalty = RankingEngine(config_penalty)
    result_penalty = engine_penalty.rank(filt_res)
    assert result_penalty.ranked_candidates[0].candidate.metadata["name"] == "Vegan Salad", "Free delivery should be first"

    # 10. Multi-Objective Optimization Tests
    print("Testing Multi-Objective Optimization...")
    obj_config_dict = {
        "objective_optimizer": "weighted",
        "objectives": {
            "protein": {"weight": 0.4, "maximize": True, "type": "numeric"},
            "price": {"weight": 0.3, "minimize": True, "type": "numeric"},
            "calories": {"weight": 0.2, "minimize": True, "type": "numeric"},
            "rating": {"weight": 0.1, "maximize": True, "type": "numeric"}
        }
    }
    config_mo = RankingConfig.from_dict(obj_config_dict)
    assert len(config_mo.objectives) == 4
    
    engine_mo = RankingEngine(config_mo)
    result_mo = engine_mo.rank(filt_res)
    assert len(result_mo.ranked_candidates) == 4
    assert result_mo.ranked_candidates[0].final_score > 0
    
    # 11. Test Conflicting Objectives (negative correlation detection)
    print("Testing Tradeoff Conflict Detection...")
    assert len(result_mo.ranked_candidates[0].explanation.objective_conflicts) > 0, "No conflicts detected"
    print("Detected conflicts:")
    for conflict in result_mo.ranked_candidates[0].explanation.objective_conflicts:
        print(f"  - {conflict}")

    # 12. Test Computed Objectives registration & auto-discovery
    print("Testing Computed Objectives...")
    class MyBusinessProfitObjective(Objective):
        def evaluate(self, candidate: Any) -> float:
            meta = _get_metadata(candidate)
            return float(meta.get("price", 0.0)) * 0.4 # 40% margin
            
    ObjectiveRegistry.register("MyBusinessObjective", MyBusinessProfitObjective)
    ObjectiveRegistry.register_computed("protein_per_dollar_custom", lambda m: float(m.get("protein", 0)) / float(m.get("price", 1)))
    
    config_comp = RankingConfig(
        objectives=[
            Objective.from_dict("margin", {"type": "MyBusinessObjective", "weight": 0.5}),
            Objective.from_dict("protein_density_calc", {"type": "computed", "expression_key": "protein_per_dollar_custom", "weight": 0.5})
        ]
    )
    engine_comp = RankingEngine(config_comp)
    result_comp = engine_comp.rank(filt_res)
    assert result_comp.ranked_candidates[0].final_score > 0

    # 13. Test Objective Serialization
    print("Testing Objective Serialization...")
    serialized = config_mo.objectives[0].__dict__
    assert serialized["name"] == "protein"
    assert serialized["maximize"] is True

    # 14. Test Future Optimizer registration
    print("Testing Future Optimizer Registration...")
    class MyFutureNSGAIIIOptimizer(BaseObjectiveOptimizer):
        def optimize(self, candidates: List[Any], objectives: List[Objective]) -> List[RankedCandidate]:
            return [RankedCandidate(candidate=c, final_score=1.0, explanation=RankingExplanation(optimization_strategy_used="NSGA-III"), original_index=i) for i, c in enumerate(candidates)]
            
    ObjectiveOptimizerRegistry.register("NSGA-III", MyFutureNSGAIIIOptimizer())
    config_nsga = RankingConfig(
        objectives=[Objective(name="protein")],
        objective_optimizer="NSGA-III"
    )
    engine_nsga = RankingEngine(config_nsga)
    result_nsga = engine_nsga.rank(filt_res)
    assert result_nsga.ranked_candidates[0].explanation.optimization_strategy_used == "NSGA-III"

    # 15. Print Enterprise Reports
    print("Testing Enterprise Report printing...")
    print(result_pref.enterprise_report)
    print(result_mo.enterprise_report)

    print("\n[SUCCESS] All Ranking and Multi-Objective Framework self-tests passed successfully!")


if __name__ == "__main__":
    run_self_tests()








