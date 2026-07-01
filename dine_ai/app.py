"""
app.py — DineAI Main Application Orchestrator
==============================================
Main orchestration facade, pipeline stages executor, and health check monitoring.
Follows SOLID, Clean Architecture, Strategy, Factory, and Registry patterns.
"""

from __future__ import annotations

import os
import time
import json
import yaml
import pickle
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, Union

# Import DineAI Core Subsystems
from dine_ai.adapters.schema_mapper import SchemaMapper
from dine_ai.adapters.validator import DataFrameValidator
from dine_ai.adapters.text_builder import TextBuilder
from dine_ai.embeddings.embedding_builder import EmbeddingBuilder
from dine_ai.embeddings.faiss_builder import FaissBuilder, FaissConfig, FaissResult
from dine_ai.retrieval.semantic_search import SemanticSearchEngine, SearchConfig
from dine_ai.retrieval.filtering import FilteringEngine, FilterRule, FilterGroup
from dine_ai.retrieval.ranking import RankingEngine, RankingConfig, RankingFactor
from dine_ai.llm.prompts import PromptConfig, PromptContext, PromptBuilder
from dine_ai.llm.generator import (
    GeneratorEngine, 
    LLMConfig, 
    GenerationConfig, 
    GenerationRequest, 
    GenerationResponse,
    GenerationStatistics
)

logger = logging.getLogger(__name__)


# ============================================================================
# APPLICATION CONFIGURATION & STATE
# ============================================================================

@dataclass
class ApplicationConfig:
    """Core configuration options governing application behaviors and subsystem flags."""
    restaurant_name: str = "restaurant_A"
    dataset_directory: str = "dine_ai/datasets"
    language: str = "English"
    # Embedding subsystem — controls dataset building and semantic search
    embedding_provider: str = "mock"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # LLM subsystem — controls text generation only
    llm_provider: str = "mock"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    execution_mode: str = "local"
    debug: bool = False
    verbose: bool = False
    stream: bool = False
    logging: bool = True
    cache_enabled: bool = True
    auto_reload: bool = False
    max_context_recipes: int = 5
    enable_reports: bool = True

    def validate(self) -> None:
        """Validates configuration parameters boundary bounds."""
        if not self.restaurant_name:
            raise ValueError("restaurant_name cannot be empty.")
        if not self.embedding_provider:
            raise ValueError("embedding_provider cannot be empty.")
        if not self.llm_provider:
            raise ValueError("llm_provider cannot be empty.")
        if self.max_context_recipes <= 0:
            raise ValueError("max_context_recipes must be positive.")

    def to_dict(self) -> Dict[str, Any]:
        """Serializes application config into a dictionary representation."""
        return {
            "restaurant_name": self.restaurant_name,
            "dataset_directory": self.dataset_directory,
            "language": self.language,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "execution_mode": self.execution_mode,
            "debug": self.debug,
            "verbose": self.verbose,
            "stream": self.stream,
            "logging": self.logging,
            "cache_enabled": self.cache_enabled,
            "auto_reload": self.auto_reload,
            "max_context_recipes": self.max_context_recipes,
            "enable_reports": self.enable_reports
        }


@dataclass
class ApplicationState:
    """Monitors active state flags, database presence, and telemetry summaries of the application."""
    application_ready: bool = False
    dataset_loaded: bool = False
    faiss_loaded: bool = False
    generator_ready: bool = False
    pipeline_ready: bool = False
    conversation_ready: bool = False
    startup_time: Optional[float] = None
    runtime_statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes active state metrics into a dictionary representation."""
        return {
            "application_ready": self.application_ready,
            "dataset_loaded": self.dataset_loaded,
            "faiss_loaded": self.faiss_loaded,
            "generator_ready": self.generator_ready,
            "pipeline_ready": self.pipeline_ready,
            "conversation_ready": self.conversation_ready,
            "startup_time": self.startup_time,
            "runtime_statistics": self.runtime_statistics
        }


# ============================================================================
# APPLICATION EVENTS & PIPELINE TELEMETRY DTOs
# ============================================================================

class ApplicationEventType(Enum):
    """Supported operational lifecycle event types for DineAIApplication."""
    STARTUP = "STARTUP"
    SHUTDOWN = "SHUTDOWN"
    QUERY_STARTED = "QUERY_STARTED"
    QUERY_COMPLETED = "QUERY_COMPLETED"
    PIPELINE_STARTED = "PIPELINE_STARTED"
    PIPELINE_FINISHED = "PIPELINE_FINISHED"
    PIPELINE_FAILED = "PIPELINE_FAILED"
    RESTAURANT_SWITCHED = "RESTAURANT_SWITCHED"
    DATASET_RELOADED = "DATASET_RELOADED"


@dataclass
class ApplicationEvent:
    """Event payload emitted by the application lifecycle transitions."""
    event_type: ApplicationEventType
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineTrace:
    """Telemetry tracker measuring latency benchmarks and warning outcomes for execution stages."""
    semantic_search_time: float = 0.0
    filtering_time: float = 0.0
    ranking_time: float = 0.0
    prompt_time: float = 0.0
    generation_time: float = 0.0
    total_time: float = 0.0
    stage_warnings: Dict[str, List[str]] = field(default_factory=dict)
    stage_errors: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes pipeline trace statistics to a dictionary."""
        return {
            "semantic_search_time": self.semantic_search_time,
            "filtering_time": self.filtering_time,
            "ranking_time": self.ranking_time,
            "prompt_time": self.prompt_time,
            "generation_time": self.generation_time,
            "total_time": self.total_time,
            "stage_warnings": self.stage_warnings,
            "stage_errors": self.stage_errors
        }


@dataclass
class PipelineContext:
    """Shared state context passed sequentially between pipeline stage executions."""
    original_query: str
    conversation_history: List[Dict[str, str]] = field(default_factory=list)
    candidates: List[Any] = field(default_factory=list)
    filtered_candidates: List[Any] = field(default_factory=list)
    ranked_candidates: List[Any] = field(default_factory=list)
    prompt_result: Optional[Any] = None
    generation_response: Optional[Any] = None
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    trace: PipelineTrace = field(default_factory=PipelineTrace)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the pipeline context and its trace payload to a dictionary representation."""
        return {
            "original_query": self.original_query,
            "conversation_history": self.conversation_history,
            "candidates_count": len(self.candidates),
            "filtered_candidates_count": len(self.filtered_candidates),
            "ranked_candidates_count": len(self.ranked_candidates),
            "prompt_result_hash": getattr(self.prompt_result, "prompt_hash", None) if self.prompt_result else None,
            "generation_response_text": getattr(self.generation_response, "generated_text", None) if self.generation_response else None,
            "warnings": self.warnings,
            "metrics": self.metrics,
            "trace": self.trace.to_dict()
        }

    def enterprise_debug_report(self) -> str:
        """Generates a detailed, factual console report tracing the execution path of the query."""
        import json
        width = 80
        sep = "=" * width
        thin = "-" * width
        
        def get_name(c) -> str:
            # Unwrap RankedCandidate and CandidateFilterResult
            val = c
            if hasattr(val, "candidate"):
                val = val.candidate
            if hasattr(val, "candidate"):
                val = val.candidate
            if hasattr(val, "metadata") and isinstance(val.metadata, dict):
                return val.metadata.get("recipe_name") or val.metadata.get("name") or "Unknown"
            if isinstance(val, dict):
                return val.get("recipe_name") or val.get("name") or "Unknown"
            return getattr(val, "recipe_name", getattr(val, "name", "Unknown"))
            
        lines = [
            sep,
            " DINEAI RETRIEVAL TRACE & DEBUG REPORT ".center(width, "="),
            sep,
            f" Raw Query              : {self.original_query}",
            f" Processed Query        : {self.original_query.strip()}",
            f" Embedding Dimension    : {self.metrics.get('embedding_dim', 384)}",
            thin,
        ]
        
        lines.append(" SEMANTIC SEARCH RETRIEVED CANDIDATES:")
        if not self.candidates:
            lines.append("  [NONE]")
        else:
            for idx, c in enumerate(self.candidates):
                name = get_name(c)
                meta = c.metadata if hasattr(c, "metadata") and isinstance(c.metadata, dict) else {}
                sim = meta.get("semantic_similarity", 0.0)
                lines.append(f"  {idx+1}. {name:<35} | Similarity Score: {sim:.4f}")
        lines.append(thin)
        
        lines.append(" FILTERING DECISIONS:")
        filtering_res = self.metrics.get("filtering_result")
        if filtering_res is None:
            lines.append("  Filtering stage was not run or returned no results.")
        else:
            lines.append(f"  * Passed Candidates Count: {len(filtering_res.passed_candidates)}")
            lines.append(f"  * Rejected Candidates Count: {len(filtering_res.rejected_candidates)}")
            for idx, rc in enumerate(filtering_res.rejected_candidates):
                name = get_name(rc)
                failed_rules = getattr(rc, "failed_rules", [])
                lines.append(f"  [REJECTED] '{name}' on rules: {failed_rules}")
            for w in filtering_res.warnings:
                lines.append(f"  [WARN] {w}")
        lines.append(thin)
        
        lines.append(" RANKING DECISIONS:")
        if not self.ranked_candidates:
            lines.append("  [NONE]")
        else:
            for idx, rc in enumerate(self.ranked_candidates[:5]):
                name = get_name(rc)
                breakdown_str = json.dumps(rc.ranking_breakdown)
                lines.append(f"  [{idx+1}] {name:<35} | Score: {rc.final_score:.4f} | Breakdown: {breakdown_str}")
        lines.append(thin)
        
        char_cnt = len(self.prompt_result.final_prompt) if self.prompt_result else 0
        est_tokens = self.prompt_result.estimated_tokens if self.prompt_result else 0
        lines.append(f" Prompt Characters Size : {char_cnt}")
        lines.append(f" Estimated Prompt Tokens: {est_tokens}")
        lines.append(thin)
        
        gen_time = self.trace.generation_time
        resp_text = self.generation_response.generated_text if self.generation_response else "None"
        lines.append(f" LLM Generation Time    : {gen_time:.4f}s")
        lines.append(f" Final Response         : {resp_text}")
        lines.append(sep)
        
        return "\n".join(lines)


# ============================================================================
# MANAGERS (RESTAURANT & CONVERSATION LOGS)
# ============================================================================

from dine_ai.embeddings.faiss_builder import FaissResult


class RestaurantManager:
    """Manages the lifecycle of loading, reloading, and switching restaurant catalogs and database assets."""

    def __init__(self, dataset_directory: str = "dine_ai/datasets"):
        self.dataset_directory = dataset_directory
        self.current_restaurant: Optional[str] = None
        self.recipes: Optional[pd.DataFrame] = None
        self.faiss_result: Optional[FaissResult] = None
        self.metadata: Optional[Any] = None
        self.capabilities: Dict[str, Any] = {}
        self.available_columns: List[str] = []
        self.restaurant_config: Dict[str, Any] = {}

    def load_restaurant(
        self, 
        restaurant_name: str, 
        embedding_provider: str = "mock",
        embedding_model: str = "BAAI/bge-small-en-v1.5"
    ) -> Dict[str, Any]:
        """Loads restaurant config, and database assets from disk, delegating to DatasetManager.
        
        Only the embedding subsystem determines the provider strategy here.
        LLM configuration must never influence dataset building.
        """
        from dine_ai.dataset.manager import DatasetManager
        from dine_ai.dataset.pipeline import DatasetPipeline, MockEmbeddingProvider, SentenceTransformerProvider
        
        # Resolve embedding provider — LLM provider must never be consulted here
        if embedding_provider == "mock":
            pipeline = DatasetPipeline(embedding_provider=MockEmbeddingProvider())
        else:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
            pipeline = DatasetPipeline(embedding_provider=SentenceTransformerProvider(
                model_name=embedding_model,
                device=device,
                normalize_embeddings=True
            ))
            
        dataset_manager = DatasetManager(dataset_directory=self.dataset_directory, pipeline=pipeline)
        dataset = dataset_manager.load_restaurant(restaurant_name)
        
        # Update manager wrapper state
        self.current_restaurant = restaurant_name
        self.recipes = dataset.recipes
        self.faiss_result = dataset.faiss_result
        self.metadata = dataset.metadata
        self.capabilities = dataset.capabilities
        self.available_columns = dataset.available_columns
        
        # Load config from config.json directly if present for backward compatibility
        config = {}
        config_path = os.path.join(self.dataset_directory, restaurant_name, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        
        self.restaurant_config = config
        
        logger.info(f"RestaurantManager: Loaded '{restaurant_name}' via DatasetManager successfully.")
        return {
            "config": config,
            "recipes": self.recipes,
            "faiss_result": self.faiss_result
        }


class ConversationManager:
    """Tracks session conversations, memories, and transactional history logs."""

    def __init__(self):
        self._histories: Dict[str, List[Dict[str, str]]] = {}

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieves or initializes conversation logs for a session ID."""
        if not session_id:
            return []
        if session_id not in self._histories:
            self._histories[session_id] = []
        return self._histories[session_id]

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Appends a user or assistant message to the session's conversation history."""
        if not session_id:
            return
        history = self.get_history(session_id)
        history.append({"role": role, "content": content})

    def reset_conversation(self, session_id: str) -> None:
        """Clears memory history for the specified session ID."""
        if session_id in self._histories:
            self._histories[session_id] = []
            logger.info(f"ConversationManager: Reset session history for '{session_id}'.")


# ============================================================================
# PIPELINE STAGE ARCHITECTURE (STRATEGY PATTERN)
# ============================================================================

class BasePipelineStage(ABC):
    """Abstract Base Class defining the pipeline execution stage protocol (Strategy Pattern)."""

    def __init__(self, is_enabled: bool = True):
        self.is_enabled = is_enabled

    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Runs the stage's primary logic on the provided PipelineContext."""
        pass

    def validate_input(self, context: PipelineContext) -> None:
        """Validates that the input context meets the stage's execution requirements."""
        pass

    def validate_output(self, context: PipelineContext) -> None:
        """Validates that the execution produced valid outcomes in the context."""
        pass

    def collect_metrics(self, context: PipelineContext, elapsed: float) -> None:
        """Appends latency or execution metrics to the context trace."""
        pass


class SemanticSearchStage(BasePipelineStage):
    """Pipeline stage executing semantic query search against FAISS indexing assets."""

    def __init__(
        self, 
        restaurant_manager: RestaurantManager,
        embedding_model: BaseEmbeddingModel,
        search_config: Optional[SearchConfig] = None
    ):
        super().__init__()
        self.restaurant_manager = restaurant_manager
        self.embedding_model = embedding_model
        self.search_config = search_config or SearchConfig(top_k=5)

    def validate_input(self, context: PipelineContext) -> None:
        if not context.original_query:
            raise ValueError("SemanticSearchStage requires a non-empty original_query.")
        if self.restaurant_manager.faiss_result is None:
            raise RuntimeError("SemanticSearchStage requires a loaded FAISS index in RestaurantManager.")

    def execute(self, context: PipelineContext) -> PipelineContext:
        self.validate_input(context)
        start = time.perf_counter()

        # Instantiate search engine dynamically with switched restaurant assets
        engine = SemanticSearchEngine(
            faiss_result=self.restaurant_manager.faiss_result,
            embedding_model=self.embedding_model,
            config=self.search_config
        )
        search_res = engine.search(context.original_query)
        if self.restaurant_manager.recipes is not None:
            import pandas as pd
            df_recipes = self.restaurant_manager.recipes
            if not isinstance(df_recipes, pd.DataFrame):
                df_recipes = pd.DataFrame(df_recipes)
            df_recipes.index = df_recipes.index.astype(str)
            search_res.attach_metadata(df_recipes)
        context.candidates = search_res.candidates
        for c in context.candidates:
            if isinstance(c.metadata, dict):
                c.metadata["semantic_similarity"] = float(c.score)
        if self.restaurant_manager.faiss_result is not None:
            context.metrics["embedding_dim"] = self.restaurant_manager.faiss_result.index.d

        elapsed = time.perf_counter() - start
        context.trace.semantic_search_time = elapsed
        
        # Pull warnings if any
        if search_res.warnings:
            context.trace.stage_warnings["SemanticSearchStage"] = search_res.warnings
            context.warnings.extend(search_res.warnings)

        self.validate_output(context)
        return context

    def validate_output(self, context: PipelineContext) -> None:
        if context.candidates is None:
            raise ValueError("SemanticSearchStage failed to populate candidates.")


class FilteringStage(BasePipelineStage):
    """Pipeline stage running metadata criteria constraints against search hits."""

    def __init__(self, filter_rules: Optional[List[Any]] = None):
        super().__init__()
        self.filter_rules = filter_rules or []

    def execute(self, context: PipelineContext) -> PipelineContext:
        self.validate_input(context)
        start = time.perf_counter()

        from dine_ai.retrieval.filtering import FilteringEngine
        engine = FilteringEngine()
        filtering_res = engine.filter(context.candidates, self.filter_rules)
        
        # Save filtered candidates
        context.filtered_candidates = filtering_res.passed_candidates
        # Save the filtering result itself under context metrics for ranking consumption
        context.metrics["filtering_result"] = filtering_res

        elapsed = time.perf_counter() - start
        context.trace.filtering_time = elapsed

        if filtering_res.warnings:
            context.trace.stage_warnings["FilteringStage"] = filtering_res.warnings
            context.warnings.extend(filtering_res.warnings)

        self.validate_output(context)
        return context

    def validate_output(self, context: PipelineContext) -> None:
        if context.filtered_candidates is None:
            raise ValueError("FilteringStage failed to populate filtered_candidates.")


class RankingStage(BasePipelineStage):
    """Pipeline stage scoring and reordering filtered candidates."""

    def __init__(self, ranking_config: Optional[RankingConfig] = None):
        super().__init__()
        self.ranking_config = ranking_config or RankingConfig(
            factors=[
                RankingFactor(name="Semantic", weight=0.4, scorer_key="SemanticScore"),
                RankingFactor(name="Rating", weight=0.3, scorer_key="RatingScore"),
                RankingFactor(name="Price", weight=0.3, scorer_key="PriceScore", invert=True)
            ]
        )

    def execute(self, context: PipelineContext) -> PipelineContext:
        self.validate_input(context)
        start = time.perf_counter()

        # Retrieve filtering result from context
        filtering_result = context.metrics.get("filtering_result")
        if filtering_result is None:
            # Fallback: wrap filtered candidates list
            from dine_ai.retrieval.filtering import FilteringResult, FilteringStatistics, CandidateFilterResult
            passed = []
            for c in context.filtered_candidates:
                if isinstance(c, CandidateFilterResult):
                    passed.append(c)
                else:
                    passed.append(CandidateFilterResult(
                        candidate=c,
                        passed=True,
                        soft_score=0.0
                    ))
            filtering_result = FilteringResult(
                passed_candidates=passed,
                rejected_candidates=[],
                statistics=FilteringStatistics(input_candidates=len(context.filtered_candidates))
            )

        engine = RankingEngine(self.ranking_config)
        ranking_res = engine.rank(filtering_result)
        context.ranked_candidates = ranking_res.ranked_candidates

        elapsed = time.perf_counter() - start
        context.trace.ranking_time = elapsed

        self.validate_output(context)
        return context

    def validate_output(self, context: PipelineContext) -> None:
        if context.ranked_candidates is None:
            raise ValueError("RankingStage failed to populate ranked_candidates.")


class PromptBuilderStage(BasePipelineStage):
    """Pipeline stage composing prompt sections and localizing headers."""

    def __init__(
        self, 
        template_name: str = "RestaurantChat",
        language: str = "English",
        persona: str = "FriendlyWaiter"
    ):
        super().__init__()
        self.template_name = template_name
        self.language = language
        self.persona = persona

    def execute(self, context: PipelineContext) -> PipelineContext:
        self.validate_input(context)
        start = time.perf_counter()

        prompt_config = PromptConfig()
        builder = PromptBuilder(prompt_config)
        builder.use_template(self.template_name)
        builder.set_language(self.language)
        builder.set_persona(self.persona)

        # Assemble prompt context DTO
        prompt_ctx = PromptContext(
            query=context.original_query,
            ranked_candidates=context.ranked_candidates,
            conversation_history=context.conversation_history
        )
        builder.set_context(prompt_ctx)

        prompt_res = builder.build()
        context.prompt_result = prompt_res

        elapsed = time.perf_counter() - start
        context.trace.prompt_time = elapsed

        if prompt_res.warnings:
            context.trace.stage_warnings["PromptBuilderStage"] = prompt_res.warnings
            context.warnings.extend(prompt_res.warnings)

        self.validate_output(context)
        return context

    def validate_output(self, context: PipelineContext) -> None:
        if context.prompt_result is None:
            raise ValueError("PromptBuilderStage failed to construct prompt_result.")


class GeneratorStage(BasePipelineStage):
    """Pipeline stage dispatching the optimized prompt to LLM executors."""

    def __init__(self, generator_config: Optional[LLMConfig] = None):
        super().__init__()
        self.generator_config = generator_config or LLMConfig(provider="mock")

    def validate_input(self, context: PipelineContext) -> None:
        if context.prompt_result is None:
            raise ValueError("GeneratorStage requires a valid prompt_result in context.")

    def execute(self, context: PipelineContext) -> PipelineContext:
        self.validate_input(context)
        start = time.perf_counter()

        engine = GeneratorEngine(self.generator_config)
        request = GenerationRequest(
            prompt_result=context.prompt_result,
            generation_config=GenerationConfig()
        )

        print("\n" + "=" * 120)
        print("RANKED CANDIDATES")
        print("=" * 120)

        if context.ranked_candidates:
            for i, candidate in enumerate(context.ranked_candidates, 1):

                meta = {}
                obj = candidate
                while hasattr(obj, "candidate"):
                    if isinstance(obj, dict):
                        break
                    obj = getattr(obj, "candidate")

                if hasattr(obj, "metadata"):
                    meta = obj.metadata

                print(f"\nCandidate #{i}")

                print("Recipe:",
                      meta.get(
                          "recipe_name",
                          meta.get("name", "UNKNOWN")
                      ))

                print("Score:",
                      getattr(candidate, "score", "N/A"))

                print("Price:",
                      meta.get("price"))

                print("Calories:",
                      meta.get("calories_per_serving"))

                print("Protein:",
                      meta.get("protein_g_per_serving"))

        else:
            print("NO RANKED CANDIDATES")

        print("=" * 120)

        print("\n" + "=" * 120)
        print("FINAL PROMPT SENT TO LLM")
        print("=" * 120)

        if context.prompt_result is not None:
            print(context.prompt_result.final_prompt)
        else:
            print("PromptResult is None")

        print("=" * 120)

        generation_start = time.perf_counter()

        gen_res = engine.generate(request)

        generation_end = time.perf_counter()

        print("=" * 120)
        print("GENERATION TIME")
        print("=" * 120)
        print(f"{generation_end - generation_start:.2f} seconds")
        print("=" * 120)
        context.generation_response = gen_res

        elapsed = time.perf_counter() - start
        context.trace.generation_time = elapsed

        if gen_res.warnings:
            context.trace.stage_warnings["GeneratorStage"] = gen_res.warnings
            context.warnings.extend(gen_res.warnings)

        self.validate_output(context)
        return context

    def validate_output(self, context: PipelineContext) -> None:
        if context.generation_response is None:
            raise ValueError("GeneratorStage failed to compile generation_response.")


# ============================================================================
# PIPELINE MANAGER (ORCHESTRATOR)
# ============================================================================

class PipelineManager:
    """Manages the registration, replacement, and sequential execution of pipeline stage strategies."""

    def __init__(self):
        self._stages: List[Tuple[str, BasePipelineStage]] = []

    def register_stage(self, stage_name: str, stage: BasePipelineStage, position: Optional[int] = None) -> None:
        """Registers a pipeline stage, optionally at a specific index sequence position."""
        if position is not None:
            self._stages.insert(position, (stage_name, stage))
        else:
            self._stages.append((stage_name, stage))
        logger.info(f"PipelineManager: Registered stage '{stage_name}'.")

    def remove_stage(self, stage_name: str) -> None:
        """Removes a registered stage by name."""
        self._stages = [s for s in self._stages if s[0] != stage_name]
        logger.info(f"PipelineManager: Removed stage '{stage_name}'.")

    def replace_stage(self, stage_name: str, new_stage: BasePipelineStage) -> None:
        """Replaces an existing stage in place with a new strategy."""
        for idx, (name, _) in enumerate(self._stages):
            if name == stage_name:
                self._stages[idx] = (stage_name, new_stage)
                logger.info(f"PipelineManager: Replaced stage '{stage_name}'.")
                return
        raise KeyError(f"Pipeline stage '{stage_name}' not found for replacement.")

    def list_stages(self) -> List[str]:
        """Returns names of registered stages in their execution sequence."""
        return [s[0] for s in self._stages]

    def execute(self, context: PipelineContext) -> PipelineContext:
        """Executes active stages sequentially, tracking telemetry and catching errors to prevent application crashes."""
        start_total = time.perf_counter()
        
        for name, stage in self._stages:
            if not stage.is_enabled:
                logger.info(f"PipelineManager: Skipping disabled stage '{name}'.")
                continue
                
            logger.info(f"PipelineManager: Executing stage '{name}'...")
            try:
                # 1. Validate input requirements
                stage.validate_input(context)
                
                # 2. Execute stage logic
                context = stage.execute(context)
                
                # 3. Validate output outcomes
                stage.validate_output(context)
            except Exception as e:
                error_msg = f"Stage '{name}' execution failed: {e}"
                logger.error(error_msg, exc_info=True)
                context.trace.stage_errors[name] = str(e)
                context.warnings.append(f"Error in {name}: {e}")
                
        context.trace.total_time = time.perf_counter() - start_total
        return context


# ============================================================================
# MOCK EMBEDDING MODEL FOR OFFLINE DEVELOPMENT
# ============================================================================

from dine_ai.embeddings.embedding_builder import BaseEmbeddingModel, EmbeddingConfig
import hashlib
import numpy as np


class MockEmbeddingModel(BaseEmbeddingModel):
    """Deterministic simulation embedding model for offline and pipeline testing."""

    def __init__(self):
        self._dim = 384

    def load(self, config: EmbeddingConfig) -> None:
        pass

    def encode(self, texts: List[str], config: EmbeddingConfig) -> np.ndarray:
        import hashlib
        import re
        vectors = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            words = re.findall(r'[a-zA-Z0-9_]+', text.lower())
            v = np.zeros(self._dim, dtype=np.float32)
            for word in words:
                for k in range(8):
                    word_hash = hashlib.sha256(f"{word}_{k}".encode("utf-8")).digest()
                    idx = int.from_bytes(word_hash[:4], "little") % self._dim
                    v[idx] += 1.0
            
            text_seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")
            rng = np.random.default_rng(text_seed)
            v += rng.random(self._dim).astype(np.float32) * 0.05
            
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            vectors[i] = v
        return vectors

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def provider_name(self) -> str:
        return "MockEmbeddingModel"


# ============================================================================
# MAIN APPLICATION FACADE (COMPOSITION OVER INHERITANCE)
# ============================================================================

class DineAIApplication:
    """Enterprise main facade orchestrating configurations, states, managers, and pipeline execution logs."""

    def __init__(self, config: Optional[ApplicationConfig] = None):
        self.config = config or ApplicationConfig()
        self.config.validate()

        self.state = ApplicationState()
        self.restaurant_manager = RestaurantManager(self.config.dataset_directory)
        self.conversation_manager = ConversationManager()
        self.pipeline_manager = PipelineManager()

        self._listeners: Dict[ApplicationEventType, List[Callable[[ApplicationEvent], None]]] = {
            t: [] for t in ApplicationEventType
        }
        
        # Initialize runtime statistics
        self.state.runtime_statistics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_query_time": 0.0,
            "total_retrieval_time": 0.0,
            "total_generation_time": 0.0,
            "total_ranking_time": 0.0
        }
        
        self.embedding_model: Optional[BaseEmbeddingModel] = None

    def register_listener(self, event_type: ApplicationEventType, callback: Callable[[ApplicationEvent], None]) -> None:
        """Subscribes an event listener callback hook to a specific application event type."""
        self._listeners[event_type].append(callback)

    def emit_event(self, event_type: ApplicationEventType, details: Optional[Dict[str, Any]] = None) -> None:
        """Broadcasts lifecycle events to registered listener triggers."""
        event = ApplicationEvent(event_type=event_type, details=details or {})
        for callback in self._listeners[event_type]:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error executing event listener callback: {e}", exc_info=True)

    def initialize(self) -> None:
        """Initializes dependencies, loads active restaurant assets, and constructs execution pipeline stages."""
        self.state.startup_time = time.time()
        logger.info("DineAIApplication: Initializing components...")

        # 1. Load active restaurant datasets (embedding_provider determines strategy; LLM config is never consulted)
        self.restaurant_manager.load_restaurant(
            self.config.restaurant_name,
            embedding_provider=self.config.embedding_provider,
            embedding_model=self.config.embedding_model
        )
        self.state.dataset_loaded = True
        self.state.faiss_loaded = True

        # 2. Setup embedding model strategy — driven solely by embedding_provider
        if self.config.embedding_provider == "mock":
            self.embedding_model = MockEmbeddingModel()
        else:
            try:
                from dine_ai.embeddings.embedding_builder import SentenceTransformerEmbedding
                self.embedding_model = SentenceTransformerEmbedding()
            except ImportError:
                logger.warning("sentence-transformers not installed. Falling back to MockEmbeddingModel.")
                self.embedding_model = MockEmbeddingModel()

        # Load the embedding model using resolved config
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
        emb_config = EmbeddingConfig(
            model_name=self.config.embedding_model,
            device=device,
            normalize_embeddings=True
        )
        self.embedding_model.load(emb_config)

        # 3. Construct default pipeline stages
        self._setup_default_pipeline()
        
        self.state.generator_ready = True
        self.state.pipeline_ready = True
        self.state.conversation_ready = True
        self.state.application_ready = True

        self.emit_event(ApplicationEventType.STARTUP, {"config": self.config.to_dict()})
        logger.info("DineAIApplication: Initialization complete.")

    def _setup_default_pipeline(self) -> None:
        """Registers default pipeline sequence stages."""
        # Semantic search stage
        search_config = SearchConfig(top_k=self.config.max_context_recipes)
        self.pipeline_manager.register_stage(
            "SemanticSearch", 
            SemanticSearchStage(self.restaurant_manager, self.embedding_model, search_config)
        )

        # Filtering stage (initially empty filters)
        self.pipeline_manager.register_stage(
            "Filtering", 
            FilteringStage()
        )

        # Ranking stage
        self.pipeline_manager.register_stage(
            "Ranking", 
            RankingStage()
        )

        # Prompt builder stage
        self.pipeline_manager.register_stage(
            "PromptBuilder", 
            PromptBuilderStage(
                template_name="RestaurantChat",
                language=self.config.language,
                persona="FriendlyWaiter"
            )
        )

        # Generator stage — driven solely by llm_provider and llm_model; embedding config is never consulted here
        gen_config = LLMConfig(
            provider=self.config.llm_provider,
            model_name=self.config.llm_model,
            execution_mode=self.config.execution_mode,
            use_cache=self.config.cache_enabled
        )
        self.pipeline_manager.register_stage(
            "Generator", 
            GeneratorStage(gen_config)
        )

    def chat(self, session_id: str, query: str, rules: Optional[List[Any]] = None) -> GenerationResponse:
        """Processes a chat conversation turn through the registered orchestration pipeline stages."""
        if not self.state.application_ready:
            raise RuntimeError("Application is not initialized. Call initialize() first.")

        start_time = time.perf_counter()
        self.emit_event(ApplicationEventType.QUERY_STARTED, {"session_id": session_id, "query": query})
        self.state.runtime_statistics["total_queries"] += 1

        # Fetch session conversation history
        history = self.conversation_manager.get_history(session_id)

        # Build execution context
        context = PipelineContext(
            original_query=query,
            conversation_history=history
        )

        # Inject runtime filtering rules if provided
        if rules is not None:
            # Dynamically replace FilteringStage with active rules
            self.pipeline_manager.replace_stage("Filtering", FilteringStage(rules))

        # Run pipeline stages sequentially
        try:
            context = self.pipeline_manager.execute(context)
            
            # Check for critical generation failures
            if context.generation_response is None:
                raise ValueError("Pipeline execution compiled no generation output.")
            
            response = context.generation_response
            
            # Append interaction to memory history logs
            self.conversation_manager.add_message(session_id, "user", query)
            self.conversation_manager.add_message(session_id, "assistant", response.generated_text)
            
            # Update telemetry stats
            elapsed = time.perf_counter() - start_time
            self.state.runtime_statistics["successful_queries"] += 1
            self.state.runtime_statistics["total_query_time"] += elapsed
            self.state.runtime_statistics["total_retrieval_time"] += context.trace.semantic_search_time
            self.state.runtime_statistics["total_generation_time"] += context.trace.generation_time
            self.state.runtime_statistics["total_ranking_time"] += context.trace.ranking_time
            
            self.emit_event(ApplicationEventType.QUERY_COMPLETED, {
                "session_id": session_id,
                "elapsed": elapsed,
                "warnings_count": len(context.warnings)
            })
            return response
        except Exception as e:
            self.state.runtime_statistics["failed_queries"] += 1
            self.emit_event(ApplicationEventType.PIPELINE_FAILED, {"session_id": session_id, "error": str(e)})
            raise RuntimeError(f"Chat execution turn failed: {e}") from e

    def reload(self) -> None:
        """Reloads active restaurant database assets from storage."""
        logger.info(f"DineAIApplication: Reloading active restaurant '{self.config.restaurant_name}'...")
        self.restaurant_manager.load_restaurant(
            self.config.restaurant_name,
            embedding_provider=self.config.embedding_provider,
            embedding_model=self.config.embedding_model
        )
        self.emit_event(ApplicationEventType.DATASET_RELOADED, {"restaurant": self.config.restaurant_name})

    def switch_restaurant(self, restaurant_name: str) -> None:
        """Switches the active catalog and index assets to another restaurant name."""
        logger.info(f"DineAIApplication: Switching restaurant to '{restaurant_name}'...")
        self.restaurant_manager.load_restaurant(
            restaurant_name,
            embedding_provider=self.config.embedding_provider,
            embedding_model=self.config.embedding_model
        )
        self.config.restaurant_name = restaurant_name
        self.emit_event(ApplicationEventType.RESTAURANT_SWITCHED, {"restaurant": restaurant_name})

    def health_check(self) -> Dict[str, Any]:
        """Runs structured diagnostics on system components."""
        recipes_len = len(self.restaurant_manager.recipes) if self.restaurant_manager.recipes is not None else 0
        return {
            "status": "healthy" if self.state.application_ready else "unhealthy",
            "uptime_seconds": time.time() - self.state.startup_time if self.state.startup_time else 0.0,
            "state": self.state.to_dict(),
            "config": self.config.to_dict(),
            "loaded_recipes_count": recipes_len,
            "faiss_vectors_count": self.restaurant_manager.faiss_result.total_vectors() if self.restaurant_manager.faiss_result else 0
        }

    def shutdown(self) -> None:
        """Performs cleanup tasks and stops operations."""
        logger.info("DineAIApplication: Shutting down system orchestrator...")
        self.state.application_ready = False
        self.emit_event(ApplicationEventType.SHUTDOWN)


# ============================================================================
# ENTERPRISE REPORTING SYSTEM
# ============================================================================

class EnterpriseApplicationReport:
    """Formats telemetry matrices and diagnostic logs into analytical reports."""

    @staticmethod
    def generate(app: DineAIApplication) -> str:
        width = 80
        sep = "=" * width
        thin = "-" * width
        
        lines = []
        lines.append(sep)
        lines.append("DineAI Application Framework - Enterprise Operational Report".center(width))
        lines.append(sep)
        
        cfg = app.config
        lines.append(f" Restaurant Catalog  : {cfg.restaurant_name.upper()}")
        lines.append(f" Dataset Location    : {cfg.dataset_directory}")
        lines.append(f" Execution Language  : {cfg.language}")
        lines.append(f" Embedding Provider  : {cfg.embedding_provider.upper()}")
        lines.append(f" Embedding Model     : {cfg.embedding_model}")
        lines.append(f" LLM Provider        : {cfg.llm_provider.upper()}")
        lines.append(f" LLM Model           : {cfg.llm_model}")
        lines.append(f" Run Mode            : {cfg.execution_mode}")
        lines.append(thin)
        
        hc = app.health_check()
        lines.append(f" Application Status  : {hc['status'].upper()}")
        lines.append(f" Loaded Recipes      : {hc['loaded_recipes_count']}")
        lines.append(f" FAISS Index Vectors : {hc['faiss_vectors_count']}")
        lines.append(f" Uptime Elapsed      : {hc['uptime_seconds']:.2f}s")
        lines.append(thin)
        
        stats = app.state.runtime_statistics
        tot = stats["total_queries"] or 1
        lines.append(" RUNTIME QUERY METRICS:")
        lines.append(f"  Total Chat Queries : {stats['total_queries']}")
        lines.append(f"  Success/Fail Ratio : {stats['successful_queries']} / {stats['failed_queries']}")
        lines.append(f"  Average Query Time : {stats['total_query_time'] / tot:.4f}s")
        lines.append(f"  Average Search Time: {stats['total_retrieval_time'] / tot:.4f}s")
        lines.append(f"  Average Rank Time  : {stats['total_ranking_time'] / tot:.4f}s")
        lines.append(f"  Average Gen Time   : {stats['total_generation_time'] / tot:.4f}s")
        lines.append(sep)
        
        return "\n".join(lines)

    @staticmethod
    def to_json(app: DineAIApplication) -> str:
        """Serializes application report information into JSON."""
        hc = app.health_check()
        report_data = {
            "restaurant": app.config.restaurant_name,
            "dataset_directory": app.config.dataset_directory,
            "language": app.config.language,
            "embedding_provider": app.config.embedding_provider,
            "embedding_model": app.config.embedding_model,
            "llm_provider": app.config.llm_provider,
            "llm_model": app.config.llm_model,
            "execution_mode": app.config.execution_mode,
            "status": hc["status"],
            "loaded_recipes_count": hc["loaded_recipes_count"],
            "faiss_vectors_count": hc["faiss_vectors_count"],
            "uptime_seconds": hc["uptime_seconds"],
            "runtime_statistics": app.state.runtime_statistics
        }
        return json.dumps(report_data, indent=2)


# ============================================================================
# SELF TESTS SUITE
# ============================================================================

def run_self_tests():
    import sys
    print("=" * 80)
    print(" Running DineAI Application Orchestrator Framework Self-Tests ".center(80, "="))
    print("=" * 80)

    # 1. Config Validation
    print("Testing ApplicationConfig validation...")
    cfg = ApplicationConfig(restaurant_name="restaurant_A")
    cfg.validate()
    assert cfg.to_dict()["restaurant_name"] == "restaurant_A"
    
    try:
        invalid_cfg = ApplicationConfig(restaurant_name="")
        invalid_cfg.validate()
        assert False, "Should have raised ValueError for empty restaurant name."
    except ValueError:
        print("  [PASS] Config validation correctly rejected empty restaurant name.")

    # 2. State & Telemetry DTO Serialization
    print("Testing State & Telemetry DTO Serialization...")
    state = ApplicationState()
    assert state.to_dict()["application_ready"] is False
    
    trace = PipelineTrace(semantic_search_time=0.1234)
    assert trace.to_dict()["semantic_search_time"] == 0.1234
    
    context = PipelineContext(original_query="healthy salad")
    assert context.to_dict()["original_query"] == "healthy salad"
    print("  [PASS] Telemetry and context classes serialization validated.")

    # 3. Dynamic Stage Registration & PipelineManager
    print("Testing Stage Registration & PipelineManager...")
    pm = PipelineManager()
    class DummyStage(BasePipelineStage):
        def execute(self, ctx: PipelineContext) -> PipelineContext:
            ctx.metrics["dummy_executed"] = True
            return ctx
            
    pm.register_stage("Dummy", DummyStage())
    assert "Dummy" in pm.list_stages()
    
    # Position insertion
    class PreDummyStage(BasePipelineStage):
        def execute(self, ctx: PipelineContext) -> PipelineContext:
            ctx.metrics["pre_dummy_executed"] = True
            return ctx
    pm.register_stage("PreDummy", PreDummyStage(), position=0)
    assert pm.list_stages() == ["PreDummy", "Dummy"]
    
    # Replacement
    class CustomDummyStage(BasePipelineStage):
        def execute(self, ctx: PipelineContext) -> PipelineContext:
            ctx.metrics["custom_dummy_executed"] = True
            return ctx
    pm.replace_stage("Dummy", CustomDummyStage())
    assert pm.list_stages() == ["PreDummy", "Dummy"]
    
    # Run test pipeline
    ctx_test = PipelineContext(original_query="test query")
    ctx_test = pm.execute(ctx_test)
    assert ctx_test.metrics.get("pre_dummy_executed") is True
    assert ctx_test.metrics.get("custom_dummy_executed") is True
    
    # Removal
    pm.remove_stage("PreDummy")
    assert pm.list_stages() == ["Dummy"]
    print("  [PASS] Pipeline stage insertions, replacements, removals and execution validated.")

    # 4. DineAIApplication lifecycle
    print("Testing DineAIApplication lifecycle...")
    app_config = ApplicationConfig(
        restaurant_name="restaurant_A",
        embedding_provider="mock",
        llm_provider="mock",
        llm_model="Qwen/Qwen2.5-7B-Instruct",
        max_context_recipes=3
    )
    app = DineAIApplication(app_config)
    
    # Register events listeners
    events_triggered = []
    app.register_listener(ApplicationEventType.STARTUP, lambda e: events_triggered.append(e.event_type))
    app.register_listener(ApplicationEventType.QUERY_STARTED, lambda e: events_triggered.append(e.event_type))
    app.register_listener(ApplicationEventType.QUERY_COMPLETED, lambda e: events_triggered.append(e.event_type))
    app.register_listener(ApplicationEventType.SHUTDOWN, lambda e: events_triggered.append(e.event_type))
    app.register_listener(ApplicationEventType.RESTAURANT_SWITCHED, lambda e: events_triggered.append(e.event_type))

    # Initialize
    app.initialize()
    assert app.state.application_ready is True
    assert ApplicationEventType.STARTUP in events_triggered
    print("  [PASS] Application initialized and STARTUP event emitted.")

    # Health Check
    hc = app.health_check()
    assert hc["status"] == "healthy"
    assert hc["loaded_recipes_count"] > 0
    assert hc["faiss_vectors_count"] > 0
    print(f"  [PASS] Diagnostics status: {hc['status']} | Loaded: {hc['loaded_recipes_count']} recipes.")

    # Chat execution
    print("Testing Chat Query Execution...")
    response = app.chat("session_123", "Tell me about the Truffle Burger")
    assert response is not None
    assert response.generated_text is not None
    assert "Truffle Burger" in response.generated_text
    
    # Conversation memory verification
    history = app.conversation_manager.get_history("session_123")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"
    
    # Event verification
    assert ApplicationEventType.QUERY_STARTED in events_triggered
    assert ApplicationEventType.QUERY_COMPLETED in events_triggered
    print("  [PASS] Query successfully executed and session history logged.")

    # ── INTEGRATION SCENARIOS (DatasetManager Integration Verification) ───────
    print("Testing DatasetManager Integration Scenarios...")
    
    test_dir_a = os.path.join(app.config.dataset_directory, "test_restaurant_A")
    test_dir_b = os.path.join(app.config.dataset_directory, "test_restaurant_B")
    test_dir_c = os.path.join(app.config.dataset_directory, "test_restaurant_C")
    
    # Helper to clean a directory
    def cleanup_dir(dpath):
        if os.path.exists(dpath):
            for fname in os.listdir(dpath):
                fpath = os.path.join(dpath, fname)
                try:
                    if os.path.isfile(fpath):
                        os.remove(fpath)
                except Exception:
                    pass
            try:
                os.rmdir(dpath)
            except Exception:
                pass
                
    cleanup_dir(test_dir_a)
    cleanup_dir(test_dir_b)
    cleanup_dir(test_dir_c)
    
    # --- Scenario 1: Only menu_items.csv exists ---
    print("  Scenario 1: Only menu_items.csv exists...")
    os.makedirs(test_dir_a, exist_ok=True)
    with open(os.path.join(test_dir_a, "menu_items.csv"), "w", encoding="utf-8") as f:
        f.write("recipe_name,price,protein_g_per_serving,calories_per_serving,cuisine_type,item_type,diet_labels,ingredients\n")
        f.write("Truffle Burger,15.0,25.0,600.0,American,main_course,halal,\"beef, truffle, bun\"\n")
        f.write("French Fries,5.0,2.0,300.0,American,side,vegan,\"potatoes, oil, salt\"\n")
        
    app_sc1 = DineAIApplication(ApplicationConfig(
        restaurant_name="test_restaurant_A",
        embedding_provider="mock",
        llm_provider="mock",
        max_context_recipes=3
    ))
    app_sc1.initialize()
    assert app_sc1.state.application_ready is True
    assert len(app_sc1.restaurant_manager.recipes) == 2
    assert os.path.exists(os.path.join(test_dir_a, "recipes.pkl"))
    assert os.path.exists(os.path.join(test_dir_a, "faiss.index"))
    assert os.path.exists(os.path.join(test_dir_a, "metadata.json"))
    print("    [PASS] Dataset generated automatically and application started successfully.")
    
    # --- Scenario 2: Artifacts already exist ---
    print("  Scenario 2: Artifacts already exist (instant load)...")
    mtime_before = os.path.getmtime(os.path.join(test_dir_a, "recipes.pkl"))
    time.sleep(0.1)
    app_sc2 = DineAIApplication(ApplicationConfig(
        restaurant_name="test_restaurant_A",
        embedding_provider="mock",
        llm_provider="mock",
        max_context_recipes=3
    ))
    app_sc2.initialize()
    mtime_after = os.path.getmtime(os.path.join(test_dir_a, "recipes.pkl"))
    assert mtime_before == mtime_after
    assert len(app_sc2.restaurant_manager.recipes) == 2
    print("    [PASS] No rebuild triggered, loaded from cache immediately.")
    
    # --- Scenario 3: menu_items.csv modified ---
    print("  Scenario 3: menu_items.csv modified (auto-rebuild)...")
    with open(os.path.join(test_dir_a, "menu_items.csv"), "a", encoding="utf-8") as f:
        f.write("Chocolate Cake,6.0,3.0,400.0,Dessert,Dessert,vegetarian,\"chocolate, sugar\"\n")
        
    app_sc3 = DineAIApplication(ApplicationConfig(
        restaurant_name="test_restaurant_A",
        embedding_provider="mock",
        llm_provider="mock",
        max_context_recipes=3
    ))
    app_sc3.initialize()
    assert len(app_sc3.restaurant_manager.recipes) == 3
    print("    [PASS] CSV modification detected, dataset rebuilt and reloaded successfully.")
    
    # --- Scenario 4: Restaurant switching ---
    print("  Scenario 4: Restaurant switching (A -> B -> C)...")
    os.makedirs(test_dir_b, exist_ok=True)
    with open(os.path.join(test_dir_b, "menu_items.csv"), "w", encoding="utf-8") as f:
        f.write("recipe_name,price,protein_g_per_serving,calories_per_serving,cuisine_type,item_type,diet_labels,ingredients\n")
        f.write("Bacon Burger,16.0,26.0,650.0,American,main_course,halal,\"beef, bacon, bun\"\n")
        
    os.makedirs(test_dir_c, exist_ok=True)
    with open(os.path.join(test_dir_c, "menu_items.csv"), "w", encoding="utf-8") as f:
        f.write("recipe_name,price,protein_g_per_serving,calories_per_serving,cuisine_type,item_type,diet_labels,ingredients\n")
        f.write("Cheese Pizza,12.0,15.0,500.0,Italian,main_course,vegetarian,\"dough, cheese, tomato\"\n")
        
    # Switch to B
    app_sc3.switch_restaurant("test_restaurant_B")
    assert app_sc3.config.restaurant_name == "test_restaurant_B"
    assert len(app_sc3.restaurant_manager.recipes) == 1
    
    # Switch to C
    app_sc3.switch_restaurant("test_restaurant_C")
    assert app_sc3.config.restaurant_name == "test_restaurant_C"
    assert len(app_sc3.restaurant_manager.recipes) == 1
    print("    [PASS] Successfully switched contexts and loaded dynamically.")
    
    # --- Scenario 5: Missing artifacts ---
    print("  Scenario 5: Missing artifacts (auto-rebuild)...")
    # Delete recipes.pkl from test_restaurant_A
    os.remove(os.path.join(test_dir_a, "recipes.pkl"))
    app_sc5 = DineAIApplication(ApplicationConfig(
        restaurant_name="test_restaurant_A",
        embedding_provider="mock",
        llm_provider="mock",
        max_context_recipes=3
    ))
    app_sc5.initialize()
    assert len(app_sc5.restaurant_manager.recipes) == 3
    assert os.path.exists(os.path.join(test_dir_a, "recipes.pkl"))
    print("    [PASS] Rebuilt missing artifacts automatically and initialized successfully.")
    
    # Cleanup test folders
    cleanup_dir(test_dir_a)
    cleanup_dir(test_dir_b)
    cleanup_dir(test_dir_c)

    # Telemetry report rendering
    print("Testing Operational Telemetry Report rendering...")
    report = EnterpriseApplicationReport.generate(app)
    assert "DineAI Application Framework" in report
    assert "RUNTIME QUERY METRICS" in report
    print(report)

    json_report = EnterpriseApplicationReport.to_json(app)
    assert json.loads(json_report)["status"] == "healthy"
    print("  [PASS] Telemetry report strings and JSON payloads successfully generated.")

    # Shutdown
    app.shutdown()
    assert app.state.application_ready is False
    assert ApplicationEventType.SHUTDOWN in events_triggered
    print("  [PASS] Application shutdown completed successfully.")

    print("\n[SUCCESS] All DineAI Application Orchestrator self-tests passed successfully!\n")


if __name__ == "__main__":
    import sys
    import argparse

    # ── Argument Parser ────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="DineAI Restaurant Intelligence Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m dine_ai.app\n"
            "  python -m dine_ai.app --embedding-provider mock --llm-provider mock\n"
            "  python -m dine_ai.app --embedding-provider sentence-transformer --llm-provider qwen\n"
        )
    )
    parser.add_argument("--test", "-t", action="store_true", help="Run framework self-tests")
    parser.add_argument("--restaurant", "-r", type=str, default="restaurant_A", help="Name of the restaurant catalog")

    # Current explicit provider flags
    parser.add_argument("--embedding-provider", "-ep", type=str, default=None,
                        help="Embedding provider: 'mock' or 'sentence-transformer'")
    parser.add_argument("--embedding-model", "-em", type=str, default=None,
                        help="Embedding model name (e.g. 'BAAI/bge-small-en-v1.5')")
    parser.add_argument("--llm-provider", "-lp", type=str, default=None,
                        help="LLM provider: 'mock', 'qwen', 'openai', 'gemini', etc.")
    parser.add_argument("--llm-model", "-lm", type=str, default=None,
                        help="LLM model name (e.g. 'Qwen/Qwen2.5-7B-Instruct')")

    # Deprecated legacy flags — kept for backward compatibility only
    parser.add_argument("--provider", "-p", type=str, default=None,
                        help="[DEPRECATED] Use --embedding-provider and --llm-provider instead.")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="[DEPRECATED] Use --llm-model instead.")

    args = parser.parse_args()

    # ── Deprecated flag resolution ─────────────────────────────────────────────
    # An LLM provider name is NEVER assumed to be a valid embedding provider.
    # Only 'mock' is safe to map to both subsystems automatically.
    _SAFE_DUAL_PROVIDERS = {"mock"}  # providers valid for both embedding and LLM

    if args.provider is not None:
        if args.provider in _SAFE_DUAL_PROVIDERS:
            print(
                f"[DEPRECATION WARNING] --provider is deprecated.\n"
                f"  Mapped '{args.provider}' to both --embedding-provider and --llm-provider.\n"
                f"  Use --embedding-provider and --llm-provider explicitly in future invocations.",
                file=sys.stderr
            )
            if args.embedding_provider is None:
                args.embedding_provider = args.provider
            if args.llm_provider is None:
                args.llm_provider = args.provider
        else:
            print(
                f"[DEPRECATED ERROR] --provider '{args.provider}' cannot be automatically mapped.\n"
                f"  Embedding and LLM providers are now configured separately.\n"
                f"  An LLM provider name is never a valid embedding provider.\n"
                f"  Use: --embedding-provider <emb_provider> --llm-provider <llm_provider>\n"
                f"  Available embedding providers : mock, sentence-transformer\n"
                f"  Available LLM providers       : mock, qwen, openai, claude, gemini, llama, mistral, ollama",
                file=sys.stderr
            )
            sys.exit(1)

    if args.model is not None:
        print(
            f"[DEPRECATION WARNING] --model is deprecated.\n"
            f"  Mapped '{args.model}' to --llm-model only (never to --embedding-model).\n"
            f"  Use --llm-model explicitly in future invocations.",
            file=sys.stderr
        )
        if args.llm_model is None:
            args.llm_model = args.model

    # ── Apply defaults for any flags still unset ───────────────────────────────
    resolved_embedding_provider = args.embedding_provider or "mock"
    resolved_embedding_model    = args.embedding_model    or "BAAI/bge-small-en-v1.5"
    resolved_llm_provider       = args.llm_provider       or "mock"
    resolved_llm_model          = args.llm_model          or "Qwen/Qwen2.5-7B-Instruct"

    # ── Test mode ─────────────────────────────────────────────────────────────
    test_mode_env = os.environ.get("DINEAI_TEST_MODE")
    if args.test or test_mode_env == "1":
        run_self_tests()
    else:
        # ── Production Interactive CLI ─────────────────────────────────────────
        print("=" * 82)
        print(" DineAI Restaurant Intelligence Framework ".center(82, "="))
        print("=" * 82)

        app_config = ApplicationConfig(
            restaurant_name=args.restaurant,
            embedding_provider=resolved_embedding_provider,
            embedding_model=resolved_embedding_model,
            llm_provider=resolved_llm_provider,
            llm_model=resolved_llm_model,
        )

        app = DineAIApplication(app_config)

        try:
            app.initialize()

            rm = app.restaurant_manager
            recipes_count = len(rm.recipes) if rm.recipes is not None else 0
            faiss_vectors = rm.faiss_result.total_vectors() if rm.faiss_result else 0

            capabilities_list = []
            if rm.capabilities and "enabled_capabilities" in rm.capabilities:
                for cap in rm.capabilities["enabled_capabilities"]:
                    capabilities_list.append(cap["name"])
            if not capabilities_list:
                capabilities_list = ["None"]

            columns_list = rm.available_columns or ["None"]

            print(f" Restaurant Name       : {app.config.restaurant_name}")
            print(f" Loaded Recipes        : {recipes_count}")
            print(f" FAISS vectors         : {faiss_vectors}")
            print(f" Capabilities detected : {', '.join(capabilities_list)}")
            print(f" Available Columns     : {', '.join(columns_list)}")
            print(f" Embedding Provider    : {app.config.embedding_provider}")
            print(f" Embedding Model       : {app.config.embedding_model}")
            print(f" LLM Provider          : {app.config.llm_provider}")
            print(f" LLM Model             : {app.config.llm_model}")
            print(f" Application Status    : READY")
            print("=" * 82)
            print("Type 'exit' or 'quit' to quit. Press Ctrl+C to force quit.")
            print("=" * 82)

            session_id = "cli_session"
            while True:
                try:
                    user_input = input("You:\n> ")
                except EOFError:
                    break

                user_input = user_input.strip()
                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit"):
                    break

                response = app.chat(session_id, user_input)
                print("\nAssistant:")
                print(response.generated_text if response and response.generated_text else "No response generated.")
                print("-" * 82)

        except KeyboardInterrupt:
            print("\nSession interrupted.")
        except Exception as e:
            print(f"\n[ERROR] Failed to start application: {e}")
            import traceback
            traceback.print_exc()
        finally:
            app.shutdown()
            print("Goodbye.")