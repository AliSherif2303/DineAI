"""
integration_test.py — DineAI Integration Test Suite
===================================================
Verifies the interface compatibility and data routing between all DineAI components.
Conforms strictly to SOLID principles, clean reports, and offline testing patterns.
"""

import os
import sys
import time
import traceback
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# DineAI Modules Imports
from dine_ai.adapters.schema_mapper import SchemaMapper
from dine_ai.adapters.validator import DataFrameValidator, ValidationResult
from dine_ai.adapters.feature_engineering import FeatureEngineer, FeatureEngineeringResult
from dine_ai.adapters.text_builder import TextBuilder, TextBuildingResult
from dine_ai.embeddings.embedding_builder import EmbeddingBuilder, EmbeddingResult, EmbeddingConfig
from dine_ai.embeddings.faiss_builder import FaissBuilder, FaissResult, FaissConfig
from dine_ai.retrieval.semantic_search import SemanticSearchEngine, SearchResult, SearchConfig
from dine_ai.retrieval.filtering import FilteringEngine, FilteringResult, FilterRule, FilterOperator
from dine_ai.retrieval.ranking import RankingEngine, RankingResult, RankingConfig, RankingFactor
from dine_ai.llm.prompts import PromptBuilder, PromptConfig, PromptContext, PromptResult
from dine_ai.llm.generator import (
    GeneratorEngine, 
    LLMConfig, 
    GenerationRequest, 
    GenerationConfig, 
    GenerationResponse
)
from dine_ai.embeddings.embedding_builder import BaseEmbeddingModel

# New Provider & Dataset Module Imports
from dine_ai.dataset.pipeline import DatasetPipeline, MockEmbeddingProvider, _MockEmbeddingModel, PipelineArtifacts
from dine_ai.dataset.manager import DatasetManager
from dine_ai.dataset.loader import RestaurantDataset
from dine_ai.capabilities.detector import CapabilityDetector, RestaurantCapabilityReport

# ============================================================================
# INTEGRATION REPORT & METRICS DTOs
# ============================================================================

@dataclass
class StageOutcome:
    """Represents the validation metrics of a single pipeline stage integration check."""
    stage_name: str
    status: str  # "PASS" | "FAIL"
    elapsed_seconds: float
    input_type: str = ""
    output_type: str = ""
    dimensions_check: str = "N/A"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class IntegrationReport:
    """Compiles and formats stage outcomes into an enterprise-style dashboard."""

    @staticmethod
    def generate(outcomes: List[StageOutcome], total_elapsed: float) -> str:
        width = 80
        sep = "=" * width
        thin = "-" * width

        # Calculations
        passed_count = sum(1 for o in outcomes if o.status == "PASS")
        failed_count = len(outcomes) - passed_count
        success_pct = (passed_count / len(outcomes)) * 100 if outcomes else 0.0

        # Health Rating lookup
        if success_pct == 100.0:
            health_rating = "EXCELLENT"
        elif success_pct >= 90.0:
            health_rating = "VERY GOOD"
        elif success_pct >= 75.0:
            health_rating = "GOOD"
        elif success_pct >= 50.0:
            health_rating = "FAIR"
        else:
            health_rating = "FAILED"

        lines = []
        lines.append(sep)
        lines.append("DINEAI INTEGRATION TEST REPORT".center(width))
        lines.append(sep)

        # Print each stage status
        for o in outcomes:
            dots = "." * (45 - len(o.stage_name))
            lines.append(f" {o.stage_name} {dots} {o.status:<8} ({o.elapsed_seconds:.4f}s)")
            if o.errors:
                for err in o.errors:
                    lines.append(f"   [ERR] {err}")
            if o.warnings:
                for warn in o.warnings:
                    lines.append(f"   [WARN] {warn}")

        lines.append(thin)
        lines.append(f" Modules Passed            : {passed_count}")
        lines.append(f" Modules Failed            : {failed_count}")
        lines.append(f" Success Percentage        : {success_pct:.2f}%")
        lines.append(f" Overall Integration Score : {success_pct:.1f}%")
        lines.append(f" Health Rating             : {health_rating}")
        lines.append(f" Pipeline Runtime          : {total_elapsed:.4f}s")
        lines.append(sep)

        return "\n".join(lines)


# ============================================================================
# INTEGRATION TESTS RUNNER
# ============================================================================

class IntegrationTestRunner:
    """Orchestrates sequential execution of all DineAI modules and verifies their interface boundaries."""

    def __init__(self):
        self.raw_df = pd.DataFrame([
            {
                "Recipe Name": "Truffle Burger",
                "Description": "Premium beef patty with gourmet truffle sauce",
                "Meal Type": "Dinner",
                "Dish Type": "Main Course",
                "Cuisine Type": "American",
                "Ingredients": ["beef", "bun", "truffle sauce", "cheese"],
                "Servings": 1.0,
                "Calories": 750.0,
                "Protein (g)": 45.0,
                "Carbs": 35.0,
                "Fat": 40.0,
                "Sodium (mg)": 980.0,
                "Diet Labels": ["High-Protein"],
                "Health Labels": ["Nut-Free"],
                "Cautions": [],
                "Menu Price": 25.0,
                "Currency": "USD",
                "Rating": 4.8,
                "Popularity": 95.0,
                "Cook Time": 15.0
            },
            {
                "Recipe Name": "Veggie Salad",
                "Description": "Crisp mixed greens with balsamic dressing",
                "Meal Type": "Lunch",
                "Dish Type": "Salad",
                "Cuisine Type": "Mediterranean",
                "Ingredients": ["lettuce", "cucumber", "tomato", "balsamic"],
                "Servings": 2.0,
                "Calories": 120.0,
                "Protein (g)": 3.0,
                "Carbs": 10.0,
                "Fat": 8.0,
                "Sodium (mg)": 150.0,
                "Diet Labels": ["Low-Fat", "Low-Calorie"],
                "Health Labels": ["Vegan", "Gluten-Free"],
                "Cautions": [],
                "Menu Price": 12.0,
                "Currency": "USD",
                "Rating": 4.5,
                "Popularity": 80.0,
                "Cook Time": 5.0
            }
        ])
        self.outcomes: List[StageOutcome] = []
        
        # Temporary directory management
        self.temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = self.temp_dir_obj.name
        self.keep_temp_dir = False
        self.available_columns: List[str] = []

    def test_schema_mapper(self) -> pd.DataFrame:
        start = time.perf_counter()
        mapper = SchemaMapper()
        df, mapping_res = mapper.fit_transform(self.raw_df)
        elapsed = time.perf_counter() - start
        
        errors = []
        warnings = []
        if not isinstance(df, pd.DataFrame):
            errors.append("Output is not a pandas DataFrame.")
        if "recipe_name" not in df.columns:
            errors.append("Required column 'recipe_name' is missing from output columns.")
        if "price" not in df.columns:
            errors.append("Required column 'price' is missing from output columns.")
            
        status = "FAIL" if errors else "PASS"
        self.outcomes.append(StageOutcome(
            stage_name="Schema Mapper",
            status=status,
            elapsed_seconds=elapsed,
            input_type=str(type(self.raw_df)),
            output_type=str(type(df)),
            dimensions_check=f"{df.shape[0]}x{df.shape[1]}",
            errors=errors,
            warnings=warnings
        ))
        if status == "FAIL":
            raise RuntimeError(f"Schema Mapper integration failed: {errors}")
        return df

    def test_validator(self, df: pd.DataFrame) -> pd.DataFrame:
        start = time.perf_counter()
        validator = DataFrameValidator()
        res = validator.validate(df)
        elapsed = time.perf_counter() - start
        
        errors = []
        warnings = []
        if not isinstance(res, ValidationResult):
            errors.append("Output is not a ValidationResult object.")
        if not res.is_valid:
            errors.append(f"Validation failed. Errors: {[e.message for e in res.errors]}")
        for w in res.warnings:
            warnings.append(w.message)
            
        status = "FAIL" if errors else "PASS"
        self.outcomes.append(StageOutcome(
            stage_name="Validator",
            status=status,
            elapsed_seconds=elapsed,
            input_type=str(type(df)),
            output_type=str(type(res)),
            dimensions_check="N/A",
            errors=errors,
            warnings=warnings
        ))
        if status == "FAIL":
            raise RuntimeError(f"Validator integration failed: {errors}")
        return df

    def test_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        start = time.perf_counter()
        from dine_ai.adapters.feature_engineering import FeatureEngineerConfig
        engineer = FeatureEngineer(config=FeatureEngineerConfig())
        enriched_df = engineer.transform(df)
        elapsed = time.perf_counter() - start
        
        errors = []
        warnings = []
        if not isinstance(enriched_df, pd.DataFrame):
            errors.append("Output is not a pandas DataFrame.")
        if "protein_density" not in enriched_df.columns:
            errors.append("Engineered column 'protein_density' is missing from output.")
        if "is_vegan" not in enriched_df.columns:
            errors.append("Engineered column 'is_vegan' is missing from output.")
            
        res = engineer.last_result
        if res:
            for w in res.warnings:
                warnings.append(f"{w.engineer}: {w.message}")
            if not res.is_success:
                errors.append("FeatureEngineer reported is_success=False")

        status = "FAIL" if errors else "PASS"
        self.outcomes.append(StageOutcome(
            stage_name="Feature Engineering",
            status=status,
            elapsed_seconds=elapsed,
            input_type=str(type(df)),
            output_type=str(type(enriched_df)),
            dimensions_check=f"{enriched_df.shape[0]}x{enriched_df.shape[1]}",
            errors=errors,
            warnings=warnings
        ))
        if status == "FAIL":
            raise RuntimeError(f"Feature Engineering integration failed: {errors}")
        return enriched_df

    def test_text_builder(self, df: pd.DataFrame) -> pd.DataFrame:
        start = time.perf_counter()
        builder = TextBuilder()
        text_df = builder.transform(df)
        elapsed = time.perf_counter() - start
        
        errors = []
        warnings = []
        if not isinstance(text_df, pd.DataFrame):
            errors.append("Output is not a pandas DataFrame.")
        if "search_text" not in text_df.columns:
            errors.append("Text column 'search_text' is missing from output.")
        if "text_chunk" not in text_df.columns:
            errors.append("Text column 'text_chunk' is missing from output.")
            
        res = builder.last_result
        if res:
            for w in res.warnings:
                warnings.append(f"{w.builder_name}: {w.message}")
            if not res.is_success:
                errors.append("TextBuilder reported is_success=False")

        status = "FAIL" if errors else "PASS"
        self.outcomes.append(StageOutcome(
            stage_name="Text Builder",
            status=status,
            elapsed_seconds=elapsed,
            input_type=str(type(df)),
            output_type=str(type(text_df)),
            dimensions_check=f"{text_df.shape[0]}x{text_df.shape[1]}",
            errors=errors,
            warnings=warnings
        ))
        if status == "FAIL":
            raise RuntimeError(f"Text Builder integration failed: {errors}")
        return text_df

    def test_capability_detector(self, df: pd.DataFrame) -> RestaurantCapabilityReport:
        start = time.perf_counter()
        detector = CapabilityDetector()
        report = detector.detect(df, "restaurant_integration")
        elapsed = time.perf_counter() - start
        
        errors = []
        warnings = []
        if not isinstance(report, RestaurantCapabilityReport):
            errors.append("Output is not a RestaurantCapabilityReport.")
        else:
            if not isinstance(report.available_columns, list):
                errors.append("available_columns is missing or not a list in report.")
            if not isinstance(report.enabled_capabilities, list):
                errors.append("enabled_capabilities is missing or not a list in report.")
            if not isinstance(report.disabled_capabilities, list):
                errors.append("disabled_capabilities is missing or not a list in report.")
                
        status = "FAIL" if errors else "PASS"
        self.outcomes.append(StageOutcome(
            stage_name="Capability Detector",
            status=status,
            elapsed_seconds=elapsed,
            input_type=str(type(df)),
            output_type=str(type(report)),
            dimensions_check=f"Score: {getattr(report, 'capability_score', 0.0):.1f}%",
            errors=errors,
            warnings=warnings
        ))
        if status == "FAIL":
            raise RuntimeError(f"Capability Detector integration failed: {errors}")
        return report

    def test_dataset_pipeline(self) -> PipelineArtifacts:
        start = time.perf_counter()
        errors = []
        warnings = []
        artifacts = None
        
        try:
            restaurant_name = "restaurant_integration"
            restaurant_dir = os.path.join(self.temp_dir, restaurant_name)
            os.makedirs(restaurant_dir, exist_ok=True)
            csv_path = os.path.join(restaurant_dir, "recipes.csv")
            self.raw_df.to_csv(csv_path, index=False)
            
            pipeline = DatasetPipeline(embedding_provider=MockEmbeddingProvider())
            artifacts = pipeline.build(
                csv_path=csv_path,
                output_dir=restaurant_dir,
                restaurant_name=restaurant_name
            )
            
            if not isinstance(artifacts, PipelineArtifacts):
                errors.append("Output is not a PipelineArtifacts instance.")
            else:
                if artifacts.enriched_df is None or artifacts.enriched_df.empty:
                    errors.append("PipelineArtifacts.enriched_df is empty.")
                if artifacts.embedding_result is None:
                    errors.append("PipelineArtifacts.embedding_result is missing.")
                if artifacts.faiss_result is None:
                    errors.append("PipelineArtifacts.faiss_result is missing.")
                if artifacts.metadata is None:
                    errors.append("PipelineArtifacts.metadata is missing.")
                if artifacts.capability_report is None:
                    errors.append("PipelineArtifacts.capability_report is missing.")
        except Exception as e:
            errors.append(f"Pipeline build crashed: {e}")
            
        elapsed = time.perf_counter() - start
        status = "FAIL" if errors else "PASS"
        self.outcomes.append(StageOutcome(
            stage_name="Dataset Pipeline",
            status=status,
            elapsed_seconds=elapsed,
            input_type="recipes.csv",
            output_type=str(type(artifacts)) if artifacts else "None",
            dimensions_check="N/A",
            errors=errors,
            warnings=warnings
        ))
        if status == "FAIL":
            raise RuntimeError(f"Dataset Pipeline integration failed: {errors}")
        return artifacts

    def test_embedding_generation(self, df: pd.DataFrame) -> EmbeddingResult:
        start = time.perf_counter()
        builder = EmbeddingBuilder(EmbeddingConfig(show_progress=False), model=_MockEmbeddingModel())
        res = builder.build(df)
        elapsed = time.perf_counter() - start
        
        errors = []
        warnings = []
        if not isinstance(res, EmbeddingResult):
            errors.append("Output is not an EmbeddingResult.")
        if not res.is_success:
            errors.append("EmbeddingResult reports is_success=False")
        if res.embeddings.shape != (2, 384):
            errors.append(f"Unexpected embeddings shape: {res.embeddings.shape}, expected (2, 384)")
        for w in res.warnings:
            warnings.append(w)
            
        status = "FAIL" if errors else "PASS"
        self.outcomes.append(StageOutcome(
            stage_name="Embedding Builder",
            status=status,
            elapsed_seconds=elapsed,
            input_type=str(type(df)),
            output_type=str(type(res)),
            dimensions_check=f"{res.embeddings.shape[0]}x{res.embeddings.shape[1]}",
            errors=errors,
            warnings=warnings
        ))
        if status == "FAIL":
            raise RuntimeError(f"Embedding Builder integration failed: {errors}")
        return res

    def test_faiss_index_build(self, embed_res: EmbeddingResult) -> FaissResult:
        start = time.perf_counter()
        builder = FaissBuilder(FaissConfig(index_type="flat_ip", normalize_before_index=True))
        res = builder.build(embed_res)
        elapsed = time.perf_counter() - start
        
        errors = []
        warnings = []
        if not isinstance(res, FaissResult):
            errors.append("Output is not a FaissResult.")
        if not res.is_success:
            errors.append("FaissResult reports is_success=False")
        if res.total_vectors() != 2:
            errors.append(f"Expected index vector count: 2, got {res.total_vectors()}")
        for w in res.warnings:
            warnings.append(w)
            
        status = "FAIL" if errors else "PASS"
        self.outcomes.append(StageOutcome(
            stage_name="FAISS Builder",
            status=status,
            elapsed_seconds=elapsed,
            input_type=str(type(embed_res)),
            output_type=str(type(res)),
            dimensions_check=f"Vectors count: {res.total_vectors()}",
            errors=errors,
            warnings=warnings
        ))
        if status == "FAIL":
            raise RuntimeError(f"FAISS Builder integration failed: {errors}")
        return res

    def test_dataset_manager(self) -> RestaurantDataset:
        start = time.perf_counter()
        errors = []
        warnings = []
        dataset = None
        
        try:
            pipeline = DatasetPipeline(embedding_provider=MockEmbeddingProvider())
            manager = DatasetManager(dataset_directory=self.temp_dir, pipeline=pipeline)
            dataset = manager.load_restaurant("restaurant_integration")
            
            if not isinstance(dataset, RestaurantDataset):
                errors.append("Output is not a RestaurantDataset instance.")
            else:
                if dataset.recipes is None or dataset.recipes.empty:
                    errors.append("RestaurantDataset.recipes is empty.")
                if dataset.faiss_result is None:
                    errors.append("RestaurantDataset.faiss_result is missing.")
                if dataset.metadata is None:
                    errors.append("RestaurantDataset.metadata is missing.")
                if not isinstance(dataset.capabilities, dict):
                    errors.append("RestaurantDataset.capabilities is missing or not a dict.")
                if not isinstance(dataset.available_columns, list):
                    errors.append("RestaurantDataset.available_columns is missing or not a list.")
        except Exception as e:
            errors.append(f"DatasetManager loading crashed: {e}")
            
        elapsed = time.perf_counter() - start
        status = "FAIL" if errors else "PASS"
        self.outcomes.append(StageOutcome(
            stage_name="Dataset Manager",
            status=status,
            elapsed_seconds=elapsed,
            input_type="restaurant name",
            output_type=str(type(dataset)) if dataset else "None",
            dimensions_check="N/A",
            errors=errors,
            warnings=warnings
        ))
        if status == "FAIL":
            raise RuntimeError(f"Dataset Manager integration failed: {errors}")
        return dataset

    def test_semantic_search(self, faiss_res: FaissResult, df: pd.DataFrame) -> SearchResult:
        start = time.perf_counter()
        engine = SemanticSearchEngine(
            faiss_result=faiss_res,
            embedding_model=_MockEmbeddingModel(),
            config=SearchConfig(top_k=5)
        )
        res = engine.search("Veggie Salad")
        elapsed = time.perf_counter() - start
        
        errors = []
        warnings = []
        if not isinstance(res, SearchResult):
            errors.append("Output is not a SearchResult.")
        if not res.is_success:
            errors.append("SearchResult reports is_success=False")
        if len(res.candidates) == 0:
            errors.append("Semantic search returned 0 candidates.")
            
        df_index = df.copy()
        df_index.index = [str(i) for i in df_index.index]
        res.attach_metadata(df_index)
        
        for w in res.warnings:
            warnings.append(w)
            
        status = "FAIL" if errors else "PASS"
        self.outcomes.append(StageOutcome(
            stage_name="Semantic Search",
            status=status,
            elapsed_seconds=elapsed,
            input_type="Query String",
            output_type=str(type(res)),
            dimensions_check=f"Candidates: {len(res.candidates)}",
            errors=errors,
            warnings=warnings
        ))
        if status == "FAIL":
            raise RuntimeError(f"Semantic Search integration failed: {errors}")
        return res

    def test_filtering(self, search_res: SearchResult) -> FilteringResult:
        start = time.perf_counter()
        engine = FilteringEngine()
        rule = FilterRule(column="price", operator=FilterOperator.LT, value=30.0, rule_type="HARD")
        res = engine.filter(search_res, [rule])
        elapsed = time.perf_counter() - start
        
        errors = []
        warnings = []
        if not isinstance(res, FilteringResult):
            errors.append("Output is not a FilteringResult.")
        if not res.is_success:
            errors.append("FilteringResult reports is_success=False")
        if len(res.passed_candidates) == 0:
            errors.append("Filtering returned 0 passed candidates.")
            
        for w in res.warnings:
            warnings.append(w)
            
        status = "FAIL" if errors else "PASS"
        self.outcomes.append(StageOutcome(
            stage_name="Filtering",
            status=status,
            elapsed_seconds=elapsed,
            input_type=str(type(search_res)),
            output_type=str(type(res)),
            dimensions_check=f"Passed: {len(res.passed_candidates)} | Rejected: {len(res.rejected_candidates)}",
            errors=errors,
            warnings=warnings
        ))
        if status == "FAIL":
            raise RuntimeError(f"Filtering integration failed: {errors}")
        return res

    def test_ranking(self, filter_res: FilteringResult) -> RankingResult:
        start = time.perf_counter()
        config = RankingConfig(
            factors=[
                RankingFactor(name="Price", weight=1.0, scorer_key="PriceScore")
            ]
        )
        engine = RankingEngine(config)
        res = engine.rank(filter_res)
        elapsed = time.perf_counter() - start
        
        errors = []
        warnings = []
        if not isinstance(res, RankingResult):
            errors.append("Output is not a RankingResult.")
        if len(res.ranked_candidates) == 0:
            errors.append("Ranking returned 0 ranked candidates.")
            
        status = "FAIL" if errors else "PASS"
        self.outcomes.append(StageOutcome(
            stage_name="Ranking",
            status=status,
            elapsed_seconds=elapsed,
            input_type=str(type(filter_res)),
            output_type=str(type(res)),
            dimensions_check=f"Ranked: {len(res.ranked_candidates)}",
            errors=errors,
            warnings=warnings
        ))
        if status == "FAIL":
            raise RuntimeError(f"Ranking integration failed: {errors}")
        return res

    def test_prompt_building(self, ranking_res: RankingResult) -> PromptResult:
        start = time.perf_counter()
        config = PromptConfig()
        builder = PromptBuilder(config)
        builder.use_template("RestaurantChat")
        builder.set_language("English")
        builder.set_persona("FriendlyWaiter")
        
        ctx = PromptContext(
            query="Healthy veggie salad",
            ranked_candidates=ranking_res.ranked_candidates,
            conversation_history=[],
            available_columns=self.available_columns
        )
        builder.set_context(ctx)
        res = builder.build()
        elapsed = time.perf_counter() - start
        
        errors = []
        warnings = []
        if not isinstance(res, PromptResult):
            errors.append("Output is not a PromptResult.")
        if not res.final_prompt:
            errors.append("Prompt text output is empty.")
        for w in res.warnings:
            warnings.append(w)
            
        status = "FAIL" if errors else "PASS"
        self.outcomes.append(StageOutcome(
            stage_name="Prompt Builder",
            status=status,
            elapsed_seconds=elapsed,
            input_type=str(type(ranking_res)),
            output_type=str(type(res)),
            dimensions_check=f"Length: {len(res.final_prompt)} chars",
            errors=errors,
            warnings=warnings
        ))
        if status == "FAIL":
            raise RuntimeError(f"Prompt Builder integration failed: {errors}")
        return res

    def test_generator(self, prompt_res: PromptResult) -> GenerationResponse:
        start = time.perf_counter()
        config = LLMConfig(provider="mock")
        engine = GeneratorEngine(config)
        
        req = GenerationRequest(
            prompt_result=prompt_res,
            generation_config=GenerationConfig()
        )
        res = engine.generate(req)
        elapsed = time.perf_counter() - start
        
        errors = []
        warnings = []
        if not isinstance(res, GenerationResponse):
            errors.append("Output is not a GenerationResponse.")
        if not res.generated_text:
            errors.append("Generated response text is empty.")
        for w in res.warnings:
            warnings.append(w)
            
        status = "FAIL" if errors else "PASS"
        self.outcomes.append(StageOutcome(
            stage_name="Generator",
            status=status,
            elapsed_seconds=elapsed,
            input_type=str(type(prompt_res)),
            output_type=str(type(res)),
            dimensions_check=f"Length: {len(res.generated_text)} chars",
            errors=errors,
            warnings=warnings
        ))
        if status == "FAIL":
            raise RuntimeError(f"Generator integration failed: {errors}")
        return res

    def run_all(self) -> bool:
        """Executes the integration tests sequentially for all DineAI pipeline modules."""
        start_total = time.perf_counter()
        self.outcomes.clear()
        
        success = False
        try:
            # Stage 1: Schema Mapper
            df = self.test_schema_mapper()
            # Stage 2: Validator
            df = self.test_validator(df)
            # Stage 3: Feature Engineering
            df = self.test_feature_engineering(df)
            # Stage 4: Text Builder
            df = self.test_text_builder(df)
            # Stage 5: Capability Detector [NEW]
            _ = self.test_capability_detector(df)
            # Stage 6: Dataset Pipeline [NEW]
            _ = self.test_dataset_pipeline()
            # Stage 7: Embedding Builder
            embed_res = self.test_embedding_generation(df)
            # Stage 8: FAISS Builder
            faiss_res = self.test_faiss_index_build(embed_res)
            # Stage 9: Dataset Manager [NEW]
            dataset = self.test_dataset_manager()
            self.available_columns = dataset.available_columns
            # Stage 10: Semantic Search
            search_res = self.test_semantic_search(dataset.faiss_result, df)
            # Stage 11: Filtering
            filter_res = self.test_filtering(search_res)
            # Stage 12: Ranking
            ranking_res = self.test_ranking(filter_res)
            # Stage 13: Prompt Builder
            prompt_res = self.test_prompt_building(ranking_res)
            # Stage 14: Generator
            _ = self.test_generator(prompt_res)
            
            success = True
        except Exception as e:
            print(f"\n[FATAL] Pipeline integration execution broke: {e}")
            traceback.print_exc()
            success = False
        finally:
            if not success:
                self.keep_temp_dir = True
            
            if self.keep_temp_dir:
                print(f"\n[INFO] Run failed or preserved. Keeping temporary directory at: {self.temp_dir}")
            else:
                try:
                    self.temp_dir_obj.cleanup()
                    print("\n[INFO] Run succeeded. Temporary directory cleaned up successfully.")
                except Exception as e:
                    print(f"\n[WARN] Failed to cleanup temporary directory: {e}")

        total_elapsed = time.perf_counter() - start_total
        
        # Compile and print report
        report = IntegrationReport.generate(self.outcomes, total_elapsed)
        print(report)
        
        return success and (len(self.outcomes) == 14) and all(o.status == "PASS" for o in self.outcomes)


if __name__ == "__main__":
    runner = IntegrationTestRunner()
    all_passed = runner.run_all()
    if all_passed:
        print("[PASS] Integration verification succeeded.")
        sys.exit(0)
    else:
        print("[FAIL] One or more integration stages failed.")
        sys.exit(1)
