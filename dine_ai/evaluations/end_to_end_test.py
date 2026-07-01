"""
end_to_end_test.py — DineAI End-to-End Test Suite
===================================================
Simulates real user queries traversing the full DineAI application and verifies
response completeness, candidate generation, and system latency.
Conforms strictly to SOLID principles, clean reports, and offline testing patterns.
"""

import os
import sys
import time
import tempfile
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# DineAI Modules Imports
from dine_ai.app import DineAIApplication, ApplicationConfig, BasePipelineStage, PipelineContext
from dine_ai.llm.generator import GenerationResponse


# ============================================================================
# INTERCEPTOR & OUTCOME DTOs
# ============================================================================

class InterceptStage(BasePipelineStage):
    """Pipeline stage to intercept pipeline execution context for verification."""

    def __init__(self):
        super().__init__()
        self.last_context: Optional[PipelineContext] = None

    def execute(self, context: PipelineContext) -> PipelineContext:
        self.last_context = context
        return context

    def validate_input(self, context: PipelineContext) -> None:
        pass

    def validate_output(self, context: PipelineContext) -> None:
        pass


@dataclass
class QueryOutcome:
    """Represents the execution metrics of a single end-to-end user query simulation."""
    query: str
    status: str  # "PASS" | "FAIL"
    latency: float
    retrieved_candidates_count: int
    top_ranked_candidate_name: str
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ============================================================================
# END-TO-END REPORT & METRICS FORMATTER
# ============================================================================

class EndToEndReport:
    """Compiles and formats query outcomes into an enterprise-style dashboard."""

    @staticmethod
    def generate(outcomes: List[QueryOutcome], total_elapsed: float) -> str:
        width = 80
        sep = "=" * width
        thin = "-" * width

        # Calculations
        passed_count = sum(1 for o in outcomes if o.status == "PASS")
        failed_count = len(outcomes) - passed_count
        success_pct = (passed_count / len(outcomes)) * 100 if outcomes else 0.0

        latencies = [o.latency for o in outcomes]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        fastest = min(latencies) if latencies else 0.0
        slowest = max(latencies) if latencies else 0.0

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
        lines.append("DINEAI END-TO-END TEST REPORT".center(width))
        lines.append(sep)

        lines.append(f" Queries Tested             : {len(outcomes)}")
        lines.append(f" Queries Passed             : {passed_count}")
        lines.append(f" Queries Failed             : {failed_count}")
        lines.append(f" Success Rate               : {success_pct:.2f}%")
        lines.append(f" Average Latency            : {avg_latency:.4f}s")
        lines.append(f" Fastest Query              : {fastest:.4f}s")
        lines.append(f" Slowest Query              : {slowest:.4f}s")
        lines.append(thin)
        lines.append("Individual Query Results".center(width))
        lines.append(thin)

        for o in outcomes:
            lines.append(f" [{o.status}] {o.query}")
            lines.append(f"  * Latency: {o.latency:.4f}s")
            lines.append(f"  * Retrieved Candidates: {o.retrieved_candidates_count}")
            lines.append(f"  * Top Ranked Candidate: {o.top_ranked_candidate_name}")
            if o.errors:
                for err in o.errors:
                    lines.append(f"    [ERR] {err}")
            if o.warnings:
                for warn in o.warnings:
                    lines.append(f"    [WARN] {warn}")
            lines.append(thin)

        lines.append(f" Overall End-to-End Score   : {success_pct:.1f}%")
        lines.append(f" Overall Health Rating       : {health_rating}")
        lines.append(sep)

        return "\n".join(lines)


# ============================================================================
# END-TO-END TESTS RUNNER
# ============================================================================

class EndToEndTestRunner:
    """Orchestrates end-to-end verification of user queries against the full DineAI Application."""

    def __init__(self):
        self.queries = [
            "High protein breakfast",
            "Low calorie lunch",
            "Cheap dinner",
            "Italian pasta",
            "Vegan breakfast",
            "Halal chicken",
            "Keto meal",
            "Low sodium soup",
            "High fiber snack",
            "Gluten free dessert"
        ]
        self.outcomes: List[QueryOutcome] = []
        
        # Configure restaurant name & dataset directory via env vars or fall back to temporary generation
        env_restaurant = os.environ.get("DINEAI_TEST_RESTAURANT")
        env_dataset_dir = os.environ.get("DINEAI_TEST_DATASET_DIR")
        
        self.temp_dir_obj = None
        self.keep_temp_dir = False
        
        if env_restaurant and env_dataset_dir:
            self.restaurant_name = env_restaurant
            self.dataset_dir = env_dataset_dir
        else:
            self.temp_dir_obj = tempfile.TemporaryDirectory()
            self.dataset_dir = self.temp_dir_obj.name
            self.restaurant_name = "restaurant_temp"
            
            # Populate temp restaurant CSV items to test Auto-build via DatasetManager
            restaurant_path = os.path.join(self.dataset_dir, self.restaurant_name)
            os.makedirs(restaurant_path, exist_ok=True)
            
            csv_path = os.path.join(restaurant_path, "recipes.csv")
            csv_content = (
                "Recipe Name,Menu Price,Protein (g),Calories,Cuisine Type,Dish Type,Diet Labels,Ingredients,Rating,Popularity\n"
                "Truffle Burger,25.0,45.0,750.0,American,main_course,High-Protein,\"beef, bun, truffle sauce\",4.8,95.0\n"
                "Veggie Salad,12.0,3.0,120.0,Mediterranean,salad,Vegan,\"lettuce, cucumber, tomato\",4.5,80.0\n"
                "Grilled Salmon,30.0,40.0,500.0,Mediterranean,main_course,High-Protein,\"salmon, lemon, herbs\",4.9,90.0\n"
                "Margherita Pizza,15.0,15.0,600.0,Italian,main_course,Vegetarian,\"dough, tomato, mozzarella\",4.6,85.0\n"
                "Pasta Primavera,18.0,12.0,450.0,Italian,main_course,Vegan,\"pasta, vegetables, olive oil\",4.4,75.0\n"
            )
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(csv_content)
                
            with open(os.path.join(restaurant_path, "config.json"), "w", encoding="utf-8") as f:
                json.dump({"restaurant_name": self.restaurant_name}, f)

        # Configure and initialize the application in offline testing mode
        self.config = ApplicationConfig(
            restaurant_name=self.restaurant_name,
            dataset_directory=self.dataset_dir,
            embedding_provider="mock",
            llm_provider="mock",
            max_context_recipes=3,
            enable_reports=False
        )
        self.app = DineAIApplication(self.config)
        self.app.initialize()

        # Register intercept stage to inspect pipeline context
        self.intercept = InterceptStage()
        self.app.pipeline_manager.register_stage("Intercept", self.intercept)

    def run_query(self, query: str) -> QueryOutcome:
        start_time = time.perf_counter()
        session_id = f"test_session_{hash(query)}"
        
        errors = []
        warnings = []
        retrieved_count = 0
        top_ranked_candidate = "None"
        
        try:
            # Execute chat turn
            response = self.app.chat(session_id=session_id, query=query)
            latency = time.perf_counter() - start_time
            
            # Fetch context from the intercept stage
            ctx = self.intercept.last_context
            if ctx is None:
                errors.append("Pipeline did not run or interceptor failed to capture context.")
            else:
                # 1. Verify semantic search candidates
                if ctx.candidates is None or len(ctx.candidates) == 0:
                    errors.append("Semantic search returned 0 candidates.")
                else:
                    retrieved_count = len(ctx.candidates)
                
                # 2. Verify filtering
                if ctx.filtered_candidates is None:
                    errors.append("Filtering stage failed to populate filtered candidates.")
                
                # 3. Verify ranking
                if ctx.ranked_candidates is None or len(ctx.ranked_candidates) == 0:
                    errors.append("Ranking stage failed to populate ranked candidates.")
                else:
                    rc = ctx.ranked_candidates[0]
                    candidate_obj = rc.candidate
                    if hasattr(candidate_obj, "candidate"):
                        inner_candidate = candidate_obj.candidate
                    else:
                        inner_candidate = candidate_obj
                        
                    if hasattr(inner_candidate, "metadata") and isinstance(inner_candidate.metadata, dict):
                        top_ranked_candidate = inner_candidate.metadata.get("recipe_name") or inner_candidate.metadata.get("name", "Unknown")
                    elif hasattr(inner_candidate, "recipe_name"):
                        top_ranked_candidate = inner_candidate.recipe_name
                    elif hasattr(inner_candidate, "name"):
                        top_ranked_candidate = inner_candidate.name
                    elif isinstance(inner_candidate, dict):
                        top_ranked_candidate = inner_candidate.get("recipe_name") or inner_candidate.get("name", "Unknown")
                    else:
                        top_ranked_candidate = "Unknown"
                    
                # 4. Verify prompt generation
                if ctx.prompt_result is None or not ctx.prompt_result.final_prompt:
                    errors.append("PromptBuilder failed to generate prompt text.")
                    
                # 5. Verify response
                if not response.generated_text:
                    errors.append("Generator returned an empty response.")
                    
                if ctx.warnings:
                    warnings.extend(ctx.warnings)
        except Exception as e:
            latency = time.perf_counter() - start_time
            errors.append(f"Execution crashed: {str(e)}")
            
        status = "FAIL" if errors else "PASS"
        return QueryOutcome(
            query=query,
            status=status,
            latency=latency,
            retrieved_candidates_count=retrieved_count,
            top_ranked_candidate_name=top_ranked_candidate,
            warnings=warnings,
            errors=errors
        )

    def run_all(self) -> bool:
        start_total = time.perf_counter()
        self.outcomes.clear()
        
        success = False
        try:
            for q in self.queries:
                outcome = self.run_query(q)
                self.outcomes.append(outcome)
            
            success = all(o.status == "PASS" for o in self.outcomes)
        except Exception as e:
            print(f"\n[FATAL] End-to-end test execution crashed: {e}")
            success = False
        finally:
            if not success:
                self.keep_temp_dir = True
            
            if self.temp_dir_obj:
                if self.keep_temp_dir:
                    print(f"\n[INFO] End-to-end run failed or preserved. Keeping temporary directory at: {self.dataset_dir}")
                else:
                    try:
                        self.temp_dir_obj.cleanup()
                        print("\n[INFO] End-to-end run succeeded. Temporary directory cleaned up successfully.")
                    except Exception as e:
                        print(f"\n[WARN] Failed to cleanup temporary directory: {e}")

        total_elapsed = time.perf_counter() - start_total
        
        # Compile and print report
        report = EndToEndReport.generate(self.outcomes, total_elapsed)
        print(report)

        return success


if __name__ == "__main__":
    runner = EndToEndTestRunner()
    all_passed = runner.run_all()
    if all_passed:
        print("[PASS] End-to-End verification succeeded.")
        sys.exit(0)
    else:
        print("[FAIL] One or more End-to-End query verifications failed.")
        sys.exit(1)
