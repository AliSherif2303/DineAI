import os
import json
import pandas as pd
from dine_ai.dataset.pipeline import DatasetPipeline, MockEmbeddingProvider
from dine_ai.dataset.manager import DatasetManager
from dine_ai.app import DineAIApplication, ApplicationConfig, PipelineContext
from dine_ai.retrieval.filtering import FilterRule, FilterOperator

def run_scenarios():
    print("=" * 80)
    print(" DETERMINISTIC CAPABILITY RETRIEVAL & RANKING SCENARIOS ".center(80, "="))
    print("=" * 80)

    # 1. Setup deterministic dataset files
    rest_dir = os.path.abspath("dine_ai/datasets/restaurant_deterministic")
    os.makedirs(rest_dir, exist_ok=True)
    
    success = False
    try:
        with open(os.path.join(rest_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"restaurant_name": "restaurant_deterministic"}, f)
            
        csv_path = os.path.join(rest_dir, "recipes.csv")
        content = (
            "recipe_name,price,protein_g_per_serving,calories_per_serving,cuisine_type,item_type,diet_labels,ingredients\n"
            "Chicken Burger,22.0,40.0,450.0,American,main_course,halal,\"chicken, bun, cheese\"\n"
            "Protein Shake,12.0,55.0,250.0,Drink,Drink,vegetarian,\"protein powder, milk, banana\"\n"
            "Pasta Primavera,8.0,12.0,350.0,Italian,main_course,vegan,\"pasta, vegetables, olive oil\"\n"
            "Greek Salad,9.5,18.0,200.0,Mediterranean,salad,vegetarian,\"lettuce, cucumber, feta, olives\"\n"
            "Diet Soda,2.5,0.0,0.0,Drink,Drink,vegan;keto,\"carbonated water, sweetener\"\n"
        )
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        # 2. Build via DatasetManager
        pipeline = DatasetPipeline(embedding_provider=MockEmbeddingProvider())
        manager = DatasetManager(pipeline=pipeline)
        manager.rebuild_all("restaurant_deterministic")
        
        # 3. Initialize app
        app = DineAIApplication(ApplicationConfig(
            restaurant_name="restaurant_deterministic",
            embedding_provider="mock",
            llm_provider="mock",
            max_context_recipes=5
        ))
        app.initialize()

        scenarios = [
            {
                "name": "high protein under 150",
                "query": "high protein under 150",
                "rules": [
                    FilterRule(column="protein_g_per_serving", operator=FilterOperator.GTE, value=30.0),
                    FilterRule(column="price", operator=FilterOperator.LTE, value=150.0)
                ],
                "expected_top": "Protein Shake"
            },
            {
                "name": "Italian pasta",
                "query": "Italian pasta",
                "rules": [],
                "expected_top": "Pasta Primavera"
            },
            {
                "name": "cheap dinner",
                "query": "cheap dinner",
                "rules": [
                    FilterRule(column="price", operator=FilterOperator.LTE, value=10.0)
                ],
                "expected_top": "Pasta Primavera"
            },
            {
                "name": "healthy salad",
                "query": "healthy salad",
                "rules": [],
                "expected_top": "Greek Salad"
            },
            {
                "name": "vegan breakfast",
                "query": "vegan breakfast",
                "rules": [
                    FilterRule(column="is_vegan", operator=FilterOperator.EQ, value=1.0)
                ],
                "expected_top": "Pasta Primavera"
            },
            {
                "name": "diet drink",
                "query": "diet drink",
                "rules": [
                    FilterRule(column="item_type", operator=FilterOperator.EXACT, value="Drink")
                ],
                "expected_top": "Diet Soda"
            }
        ]

        def get_name(c) -> str:
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

        for sc in scenarios:
            ctx = PipelineContext(
                original_query=sc["query"],
                conversation_history=[]
            )
            # Apply filtering rules via stage replacement
            from dine_ai.app import FilteringStage
            app.pipeline_manager.replace_stage("Filtering", FilteringStage(sc["rules"]))
                
            ctx = app.pipeline_manager.execute(ctx)
            
            # Output Trace & Debug Report
            print(ctx.enterprise_debug_report())
            
            # Verify correctness
            top_name = "None"
            if ctx.ranked_candidates:
                top_name = get_name(ctx.ranked_candidates[0])
                
            print(f"Expected Top Candidate: {sc['expected_top']}")
            print(f"Actual Top Candidate:   {top_name}")
            assert top_name == sc["expected_top"], f"Scenario failed! Expected {sc['expected_top']}, got {top_name}"
            print(f"[PASS] Scenario '{sc['name']}' resolved correctly.")
            print("=" * 80)

        success = True
        print("\n[SUCCESS] All deterministic recommendation scenarios validated successfully!")
    finally:
        if success:
            import shutil
            try:
                shutil.rmtree(rest_dir)
                print(f"[INFO] Cleanup succeeded. Removed deterministic restaurant directory: {rest_dir}")
            except Exception as e:
                print(f"[WARN] Failed to clean up deterministic directory: {e}")
        else:
            print(f"[INFO] Run failed or preserved. Keeping deterministic restaurant directory at: {rest_dir}")

if __name__ == "__main__":
    run_scenarios()
