# 09 — Adding New Features (Extensibility Guide)

DineAI is designed so you can add new capabilities without touching the core pipeline. This document explains how.

The key concept: **registries**. Almost everything in DineAI that can vary (AI providers, filter types, ranking factors, capabilities) is registered in a list. To add something new, you register it and the system picks it up automatically.

---

## Adding a New AI Provider (e.g. Claude, Vertex AI)

**File to edit:** `dine_ai/llm/generator.py`

**Step 1 — Create a new provider class:**
```python
class ClaudeProvider(BaseLLMProvider):
    """Anthropic Claude integration."""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            messages=[{"role": "user", "content": request.prompt}]
        )
        return GenerationResponse(
            generated_text=response.content[0].text,
            model_id="claude-3-sonnet"
        )
```

**Step 2 — Register it:**
```python
# In the ProviderRegistry class
ProviderRegistry.register("claude", ClaudeProvider)
```

**Step 3 — Use it:**
```python
ApplicationConfig(llm_provider="claude")
```

---

## Adding a New Filter Operator (e.g. "starts_with")

**File to edit:** `dine_ai/retrieval/filtering.py`

**Step 1 — Create the filter class:**
```python
class StartsWithFilter(BaseFilter):
    """Checks if a text field starts with a given prefix."""
    
    operator = "starts_with"
    
    def evaluate(self, candidate_value: Any, rule_value: Any) -> bool:
        if not isinstance(candidate_value, str):
            return False
        return str(candidate_value).lower().startswith(str(rule_value).lower())
```

**Step 2 — Register it:**
```python
FilterRegistry.register(StartsWithFilter)
```

**Step 3 — Use it in queries:**
```python
FilterRule(field="recipe_name", operator="starts_with", value="Grilled")
```

---

## Adding a New Ranking Factor (e.g. "freshness score")

**File to edit:** `dine_ai/retrieval/ranking.py`

**Step 1 — Add the factor to `RankingFactor`:**
```python
class RankingFactor(Enum):
    SEMANTIC_SIMILARITY = "semantic_similarity"
    PRICE_NORMALIZED = "price_normalized"
    RATING_NORMALIZED = "rating_normalized"
    FRESHNESS = "freshness"   # ← add this
```

**Step 2 — Add the scoring logic in `RankingEngine._compute_score()`:**
```python
elif factor == RankingFactor.FRESHNESS:
    freshness = metadata.get("days_since_added", 0)
    score = max(0, 1 - freshness / 30)  # higher score = added recently
```

**Step 3 — Enable it in your config:**
```python
RankingConfig(factors=[RankingFactor.SEMANTIC_SIMILARITY, RankingFactor.FRESHNESS])
```

---

## Adding a New Dietary Capability (e.g. "nut-free")

**File to edit:** `dine_ai/capabilities/definitions.py`

**Step 1 — Add the capability definition:**
```python
class NutFreeCapability(BaseCapability):
    name = "nut_free_filtering"
    description = "Enables filtering by nut-free status"
    required_columns = ["is_nut_free"]
    
    def check(self, df: pd.DataFrame) -> bool:
        return "is_nut_free" in df.columns
```

**Step 2 — Register it:**
```python
# In definitions.py
ALL_CAPABILITIES = [
    NutritionCapability,
    BudgetCapability,
    RatingCapability,
    DietaryCapability,
    CuisineCapability,
    NutFreeCapability,   # ← add this
]
```

**Step 3 — Add the detection logic in feature engineering:**
```python
# In dine_ai/adapters/feature_engineering.py
df["is_nut_free"] = df["ingredients"].apply(
    lambda x: 1 if "nut" not in str(x).lower() else 0
)
```

---

## Adding a New Computed Feature (e.g. "fiber density")

**File to edit:** `dine_ai/adapters/feature_engineering.py`

**Step 1 — Find the `FeatureEngineer` class and add a method:**
```python
def compute_fiber_density(self, df: pd.DataFrame) -> pd.DataFrame:
    """Compute fiber per 100g of food."""
    if "fiber_g" in df.columns and "weight_g" in df.columns:
        df["fiber_density"] = df["fiber_g"] / (df["weight_g"] / 100)
    return df
```

**Step 2 — Call it in the main `transform()` method:**
```python
def transform(self, df: pd.DataFrame) -> pd.DataFrame:
    df = self.compute_protein_density(df)
    df = self.compute_dietary_flags(df)
    df = self.compute_fiber_density(df)    # ← add this line
    return df
```

---

## Adding a New Persona (AI Personality Style)

**File to edit:** `dine_ai/llm/prompts.py`

**Step 1 — Find `PersonaRegistry` and add a new persona:**
```python
PersonaRegistry.register(
    name="health_coach",
    system_prompt="""You are a certified nutritionist and health coach.
    You give menu recommendations focused on health goals.
    You mention calorie counts, macros, and nutritional benefits in every response.
    You are enthusiastic but not pushy."""
)
```

**Step 2 — Use it when building prompts:**
```python
PromptConfig(persona="health_coach")
```

---

## Adding a New Column Alias (e.g. "Cost" → "price")

**File to edit:** `dine_ai/adapters/schema_mapper.py`

**Where:** Find the `ALIAS_REGISTRY` dictionary and add your alias:

```python
ALIAS_REGISTRY = {
    "price": ["price", "cost", "menu price", "item price", "Cost", "PRICE"],
    #                                                          ^^^^^ add here
    ...
}
```

This is the simplest type of extension — just add a string to a list.

---

## Testing Your Extension

After any extension:

1. Run integration tests to verify the wiring:
   ```bash
   python dine_ai/evaluations/integration_test.py
   ```

2. Run a quick self-test:
   ```bash
   python -m dine_ai.app --test
   ```

3. For AI provider changes, run end-to-end tests:
   ```bash
   python dine_ai/evaluations/end_to_end_test.py
   ```
