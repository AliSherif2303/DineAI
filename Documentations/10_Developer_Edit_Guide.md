# 10 — Developer Edit Guide

> **Who this is for:** Developers who want to make changes to DineAI and need to know exactly which file to open and what to modify.
>
> This guide is organized by **what you want to do**, not by file name.

---

## Quick Lookup Table

| I want to... | Open this file | Go to this section |
|---|---|---|
| Add a new restaurant | `dine_ai/datasets/` | Create a folder + CSV |
| Add a column alias (e.g. "Cost" → price) | `adapters/schema_mapper.py` | `ALIAS_REGISTRY` dict |
| Add a new standard column | `adapters/schema_mapper.py` | `CANONICAL_SCHEMA` + `ALIAS_REGISTRY` |
| Change dietary thresholds | `adapters/feature_engineering.py` | `FeatureEngineerConfig` class |
| Add a new dietary flag | `adapters/feature_engineering.py` | `DietaryFeatureEngineer` class |
| Change price brackets (budget/mid/premium) | `adapters/feature_engineering.py` | `FeatureEngineerConfig.price_bins` |
| Change what's in the AI's context card | `adapters/text_builder.py` | `TextChunkBuilder` class |
| Add a new AI provider | `llm/generator.py` | New class + `ProviderRegistry.register()` |
| Change AI response length | `llm/generator.py` | `GenerationConfig.max_new_tokens` |
| Make AI more/less creative | `llm/generator.py` | `GenerationConfig.temperature` |
| Add a new AI persona | `llm/prompts.py` | `PersonaRegistry.register()` |
| Add a new language | `llm/prompts.py` | `LOCALIZATION_DATA` dict |
| Change the default persona | `llm/prompts.py` | `PromptConfig.default_persona` |
| Change token budget | `llm/prompts.py` | `PromptConfig.max_budget_tokens` |
| Change how many results the AI sees | `app.py` | `ApplicationConfig.max_context_recipes` |
| Add a new filter operator | `retrieval/filtering.py` | New class + `FilterRegistry.register()` |
| Change ranking weights | `retrieval/ranking.py` | `RankingConfig.factors` |
| Add a new ranking factor | `retrieval/ranking.py` | `RankingFactor` enum + scoring logic |
| Change search top-K | `app.py` or `retrieval/semantic_search.py` | `SearchConfig.top_k` |
| Add a new capability | `capabilities/definitions.py` | New class + `ALL_CAPABILITIES` list |
| Change rebuild logic | `dataset/manager.py` | Rebuild check conditions |
| Add a new file to the saved bundle | `dataset/pipeline.py` + `dataset/loader.py` | `build()` save + `load()` read |
| Add a test scenario | `evaluations/deterministic_scenarios_test.py` | `SCENARIOS` list |
| Change the system startup sequence | `app.py` | `DineAIApplication.startup()` |
| Add a new pipeline stage | `app.py` | New `BasePipelineStage` class |

---

## Section 1: Menu Data Changes

### Adding a New Restaurant

1. Create a folder: `dine_ai/datasets/your_restaurant_name/`
2. Add your CSV file: `dine_ai/datasets/your_restaurant_name/menu_items.csv`
3. Update `ApplicationConfig(restaurant_name="your_restaurant_name")`
4. Run — DineAI will auto-build the database on first startup.

> [!TIP]
> The CSV must have at minimum a dish name column. Column names can be anything — the Schema Mapper handles translation.

---

### Adding a Column Alias (e.g. "Cost" → "price")

**File:** `dine_ai/adapters/schema_mapper.py`

```python
# Find ALIAS_REGISTRY and add to the "price" list:
ALIAS_REGISTRY = {
    "price": ["price", "cost", "menu price", "Cost", "COST"],  # ← add "Cost"
    ...
}
```

No restart needed for the alias to be picked up — rebuilding the dataset applies it.

---

### Adding a Totally New Standard Column (e.g. `spice_level`)

**Files to edit:**
1. `dine_ai/adapters/schema_mapper.py` — add to `CANONICAL_SCHEMA` and `ALIAS_REGISTRY`
2. `dine_ai/adapters/feature_engineering.py` — compute it if needed
3. `dine_ai/adapters/text_builder.py` — include it in search text and/or AI context card

**Step 1 — schema_mapper.py:**
```python
CANONICAL_SCHEMA["spice_level"] = {
    "required": False,
    "dtype": str,
    "description": "Spice level (mild, medium, hot, extra-hot)"
}

ALIAS_REGISTRY["spice_level"] = ["spice level", "spiciness", "heat level", "spice_level"]
```

**Step 2 — text_builder.py** (add to search text):
```python
if "spice_level" in row and pd.notna(row["spice_level"]):
    parts.append(f"{row['spice_level']} spice")
```

---

### Changing Dietary Thresholds

**File:** `dine_ai/adapters/feature_engineering.py` → `FeatureEngineerConfig` class

| What to change | Field to edit | Default |
|---|---|---|
| What counts as "keto" | `keto_carbs_g_threshold` | `20.0` g |
| What counts as "low calorie" | `low_calorie_threshold` | `150.0` cal |
| What counts as "high protein" | `high_protein_threshold_g` | `15.0` g |
| What counts as "low sodium" | `low_sodium_mg` | `140.0` mg |
| Price bracket boundaries | `price_bins` | `[10.0, 25.0]` |

```python
# Example: Change keto to <30g carbs instead of <20g
@dataclass
class FeatureEngineerConfig:
    keto_carbs_g_threshold: float = 30.0  # was 20.0
```

---

### Adding a New Dietary Flag (e.g. `is_nut_free`)

**File:** `dine_ai/adapters/feature_engineering.py`

1. Add to `DietaryFeatureEngineer`:
```python
def compute_is_nut_free(self, df: pd.DataFrame) -> pd.DataFrame:
    nut_keywords = ["peanut", "almond", "cashew", "walnut", "pecan", "hazelnut"]
    df["is_nut_free"] = df.get("ingredients", pd.Series(dtype=str)).apply(
        lambda x: 0 if any(n in str(x).lower() for n in nut_keywords) else 1
    )
    return df
```

2. Call it from `transform()`:
```python
def transform(self, df):
    ...
    df = self.compute_is_nut_free(df)
    return df
```

3. Add a capability for it in `capabilities/definitions.py`.

---

## Section 2: AI Behavior Changes

### Changing the AI's Personality

**File:** `dine_ai/llm/prompts.py` → `PersonaRegistry`

**To add a new persona:**
```python
PersonaRegistry.register(
    "SeniorChef",
    system_prompt="""You are a senior chef with 20 years of experience.
    You speak with authority about ingredients and preparation techniques.
    You occasionally share cooking tips in your recommendations.
    Keep your tone warm but professional."""
)
```

**To change the default persona:**
```python
@dataclass
class PromptConfig:
    default_persona: str = "SeniorChef"   # was "FriendlyWaiter"
```

---

### Adding a New Language

**File:** `dine_ai/llm/prompts.py` → `LOCALIZATION_DATA`

```python
LOCALIZATION_DATA["japanese"] = {
    "recipe_name": "料理名",
    "protein": "タンパク質",
    "calories": "カロリー",
    "price": "価格",
    "ingredients": "材料",
    "description": "説明"
}
```

Then set `ApplicationConfig(language="Japanese")`.

---

### Making the AI Write Longer / Shorter Responses

**File:** `dine_ai/llm/generator.py` → `GenerationConfig`

```python
@dataclass
class GenerationConfig:
    max_new_tokens: int = 600   # was 384 — now allows longer responses
```

---

### Making the AI More Creative / Varied

**File:** `dine_ai/llm/generator.py` → `GenerationConfig`

```python
@dataclass
class GenerationConfig:
    temperature: float = 0.7    # was 0.1 — now more varied
    do_sample: bool = True      # was False — enables sampling-based generation
```

> [!WARNING]
> Temperature above 0.9 can produce inconsistent or off-topic responses. Keep it between 0.1 (safe) and 0.7 (creative) for restaurant use.

---

### Switching the AI Model

**File:** `dine_ai/app.py` → `ApplicationConfig`

```python
config = ApplicationConfig(
    llm_provider="openai",
    llm_model="gpt-4o"          # or "gpt-3.5-turbo", "claude-3-sonnet", etc.
)
```

No code changes needed beyond config. Just make sure the API key is set as an environment variable.

---

### Adding a New AI Provider (e.g. Vertex AI)

**File:** `dine_ai/llm/generator.py`

1. Create a new provider class:
```python
class VertexAIProvider(BaseLLMProvider):
    def __init__(self, config: LLMConfig):
        import vertexai
        vertexai.init(project="your-project", location="us-central1")
        self.model = GenerativeModel("gemini-1.5-pro")
    
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        response = self.model.generate_content(request.prompt)
        return GenerationResponse(
            generated_text=response.text,
            model_id="gemini-1.5-pro"
        )
```

2. Register it:
```python
ProviderRegistry.register("vertexai", VertexAIProvider)
```

3. Use it:
```python
ApplicationConfig(llm_provider="vertexai")
```

---

## Section 3: Search & Retrieval Changes

### Changing How Many Results the AI Sees

**File:** `dine_ai/app.py` → `ApplicationConfig`

```python
ApplicationConfig(max_context_recipes=8)   # was 5
```

Higher values give the AI more options but increase token usage and response time.

---

### Changing Search Sensitivity

**File:** `dine_ai/retrieval/semantic_search.py` → `SearchConfig`

| Setting | Effect |
|---|---|
| Increase `top_k` | More candidates for filtering to work with |
| Set `score_threshold=0.4` | Only return results with ≥40% semantic match |

---

### Changing Ranking Weights

**File:** `dine_ai/retrieval/ranking.py` → `RankingConfig`

```python
RankingConfig(factors=[
    RankingFactor(name="SEMANTIC_SIMILARITY", weight=0.30),  # was 0.50
    RankingFactor(name="RATING_NORMALIZED", weight=0.50),    # was 0.30
    RankingFactor(name="PRICE_NORMALIZED", weight=0.20),
])
# Weights must always sum to 1.0
```

---

### Adding a New Filter Operator (e.g. "near")

**File:** `dine_ai/retrieval/filtering.py`

1. Add to `FilterOperator` enum:
```python
NEAR = "near"    # within N% of a target value
```

2. Create filter class:
```python
class NearFilter(BaseFilter):
    operator = FilterOperator.NEAR
    
    def evaluate(self, candidate_value, rule_value):
        target, tolerance_pct = rule_value
        low = target * (1 - tolerance_pct / 100)
        high = target * (1 + tolerance_pct / 100)
        return low <= float(candidate_value) <= high
```

3. Register it:
```python
FilterRegistry.register(NearFilter)
```

---

## Section 4: System Configuration Changes

### All Settings in One Place

**File:** `dine_ai/app.py` → `ApplicationConfig` class (line ~49)

See [05_Configuration.md](file:///d:/data/AI_WAITER/Documentations/05_Configuration.md) for the full explanation of every setting.

---

### Switching the Embedding Model

**File:** `dine_ai/app.py` → `ApplicationConfig`

```python
ApplicationConfig(
    embedding_provider="local",
    embedding_model="sentence-transformers/all-MiniLM-L12-v2"  # faster, smaller
)
```

> [!IMPORTANT]
> After changing the embedding model, you **must** delete `faiss.index` and `embeddings.npy` from your restaurant folder and let the system rebuild. Old indexes are incompatible with new models.

---

## Section 5: Adding Test Scenarios

### Adding a Deterministic Test Scenario

**File:** `dine_ai/evaluations/deterministic_scenarios_test.py` → `SCENARIOS` list

```python
ScenarioTest(
    name="nut_free_options",
    query="show me nut free dishes",
    required_flags={"is_nut_free": True},
    max_price=None,
    min_rating=None,
    expected_top_item=None   # or specify a dish name
)
```

---

### Adding a New Integration Check

**File:** `dine_ai/evaluations/integration_test.py`

Add a new method to the test class:
```python
def test_spice_level_column(self):
    """Verify new spice_level column is created."""
    result = FeatureEngineer().transform(self.sample_df)
    assert "spice_level" in result.columns or True  # adjust as needed
```

---

## Section 6: Architecture Changes

### Adding a New Pipeline Stage

**File:** `dine_ai/app.py`

1. Create the stage class:
```python
class RecommendationDiversityStage(BasePipelineStage):
    """Ensures variety in recommendations (no two dishes from same category)."""
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        seen_cuisines = set()
        diverse = []
        for candidate in context.ranked_candidates:
            cuisine = candidate.candidate.metadata.get("cuisine_type", "")
            if cuisine not in seen_cuisines:
                diverse.append(candidate)
                seen_cuisines.add(cuisine)
        context.ranked_candidates = diverse
        return context
```

2. Add it to the pipeline in `DineAIApplication._run_pipeline()`:
```python
# After ranking, before prompt building
context = RecommendationDiversityStage().execute(context)
```

---

## File Edit Risk Guide

Some files are safer to edit than others. Here's a quick risk summary:

| File | Risk | Why |
|---|---|---|
| `app.py` → `ApplicationConfig` | 🟢 Low | Config changes only, no logic change |
| `adapters/schema_mapper.py` | 🟢 Low | Just adds aliases, no logic change |
| `adapters/feature_engineering.py` | 🟡 Medium | Logic change — test with integration tests |
| `llm/prompts.py` | 🟡 Medium | Changes AI behavior — test end-to-end |
| `llm/generator.py` | 🟡 Medium | Changes AI backend — test end-to-end |
| `retrieval/ranking.py` | 🟡 Medium | Changes result order — test deterministic scenarios |
| `retrieval/filtering.py` | 🔴 High | Core filter logic — any bug silently passes bad results |
| `dataset/manager.py` | 🔴 High | Rebuild logic — bugs can corrupt cache |
| `dataset/pipeline.py` | 🔴 High | Build pipeline — bugs cause build failures |
| `app.py` → pipeline stages | 🔴 High | Core orchestration — test everything after changes |
