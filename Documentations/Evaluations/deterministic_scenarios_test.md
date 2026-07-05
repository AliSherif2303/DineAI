# Evaluation: Deterministic Scenario Tests

**File:** `dine_ai/evaluations/deterministic_scenarios_test.py`

---

## What This Test Does (Plain English)

A "deterministic scenario" means: given a specific, unambiguous query with a known correct answer, does the system return the right result?

These tests don't check AI creativity or response quality. They check whether the retrieval, filtering, and ranking pipeline is logically correct.

**Example scenarios:**
- *"highest protein item"* → the result must have the highest protein value in the menu
- *"vegan items only"* → every returned item must have `is_vegan = 1`
- *"cheapest option"* → the top result must have the lowest price
- *"Italian cuisine under $20"* → all results must have `cuisine_type = "Italian"` and `price < 20`

---

## Why These Tests Matter

These are the hardest tests to pass because they're exact. If your filtering logic has a bug, this test will catch it immediately. If your ranking weights are wrong and the cheapest item isn't appearing first when asked for "cheapest", this test reveals that.

---

## How to Run

```bash
python dine_ai/evaluations/deterministic_scenarios_test.py
```

Expected output:
```
[PASS] Scenario 1: highest_protein_item — Top result: Grilled Chicken (45g protein)
[PASS] Scenario 2: vegan_only — All 3 results are vegan
[PASS] Scenario 3: cheapest_option — Top result: $8.50 Garden Salad
...
All 12 scenarios passed.
```

---

## When to Run

- After changing `filtering.py`
- After changing `ranking.py`
- After changing `feature_engineering.py` thresholds
- After adding a new filter capability

---

## How to Add a New Scenario

Find the `SCENARIOS` list in the test file and add:

```python
ScenarioTest(
    name="gluten_free_options",
    query="show me gluten free dishes",
    required_flags={"is_gluten_free": True},
    max_price=None,
    min_rating=None
)
```
