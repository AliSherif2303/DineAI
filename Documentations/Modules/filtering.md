# Module: `filtering.py` — Rules Engine

**Location:** `dine_ai/retrieval/filtering.py`

---

## What This Does (Plain English)

After the semantic search finds the most relevant menu items, the Filtering Engine applies hard rules. Think of it like a bouncer at a door — if an item doesn't meet a rule, it gets kicked out and never shown to the customer.

For example: a customer asks for "vegan options under $10." Semantic search might return 10 candidates. Filtering removes any item that costs more than $10 or isn't vegan. The remaining items pass through to the ranking stage.

---

## Two Types of Filters

### Hard Rules (strict pass/fail)
Items either pass or fail. If they fail, they're gone. Example: `price <= 10.0` — any item costing more is excluded.

### Soft Rules (scoring adjustment)
Items don't get excluded, but their final score is affected. Example: a "prefer Italian cuisine" soft rule doesn't remove Thai food, but it boosts Italian items' scores.

---

## All Supported Operators

| Category | Operator | Example Use |
|---|---|---|
| **Numeric** | `==`, `!=`, `>`, `>=`, `<`, `<=` | `price <= 15.0` |
| **Numeric range** | `between`, `outside` | `calories between [300, 600]` |
| **Approximate** | `approximately_equal` | `price approximately_equal 20` |
| **Text** | `contains`, `icontains` | `ingredients contains "chicken"` |
| **Text** | `startswith`, `endswith`, `exact` | `cuisine_type exact "Italian"` |
| **Text** | `regex` | `recipe_name regex "^Grilled.*"` |
| **List** | `contains_any`, `contains_all` | `tags contains_any ["vegan", "keto"]` |
| **List** | `overlap`, `subset`, `superset` | |
| **Existence** | `exists`, `not_exists` | Check if a field is present |
| **Null** | `is_null`, `is_not_null` | Check if a field has a value |
| **Boolean** | `is_true`, `is_false` | `is_vegan is_true` |

---

## FilterGroup (Combining Rules with AND / OR)

You can combine multiple rules using `FilterGroup`:

```python
# Items must be (vegan OR vegetarian) AND (price under $15)
FilterGroup(
    operator="AND",
    groups=[
        FilterGroup(operator="OR", rules=[
            FilterRule(field="is_vegan", operator="is_true"),
            FilterRule(field="is_vegetarian", operator="is_true")
        ]),
        FilterRule(field="price", operator="lte", value=15.0)
    ]
)
```

---

## Computed Metrics

The filter engine can evaluate "virtual columns" that don't actually exist in the data but can be computed on the fly:

| Virtual Field | What it computes |
|---|---|
| `price_per_100cal` | price / (calories / 100) |
| `protein_per_dollar` | protein_g / price |
| `calorie_to_price_ratio` | calories / price |

---

## How to Edit This Module

**To add a new filter operator** (e.g. `"between_pct"` — within X% of a value):
1. Add the new operator to the `FilterOperator` enum
2. Create a class inheriting `BaseFilter`:
   ```python
   class BetweenPercentFilter(BaseFilter):
       operator = FilterOperator.BETWEEN_PCT
       def evaluate(self, candidate_value, rule_value):
           center, pct = rule_value
           low = center * (1 - pct/100)
           high = center * (1 + pct/100)
           return low <= candidate_value <= high
   ```
3. Register it: `FilterRegistry.register(BetweenPercentFilter)`

**To change what happens when no items pass filtering:**
1. Find `FilteringEngine.filter()` in `filtering.py`
2. Look for the fallback logic when `passed_candidates` is empty
