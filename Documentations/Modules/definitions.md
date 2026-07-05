# Module: `definitions.py` — Capability Definitions

**Location:** `dine_ai/capabilities/definitions.py`

---

## What This Does (Plain English)

This file is simply a catalog of every "capability" that DineAI knows how to support. Each capability is a feature that might or might not be available depending on the restaurant's menu data.

Think of it like a checklist of superpowers — the system checks each one to see if it applies to the current restaurant.

---

## Built-In Capabilities

| Capability Class | Name | What it means | Required data |
|---|---|---|---|
| `NutritionCapability` | `nutrition_filtering` | Can filter by calories, protein, fat | Any nutrition column |
| `BudgetCapability` | `budget_filtering` | Can filter by price | `price` column |
| `RatingCapability` | `rating_sorting` | Can sort by rating | `rating` column |
| `DietaryCapability` | `dietary_filtering` | Can filter vegan/halal/keto etc. | Any `is_*` dietary column |
| `CuisineCapability` | `cuisine_filtering` | Can filter by cuisine type | `cuisine_type` column |

---

## The `ALL_CAPABILITIES` List

At the bottom of `definitions.py`:

```python
ALL_CAPABILITIES = [
    NutritionCapability,
    BudgetCapability,
    RatingCapability,
    DietaryCapability,
    CuisineCapability,
]
```

This is what the capability detector iterates over. **To add a new capability, add it to this list.**

---

## How to Edit This Module

**To add a new capability** (e.g. "delivery time filtering"):

```python
class DeliveryTimeCapability(BaseCapability):
    name = "delivery_filtering"
    description = "Enables filtering by estimated delivery time"
    required_columns = ["estimated_delivery_minutes"]
    
    def check(self, df: pd.DataFrame) -> bool:
        return "estimated_delivery_minutes" in df.columns

# Then add to the list:
ALL_CAPABILITIES = [
    ...,
    DeliveryTimeCapability,
]
```

**To disable a built-in capability:**
Remove it from `ALL_CAPABILITIES`. The system will treat that capability as always unavailable.
