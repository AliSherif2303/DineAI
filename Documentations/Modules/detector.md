# Module: `detector.py` — Capability Scanner

**Location:** `dine_ai/capabilities/detector.py`

---

## What This Does (Plain English)

Not every restaurant menu has every piece of information. One menu might have calorie counts, another might not. One might have ratings, another might not.

The Capability Detector scans the enriched menu data and figures out which features the system can support for *this specific restaurant's menu*. It then writes a capability report to `metadata.json`.

This matters because:
- It prevents the system from trying to filter by calories when there's no calorie data
- It lets the AI know what questions it can and can't answer
- It allows the UI to show only the filters that are actually available

---

## What It Checks For

| Capability | What column it needs | What it enables |
|---|---|---|
| `nutrition_filtering` | `calories_per_serving` or `protein_g_per_serving` | Filter by calories, protein, etc. |
| `budget_filtering` | `price` | Filter by price range |
| `rating_sorting` | `rating` | Sort by highest rating |
| `dietary_filtering` | `is_vegan` or `is_halal` etc. | Filter by dietary restrictions |
| `cuisine_filtering` | `cuisine_type` | Filter by cuisine |

---

## What It Outputs

A dictionary like this, saved into `metadata.json`:

```json
{
    "nutrition_filtering": true,
    "budget_filtering": true,
    "rating_sorting": false,
    "dietary_filtering": true,
    "cuisine_filtering": true
}
```

---

## How to Edit This Module

**To add a check for a new capability:**
1. First, define the capability in `definitions.py`
2. Register it in `registry.py`
3. The detector automatically picks it up from the registry and runs the check

You don't need to edit `detector.py` directly unless you want to change how checks are executed.
