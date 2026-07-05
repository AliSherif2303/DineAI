# Module: `schema_mapper.py` — Column Name Normalizer

**Location:** `dine_ai/adapters/schema_mapper.py`

---

## What This Does (Plain English)

Every restaurant's spreadsheet looks different. One restaurant might call the dish name column `"Recipe Name"`, another uses `"Item"`, another uses `"Dish Title"`. The Schema Mapper's job is to look at whatever column names came in and rename them to the standard names that the rest of the system expects.

Think of it like a translator. It doesn't change the data — it just relabels the columns.

---

## Why It Exists

Without this module, every other module would need to handle hundreds of possible column name variations. Instead, they only need to know one name (`recipe_name`, `price`, `calories_per_serving`, etc.) and the Schema Mapper guarantees those names are always used.

---

## What It Reads In

A raw pandas DataFrame with whatever column names your CSV file has. For example:
```
| Recipe Name | Menu Price | Protein (g) |
```

## What It Outputs

The same DataFrame with standardized column names:
```
| recipe_name | price | protein_g_per_serving |
```

---

## The Alias Registry

The `ALIAS_REGISTRY` is a dictionary that maps standard names to a list of acceptable alternative names:

```python
"price": ["price", "cost", "menu price", "item price", "selling price", "unit price"]
"recipe_name": ["recipe name", "item name", "dish name", "item", "food name", "name"]
"calories_per_serving": ["calories", "cal", "kcal", "energy", "calories per serving"]
```

When a column doesn't match any alias, it's left with its original name (lowercased and normalized).

---

## The Canonical Schema

`CANONICAL_SCHEMA` defines every standard internal column and whether it's required or optional:

| Standard Name | Required? | What it stores |
|---|---|---|
| `recipe_name` | ✅ Yes | Dish name |
| `price` | ❌ No | Price per item |
| `calories_per_serving` | ❌ No | Calorie count |
| `protein_g_per_serving` | ❌ No | Protein in grams |
| `cuisine_type` | ❌ No | Cuisine category |
| `meal_type` | ❌ No | Breakfast / Lunch / Dinner |
| `description` | ❌ No | Free-text description |
| `ingredients` | ❌ No | Ingredient list |

Only `recipe_name` is required. Everything else is optional.

---

## How to Edit This Module

**To add a new column alias** (e.g. "Cost" should map to "price"):
1. Open `schema_mapper.py`
2. Find `ALIAS_REGISTRY` 
3. Add your alias string to the matching list

**To add a new standard column** (e.g. add `spice_level`):
1. Add an entry to `CANONICAL_SCHEMA`
2. Add aliases to `ALIAS_REGISTRY`
3. Update `feature_engineering.py` and `text_builder.py` to use the new column
