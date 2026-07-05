# Module: `validator.py` — Data Quality Checker

**Location:** `dine_ai/adapters/validator.py`

---

## What This Does (Plain English)

Before DineAI processes a menu, it runs a quick quality check. Think of this like a spell-checker for your spreadsheet — it looks for obvious problems and reports them.

It doesn't delete or fix bad data. It just reports problems so you know about them.

---

## What It Checks

### Required Columns
- `recipe_name` **must** exist. If it doesn't, the system stops and reports an error.

### Optional Column Quality Checks
| Check | What it looks for |
|---|---|
| Negative prices | A price value below 0 |
| Negative calories | Calories below 0 |
| Empty dish names | Rows where the dish name is blank |
| Missing columns | Important columns like `price` or `calories` that aren't present |

---

## What It Returns

A `ValidationReport` object with two lists:
- `errors` — serious problems (can block processing)
- `warnings` — non-fatal issues (reported but processing continues)

Example:
```python
ValidationReport(
    errors=[],
    warnings=[
        "Column 'calories_per_serving' not found — calorie filtering will be disabled",
        "3 rows have missing recipe_name — these rows will be skipped"
    ]
)
```

---

## What Happens If It Finds Problems

- **Errors** → Pipeline stops. You need to fix the CSV before the system can proceed.
- **Warnings** → Pipeline continues. The issue is noted, and the capability that depended on that data is marked as unavailable.

So if your menu has no calorie data, the system still loads and works — it just disables calorie-based filtering automatically.

---

## How to Edit This Module

**To add a new validation check** (e.g. verify that price is never 0):
1. Open `validator.py`
2. Find the `DataFrameValidator` class
3. Add a check method:
   ```python
   def check_no_zero_prices(self, df):
       if "price" in df.columns:
           zero_prices = (df["price"] == 0).sum()
           if zero_prices > 0:
               self.report.warnings.append(
                   f"{zero_prices} items have $0 price — check if intentional"
               )
   ```
4. Call it from the main `validate()` method

**To make a warning into an error** (stricter validation):
1. Find the check that currently appends to `warnings`
2. Change it to append to `errors` instead
