# Module: `feature_engineering.py` — Data Enrichment

**Location:** `dine_ai/adapters/feature_engineering.py`

---

## What This Does (Plain English)

Your CSV menu probably doesn't have every piece of information DineAI needs to do smart filtering. For example:
- It won't have a column called `is_vegan` — that needs to be calculated from the ingredients.
- It won't have `protein_to_calorie_ratio` — that needs to be computed from the protein and calorie columns.
- It won't have `price_bracket` — the system needs to decide if $8.50 is "budget" or "mid-range".

Feature Engineering does all of these calculations and adds the results as new columns to the menu data.

---

## What New Columns It Creates

### Nutrition Ratios
| New Column | How it's calculated | What it means |
|---|---|---|
| `protein_density` | protein_g / weight_g × 100 | How much protein per 100g of food |
| `protein_to_calorie_ratio` | protein_g / calories | Good for "high protein, low calorie" requests |
| `nutrition_score` | weighted formula | Overall healthiness score |

### Dietary Flags (0 = No, 1 = Yes)
| New Column | Logic |
|---|---|
| `is_vegan` | No animal products in ingredients or labels |
| `is_vegetarian` | No meat, but may have dairy/eggs |
| `is_halal` | Explicitly labeled halal |
| `is_keto` | Carbs below 20g threshold |
| `is_low_calorie` | Calories below 150 threshold |
| `is_high_protein` | Protein above 15g threshold |
| `is_low_sodium` | Sodium below 140mg threshold |

### Price Categories
| New Column | Values | Thresholds |
|---|---|---|
| `price_bracket` | "budget" / "mid-range" / "premium" | Under $10 / $10–$25 / Over $25 |
| `price_level` | "$" / "$$" / "$$$" | Same thresholds |

---

## Configuration Thresholds (`FeatureEngineerConfig`)

The exact numbers that define categories are all in `FeatureEngineerConfig` at the top of the file. This is where to change the thresholds if they don't match your restaurant's menu:

| Threshold | Default | Change it if... |
|---|---|---|
| `high_protein_threshold_g` | 15g | Your restaurant serves smaller portions |
| `low_calorie_threshold` | 150 cal | Your menu has mostly light dishes |
| `keto_carbs_g_threshold` | 20g | You need stricter keto definition |
| `price_bins` | [10, 25] | Your menu has different price distribution |

---

## The Registry Pattern

Each group of features (nutrition, dietary, price) is implemented as a separate `BaseFeatureEngineer` class and registered in `FeatureRegistry`. This means you can add a new feature group without changing existing ones.

---

## How to Edit This Module

**To change a dietary threshold** (e.g. change what counts as "keto"):
1. Open `feature_engineering.py`
2. Find `FeatureEngineerConfig`
3. Change `keto_carbs_g_threshold`

**To add a new dietary flag** (e.g. `is_nut_free`):
1. Find `DietaryFeatureEngineer` class
2. Add a new method:
   ```python
   def compute_is_nut_free(self, df):
       nuts = ["peanut", "almond", "cashew", "walnut"]
       df["is_nut_free"] = df["ingredients"].apply(
           lambda x: 0 if any(n in str(x).lower() for n in nuts) else 1
       )
       return df
   ```
3. Call it from `transform()`

**To change price brackets** (e.g. budget is under $15):
1. Find `FeatureEngineerConfig`
2. Change `price_bins` to `[-inf, 15.0, 30.0, inf]`
3. Update `price_labels` accordingly
