# 02 — How Data Flows Through DineAI

This document traces exactly what happens to data at each step — both when you set up a restaurant and when a customer sends a question.

---

## Part 1: Loading a Restaurant (The One-Time Setup)

This runs the first time you add a restaurant, and again automatically whenever you update the menu spreadsheet.

### Step 1 — Read the Raw Spreadsheet
Your input is a CSV file like this:

| Recipe Name | Menu Price | Protein (g) | Calories | Cuisine | Diet Labels |
|---|---|---|---|---|---|
| Truffle Burger | 25.00 | 45 | 750 | American | High-Protein |
| Veggie Salad | 12.00 | 3 | 120 | Mediterranean | Vegan |

Column names don't need to be exact. DineAI understands common variations.

---

### Step 2 — Normalize Column Names
**What happens:** The Schema Mapper renames your columns to standard internal names.

| Your column name | Internal name used |
|---|---|
| `Recipe Name` | `recipe_name` |
| `Menu Price` | `price` |
| `Protein (g)` | `protein_g_per_serving` |
| `Calories` | `calories_per_serving` |

Unknown columns (like `"Chef's Notes"`) are kept as-is.

---

### Step 3 — Validate the Data
**What happens:** The Validator checks that the data makes sense. It will warn you (but not crash) if:
- A price is negative
- A dish name is missing
- A recommended column like `calories` is absent

---

### Step 4 — Compute Extra Features
**What happens:** The Feature Engineer calculates things your spreadsheet probably doesn't have, like:

| New column | What it means |
|---|---|
| `protein_density` | How much protein per gram of food |
| `protein_to_calorie_ratio` | Useful for "high protein, low calorie" searches |
| `is_vegan` | `1` if the item has no animal products, `0` otherwise |
| `is_keto` | `1` if carbs are very low, `0` otherwise |
| `is_halal` | `1` if labeled halal, `0` otherwise |
| `price_bracket` | "budget" / "mid-range" / "premium" |

---

### Step 5 — Build Text Strings
**What happens:** The Text Builder creates three text versions of each menu item:

- **`search_text`** — A compact summary used to build the search index. Example: *"Truffle Burger American beef bun truffle sauce halal 750 calories 45g protein $25"*
- **`text_chunk`** — A detailed markdown block sent to the AI when recommending this item.
- **`summary_text`** — A short one-liner for UI previews.

---

### Step 6 — Detect Capabilities
**What happens:** The Capability Detector scans the enriched data and figures out what features this restaurant supports.

For example: *"This menu has calorie data → calorie filtering is ENABLED. This menu has no rating data → rating sorting is DISABLED."*

This gets saved to `metadata.json` so the rest of the system knows what it can offer.

---

### Step 7 — Convert Text to Numbers (Embeddings)
**What happens:** The Embedding Builder takes the `search_text` for every item and runs it through an AI model that converts it into a list of 384 numbers. This is called an "embedding" or "vector."

These numbers capture the *meaning* of the text. Two semantically similar items (like "Chicken Salad" and "Grilled Chicken on Greens") will have very similar vectors.

---

### Step 8 — Build the Search Index
**What happens:** All those vectors are loaded into a FAISS index — think of it as a highly optimized lookup table that can find the closest matches in milliseconds, even with thousands of items.

---

### Step 9 — Save Everything to Disk
**What's saved:**
- `recipes.pkl` — The enriched DataFrame (all rows with all columns)
- `faiss.index` — The search index
- `metadata.json` — Hashes and capability report

Next time the system starts, it checks if these files are current before deciding whether to rebuild.

---

## Part 2: Answering a Customer Question (Every Query)

### Step 1 — Receive the Question
Customer types: *"Something low calorie and Italian please"*

### Step 2 — Convert Question to Numbers
The same embedding model converts the question into a vector.

### Step 3 — Search the Index
FAISS finds the top N menu items whose vectors are closest to the question vector. "Closeness" means semantic similarity, not exact word matching.

### Step 4 — Attach Menu Data
The positional results from FAISS (which are just row numbers) get matched back to the actual menu rows so we have names, prices, calories, etc.

### Step 5 — Apply Filters
If any rules were set (e.g. "must be under $15", "must be vegan"), items that don't pass are removed.

### Step 6 — Rank the Survivors
Remaining items are scored and sorted based on configured weights (e.g. 40% semantic similarity + 30% rating + 30% price).

### Step 7 — Build the AI Prompt
The Prompt Builder assembles an instruction like:
```
You are a friendly waiter. The customer asked: "Something low calorie and Italian please"
Here are the best matching menu items:
1. Pasta Primavera — 450 cal, $18, vegan
2. Caprese Salad — 220 cal, $14, vegetarian
Respond helpfully in English.
```

### Step 8 — Run the AI
The AI model reads the prompt and generates a natural reply. The system then checks the reply for problems (repetition, prompt leakage) before returning it.

### Step 9 — Return the Response + Update Memory
The reply is returned to the caller. The conversation is saved to session memory so follow-up questions can reference it.
