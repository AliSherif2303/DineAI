# Module: `semantic_search.py` — Find Relevant Menu Items

**Location:** `dine_ai/retrieval/semantic_search.py`

---

## What This Does (Plain English)

When a customer types a question, the system needs to find which menu items are most relevant — even if the customer uses completely different words from what's in the menu.

For example: a customer asks *"something light and refreshing"* — the menu doesn't have a dish called "light and refreshing." But a Caprese Salad (120 cal, fresh ingredients) is semantically close to what the customer means.

Semantic Search converts the customer's question into the same kind of vector as the menu items, then asks FAISS: *"which stored vectors are closest to this one?"* The closest matches are the most relevant dishes.

---

## What Goes In / What Comes Out

**Input:**
- Customer query string (e.g. `"I want something vegan and light"`)
- The FAISS index (already loaded)

**Output:**
- A list of `SearchCandidate` objects — each one is a menu item with a similarity score

```python
[
    SearchCandidate(row_id=42, score=0.89, metadata={"recipe_name": "Garden Salad", ...}),
    SearchCandidate(row_id=17, score=0.82, metadata={"recipe_name": "Tofu Stir-Fry", ...}),
    ...
]
```

---

## The `score` Field

The score is a number between 0 and 1:
- `1.0` = perfect semantic match
- `0.5` = loosely related
- `0.0` = completely unrelated

Higher is better. The default threshold is `0.0` — the system returns all top-K results regardless of score, and the Ranking Engine handles sorting by score later.

---

## SearchConfig Settings

| Setting | Default | What it does |
|---|---|---|
| `top_k` | `10` | How many candidates to retrieve from FAISS |
| `score_threshold` | `0.0` | Minimum score to include (0 = include everything) |
| `normalize_query` | `True` | Normalize query vector before search (recommended) |

> [!TIP]
> `top_k` is not the same as `max_context_recipes`. The system retrieves `top_k` candidates for filtering and ranking, then the ranking stage reduces them to `max_context_recipes` for the AI.

---

## How to Edit This Module

**To retrieve more candidates** (gives filtering more to work with):
1. Open `app.py`
2. Find where `SearchConfig` is instantiated
3. Increase `top_k` (e.g. from 10 to 20)

**To add a minimum quality threshold** (only return items above certain relevance):
1. Find `SearchConfig`
2. Set `score_threshold=0.5` — items scoring below 0.5 won't be returned

**To change the embedding model for queries:**
The same model used to build the index must be used for queries. Change `embedding_model` in `ApplicationConfig` and rebuild the index.
