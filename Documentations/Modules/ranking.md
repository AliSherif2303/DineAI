# Module: `ranking.py` — Results Sorter

**Location:** `dine_ai/retrieval/ranking.py`

---

## What This Does (Plain English)

After filtering, you might still have 8 items left. They all passed the rules — but which one should appear first? That's what the Ranking Engine decides.

It gives each item a final score based on multiple factors. The item with the highest score goes first — and that's typically what the AI leads with in its response.

---

## The Scoring Factors

Each factor is a signal about how good an item is for this query:

| Factor | What it measures | Direction |
|---|---|---|
| `SEMANTIC_SIMILARITY` | How closely the dish matches what the customer asked | Higher = better |
| `PRICE_NORMALIZED` | How affordable the item is relative to others | Lower price = higher score |
| `RATING_NORMALIZED` | How highly rated the item is | Higher rating = better |

Each factor has a **weight** — its percentage contribution to the final score.

**Default weights:**
- Semantic Similarity: **50%**
- Rating: **30%**
- Price: **20%**

These weights add up to 100%.

---

## How the Final Score is Calculated

For each item:
1. Each factor's raw value is normalized to a 0–1 scale (so different metrics can be compared fairly)
2. Each normalized value is multiplied by its weight
3. The weighted values are summed to produce the final score

Example for a $12 dish with 0.85 similarity and 4.5-star rating:
```
similarity score: 0.85 × 50% = 0.425
rating score:     0.90 × 30% = 0.270   (4.5/5 = 0.9)
price score:      0.70 × 20% = 0.140   (mid-range pricing → 0.7)
                                ─────
final score:                   0.835
```

---

## Normalization Methods

Since raw values are on different scales (price in dollars, rating out of 5, similarity 0–1), the engine normalizes them before combining:

| Method | When it's used |
|---|---|
| `MIN_MAX` | Scale values to exactly 0–1 range based on the current set of candidates |
| `Z_SCORE` | Scale based on mean and standard deviation (handles outliers better) |
| `IDENTITY` | No normalization — use raw value as-is |

---

## RankedCandidate Object

After ranking, each item is wrapped in a `RankedCandidate` with:

| Field | What it contains |
|---|---|
| `candidate` | The original search candidate |
| `final_score` | The combined weighted score (0.0–1.0) |
| `ranking_breakdown` | Dict showing each factor's contribution |
| `rank_position` | Position in the sorted list (1 = best) |

---

## How to Edit This Module

**To change ranking weights** (e.g. make rating matter more):
1. Open `ranking.py`
2. Find `RankingConfig` and modify `factors`:
   ```python
   RankingConfig(factors=[
       RankingFactor(name="SEMANTIC_SIMILARITY", weight=0.40),
       RankingFactor(name="RATING_NORMALIZED", weight=0.45),    # increased
       RankingFactor(name="PRICE_NORMALIZED", weight=0.15)
   ])
   ```
   Weights must sum to 1.0.

**To add a new ranking factor** (e.g. "freshness"):
1. Add it to the `RankingFactor` enum
2. Add scoring logic in `RankingEngine._compute_factor_score()`
3. Add it to the `RankingConfig`

**To disable price ranking entirely** (e.g. price isn't relevant):
1. Remove `PRICE_NORMALIZED` from the factors list
2. Redistribute its weight to other factors so they total 1.0
