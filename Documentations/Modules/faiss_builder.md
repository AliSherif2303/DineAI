# Module: `faiss_builder.py` — Search Index Builder

**Location:** `dine_ai/embeddings/faiss_builder.py`

---

## What This Does (Plain English)

FAISS (Facebook AI Similarity Search) is a library that can find the most similar items in a list of vectors — extremely fast, even with millions of items.

Think of it like a super-powered lookup table. Instead of searching through every item one by one, FAISS organizes the vectors in a smart structure so that finding the closest 5 items takes milliseconds instead of seconds.

The FAISS Builder takes the vector matrix from the Embedding Builder and packages it into this searchable structure.

---

## What Goes In / What Comes Out

**Input:** A NumPy matrix of shape `(N, 384)` — N items, each represented as 384 numbers.

**Output:**
- `faiss.index` file on disk — the searchable index
- A `FaissResult` object in memory — the loaded index ready to use

---

## Index Types

DineAI uses the `IndexFlatIP` index type by default. In plain terms:
- **Flat** = exact search (no approximation — finds the true closest match)
- **IP** = inner product (used for normalized vectors, equivalent to cosine similarity)

This is the most accurate type. For very large menus (10,000+ items), a faster approximate index (like `IndexIVFFlat`) could be used instead, at the cost of occasionally missing the best match.

---

## FaissConfig Settings

| Setting | Default | What it does |
|---|---|---|
| `index_type` | `"IndexFlatIP"` | Type of FAISS index (accuracy vs. speed tradeoff) |
| `normalize_vectors` | `True` | Normalizes vectors before indexing (required for cosine similarity) |
| `top_k` | `10` | Default number of results to return per search |

---

## FaissResult Object

This is what the FAISS Builder returns, and what the search engine uses:

| Field | What it contains |
|---|---|
| `index` | The loaded FAISS index object |
| `id_to_row` | Map from FAISS position → DataFrame row index |
| `dimension` | Number of dimensions (384 by default) |
| `total_vectors` | Total number of menu items in the index |

---

## How to Edit This Module

**To change the index type** (e.g. for very large menus):
1. Open `faiss_builder.py`
2. Find `FaissConfig` and change `index_type` to `"IndexIVFFlat"`
3. Note: `IndexIVFFlat` requires an additional `nlist` (number of clusters) parameter

**To change the similarity metric** (e.g. L2 distance instead of cosine):
1. Change `index_type` to `"IndexFlatL2"`
2. Set `normalize_vectors=False` in `FaissConfig`
3. You'll also need to update `embedding_builder.py` to not normalize embeddings

> [!WARNING]
> Changing index type or similarity metric requires deleting and rebuilding the index. The old index file is incompatible.
