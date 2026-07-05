# Module: `manager.py` — Rebuild Decision Engine

**Location:** `dine_ai/dataset/manager.py`

---

## What This Does (Plain English)

Processing a menu from scratch (reading the CSV, computing features, building embeddings, building the FAISS index) is slow. It would be wasteful to do it every single time the system starts.

The Dataset Manager is the smart layer that decides: *"Do we need to rebuild, or can we load from the files we already saved?"*

It checks the saved files against the current state of the CSV and the current model settings. If everything matches, it loads from disk (fast). If anything changed, it triggers a rebuild.

---

## The Decision Logic

When you call `DatasetManager.load_restaurant()`, here's what happens:

```
1. Check if recipes.pkl, faiss.index, metadata.json all exist
   → If any file is missing: REBUILD
   
2. Compute the SHA-256 hash of the current menu_items.csv
   → Compare to the hash stored in metadata.json
   → If different: REBUILD (menu changed)
   
3. Check the embedding model name stored in metadata
   → Compare to the currently configured model
   → If different: REBUILD (model changed, vectors incompatible)
   
4. All checks pass → LOAD from cached files
```

---

## What Is a Hash?

A "hash" (or SHA-256 checksum) is like a fingerprint for a file. If even one character changes in the CSV file, the hash is completely different. This lets the manager detect changes without reading the entire file.

---

## What It Stores in `metadata.json`

```json
{
    "csv_hash": "a3f4c2...",           // fingerprint of the CSV file
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "build_timestamp": "2024-01-15T10:30:00",
    "total_items": 150,
    "capabilities": {
        "nutrition_filtering": true,
        "budget_filtering": true,
        "rating_sorting": false
    }
}
```

This file is updated every time a rebuild completes.

---

## How to Edit This Module

**To force a full rebuild even if nothing changed:**
Delete all three generated files from your restaurant folder:
```
dine_ai/datasets/restaurant_A/recipes.pkl
dine_ai/datasets/restaurant_A/faiss.index
dine_ai/datasets/restaurant_A/metadata.json
```
Next startup will rebuild everything.

**To add a new rebuild trigger** (e.g. rebuild if the Python version changed):
1. Open `manager.py`
2. Find the rebuild check logic
3. Add a new condition to the check block

**To disable caching entirely** (always rebuild):
1. Open `app.py`
2. Set `cache_enabled=False` in `ApplicationConfig`
