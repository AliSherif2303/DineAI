# Module: `loader.py` — File Reader

**Location:** `dine_ai/dataset/loader.py`

---

## What This Does (Plain English)

After the Dataset Manager decides that the cached files are valid (no rebuild needed), the Dataset Loader actually reads those files from disk and loads them into memory.

It's a simple but critical step: the loader reads three files and assembles them into one `RestaurantDataset` object that the rest of the system uses.

---

## What It Reads

| File | Format | What's inside |
|---|---|---|
| `recipes.pkl` | Python binary (pickle) | The enriched menu DataFrame — all rows and columns |
| `faiss.index` | FAISS binary | The search index |
| `metadata.json` | JSON text | Checksums, capabilities, build info |

---

## The `RestaurantDataset` Object

This is what the loader returns — a bundle containing everything:

| Field | Type | What it contains |
|---|---|---|
| `recipes` | `pd.DataFrame` | All menu items with all computed columns |
| `faiss_result` | `FaissResult` | Loaded search index |
| `metadata` | `dict` | The metadata JSON contents |
| `capabilities` | `dict` | Which capabilities are available |
| `available_columns` | `list` | Column names present in this restaurant's data |

---

## Error Handling

If any file is missing or corrupted:
- The loader raises a specific error identifying which file failed
- The Dataset Manager catches this error and triggers a rebuild
- After rebuilding, the loader runs again on the fresh files

This means: if you accidentally corrupt a file, the system auto-recovers on next startup.

---

## How to Edit This Module

**You rarely need to edit this module directly.** The loader's behavior is straightforward.

**If you add a new saved artifact** (e.g. a `clusters.npy` file from a new clustering step):
1. Open `loader.py`
2. Find the `DatasetLoader.load()` method
3. Add a line to read your new file:
   ```python
   clusters_path = os.path.join(restaurant_path, "clusters.npy")
   if os.path.exists(clusters_path):
       dataset.clusters = np.load(clusters_path)
   ```
4. Add the field to `RestaurantDataset` dataclass
