# Module: `pipeline.py` — Build Sequence Runner

**Location:** `dine_ai/dataset/pipeline.py`

---

## What This Does (Plain English)

The Dataset Pipeline is the assembly line that processes a menu from raw CSV to finished database. It runs when a new restaurant is added or when an update is detected.

It connects all the data processing steps together and runs them in order:
Schema Mapper → Validator → Feature Engineer → Text Builder → Capability Detector → Embedding Builder → FAISS Builder → Save to Disk.

---

## Embedding Providers

The pipeline is designed to swap out the embedding model without changing anything else. It uses a "provider" pattern:

| Provider | Behavior |
|---|---|
| `MockEmbeddingProvider` | Returns random vectors. No model needed. Used in testing. |
| `SentenceTransformerProvider` | Loads `BAAI/bge-small-en-v1.5` (or any specified model) and runs it locally. |

The correct provider is chosen in `app.py` based on the `embedding_provider` config setting.

---

## What the Pipeline Produces

At the end of a successful build, three files are saved in the restaurant folder:

```
dine_ai/datasets/restaurant_A/
├── recipes.pkl        ← enriched menu DataFrame
├── faiss.index        ← searchable vector index
└── metadata.json      ← checksums, capabilities, build info
```

---

## Error Handling

If any step in the pipeline fails:
- A `DatasetBuildError` exception is raised with the stage name that failed
- No partial files are saved (the build is atomic — all or nothing)
- The old files (if they existed) remain intact
- The error is logged with full stack trace

---

## How to Edit This Module

**To add a new step to the build pipeline:**
1. Open `pipeline.py`
2. Find the `DatasetPipeline.build()` method
3. Add your step in the right position:
   ```python
   # After feature engineering, before text building
   df = MyNewStep().transform(df)
   ```

**To skip a step for testing** (temporarily):
1. Comment out the step call in `build()`
2. Re-enable it before committing

**To change what gets saved:**
1. Find the save section at the end of `build()`
2. Add your new artifact save call
3. Update `loader.py` to read it back
