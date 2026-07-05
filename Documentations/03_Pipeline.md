# 03 — The Two Pipelines Explained

A "pipeline" is just a sequence of steps that run in a fixed order, where each step's output becomes the next step's input.

DineAI has **two separate pipelines** that serve completely different purposes.

---

## Pipeline 1 — "Build the Database" (Dataset Build Pipeline)

**When does it run?**
- The first time you add a new restaurant
- When the menu spreadsheet changes
- When you change the embedding model
- When you manually force a rebuild

**Who triggers it?** The `DatasetManager` — it checks automatically on startup.

**What does it produce?** Three files saved to disk:
- `recipes.pkl` — the enriched menu data
- `faiss.index` — the search index
- `metadata.json` — checksums and capability info

**Is it fast?** No — it can take seconds to minutes depending on menu size and whether you're using a real AI model for embeddings. But it only runs when something actually changed.

### The 7 Steps

```
1. READ CSV        → load the spreadsheet from disk
       ↓
2. NORMALIZE       → rename columns to standard names
       ↓
3. VALIDATE        → check for missing or broken data
       ↓
4. ENRICH          → compute protein ratios, dietary flags, price brackets
       ↓
5. BUILD TEXT      → create search_text, text_chunk, summary_text per item
       ↓
6. DETECT CAPS     → figure out what filtering capabilities are available
       ↓
7. EMBED + INDEX   → convert to numbers and build the FAISS search index
       ↓
8. SAVE            → write everything to disk
```

### What Triggers a Rebuild?

DineAI is smart about when to rebuild. It won't redo work unnecessarily.

| Situation | What happens |
|---|---|
| Menu CSV hasn't changed, model is the same | Load from cache — no rebuild |
| Menu CSV was modified | Full rebuild |
| Embedding model was changed | Rebuild embeddings + index |
| A binary file is missing | Rebuild that specific artifact |
| Checksum mismatch (file corrupted) | Full rebuild |

---

## Pipeline 2 — "Answer a Question" (Query Pipeline)

**When does it run?** Every time a customer sends a message.

**Who triggers it?** `DineAIApplication.chat()` — called from your application.

**What does it produce?** A `GenerationResponse` containing the AI's reply text.

**Is it fast?** Yes — for mock mode it's near-instant. With a real local model, expect 1–10 seconds.

### The 5 Stages

```
1. SEMANTIC SEARCH    → find the top-K most relevant menu items
       ↓
2. FILTERING          → remove anything that violates rules (price, dietary, etc.)
       ↓
3. RANKING            → score and sort the remaining items
       ↓
4. PROMPT BUILDING    → write the AI instruction using the ranked items
       ↓
5. GENERATION         → run the AI model and return the reply
```

### Each Stage Has a Contract

Every stage must:
- Accept a `PipelineContext` object (the shared state bag)
- Do its work and update the context
- Return the updated context for the next stage

If a stage crashes, the error is logged and added to the context's warnings, but the pipeline continues (it doesn't crash your application).

---

## Stage Breakdown — Build Pipeline

| # | Stage | Input | Output | What breaks if skipped |
|---|---|---|---|---|
| 1 | Schema Mapper | Raw CSV | Normalized DataFrame | All downstream columns will have wrong names |
| 2 | Validator | Normalized DataFrame | Same + report | Silent bad data passes through |
| 3 | Feature Engineer | Validated DataFrame | Enriched DataFrame | Dietary flags, macro ratios won't exist |
| 4 | Text Builder | Enriched DataFrame | + text columns | No text for embedding; search index can't be built |
| 5 | Capability Detector | Enriched DataFrame | Capability report | System won't know what filters are available |
| 6 | Embedding Builder | `search_text` column | Vector matrix | No embeddings = no search index |
| 7 | FAISS Builder | Vector matrix | `faiss.index` | Semantic search is impossible |

---

## Stage Breakdown — Query Pipeline

| # | Stage | Input | Output | What breaks if skipped |
|---|---|---|---|---|
| 1 | Semantic Search | Question text | Top-K candidates | Nothing to filter/rank/show |
| 2 | Filtering | Candidates | Passed candidates | Dietary/price violations reach the AI |
| 3 | Ranking | Passed candidates | Ordered candidates | Worst match could appear first |
| 4 | Prompt Builder | Ordered candidates | Prompt text | AI gets no context; hallucinates |
| 5 | Generator | Prompt text | Reply text | No response returned |

---

## Error Handling

- **Build pipeline errors** — If a stage fails during build, it raises an exception and the build stops. The restaurant is not updated.
- **Query pipeline errors** — If a stage fails during a query, the error is captured in `PipelineTrace.stage_errors`, warnings are added to the context, and the pipeline tries to continue if possible.
