# 04 — Where Files Live (Folder Structure)

This document explains every folder and file in the project. If you're looking for where to make a change, start here.

---

## Top-Level Layout

```
d:\data\AI_WAITER\
│
├── dine_ai/               ← ALL source code lives here
├── Documentations/        ← This documentation
├── requirements.txt       ← List of Python libraries needed
└── setup.py               ← How to install the package
```

---

## Inside `dine_ai/` — The Main Source Code

```
dine_ai/
│
├── app.py                 ← THE MAIN ENTRY POINT. This is what you call to use DineAI.
│
├── adapters/              ← "Prepare the menu data"
│   ├── schema_mapper.py       → Rename messy column headers to standard names
│   ├── validator.py           → Check data quality and required fields
│   ├── feature_engineering.py → Calculate protein ratios, dietary flags, price brackets
│   └── text_builder.py        → Build text strings for search and AI context
│
├── capabilities/          ← "What can this restaurant's menu support?"
│   ├── base.py                → Abstract rules all capabilities must follow
│   ├── definitions.py         → The actual list of capabilities (nutrition, budget, etc.)
│   ├── detector.py            → Scans a menu and reports which capabilities are available
│   └── registry.py            → The master list; add new capabilities here
│
├── dataset/               ← "Save and load restaurant databases"
│   ├── loader.py              → Reads saved .pkl, .index, and .json files from disk
│   ├── manager.py             → Decides when to rebuild vs. load from cache
│   ├── metadata.py            → Computes and checks file checksums
│   └── pipeline.py            → Runs the full build sequence (schema → embed → index)
│
├── embeddings/            ← "Convert text to searchable numbers"
│   ├── embedding_builder.py   → Batch-encodes text using AI embedding models
│   └── faiss_builder.py       → Builds the FAISS search index from those numbers
│
├── retrieval/             ← "Find + filter + rank results"
│   ├── semantic_search.py     → Searches FAISS for closest matches to a query
│   ├── filtering.py           → Removes candidates that don't meet rules
│   └── ranking.py             → Scores and sorts candidates by priority
│
├── llm/                   ← "Generate the reply"
│   ├── prompts.py             → Assembles the instruction sent to the AI model
│   └── generator.py           → Runs the AI model and returns its response
│
├── evaluation/            ← Thin wrappers for library compatibility
│   ├── end_to_end_test.py     → Points to evaluations/end_to_end_test.py
│   └── integration_test.py    → Points to evaluations/integration_test.py
│
├── evaluations/           ← The actual test suites
│   ├── deterministic_scenarios_test.py  → "Does the right dish win?"
│   ├── end_to_end_test.py     → "Does a full query work from start to finish?"
│   ├── end_to_end.py          → CLI runner for end_to_end_test
│   └── integration_test.py    → "Does each module connect to the next correctly?"
│
└── datasets/              ← WHERE RESTAURANT DATA IS STORED
    ├── restaurant_A/
    │   ├── menu_items.csv     ← Your source menu spreadsheet (INPUT)
    │   ├── recipes.pkl        ← Processed menu data (auto-generated)
    │   ├── faiss.index        ← Search index (auto-generated)
    │   └── metadata.json      ← Checksums + capability report (auto-generated)
    └── restaurant_B/
        └── ...
```

---

## The `datasets/` Folder — What Goes Where

This is the most important folder for day-to-day use.

### What you put in:
- A folder named after your restaurant (e.g. `restaurant_A`)
- Inside that folder: a CSV file named `menu_items.csv` (or `recipes.csv`)

### What DineAI generates automatically:
- `recipes.pkl` — the cleaned, enriched menu data as a binary file
- `faiss.index` — the vector search index
- `metadata.json` — fingerprints of all files (so it knows when to rebuild)
- `embeddings.npy` — the raw embedding vectors (used for rebuilding the index)

> [!WARNING]
> Don't manually edit or delete `recipes.pkl`, `faiss.index`, or `metadata.json`. Let DineAI manage these. If you suspect they're corrupted, delete them all and restart — DineAI will rebuild from your CSV.

---

## Quick Reference: "Where do I put X?"

| What you want to add/change | Where |
|---|---|
| A new restaurant's menu | `dine_ai/datasets/your_restaurant_name/menu_items.csv` |
| A new column alias (e.g. "Cost" → "price") | `dine_ai/adapters/schema_mapper.py` |
| A new dietary flag calculation | `dine_ai/adapters/feature_engineering.py` |
| A new capability (e.g. "delivery time filtering") | `dine_ai/capabilities/definitions.py` |
| A new AI provider (e.g. Vertex AI) | `dine_ai/llm/generator.py` |
| A new ranking factor | `dine_ai/retrieval/ranking.py` |
| A new filter operator | `dine_ai/retrieval/filtering.py` |
| The main configuration | `dine_ai/app.py` → `ApplicationConfig` class |
