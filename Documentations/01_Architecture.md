# 01 — How DineAI is Built (System Architecture)

## Think of It Like a Kitchen

A restaurant kitchen has stations — the grill station, the prep station, the plating station. Each station does one job, passes the result to the next, and nobody skips steps.

DineAI is the same. It has **5 layers**, and data moves through them in a defined order. No layer does another layer's job.

---

## The 5 Layers

### Layer 1 — Data Preparation
*"Get the menu ready"*

This is the first thing that runs when you add a restaurant. It takes your raw spreadsheet and does all the cleaning, calculating, and formatting work needed before anything else can happen.

Modules in this layer: Schema Mapper → Validator → Feature Engineer → Text Builder

### Layer 2 — Search Index
*"Build the search engine"*

Once the menu is clean and enriched, this layer converts every menu item into a mathematical fingerprint (a list of numbers), then stores them in a blazing-fast search database. This is what makes "semantic search" possible — searching by *meaning* rather than exact keywords.

Modules in this layer: Embedding Builder → FAISS Index Builder

### Layer 3 — Retrieval (Find + Filter + Rank)
*"Answer: what's relevant to this customer?"*

When a customer asks something, this layer handles finding, filtering, and sorting results. It runs every time a question comes in.

Modules in this layer: Semantic Search → Filtering Engine → Ranking Engine

### Layer 4 — Language Generation
*"Write a nice reply"*

Takes the ranked, filtered results and uses them to write a natural-sounding AI response. Also manages the conversation memory (so the AI remembers what was said earlier in the chat).

Modules in this layer: Prompt Builder → AI Generator

### Layer 5 — The Main Controller
*"Orchestrate everything"*

The top-level coordinator. When you call `app.chat("session-1", "I want pasta")`, this layer is what you're talking to. It starts up the system, runs the query through layers 1–4 in the right order, tracks statistics, and handles events.

Module: `app.py` (DineAIApplication)

---

## Visual Layout

```
┌──────────────────────────────────────────────────────────┐
│              LAYER 5 — Main Controller (app.py)          │
├──────────────────────────────────────────────────────────┤
│         LAYER 4 — Prompt Builder + AI Generator          │
├──────────────────────────────────────────────────────────┤
│    LAYER 3 — Semantic Search + Filtering + Ranking       │
├──────────────────────────────────────────────────────────┤
│        LAYER 2 — Embedding Builder + FAISS Index         │
├──────────────────────────────────────────────────────────┤
│   LAYER 1 — Schema Mapper → Validator → Feature          │
│              Engineer → Text Builder                     │
└──────────────────────────────────────────────────────────┘
```

---

## Two Separate Workflows

DineAI has two distinct workflows. It's important to understand that they are **completely separate**.

### Workflow A — "Build the menu database" (runs once, or when menu changes)
```
Your CSV file
    → Clean up columns
    → Validate data
    → Compute features
    → Build text strings
    → Scan capabilities
    → Convert to numbers (embeddings)
    → Build search index (FAISS)
    → Save everything to disk
```
This workflow is slow (seconds to minutes) but only runs when needed.

### Workflow B — "Answer a customer question" (runs every time a customer asks something)
```
Customer's question
    → Search the index
    → Filter results by rules
    → Rank results
    → Build AI prompt
    → Run AI model
    → Return reply
```
This workflow is fast (milliseconds to seconds) and runs constantly.

---

## Plugin Points (Where You Can Extend Things)

DineAI uses "registries" — think of them as plug-in boards. You can add new capabilities without changing the core system.

| Registry | What it lets you add |
|---|---|
| `FeatureRegistry` | New calculated columns (e.g. "fiber density") |
| `FaissBuilder Registry` | New search index types |
| `RankingRegistry` | New scoring criteria |
| `FilterRegistry` | New filter operators |
| `PersonaRegistry` | New AI personality styles |
| `ProviderRegistry` | New AI model providers |

See [09_Extensibility.md](file:///d:/data/AI_WAITER/Documentations/09_Extensibility.md) for how to use these.

---

## How a Customer Query Moves Through the System

```
Customer: "I want something vegan under $10"
    │
    ▼
[Main Controller] creates a "context" object to track everything
    │
    ▼
[Semantic Search] searches the index → finds 5 closest menu items
    │
    ▼
[Filtering Engine] removes items that aren't vegan or cost more than $10
    │
    ▼
[Ranking Engine] sorts remaining items by best match
    │
    ▼
[Prompt Builder] writes the AI instruction: "Here are 3 items. Customer wants vegan under $10. Reply helpfully."
    │
    ▼
[AI Generator] writes the actual reply
    │
    ▼
Customer receives: "Great choice! We have the Garden Wrap ($8, fully vegan)..."
```
