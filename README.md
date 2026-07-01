# DineAI Backend Architecture

This repository contains the baseline project structure for **DineAI**, a semantic search and LLM-powered dining assistant.

## Project Structure

```text
dine_ai/
├── adapters/                  # Data mapping, validation, text formatting, and feature engineering
│   ├── __init__.py
│   ├── schema_mapper.py       # Maps raw source data to internal schema
│   ├── validator.py           # Validates input datasets and schemas
│   ├── feature_engineering.py # Extracts attributes (e.g. dietary tags, prep time tags)
│   └── text_builder.py        # Compiles structured fields into search-optimized text blocks
│
├── embeddings/                # Vector representation generation and indexing
│   ├── __init__.py
│   ├── embedding_builder.py   # Handles conversion of text to vector embeddings
│   └── faiss_builder.py       # Builds, stores, and loads local FAISS vector databases
│
├── retrieval/                 # Search, filtering, and candidate ranking logic
│   ├── __init__.py
│   ├── semantic_search.py     # Performs vector search against the database
│   ├── filtering.py           # Filters retrieved candidates based on dietary/metadata parameters
│   └── ranking.py             # Re-ranks items using preference alignment or popularity
│
├── llm/                       # LLM prompt templates and generations
│   ├── __init__.py
│   ├── prompts.py             # System and template instructions
│   └── generator.py           # Connects to model APIs to form natural language recommendations
│
├── datasets/                  # Datasets and local databases grouped by restaurant
│   ├── restaurant_A/
│   │   ├── recipes.pkl        # Pickle database of recipes for Restaurant A
│   │   ├── faiss.bin          # Binary FAISS vector search index
│   │   └── config.json        # Configuration file for Restaurant A
│   └── restaurant_B/
│
└── app.py                     # Main application entry point
```

## Getting Started

### Prerequisites

Ensure you have Python installed (3.8+ recommended).

### Running the App

To run the application, execute the module command from the root workspace directory:

```bash
python -m dine_ai.app
```

### Extending with Restaurant B

To configure Restaurant B:
1. Place a corresponding `config.json`, `recipes.pkl`, and `faiss.bin` in `dine_ai/datasets/restaurant_B/`.
2. Update the initialization logic in `dine_ai/app.py` to target `restaurant_B`.
