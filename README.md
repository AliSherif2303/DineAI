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
├── capabilities/              # Feature detection for a restaurant's menu
│   ├── __init__.py
│   ├── base.py                # BaseCapability abstract class and result types
│   ├── definitions.py         # Concrete capability rules (dietary, item type, nutrition, etc.)
│   ├── detector.py            # CapabilityDetector: runs all rules, produces a report
│   └── registry.py            # CapabilityRegistry: registers and manages capability definitions
│
├── dataset/                   # Dataset lifecycle: building, loading, and managing artifacts
│   ├── __init__.py
│   ├── pipeline.py            # DatasetPipeline: full build from CSV to FAISS + embeddings
│   ├── manager.py             # DatasetManager: smart loader with auto-rebuild on change
│   ├── loader.py              # RestaurantDataset DTO: recipes, FAISS, metadata, capabilities
│   └── metadata.py            # DatasetMetadata: artifact hashes, version, build provenance
│
├── embeddings/                # Vector representation generation and indexing
│   ├── __init__.py
│   ├── embedding_builder.py   # EmbeddingBuilder + SentenceTransformerEmbedding + MockEmbeddingModel
│   └── faiss_builder.py       # Builds, stores, and queries FAISS vector indexes
│
├── retrieval/                 # Search, filtering, and candidate ranking logic
│   ├── __init__.py
│   ├── semantic_search.py     # SemanticSearchEngine: vector similarity search via FAISS
│   ├── filtering.py           # FilteringEngine: metadata rule-based candidate filtering
│   └── ranking.py             # RankingEngine: multi-factor scoring and reordering
│
├── llm/                       # LLM prompt templates and generation
│   ├── __init__.py
│   ├── prompts.py             # PromptBuilder, PromptConfig, PromptContext, PromptResult
│   └── generator.py           # GeneratorEngine + ProviderRegistry (mock, qwen, openai, etc.)
│
├── evaluations/               # Active evaluation suite (modernized)
│   ├── integration_test.py    # 14-stage pipeline integration test with enterprise report
│   ├── end_to_end_test.py     # Full application E2E test across 10 user queries
│   └── deterministic_scenarios_test.py  # Fixed-input retrieval and ranking scenarios
│
├── datasets/                  # Restaurant data grouped by restaurant name
│   └── restaurant_A/
│       ├── recipes.csv        # Source menu CSV (raw or normalized)
│       ├── recipes.pkl        # Built pickle dataset artifact
│       ├── embeddings.npy     # Saved embedding matrix
│       ├── faiss.index        # FAISS binary index
│       ├── metadata.json      # Build provenance, hashes, capability report
│       └── config.json        # Restaurant config
│
└── app.py                     # Main application facade, pipeline orchestrator, and CLI
```

## Architecture Overview

DineAI processes a user query through the following pipeline stages:

```
User Query
    │
    ▼
SemanticSearchStage      — embeds the query and retrieves top-K candidates via FAISS
    │
    ▼
FilteringStage           — applies metadata rule constraints (diet, price, type, etc.)
    │
    ▼
RankingStage             — scores candidates using semantic similarity + price + rating
    │
    ▼
PromptBuilderStage       — formats ranked candidates into a structured LLM prompt
    │
    ▼
GeneratorStage           — dispatches the prompt to the configured LLM provider
    │
    ▼
Final Response
```

Dataset artifacts are built automatically by `DatasetManager` the first time a restaurant is loaded, or whenever the source CSV changes.

## Configuration

`ApplicationConfig` controls the full application. The two main provider systems are **fully independent**:

```python
from dine_ai.app import DineAIApplication, ApplicationConfig

app = DineAIApplication(ApplicationConfig(
    restaurant_name="restaurant_A",

    # Embedding subsystem — controls dataset build and semantic search
    embedding_provider="mock",               # "mock" | "sentence-transformer"
    embedding_model="BAAI/bge-small-en-v1.5",

    # LLM subsystem — controls text generation only
    llm_provider="mock",                     # "mock" | "qwen" | "openai" | "gemini" | ...
    llm_model="Qwen/Qwen2.5-7B-Instruct",

    max_context_recipes=5,
))
app.initialize()
response = app.chat("session_1", "High protein breakfast under $20")
print(response.generated_text)
```

> An LLM provider name is **never** assumed to be a valid embedding provider. Each subsystem must be configured independently.

## Getting Started

### Prerequisites

Python 3.10+ is required.

```bash
# Core install (mock/offline mode — no GPU, no API keys needed)
pip install numpy pandas faiss-cpu PyYAML

# Optional: real sentence-transformer embeddings
pip install sentence-transformers

# Optional: local HuggingFace LLM inference (Qwen, LLaMA, Mistral)
pip install torch transformers accelerate

# Optional: cloud LLM APIs (install the one you need)
pip install openai          # OpenAI / GPT
pip install anthropic       # Anthropic Claude
pip install google-generativeai  # Google Gemini
pip install ollama          # Local Ollama server
```

Or use the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Running the Application

```bash
# Interactive CLI — uses mock providers by default
python -m dine_ai.app

# Explicit provider flags (recommended)
python -m dine_ai.app --embedding-provider mock --llm-provider mock
python -m dine_ai.app --embedding-provider sentence-transformer --llm-provider qwen

# Target a specific restaurant
python -m dine_ai.app --restaurant restaurant_A --embedding-provider mock --llm-provider mock

# Run framework self-tests
python -m dine_ai.app --test
```

#### CLI Flags

| Flag | Short | Description |
|:---|:---|:---|
| `--embedding-provider` | `-ep` | `mock` or `sentence-transformer` |
| `--embedding-model` | `-em` | Embedding model name (e.g. `BAAI/bge-small-en-v1.5`) |
| `--llm-provider` | `-lp` | `mock`, `qwen`, `openai`, `gemini`, `claude`, `ollama`, etc. |
| `--llm-model` | `-lm` | LLM model name (e.g. `Qwen/Qwen2.5-7B-Instruct`) |
| `--restaurant` | `-r` | Restaurant dataset name (default: `restaurant_A`) |
| `--test` | `-t` | Run framework self-tests |

> **Deprecated flags**: `--provider` and `--model` are retained for backward compatibility.  
> `--provider mock` maps to both providers automatically.  
> `--provider <llm-name>` (e.g. `--provider qwen`) will **exit with an error** and ask you to use explicit flags.

---

## Adding a New Restaurant

1. Create a directory under `dine_ai/datasets/<restaurant_name>/`.
2. Place your menu CSV inside it (either `recipes.csv` or `menu_items.csv`).
3. Run the application — `DatasetManager` will auto-detect and build all artifacts on first load.

The CSV can use any column names; `SchemaMapper` will automatically map them to the internal schema.

---

## Evaluation Suite

Three test suites verify the full pipeline:

```bash
# 14-stage integration test (covers every pipeline component independently)
python -m dine_ai.evaluations.integration_test

# End-to-end test (10 real user queries through the full application)
python -m dine_ai.evaluations.end_to_end_test

# Deterministic retrieval and ranking scenario tests
python -m dine_ai.evaluations.deterministic_scenarios_test
```

All three suites:
- Run fully **offline** using the mock embedding and mock LLM providers
- Auto-generate temporary restaurant datasets — no pre-existing data required
- Clean up generated files on success; preserve them on failure for debugging
- Support `DINEAI_TEST_RESTAURANT` and `DINEAI_TEST_DATASET_DIR` environment variables to run against a real dataset

---

## LLM Providers

| Provider key | Type | Status |
|:---|:---|:---|
| `mock` | Built-in | ✅ Fully implemented |
| `qwen` | Local HuggingFace | ✅ Fully implemented |
| `openai` | API | 🔧 Stub — implement `OpenAIProvider` |
| `claude` | API | 🔧 Stub — implement `ClaudeProvider` |
| `gemini` | API | 🔧 Stub — implement `GeminiProvider` |
| `llama` | Local HuggingFace | 🔧 Stub — implement `LlamaProvider` |
| `mistral` | Local/API | 🔧 Stub — implement `MistralProvider` |
| `ollama` | Local server | 🔧 Stub — implement `OllamaProvider` |

To implement a provider, subclass `BaseLLMProvider` in `dine_ai/llm/generator.py` and register it with `ProviderRegistry.register("my_provider", MyProvider)`.
