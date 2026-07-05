# 05 — All Settings Explained (Configuration)

All DineAI settings are controlled through one configuration object called `ApplicationConfig`. This lives in `dine_ai/app.py`.

This document explains every setting in plain English: what it does, what values it accepts, and when you'd want to change it.

---

## Where to Change Settings

Open this file:
```
d:\data\AI_WAITER\dine_ai\app.py
```

Find the `ApplicationConfig` class (around line 49). Change the default values there, or pass new values when creating the application:

```python
app = DineAIApplication(ApplicationConfig(
    restaurant_name="my_restaurant",
    language="Arabic",
    llm_provider="openai"
))
```

---

## All Settings

### 🏪 Restaurant Settings

| Setting | Default | What it does |
|---|---|---|
| `restaurant_name` | `"restaurant_A"` | The name of the folder in `dine_ai/datasets/` to use. Change this to switch restaurants. |
| `dataset_directory` | `"dine_ai/datasets"` | The parent folder where all restaurant data lives. Only change if you moved the data somewhere else. |

---

### 🌍 Language Settings

| Setting | Default | What it does |
|---|---|---|
| `language` | `"English"` | The language the AI will reply in. Accepted values: `"English"`, `"Arabic"`, `"French"`, `"Spanish"`, `"German"`, `"Italian"` |

> [!NOTE]
> This changes the AI's *output* language. The menu data itself doesn't need to be in that language.

---

### 🤖 Embedding Settings (for menu processing)

These settings control how the menu is converted into searchable numbers. They affect the build pipeline, not the chat response.

| Setting | Default | What it does |
|---|---|---|
| `embedding_provider` | `"mock"` | Which provider to use for embeddings. Options: `"mock"` (fake, for testing), `"local"` (runs on your machine), `"openai"`, `"google"` |
| `embedding_model` | `"BAAI/bge-small-en-v1.5"` | Which specific model to use. Only matters when `embedding_provider` is `"local"`. |

> [!IMPORTANT]
> If you change `embedding_model`, you **must** trigger a full rebuild. The old index won't be compatible.

---

### 💬 AI Language Model Settings (for generating replies)

These settings control the actual AI that writes responses.

| Setting | Default | What it does |
|---|---|---|
| `llm_provider` | `"mock"` | Which AI to use for responses. Options: `"mock"` (canned responses, no model needed), `"local"` (runs Qwen on your machine), `"openai"` (calls OpenAI API), `"google"` (calls Gemini API) |
| `llm_model` | `"Qwen/Qwen2.5-7B-Instruct"` | The specific model name. Used when `llm_provider = "local"`. |
| `execution_mode` | `"local"` | How the model runs. Options: `"local"` (your GPU/CPU), `"api"` (remote service). |
| `stream` | `False` | When `True`, the reply is streamed word-by-word (like ChatGPT typing effect). |

---

### 🔍 Retrieval Settings (how many results the AI sees)

| Setting | Default | What it does |
|---|---|---|
| `max_context_recipes` | `5` | How many menu items to include in the AI's context. Higher = more options for the AI to choose from, but slower and uses more memory. Range: 1–20. |

---

### 🔧 System Behavior Settings

| Setting | Default | What it does |
|---|---|---|
| `cache_enabled` | `True` | When `True`, the system loads from saved files instead of rebuilding every time. Set to `False` only for debugging. |
| `auto_reload` | `False` | When `True`, the system checks for menu changes on every query and rebuilds if needed. Performance cost: ~100ms per query. |
| `debug` | `False` | Enables detailed internal logging. Only for developers investigating bugs. |
| `verbose` | `False` | Prints extra information to console during operations. |
| `logging` | `True` | Enables the logging system overall. |
| `enable_reports` | `True` | Generates a capability report after loading a restaurant. |

---

## Recommended Settings by Use Case

### During Development / Testing
```python
ApplicationConfig(
    restaurant_name="restaurant_A",
    embedding_provider="mock",
    llm_provider="mock",
    debug=True,
    verbose=True
)
```
No AI models loaded. Super fast. Great for testing logic without waiting for model loads.

---

### Local Demo (GPU Machine)
```python
ApplicationConfig(
    restaurant_name="restaurant_A",
    embedding_provider="local",
    embedding_model="BAAI/bge-small-en-v1.5",
    llm_provider="local",
    llm_model="Qwen/Qwen2.5-7B-Instruct",
    execution_mode="local"
)
```
Everything runs on your machine. No API keys needed.

---

### Production (Cloud API)
```python
ApplicationConfig(
    restaurant_name="my_restaurant",
    embedding_provider="openai",
    llm_provider="openai",
    language="English",
    cache_enabled=True,
    max_context_recipes=5
)
```
Uses OpenAI for both embeddings and replies. Requires `OPENAI_API_KEY` environment variable.

---

### Multi-Language Restaurant
```python
ApplicationConfig(
    restaurant_name="restaurant_arabic",
    language="Arabic",
    llm_provider="local"
)
```
AI replies will be written in Arabic.

---

## Environment Variables (Secrets)

Never put API keys directly in code. Use environment variables:

| Variable | Used when |
|---|---|
| `OPENAI_API_KEY` | `llm_provider="openai"` or `embedding_provider="openai"` |
| `GOOGLE_API_KEY` | `llm_provider="google"` |
