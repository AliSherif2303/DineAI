# 06 — Running DineAI (Deployment Guide)

This guide explains how to actually run DineAI — locally on your laptop, on a team server, or in the cloud.

---

## Prerequisites

Before running anything, make sure you have:
- Python 3.9 or higher installed
- Pip installed (comes with Python)
- The project files downloaded to your machine

---

## Step 1 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all the Python libraries DineAI needs (FAISS, SentenceTransformers, HuggingFace, etc.).

> [!NOTE]
> First install can take 5–15 minutes because it downloads AI model libraries. This is a one-time step.

---

## Step 2 — Add Your Restaurant Data

Create a folder with your restaurant's name inside `dine_ai/datasets/`:

```
dine_ai/datasets/
└── my_restaurant/
    └── menu_items.csv      ← your menu spreadsheet goes here
```

The CSV file must have at minimum:
- A column for **dish name** (can be called anything — DineAI will recognize it)
- Ideally also: price, calories, ingredients, cuisine type

> [!TIP]
> You can use the existing `restaurant_A` dataset to test before adding your own data.

---

## Step 3 — Configure Your Settings

Open `dine_ai/app.py` and update the `ApplicationConfig` defaults (or pass them when creating the app).

Minimum configuration to run:
```python
config = ApplicationConfig(
    restaurant_name="my_restaurant",   # must match your folder name
    embedding_provider="mock",         # use "mock" for testing, "local" for real
    llm_provider="mock"                # use "mock" for testing, "local" for real
)
```

---

## Step 4 — Run DineAI

### Option A — Self-Test Mode (Recommended First)
Verifies the system is working without needing any user input:
```bash
python -m dine_ai.app --test
```

If you see `✓ All systems operational`, you're good to go.

### Option B — Interactive CLI
Start a conversation from the command line:
```bash
python -m dine_ai.app
```
You'll be prompted to type questions. Type `exit` to quit.

### Option C — Use It in Your Own Code
```python
from dine_ai.app import DineAIApplication, ApplicationConfig

config = ApplicationConfig(restaurant_name="my_restaurant")
app = DineAIApplication(config)
app.startup()

response = app.chat("session-1", "What do you recommend for a vegan?")
print(response.text)
```

---

## Deployment Profiles

### Profile 1 — Local Development (No AI Models)
Best for: Testing logic, developing new features, running tests fast.

```python
ApplicationConfig(
    restaurant_name="restaurant_A",
    embedding_provider="mock",
    llm_provider="mock",
    debug=True
)
```

- ✅ Instant startup
- ✅ No GPU required
- ✅ No internet required
- ❌ Responses are placeholder text, not real AI

---

### Profile 2 — Local with Real AI (GPU Machine)
Best for: Full demo, development with real AI responses.

```python
ApplicationConfig(
    restaurant_name="restaurant_A",
    embedding_provider="local",
    llm_provider="local",
    llm_model="Qwen/Qwen2.5-7B-Instruct"
)
```

- ✅ Real AI responses
- ✅ No API costs
- ⚠️ Requires 8–16GB RAM (more with larger models)
- ⚠️ Slow cold start (~30–60 seconds first time, model downloads)

---

### Profile 3 — Cloud API (Production)
Best for: Deployed applications with real users.

```python
ApplicationConfig(
    restaurant_name="my_restaurant",
    embedding_provider="openai",
    llm_provider="openai",
    cache_enabled=True
)
```

Required environment variables:
```bash
# Windows
set OPENAI_API_KEY=sk-...

# Linux / Mac
export OPENAI_API_KEY=sk-...
```

- ✅ Fast, scalable, high quality
- ✅ No GPU required on your server
- ❌ API costs money per query
- ❌ Requires internet

---

## What Happens on First Startup

When you run DineAI for the first time with a new restaurant:

1. It checks if `recipes.pkl`, `faiss.index`, and `metadata.json` exist.
2. If they don't → it runs the full build pipeline (reads CSV → processes → saves).
3. On subsequent runs → it loads from the saved files (fast).

This means the first run is always slower than subsequent runs.

---

## Updating the Menu

If you change your `menu_items.csv`:
1. DineAI will automatically detect the change on next startup (or query if `auto_reload=True`).
2. It will rebuild the database.
3. The new menu is active immediately after the rebuild completes.

You don't need to delete anything manually.

---

## Health Check

To verify everything is running correctly at any time:

```python
health = app.get_health()
print(health)
```

Returns:
```json
{
  "status": "healthy",
  "application_ready": true,
  "dataset_loaded": true,
  "faiss_loaded": true,
  "generator_ready": true
}
```

---

## Common Problems

| Problem | Likely Cause | Fix |
|---|---|---|
| `FileNotFoundError: menu_items.csv` | Wrong restaurant name in config | Check that `restaurant_name` matches your folder name |
| `CUDA out of memory` | GPU too small for model | Try a smaller model, or switch to `embedding_provider="mock"` |
| `ImportError: faiss` | FAISS not installed | Run `pip install faiss-cpu` |
| Very slow first query | Model is loading | Wait — subsequent queries will be fast |
| Replies are placeholder text | `llm_provider="mock"` is set | Switch to `"local"` or `"openai"` for real responses |
