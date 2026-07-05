# Module: `embedding_builder.py` — Text-to-Numbers Converter

**Location:** `dine_ai/embeddings/embedding_builder.py`

---

## What This Does (Plain English)

To search for menu items by *meaning* (not just keywords), the system needs to convert text into numbers. The Embedding Builder does exactly this — it runs each menu item's `search_text` through an AI model and gets back a list of 384 numbers called a "vector" or "embedding."

Two pieces of text that mean similar things will have similar numbers. This is how the system can find "grilled chicken" when a customer asks for "something with lean protein."

---

## What Goes In / What Comes Out

**Input:** A list of text strings (one per menu item):
```python
["Truffle Burger American beef 750cal $25", "Garden Salad vegan 120cal $10", ...]
```

**Output:** A matrix of numbers, one row per item:
```python
[
    [0.12, -0.34, 0.87, ...],   # 384 numbers for "Truffle Burger"
    [0.44, 0.21, -0.56, ...],   # 384 numbers for "Garden Salad"
    ...
]
```

This matrix gets saved as `embeddings.npy` and used to build the FAISS index.

---

## Supported Providers

| Provider | When to use | Requires |
|---|---|---|
| `MockEmbeddingProvider` | Testing (returns random vectors) | Nothing |
| `SentenceTransformerProvider` | Local deployment | `sentence-transformers` library |
| OpenAI Embedding | Cloud deployment | `OPENAI_API_KEY` env var |
| Google Embedding | Cloud deployment | `GOOGLE_API_KEY` env var |

---

## The Default Model

When using the local provider, the default model is `BAAI/bge-small-en-v1.5`. This model:
- Is free to use
- Downloads automatically on first run (~100MB)
- Produces 384-dimensional vectors
- Works well for English menu text

> [!NOTE]
> If you change the model, you must delete the old `faiss.index` and `embeddings.npy` files and let the system rebuild. Different models produce incompatible vector formats.

---

## Batch Processing

Rather than processing one item at a time, the Embedding Builder processes all items in batches (default: 64 items per batch). This is more efficient and avoids memory issues with large menus.

---

## How to Edit This Module

**To change the embedding model:**
1. Open `dine_ai/app.py`
2. Change `embedding_model` in `ApplicationConfig`
3. Delete `faiss.index` and `embeddings.npy` from your restaurant folder
4. Restart — it will rebuild

**To change batch size** (if running out of memory):
1. Open `embedding_builder.py`
2. Find `batch_size` parameter (default: 64)
3. Reduce it (e.g. 32 or 16)

**To add a new embedding provider:**
1. Create a class implementing `BaseEmbeddingProvider`
2. Implement `encode(texts: List[str]) -> np.ndarray`
3. Register it in `app.py`'s provider resolution logic
