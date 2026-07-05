# 08 — Developer API Reference

This document is for developers who are integrating DineAI into their own application. It covers the public functions you can call, what they need, and what they return.

---

## The Main Class: `DineAIApplication`

**File:** `dine_ai/app.py`

This is the only class you need to import. Everything else is handled internally.

```python
from dine_ai.app import DineAIApplication, ApplicationConfig
```

---

## Creating the App

```python
config = ApplicationConfig(
    restaurant_name="my_restaurant",
    language="English",
    llm_provider="mock"        # or "local", "openai", "google"
)

app = DineAIApplication(config)
```

---

## `app.startup()` — Start the System

**What it does:** Loads all subsystems, reads the restaurant database, prepares the AI model.

```python
app.startup()
```

Must be called before any other method. Raises `RuntimeError` if the restaurant folder doesn't exist.

---

## `app.chat(session_id, query)` — Send a Customer Message

**What it does:** Takes a customer's question, runs it through the full pipeline, and returns an AI-generated reply.

```python
response = app.chat("session-abc-123", "What's a good low-calorie lunch?")
print(response.text)
```

| Parameter | Type | Description |
|---|---|---|
| `session_id` | `str` | Unique identifier for this conversation. Use the same ID for follow-up messages. |
| `query` | `str` | The customer's question or message. |

**Returns:** `GenerationResponse` object with these fields:

| Field | Type | What it is |
|---|---|---|
| `text` | `str` | The AI's reply — this is what you show the user. |
| `generated_text` | `str` | Same as `text`. Alternative access. |
| `model_id` | `str` | Which AI model generated this. |
| `latency_ms` | `float` | How long the full query took in milliseconds. |
| `tokens_generated` | `int` | Approximate number of words generated. |

---

## `app.chat_with_filters(session_id, query, filter_rules)` — Chat With Rules

**What it does:** Same as `chat()`, but with hard rules applied before the AI responds.

```python
from dine_ai.retrieval.filtering import FilterRule

rules = [
    FilterRule(field="price", operator="lte", value=15.0),
    FilterRule(field="is_vegan", operator="eq", value=1)
]

response = app.chat_with_filters("session-1", "What do you recommend?", rules)
```

The AI can only see items that pass all filters. Items that don't pass are never considered.

**Available operators:**
| Operator | Meaning | Example |
|---|---|---|
| `eq` | equals | `price eq 15.0` |
| `neq` | not equals | `cuisine neq "Italian"` |
| `gt` | greater than | `rating gt 4.0` |
| `gte` | greater than or equal | `calories gte 300` |
| `lte` | less than or equal | `price lte 20.0` |
| `lt` | less than | `calories lt 500` |
| `contains` | text contains | `ingredients contains "chicken"` |
| `in` | value in list | `cuisine in ["Italian", "Mexican"]` |

---

## `app.switch_restaurant(restaurant_name)` — Change Active Restaurant

**What it does:** Loads a different restaurant's database without restarting the app.

```python
app.switch_restaurant("restaurant_B")
```

Useful if you're managing multiple restaurants in one application instance.

---

## `app.reset_conversation(session_id)` — Clear Chat History

**What it does:** Wipes the conversation memory for a session. Next message will be treated as a fresh conversation.

```python
app.reset_conversation("session-abc-123")
```

---

## `app.get_health()` — Check System Status

**What it does:** Returns a dictionary showing whether all subsystems are ready.

```python
status = app.get_health()
```

Returns:
```python
{
    "status": "healthy",           # or "degraded" or "not_ready"
    "application_ready": True,
    "dataset_loaded": True,
    "faiss_loaded": True,
    "generator_ready": True,
    "pipeline_ready": True
}
```

---

## `app.get_statistics()` — Get Usage Statistics

**What it does:** Returns query count, average response time, and error rate since startup.

```python
stats = app.get_statistics()
```

---

## `app.get_capabilities()` — What Can This Restaurant Do?

**What it does:** Returns the capability report for the currently loaded restaurant.

```python
caps = app.get_capabilities()
```

Returns a dict like:
```python
{
    "nutrition_filtering": True,    # has calorie/protein data
    "budget_filtering": True,       # has price data
    "rating_sorting": False,        # no rating column in this menu
    "dietary_filtering": True,      # has vegan/halal/etc labels
    "cuisine_filtering": True       # has cuisine type column
}
```

---

## `app.shutdown()` — Graceful Shutdown

```python
app.shutdown()
```

Clears resources, saves any pending state, emits a `SHUTDOWN` event.

---

## Complete Working Example

```python
from dine_ai.app import DineAIApplication, ApplicationConfig
from dine_ai.retrieval.filtering import FilterRule

# 1. Configure and start
config = ApplicationConfig(
    restaurant_name="restaurant_A",
    language="English",
    llm_provider="local"
)
app = DineAIApplication(config)
app.startup()

# 2. Simple conversation
session = "user-42"
r1 = app.chat(session, "Hi, what do you have today?")
print(r1.text)

r2 = app.chat(session, "Do you have anything vegan?")   # remembers previous message
print(r2.text)

# 3. With price filter
rules = [FilterRule(field="price", operator="lte", value=12.0)]
r3 = app.chat_with_filters(session, "Any cheap options?", rules)
print(r3.text)

# 4. Clean up
app.shutdown()
```
