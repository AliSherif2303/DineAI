# Module: `app.py` — The Main Controller

**Location:** `dine_ai/app.py`

---

## What This Does (Plain English)

`app.py` is the front door to DineAI. It's the only file you need to import when using DineAI in your own code.

It coordinates everything else: starts the system, loads the restaurant data, runs queries through the pipeline, tracks conversation history, and handles shutting down cleanly.

Think of it like the manager of a restaurant — they don't cook the food or seat the customers themselves, but they make sure everyone else does their job in the right order.

---

## Key Classes

### `ApplicationConfig`
The settings object. Contains all the knobs and dials for the system.
→ See [05_Configuration.md](file:///d:/data/AI_WAITER/Documentations/05_Configuration.md) for full details.

### `DineAIApplication`
The main class you instantiate. Has these public methods:

| Method | What it does |
|---|---|
| `startup()` | Starts up the whole system |
| `chat(session_id, query)` | Sends a question through the pipeline |
| `chat_with_filters(session_id, query, rules)` | Chat with hard filter rules |
| `switch_restaurant(name)` | Load a different restaurant |
| `reset_conversation(session_id)` | Clear session memory |
| `get_health()` | Check if everything is running |
| `get_statistics()` | Get usage stats |
| `get_capabilities()` | See what features are available |
| `shutdown()` | Shut down cleanly |

### `PipelineContext`
The "shared bag" that travels through each pipeline stage. Contains:
- The customer's original query
- The search candidates
- The filtered candidates
- The ranked candidates
- The generated response
- Warnings and timing information

### `ApplicationState`
Tracks whether each subsystem has loaded successfully. The system won't accept queries until all state flags are `True`.

---

## The Pipeline Stages (Inside `app.py`)

`app.py` defines the pipeline stages as classes:

| Stage Class | What it runs |
|---|---|
| `SemanticSearchStage` | Calls `semantic_search.py` |
| `FilteringStage` | Calls `filtering.py` |
| `RankingStage` | Calls `ranking.py` |
| `PromptBuildingStage` | Calls `prompts.py` |
| `GenerationStage` | Calls `generator.py` |

Each stage has three methods: `validate_input()`, `execute()`, `validate_output()`. If any stage fails, the error is recorded but execution continues where possible.

---

## Events System

The application emits events at key lifecycle moments:

| Event | When it fires |
|---|---|
| `STARTUP` | After successful startup |
| `SHUTDOWN` | Before shutting down |
| `QUERY_STARTED` | When a chat() call begins |
| `QUERY_COMPLETED` | When a chat() call finishes |
| `PIPELINE_FAILED` | If a pipeline stage throws an error |
| `RESTAURANT_SWITCHED` | After a restaurant is loaded |

You can attach event handlers to respond to these events.

---

## How to Edit This Module

**To change default settings:**
Modify the default values inside `ApplicationConfig`.

**To add a new pipeline stage:**
1. Create a class inheriting `BasePipelineStage`
2. Implement `execute(context: PipelineContext) -> PipelineContext`
3. Add it to the pipeline sequence in `DineAIApplication._run_pipeline()`

**To add a new event type:**
1. Add the new event to `ApplicationEventType` enum
2. Emit it at the appropriate place using `self._emit_event(ApplicationEventType.YOUR_EVENT, {...})`
