# 07 — Testing & Quality Checks

This document explains how we verify that DineAI is working correctly — and how you can run those checks yourself.

---

## Why Testing Matters Here

DineAI has multiple moving parts: data processing, search, filtering, ranking, and AI generation. A bug in any layer could silently produce wrong results — like recommending a meat dish to a vegan customer, or showing a $50 dish when the filter said "under $15."

Our test suite is designed to catch these problems before they reach real users.

---

## The Three Test Types

### Test 1 — Deterministic Scenario Tests
**File:** `dine_ai/evaluations/deterministic_scenarios_test.py`

**What it checks:** *"Does the system return the correct result for a specific, unambiguous query?"*

These tests define exact situations with a known right answer. For example:
- Query: *"highest protein item"* → the test verifies the top result has the highest protein on the menu
- Query: *"vegan items under $10"* → all results must be vegan and under $10
- Query: *"most popular dish"* → the top result must have the highest rating

If the retrieval + filtering + ranking pipeline is working correctly, these tests pass.

**When to run:** After making any changes to `retrieval/`, `adapters/feature_engineering.py`, or `retrieval/ranking.py`.

```bash
python dine_ai/evaluations/deterministic_scenarios_test.py
```

---

### Test 2 — Integration Tests
**File:** `dine_ai/evaluations/integration_test.py`

**What it checks:** *"Does each module correctly pass its outputs to the next module?"*

These tests verify the "wiring" — that the Schema Mapper's output is what the Validator expects, that the Validator's output is what the Feature Engineer expects, and so on.

It's like checking that all the pipes in a plumbing system connect correctly, without necessarily turning the water on.

**When to run:** After modifying any module's input/output structure, or after adding a new pipeline stage.

```bash
python dine_ai/evaluations/integration_test.py
```

---

### Test 3 — End-to-End Tests
**File:** `dine_ai/evaluations/end_to_end_test.py`

**What it checks:** *"Does a full query work from start to finish?"*

These tests send a real query through the entire system and verify:
- A response is returned (not empty, not an error)
- The response doesn't leak the AI's internal instructions
- The response doesn't repeat itself
- Session history is preserved across multiple messages

**When to run:** Before deploying any changes. This is the final sanity check.

```bash
python dine_ai/evaluations/end_to_end_test.py
```

---

## Quick Self-Test (Run All Checks at Once)

The easiest way to verify everything is working:

```bash
python -m dine_ai.app --test
```

This runs a lightweight internal test that checks:
- The system starts up without errors
- The dataset loads correctly
- A sample query returns a non-empty response

Output should look like:
```
✓ Application startup successful
✓ Dataset loaded: 150 items
✓ FAISS index ready: 150 vectors
✓ Query pipeline functional
✓ All systems operational
```

---

## What "Mock Mode" Means for Testing

Many tests can be run without any actual AI model. When `llm_provider="mock"` and `embedding_provider="mock"`:
- The embedding builder returns fake but deterministic vectors
- The generator returns a canned response string instead of real AI text

This means tests are:
- Fast (no model loading)
- Repeatable (same input always produces same output)
- Cheap (no API calls)

For testing logic (routing, filtering, ranking), mock mode is preferred. For testing AI response quality, you need a real provider.

---

## Test Coverage by Layer

| Layer | Covered by |
|---|---|
| Schema Mapper | Integration Test |
| Validator | Integration Test |
| Feature Engineering | Integration Test + Deterministic Scenarios |
| Text Builder | Integration Test |
| Embedding Builder | Integration Test |
| FAISS Builder | Integration Test |
| Semantic Search | Deterministic Scenarios |
| Filtering | Deterministic Scenarios |
| Ranking | Deterministic Scenarios |
| Prompt Builder | End-to-End |
| AI Generator | End-to-End |
| Session Memory | End-to-End |

---

## Adding Your Own Tests

If you add a new feature (e.g. a new dietary filter), add a test scenario to:

```
dine_ai/evaluations/deterministic_scenarios_test.py
```

Follow the existing pattern:
```python
ScenarioTest(
    query="gluten free options",
    expected_top_item="Rice Bowl",   # must be in your test dataset
    required_flags={"is_gluten_free": True}
)
```
