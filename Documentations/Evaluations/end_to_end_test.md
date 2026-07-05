# Evaluation: End-to-End Tests

**File:** `dine_ai/evaluations/end_to_end_test.py`

---

## What This Test Does (Plain English)

End-to-end tests run the entire system from start to finish — exactly like a real user would use it. They start the app, send a real query, and verify that a sensible response comes back.

These are the most comprehensive tests. If they pass, you can be confident the whole chain works.

**What they check:**
1. The app starts without errors
2. A query returns a non-empty response
3. The response doesn't repeat the system prompt back to the user (prompt leakage)
4. The response doesn't contain obvious repetition
5. Sending a follow-up question works (session memory is maintained)
6. Switching restaurants mid-session works

---

## Why These Tests Matter

Unit and integration tests can pass while the end product is still broken — perhaps the AI prompt is malformed, or the session memory isn't being maintained. End-to-end tests are the only way to verify the complete experience.

---

## How to Run

```bash
python dine_ai/evaluations/end_to_end_test.py
```

Expected output:
```
[PASS] Startup: Application ready in 0.8s
[PASS] Query 1: Response received (127 tokens, 2.1s)
[PASS] Query 1: No prompt leakage detected
[PASS] Query 1: No repetition detected
[PASS] Query 2 (follow-up): Session memory preserved
[PASS] Restaurant switch: restaurant_B loaded successfully
End-to-end test: ALL PASSED
```

---

## When to Run

- Before merging any changes into main/production
- After changing `prompts.py` or `generator.py`
- After changing the session memory system
- After adding a new AI provider

---

## Running in Mock Mode vs. Real Mode

By default, end-to-end tests run with `llm_provider="mock"`. This means:
- ✅ Fast (no model loading)
- ✅ Repeatable
- ❌ Doesn't test actual AI output quality

To test with a real model:
```python
# In end_to_end_test.py, change:
config = ApplicationConfig(llm_provider="local")
```

> [!NOTE]
> Running with a real model requires the model to be downloaded and a GPU (or a lot of patience). Keep mock mode for CI/CD pipelines.
