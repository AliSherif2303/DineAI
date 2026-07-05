# Evaluation: End-to-End CLI Runner

**File:** `dine_ai/evaluations/end_to_end.py`

---

## What This Does (Plain English)

This is the command-line script that runs the end-to-end tests. It's essentially the entry point that sets up the test environment and calls `end_to_end_test.py`.

Think of it as the "run button" for the end-to-end test suite.

---

## How to Run

```bash
python dine_ai/evaluations/end_to_end.py
```

---

## Difference from `end_to_end_test.py`

| File | Purpose |
|---|---|
| `end_to_end_test.py` | Contains the actual test cases and assertions |
| `end_to_end.py` | The CLI entry point — configures and runs the test suite |

You typically only run `end_to_end.py`. You edit `end_to_end_test.py` when you want to change what gets tested.
