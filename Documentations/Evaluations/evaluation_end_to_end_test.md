# Evaluation: `evaluation/end_to_end_test.py` (Wrapper)

**File:** `dine_ai/evaluation/end_to_end_test.py`

---

## What This Is

This is a compatibility wrapper in the `evaluation/` folder (singular). It re-exports from the real implementation in `evaluations/` (plural).

> [!NOTE]
> The real end-to-end test logic is in:
> `dine_ai/evaluations/end_to_end_test.py`
>
> See [end_to_end_test.md](file:///d:/data/AI_WAITER/Documentations/Evaluations/end_to_end_test.md) for the full documentation.

This wrapper exists so code that imports `from dine_ai.evaluation.end_to_end_test import ...` continues to work.

---

## Do You Need to Edit This?

No — only edit the real file in `evaluations/`.
