# Evaluation: `evaluation/` Wrapper Modules

**Files:**
- `dine_ai/evaluation/end_to_end_test.py`
- `dine_ai/evaluation/integration_test.py`

---

## What These Do (Plain English)

These files in the `evaluation/` folder (singular, no 's') are thin wrappers around the real test files in `evaluations/` (plural, with 's').

They exist for backwards compatibility and library import paths. If some external code imports from `dine_ai.evaluation`, these files ensure it still works.

---

## Important Note

> [!NOTE]
> These are **not** the real test files. The actual test logic lives in:
> - `dine_ai/evaluations/integration_test.py`
> - `dine_ai/evaluations/end_to_end_test.py`
>
> When adding or editing tests, always edit the files in `evaluations/` (with the 's').

---

## Do You Need to Edit These?

**Rarely.** Only if:
- You rename or move the real test files
- You want to add an additional import alias

Otherwise, leave them alone.
