# Evaluation: Integration Tests

**File:** `dine_ai/evaluations/integration_test.py`

---

## What This Test Does (Plain English)

Integration tests check the "wiring" between modules. They verify that each module's output is correctly shaped and typed for the next module to consume.

Think of it like checking that all the pipes in a plumbing system fit together correctly — before turning on the water.

**What it checks:**
- Does Schema Mapper output a DataFrame with the right column names?
- Does Validator accept the Schema Mapper's output without crashing?
- Does Feature Engineer produce the expected new columns?
- Does Text Builder create non-empty `search_text` for every row?
- Does Embedding Builder produce vectors of the correct dimension?
- Does FAISS Builder create a searchable index from those vectors?

---

## Why These Tests Matter

Individual unit tests check each module in isolation. Integration tests check that they work *together*. A module could work perfectly in isolation but fail because it expects a column name that the previous module uses a different name for.

---

## How to Run

```bash
python dine_ai/evaluations/integration_test.py
```

Expected output:
```
[PASS] Schema Mapper: 6 columns normalized
[PASS] Validator: 0 errors, 2 warnings (non-fatal)
[PASS] Feature Engineer: 8 new columns added
[PASS] Text Builder: search_text, text_chunk, summary_text created
[PASS] Embedding Builder: vectors shape (150, 384)
[PASS] FAISS Builder: index contains 150 vectors
Integration test: ALL PASSED
```

---

## When to Run

- After modifying any module's input or output structure
- After adding a new column to the pipeline
- After adding a new pipeline stage
- Before any commit touching the `adapters/`, `embeddings/`, or `dataset/` folders

---

## How to Add a New Integration Check

Find the test class and add a new `test_*` method:

```python
def test_my_new_module(self):
    # Feed it valid input
    result = MyNewModule().transform(self.sample_df)
    
    # Check the output is correct
    assert "my_new_column" in result.columns
    assert result["my_new_column"].notna().all()
```
