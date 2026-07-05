# Module: `metadata.py` — File Integrity Checker

**Location:** `dine_ai/dataset/metadata.py`

---

## What This Does (Plain English)

This module is responsible for creating and reading the `metadata.json` file that sits in each restaurant's folder.

Its main job is file integrity: computing a SHA-256 "fingerprint" of the menu CSV so that the system can detect if anything changed. It also stores the capability report and build timestamp.

---

## What SHA-256 Means

SHA-256 is a mathematical algorithm that takes any file and produces a 64-character string. If even a single character in the file changes, the output is completely different. This makes it perfect for detecting changes without comparing the full file content.

Example:
```
File: "Truffle Burger, $25, 750 cal"
Hash: "a3f9c2b4e7d1..."

After change: "Truffle Burger, $26, 750 cal"  (price went up $1)
Hash: "87b4a1d9f3e2..."    (completely different)
```

---

## What's in `metadata.json`

```json
{
    "csv_hash": "a3f9c2b4e7d1f8...",
    "embedding_model": "BAAI/bge-small-en-v1.5",
    "build_timestamp": "2024-01-15T10:30:00Z",
    "total_items": 150,
    "capabilities": {
        "nutrition_filtering": true,
        "budget_filtering": true,
        "rating_sorting": false,
        "dietary_filtering": true,
        "cuisine_filtering": true
    },
    "schema_version": "1.0"
}
```

---

## How to Edit This Module

**To add a new field to metadata** (e.g. track the Python version):
1. Open `metadata.py`
2. Find where the metadata dict is assembled
3. Add your field:
   ```python
   import sys
   metadata["python_version"] = sys.version
   ```
4. Add a check in `manager.py` if this field should trigger a rebuild when it changes

**To change what triggers a hash mismatch:**
The hash is computed from the CSV file content. You can also hash additional files (e.g. a config file) by passing them to the hash function.
