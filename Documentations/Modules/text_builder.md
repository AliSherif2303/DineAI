# Module: `text_builder.py` — Text Generator

**Location:** `dine_ai/adapters/text_builder.py`

---

## What This Does (Plain English)

Each menu item in your spreadsheet is a row of numbers and codes. But the search system and the AI need text. The Text Builder reads each row and writes three different text versions of each item.

Think of it like writing three different labels for the same product:
1. A **search tag** (for the search engine to index)
2. A **detailed card** (for the AI to read when recommending)
3. A **one-liner** (for quick previews)

---

## The Three Text Formats

### 1. `search_text` — Compact Search String
This is what gets embedded and indexed for semantic search. It packs all the key facts about a dish into one compact string.

**Example:**
```
"Truffle Burger American beef lettuce tomato truffle aioli halal 750 calories 45g protein $25 premium"
```

### 2. `text_chunk` — Rich AI Context Block
This is what gets sent to the AI when it needs to recommend this dish. Formatted as a mini data card.

**Example:**
```markdown
**Truffle Burger** | Cuisine: American | Price: $25.00
- Calories: 750 | Protein: 45g | Carbs: 40g
- Dietary: Halal, High-Protein
- Description: Grass-fed beef patty with black truffle aioli on brioche bun
```

### 3. `summary_text` — One-Liner
A single short sentence for quick display.

**Example:**
```
"Truffle Burger — American cuisine — $25.00 — 750 cal"
```

---

## Why Three Formats?

| Format | Used by | Optimized for |
|---|---|---|
| `search_text` | Embedding builder | Dense, keyword-rich — good for semantic similarity |
| `text_chunk` | Prompt builder | Readable, structured — good for AI comprehension |
| `summary_text` | UI / preview | Short — good for listing results |

---

## How to Edit This Module

**To add a new field to the search text** (e.g. add spice level):
1. Open `text_builder.py`
2. Find the `SearchTextBuilder` or equivalent class
3. Append your field to the output string:
   ```python
   if "spice_level" in row:
       parts.append(f"spice level {row['spice_level']}")
   ```

**To change the AI context card format** (e.g. add chef's notes):
1. Find the `TextChunkBuilder` class
2. Modify the template string to include the new field
3. Make sure to handle the case where the field is missing (`None` or `NaN`)

**To change the one-liner format:**
1. Find `SummaryTextBuilder`
2. Modify the f-string template
