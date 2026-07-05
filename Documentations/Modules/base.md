# Module: `base.py` — Capability Contract

**Location:** `dine_ai/capabilities/base.py`

---

## What This Does (Plain English)

`base.py` defines the rules that every capability must follow. It's an abstract template — you can't use it directly, but every capability in `definitions.py` inherits from it and must implement its interface.

Think of it like a job description: any "capability" applying for the job must be able to answer two questions:
1. What columns do you need?
2. Given this menu data, are you available?

---

## The `BaseCapability` Class

Every capability inherits from this class and must implement:

| What it must have | Type | Description |
|---|---|---|
| `name` | `str` | Unique identifier (e.g. `"nutrition_filtering"`) |
| `description` | `str` | Human-readable description |
| `required_columns` | `list[str]` | Which columns this capability needs to work |
| `check(df)` | `method` | Returns `True` if this capability is available for the given DataFrame |

---

## Why This Exists

Without a base class, each capability might define its interface differently. The base class enforces consistency. The `CapabilityDetector` can then loop through all capabilities and call `check()` on each one without needing to know what kind of capability it is.

This is the "Liskov Substitution Principle" — any capability can be swapped in or out without changing the detector.

---

## How to Edit This Module

**Rarely needed.** Only edit `base.py` if you want to add a new required field or method to ALL capabilities. For example, if you wanted every capability to report its confidence level:
1. Add `confidence: float` as an abstract property to `BaseCapability`
2. All existing capability classes would then need to implement it
