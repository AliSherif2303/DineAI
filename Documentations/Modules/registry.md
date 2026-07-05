# Module: `registry.py` — Capability Master List

**Location:** `dine_ai/capabilities/registry.py`

---

## What This Does (Plain English)

The Capability Registry is a central catalog that holds all registered capabilities. The Capability Detector asks the Registry for the full list, then checks each one against the menu data.

It's like a phone directory — it doesn't know the details of each capability, it just knows where they all are so the detector can find them.

---

## How It Works

```python
# Registry is populated by definitions.py
CapabilityRegistry.register(NutritionCapability)
CapabilityRegistry.register(BudgetCapability)
CapabilityRegistry.register(RatingCapability)

# Detector asks the registry for all capabilities
capabilities = CapabilityRegistry.get_all()
for cap in capabilities:
    is_available = cap.check(df)
```

---

## How to Edit This Module

**You don't usually edit `registry.py` directly.** The registry registration happens in `definitions.py` — when you add a new capability there and add it to `ALL_CAPABILITIES`, it automatically gets registered.

**If you need to explicitly register a capability** from a different file:
```python
from dine_ai.capabilities.registry import CapabilityRegistry
CapabilityRegistry.register(MyCustomCapability)
```

**To unregister a capability at runtime:**
```python
CapabilityRegistry.unregister("nutrition_filtering")
```
