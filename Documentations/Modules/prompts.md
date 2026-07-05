# Module: `prompts.py` — AI Instruction Builder

**Location:** `dine_ai/llm/prompts.py`

---

## What This Does (Plain English)

Before the AI can write a response, someone needs to write the instruction for it. That's the Prompt Builder's job.

It takes:
- The customer's question
- The ranked list of relevant menu items
- The current conversation history
- The language setting
- The persona (personality style)

And assembles them into a complete, structured instruction that gets sent to the AI model.

Think of it like writing a briefing for an employee: *"Here's who you are, here's what the customer wants, here's the relevant information. Now write a helpful reply."*

---

## What a Prompt Looks Like (Simplified)

```
SYSTEM:
You are a friendly, helpful restaurant waiter named Zara.
Respond only in English. Be warm and concise.

CONTEXT:
The customer asked: "Something healthy under $15"

Here are the best matching menu items:

1. Grilled Salmon — Mediterranean — $14.50
   Calories: 380 | Protein: 42g | Fat: 14g
   Dietary: Gluten-Free, High-Protein
   
2. Quinoa Buddha Bowl — Vegan — $12.00
   Calories: 290 | Protein: 12g | Carbs: 45g

CONVERSATION HISTORY:
User: "I want something healthy"
Assistant: "Of course! What's your budget?"

USER MESSAGE:
Something healthy under $15

INSTRUCTION:
Recommend from the items above. Do not invent dishes.
```

---

## Personas (AI Personality Styles)

DineAI supports multiple built-in personas:

| Persona Name | Personality |
|---|---|
| `FriendlyWaiter` | Warm, conversational, helpful waiter |
| `HealthCoach` | Focused on nutrition, macros, health goals |
| `Chef` | Talks about cooking style, flavors, techniques |
| `BudgetAdvisor` | Emphasizes value, price comparisons |

The default is `FriendlyWaiter`.

---

## Token Budget Management

A "token" is roughly a word. AI models have a limit on how much text they can read at once (typically 2048–8192 tokens).

The Prompt Builder actively manages this budget:
1. It counts how many tokens the system message, history, and menu items will use
2. If the total exceeds the budget, it compresses the menu items (using configurable strategies)
3. It never sends a prompt that exceeds the model's limit

**Compression Strategies:**

| Strategy | What it does |
|---|---|
| `metadata` (default) | Trims detailed text, keeps name/price/calories |
| `top-k` | Reduces to fewer menu items |
| `summary` | Replaces full details with one-line summaries |
| `nutrition` | Keeps only nutrition data, removes description |
| `ingredient` | Keeps only ingredients, removes other details |

---

## Localization

All field labels in the AI context card (like "Protein:", "Calories:", "Price:") are translated based on the `language` setting. Supported languages and their translations are defined in `LOCALIZATION_DATA` at the top of `prompts.py`.

---

## How to Edit This Module

**To add a new persona:**
1. Open `prompts.py`
2. Find `PersonaRegistry`
3. Register a new persona:
   ```python
   PersonaRegistry.register("MasterChef", 
       system_prompt="""You are a Michelin-starred chef assistant.
       Focus on ingredients, technique, and flavor profiles.
       Be sophisticated but approachable."""
   )
   ```

**To add a new language:**
1. Open `prompts.py`
2. Find `LOCALIZATION_DATA`
3. Add a new language block:
   ```python
   "italian": {
       "recipe_name": "Nome della ricetta",
       "protein": "Proteine",
       "calories": "Calorie",
       "price": "Prezzo",
       ...
   }
   ```

**To change the default token budget:**
1. Find `PromptConfig`
2. Change `max_budget_tokens` (default: 2048)
