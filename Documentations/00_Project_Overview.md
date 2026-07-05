# 00 — What is DineAI? (Project Overview)

## The Problem It Solves

Imagine you're a restaurant owner and you want to add a chatbot to your website. The chatbot needs to answer questions like:
- *"What's on the menu?"*
- *"Do you have anything gluten-free under $12?"*
- *"What's the most popular dish?"*

The obvious answer is to use a general-purpose AI like ChatGPT — but there's a serious problem: **general AI models make things up.** They'll confidently recommend a "Truffle Burger" that doesn't exist, invent a price, or suggest a vegan option that contains bacon.

DineAI was built to solve exactly this. It wraps AI generation inside a strict, verifiable retrieval system. Before the AI writes a single word, DineAI has already:
1. Loaded your actual menu
2. Found the most relevant items
3. Filtered out anything that doesn't match the customer's requirements
4. Ranked the remaining options
5. Only then handed the results to the AI to write a nice reply

**The AI can only talk about what's actually on the menu.**

---

## What DineAI Does, Step by Step

### Step 1 — You Give It a Menu
You provide a spreadsheet (CSV file) with your menu items. Each row is a dish. Columns can include name, price, calories, ingredients, cuisine type, dietary labels, and more.

### Step 2 — It Learns the Menu
DineAI processes the spreadsheet: cleans it up, computes things it doesn't have (like "protein per calorie"), builds searchable text for each item, and creates a fast search index. This is saved to disk so it doesn't repeat the work every time.

### Step 3 — A Customer Asks Something
A customer types something natural like *"I want something cheap and healthy"*. DineAI converts that into a mathematical search, finds the most relevant menu items, filters by any active rules (price limits, dietary restrictions), ranks the results, and sends only the best matches to the AI.

### Step 4 — The AI Writes a Reply
The AI reads the pre-filtered, ranked menu items and writes a natural, conversational response. It cannot invent dishes because it was only given real ones.

---

## What Makes It Different

| Regular Chatbot | DineAI |
|---|---|
| Can hallucinate menu items | Can only reference real menu items |
| Ignores price/dietary rules | Enforces hard rules before responding |
| One-size-fits-all personality | Configurable persona (waiter, chef, health coach) |
| Needs manual updates | Auto-detects when the menu changes and rebuilds |
| Works with one AI only | Swappable AI backends (local, OpenAI, Google, etc.) |

---

## Key Features (Plain English)

- **Auto-Column Matching** — Your spreadsheet can use any column names. DineAI figures out what `"Dish Name"`, `"Cost"`, and `"Calories Per Serving"` mean automatically.
- **Smart Dietary Detection** — It reads the ingredients and labels to figure out if something is vegan, keto, halal, gluten-free, etc.
- **Offline Testing Mode** — You can run the entire system without any internet connection or AI model using "mock mode". Good for testing.
- **Multi-Language Replies** — The assistant can respond in English, Arabic, French, Spanish, German, and more.
- **Plug-and-Play AI Models** — Switch between a local Qwen model, OpenAI's API, Google Gemini, or others by changing one config line.
- **Auto-Rebuild When Menu Changes** — If you update the spreadsheet, DineAI automatically detects the change and rebuilds everything.

---

## Technology Used (for the curious)

| Tool | What it does in DineAI |
|---|---|
| **Python** | The programming language everything is written in |
| **Pandas** | Reads and processes the menu spreadsheet |
| **FAISS** | Ultra-fast search index (like a Google for your menu) |
| **SentenceTransformers** | Converts text to numbers for semantic search |
| **HuggingFace Transformers** | Runs AI models locally on your computer |
| **NumPy** | Math operations for ranking and scoring |
