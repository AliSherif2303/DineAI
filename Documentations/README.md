# DineAI — Documentation Home

> **Who this is for:** Anyone working with this project — whether you're running it, editing it, or just trying to understand what it does. No coding experience required to read most of this.

---

## 🍽️ What is DineAI in Plain English?

DineAI is the brain behind an AI-powered restaurant assistant. You give it a menu (a spreadsheet of food items), it learns the menu, and then it can answer customer questions like:

- *"What's a good high-protein meal under $20?"*
- *"Do you have anything vegan and spicy?"*
- *"What's your most popular low-calorie option?"*

It doesn't just guess — it actually reads the menu, understands it, searches it intelligently, applies your rules (like price limits or dietary filters), and then generates a natural-sounding reply using an AI language model.

---

## 📚 How the Documentation is Organized

### 📖 Section 1 — Understanding the System (Start Here)

These files explain what DineAI does and how it works. Great for anyone on the team.

| File | What it answers |
|---|---|
| [00. What is DineAI?](file:///d:/data/AI_WAITER/Documentations/00_Project_Overview.md) | Why does this exist? What problem does it solve? |
| [01. How it's Built](file:///d:/data/AI_WAITER/Documentations/01_Architecture.md) | What are the main parts and how do they fit together? |
| [02. How Data Flows](file:///d:/data/AI_WAITER/Documentations/02_Data_Flow.md) | What happens from "load the menu" to "generate a reply"? |
| [03. The Two Pipelines](file:///d:/data/AI_WAITER/Documentations/03_Pipeline.md) | What runs when you add a restaurant? What runs when a customer asks something? |
| [04. Folder Layout](file:///d:/data/AI_WAITER/Documentations/04_Folder_Structure.md) | Where do files live? What is each folder for? |
| [05. All Settings Explained](file:///d:/data/AI_WAITER/Documentations/05_Configuration.md) | Every knob and dial the system has, explained simply. |
| [06. Running It in Production](file:///d:/data/AI_WAITER/Documentations/06_Deployment.md) | How to actually run this — on your laptop, a server, or the cloud. |
| [07. Testing & Quality Checks](file:///d:/data/AI_WAITER/Documentations/07_Testing.md) | How do we know the system is working correctly? |
| [08. Developer API Reference](file:///d:/data/AI_WAITER/Documentations/08_API.md) | For developers: the exact functions, inputs, and outputs. |
| [09. Adding New Features](file:///d:/data/AI_WAITER/Documentations/09_Extensibility.md) | How to extend DineAI without breaking things. |
| [10. 🛠️ Developer Edit Guide](file:///d:/data/AI_WAITER/Documentations/10_Developer_Edit_Guide.md) | **"I want to change X — which file do I open?"** |

---

### 🔩 Section 2 — Individual Component References

Each file in `Modules/` explains one specific piece of the system. Think of these as the "instruction manual for each part."

| What It Does | Reference File |
|---|---|
| **The Main Controller** — runs everything | [app.md](file:///d:/data/AI_WAITER/Documentations/Modules/app.md) |
| **Column Normalizer** — cleans up messy spreadsheet headers | [schema_mapper.md](file:///d:/data/AI_WAITER/Documentations/Modules/schema_mapper.md) |
| **Data Checker** — validates the menu before using it | [validator.md](file:///d:/data/AI_WAITER/Documentations/Modules/validator.md) |
| **Feature Calculator** — computes things like "protein per calorie" | [feature_engineering.md](file:///d:/data/AI_WAITER/Documentations/Modules/feature_engineering.md) |
| **Text Formatter** — builds searchable and readable text from each menu item | [text_builder.md](file:///d:/data/AI_WAITER/Documentations/Modules/text_builder.md) |
| **Capability Detector** — figures out what the menu supports (e.g. calorie filtering) | [detector.md](file:///d:/data/AI_WAITER/Documentations/Modules/detector.md) |
| **Capability Definitions** — the list of features the system can detect | [definitions.md](file:///d:/data/AI_WAITER/Documentations/Modules/definitions.md) |
| **Capability Base Rules** — the contract every capability must follow | [base.md](file:///d:/data/AI_WAITER/Documentations/Modules/base.md) |
| **Capability Registry** — the master list of enabled capabilities | [registry.md](file:///d:/data/AI_WAITER/Documentations/Modules/registry.md) |
| **Menu Loader** — reads the saved menu files from disk | [loader.md](file:///d:/data/AI_WAITER/Documentations/Modules/loader.md) |
| **Rebuild Manager** — decides when to re-process the menu vs. use the cached version | [manager.md](file:///d:/data/AI_WAITER/Documentations/Modules/manager.md) |
| **File Integrity Checker** — detects if menu files were changed or corrupted | [metadata.md](file:///d:/data/AI_WAITER/Documentations/Modules/metadata.md) |
| **Build Pipeline** — the sequence that processes a menu from scratch | [pipeline.md](file:///d:/data/AI_WAITER/Documentations/Modules/pipeline.md) |
| **Embedding Builder** — converts text into numbers the AI can search | [embedding_builder.md](file:///d:/data/AI_WAITER/Documentations/Modules/embedding_builder.md) |
| **Search Index Builder** — builds a fast searchable database from those numbers | [faiss_builder.md](file:///d:/data/AI_WAITER/Documentations/Modules/faiss_builder.md) |
| **Semantic Search** — finds relevant menu items based on what the customer means | [semantic_search.md](file:///d:/data/AI_WAITER/Documentations/Modules/semantic_search.md) |
| **Filter Engine** — applies rules like "must be under $15" or "must be vegan" | [filtering.md](file:///d:/data/AI_WAITER/Documentations/Modules/filtering.md) |
| **Ranking Engine** — sorts results by best match (price + rating + relevance) | [ranking.md](file:///d:/data/AI_WAITER/Documentations/Modules/ranking.md) |
| **Prompt Builder** — writes the instruction that gets sent to the AI | [prompts.md](file:///d:/data/AI_WAITER/Documentations/Modules/prompts.md) |
| **AI Generator** — runs the language model and returns the final reply | [generator.md](file:///d:/data/AI_WAITER/Documentations/Modules/generator.md) |

---

### 🧪 Section 3 — Test Suites

| Test | What it checks |
|---|---|
| [Deterministic Scenario Tests](file:///d:/data/AI_WAITER/Documentations/Evaluations/deterministic_scenarios_test.md) | Does "high protein under $15" actually return the right dish? |
| [End-to-End Tests](file:///d:/data/AI_WAITER/Documentations/Evaluations/end_to_end_test.md) | Does the full system work from query to response? |
| [Integration Tests](file:///d:/data/AI_WAITER/Documentations/Evaluations/integration_test.md) | Does each module pass the right data to the next one? |

---

## 🧭 Suggested Reading Order

**If you're completely new** → Read `00` through `04` in order.

**If you want to make changes** → Go straight to [10. Developer Edit Guide](file:///d:/data/AI_WAITER/Documentations/10_Developer_Edit_Guide.md).

**If something broke** → Check `07_Testing.md` and the relevant `Modules/` page.
