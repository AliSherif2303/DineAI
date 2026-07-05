# Module: `generator.py` — AI Response Engine

**Location:** `dine_ai/llm/generator.py`

---

## What This Does (Plain English)

The Generator is the last step in the pipeline. It takes the assembled prompt from the Prompt Builder, sends it to the configured AI model, and returns the AI's response.

It also handles:
- Loading the model (which can be slow the first time)
- Memory management (unloading the model when not in use to free GPU memory)
- Response validation (checking the output for problems)
- Caching (so identical prompts don't cost extra API calls)
- Response statistics (token count, latency)

---

## Supported AI Providers

| Provider Key | What runs | Where it runs |
|---|---|---|
| `"mock"` | Returns a canned placeholder text | Locally, instant |
| `"local"` / `"qwen"` | Runs Qwen or another local model | Your machine (GPU/CPU) |
| `"openai"` | Calls OpenAI API (GPT-4, etc.) | OpenAI servers |
| `"gemini"` | Calls Google Gemini API | Google servers |
| `"claude"` | Calls Anthropic Claude API | Anthropic servers |
| `"ollama"` | Calls local Ollama server | Your machine via HTTP |
| `"mistral"` | Runs Mistral model | Locally or API |

---

## LLMConfig — Model Setup Settings

| Setting | Default | Meaning |
|---|---|---|
| `provider` | `"mock"` | Which AI backend to use |
| `model_name` | `"Qwen/Qwen2.5-7B-Instruct"` | Specific model identifier |
| `execution_mode` | `"local"` | `"local"` = your machine, `"api"` = remote |
| `device` | `"auto"` | `"auto"` = pick GPU if available, else CPU |
| `lazy_loading` | `True` | Load model only when first query arrives |
| `quantization` | `None` | `"4bit"` or `"8bit"` to reduce memory usage |
| `auto_unload` | `False` | Unload model after each response to free memory |
| `fallback_model` | `None` | Use this model if the primary fails to load |

---

## GenerationConfig — Response Style Settings

| Setting | Default | What it controls |
|---|---|---|
| `temperature` | `0.1` | Creativity: 0 = deterministic, 2 = very creative |
| `max_new_tokens` | `384` | Maximum length of the AI's response (in tokens/words) |
| `repetition_penalty` | `1.1` | Discourages the AI from repeating phrases |
| `do_sample` | `False` | Whether to use random sampling (False = deterministic) |
| `stream` | `False` | Stream response word by word |
| `timeout` | `60.0` | Maximum seconds to wait for a response |

---

## GenerationResponse — What Comes Out

| Field | What it contains |
|---|---|
| `generated_text` | The AI's response text |
| `model_id` | Which model produced the response |
| `latency_ms` | Time taken in milliseconds |
| `tokens_generated` | Approximate number of tokens in the response |
| `finish_reason` | Why generation stopped (`"stop"`, `"length"`, `"timeout"`) |
| `cached` | `True` if response came from cache |

---

## Response Validation

Before returning the response, the engine checks for:
- **Prompt leakage** — Did the AI accidentally repeat the system instructions?
- **Empty response** — Is the response blank?
- **Repetition** — Is the AI repeating the same sentence multiple times?

If any of these are detected, a warning is added to the context and the response may be cleaned up.

---

## How to Edit This Module

**To make the AI give longer responses:**
1. Open `generator.py`
2. Find `GenerationConfig`
3. Increase `max_new_tokens` (e.g. from 384 to 600)

**To make the AI more creative/varied:**
1. Find `GenerationConfig`
2. Increase `temperature` (e.g. from 0.1 to 0.7)
3. Set `do_sample=True`

**To add a new AI provider:**
1. Create a class inheriting `BaseLLMProvider`
2. Implement `generate(request: GenerationRequest) -> GenerationResponse`
3. Register it in `ProviderRegistry`:
   ```python
   ProviderRegistry.register("my_provider", MyProviderClass)
   ```

**To enable quantization** (reduce GPU memory by ~50%):
1. Find `LLMConfig` in `app.py`
2. Set `quantization="4bit"` — requires `bitsandbytes` library
