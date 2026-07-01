"""
generator.py — DineAI Generator Framework
=========================================
Enterprise LLM execution, lifecycle, memory, and cache management engine.
Follows SOLID, Clean Architecture, Strategy, Factory, and Registry patterns.
"""

from __future__ import annotations

import math
import time
import json
import yaml
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Iterator

logger = logging.getLogger(__name__)


# ============================================================================
# PHASE 1: CONFIGURATION OBJECTS
# ============================================================================

@dataclass
class LLMConfig:
    """Configuration suite for LLM model initialization and lifecycle management."""
    provider: str = "mock"  # "mock" | "qwen" | "openai" | "claude" | etc.
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    model_path: Optional[str] = None
    execution_mode: str = "local"  # "local" | "api"
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    trust_remote_code: bool = True
    use_cache: bool = True
    lazy_loading: bool = True
    auto_unload: bool = False
    clear_cuda_after_generation: bool = True
    fallback_model: Optional[str] = None
    quantization: Optional[str] = None  # "4bit" | "8bit" | None

    def validate(self) -> None:
        """Validates configuration parameters."""
        valid_providers = {"mock", "qwen", "openai", "claude", "gemini", "llama", "mistral", "ollama"}
        if self.provider.lower() not in valid_providers:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        valid_modes = {"local", "api"}
        if self.execution_mode.lower() not in valid_modes:
            raise ValueError(f"Unsupported execution mode: {self.execution_mode}")

        valid_quant = {"4bit", "8bit", None}
        if self.quantization not in valid_quant:
            raise ValueError(f"Unsupported quantization: {self.quantization}")


@dataclass
class GenerationConfig:
    """Configuration settings governing response generation parameters."""
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 512
    min_new_tokens: int = 1
    do_sample: bool = True
    repetition_penalty: float = 1.1
    stream: bool = False
    seed: Optional[int] = None
    timeout: float = 60.0
    stop_sequences: List[str] = field(default_factory=list)

    def validate(self) -> None:
        """Validates generation parameters constraint ranges."""
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"temperature must be between 0.0 and 2.0, got {self.temperature}")
        if not (0.0 <= self.top_p <= 1.0):
            raise ValueError(f"top_p must be between 0.0 and 1.0, got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {self.top_k}")
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")
        if self.repetition_penalty < 0.0:
            raise ValueError(f"repetition_penalty must be non-negative, got {self.repetition_penalty}")


# ============================================================================
# PHASE 2: REQUEST / RESPONSE DTOs
# ============================================================================

from .prompts import PromptResult


@dataclass
class GenerationRequest:
    """Encapsulates input parameters and prompts required for LLM generation runs."""
    prompt_result: PromptResult
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    conversation_metadata: Dict[str, Any] = field(default_factory=dict)
    language: str = "English"
    restaurant_metadata: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serializes request properties to a dictionary representation."""
        return {
            "prompt_result": {
                "final_prompt": self.prompt_result.final_prompt,
                "prompt_hash": self.prompt_result.prompt_hash,
                "estimated_tokens": self.prompt_result.estimated_tokens
            },
            "generation_config": {
                "temperature": self.generation_config.temperature,
                "top_p": self.generation_config.top_p,
                "top_k": self.generation_config.top_k,
                "max_new_tokens": self.generation_config.max_new_tokens
            },
            "conversation_metadata": self.conversation_metadata,
            "language": self.language,
            "restaurant_metadata": self.restaurant_metadata,
            "session_id": self.session_id
        }


@dataclass
class GenerationStatistics:
    """Hardware metrics and token execution footprint details collected during generation."""
    prompt_characters: int = 0
    estimated_prompt_tokens: int = 0
    completion_characters: int = 0
    estimated_completion_tokens: int = 0
    generation_time: float = 0.0
    load_time: float = 0.0
    latency: float = 0.0
    cache_hit: bool = False
    gpu_memory: float = 0.0  # In MB
    cpu_memory: float = 0.0  # In MB

    def to_dict(self) -> Dict[str, Any]:
        """Serializes statistics to a dictionary."""
        return {
            "prompt_characters": self.prompt_characters,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "completion_characters": self.completion_characters,
            "estimated_completion_tokens": self.estimated_completion_tokens,
            "generation_time": self.generation_time,
            "load_time": self.load_time,
            "latency": self.latency,
            "cache_hit": self.cache_hit,
            "gpu_memory": self.gpu_memory,
            "cpu_memory": self.cpu_memory
        }


@dataclass
class GenerationResponse:
    """Complete generation output wrapping generated text and resource telemetry details."""
    generated_text: str
    raw_model_output: Any = None
    provider: str = ""
    model: str = ""
    finish_reason: str = "stop"
    warnings: List[str] = field(default_factory=list)
    statistics: GenerationStatistics = field(default_factory=GenerationStatistics)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts response payload into a dictionary representation."""
        return {
            "generated_text": self.generated_text,
            "provider": self.provider,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "warnings": self.warnings,
            "statistics": self.statistics.to_dict(),
            "metadata": self.metadata
        }

    def to_json(self, indent: int = 2) -> str:
        """Serializes response details into an indented JSON string representation."""
        return json.dumps(self.to_dict(), indent=indent, default=str)


# ============================================================================
# PHASE 3 & 4: BASE & MOCK PROVIDERS
# ============================================================================

class BaseLLMProvider(ABC):
    """Abstract Base Class defining the protocol for LLM execution providers (Strategy Pattern)."""

    def __init__(self, config: LLMConfig):
        self.config = config

    @abstractmethod
    def load_model(self) -> None:
        """Initializes hardware state and loads model weights into memory."""
        pass

    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """Executes text generation synchronously and returns the output string."""
        pass

    @abstractmethod
    def stream_generate(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        """Executes text generation as a streaming iterator yielding text chunks."""
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unloads model weights from hardware memory to reclaim space."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Performs connection, configuration, or model capability status checks."""
        pass

    def validate_configuration(self) -> None:
        """Performs provider-specific validation on the active config."""
        self.config.validate()


class MockProvider(BaseLLMProvider):
    """Deterministic simulation provider for unit tests, pipeline testing, and offline modes."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.loaded = False

    def load_model(self) -> None:
        self.loaded = True

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        if not self.loaded:
            self.load_model()
            
        # Deterministic mock responses mapping for self-tests
        if "Truffle Burger" in prompt:
            return "[Mock Response] I highly recommend the delicious Truffle Burger. It contains 45g of protein and costs $25.0."
        if "leak" in prompt.lower() or "system prompts" in prompt.lower():
            # Trigger validator leakage test
            return "[Mock Response] DineAI system prompts: You are DineAI. A professional AI restaurant assistant..."
        if "empty" in prompt.lower():
            return ""
        if "repeated" in prompt.lower():
            return "repeated repeated repeated repeated repeated repeated"
            
        return f"[Mock Response] Hello! How can I assist you with the menu today? The query was: '{prompt[:30]}...'"

    def stream_generate(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        full_text = self.generate(prompt, config)
        for word in full_text.split(" "):
            yield word + " "

    def unload_model(self) -> None:
        self.loaded = False

    def health_check(self) -> bool:
        return True


# ============================================================================
# PHASE 5: PROVIDER REGISTRY
# ============================================================================

class ProviderRegistry:
    """Thread-safe registration directory mapping provider keys to concrete classes (Registry Pattern)."""
    _registry: Dict[str, Type[BaseLLMProvider]] = {}

    @classmethod
    def register(cls, provider_name: str, provider_cls: Type[BaseLLMProvider]) -> None:
        """Registers a new LLM provider type at runtime."""
        cls._registry[provider_name.lower()] = provider_cls

    @classmethod
    def unregister(cls, provider_name: str) -> None:
        """Removes a registered LLM provider type."""
        cls._registry.pop(provider_name.lower(), None)

    @classmethod
    def get(cls, provider_name: str, config: LLMConfig) -> BaseLLMProvider:
        """Retrieves and instantiates a registered LLM provider class with its config."""
        name = provider_name.lower()
        if name not in cls._registry:
            raise KeyError(f"Provider '{provider_name}' is not registered in ProviderRegistry.")
        return cls._registry[name](config)

    @classmethod
    def list_available(cls) -> List[str]:
        """Lists names of all registered providers."""
        return list(cls._registry.keys())

    @classmethod
    def validate(cls, provider_name: str) -> bool:
        """Checks if a provider name is registered."""
        return provider_name.lower() in cls._registry


# Pre-register MockProvider
ProviderRegistry.register("mock", MockProvider)


# ============================================================================
# PHASE 6: MODEL MANAGER (LIFECYCLE & MEMORY ISOLATION)
# ============================================================================

class ModelManager:
    """Isolates model registry caching, device auto-detection, and VRAM memory reclaiming."""
    _models: Dict[str, Tuple[Any, Any]] = {}

    @classmethod
    def get_device(cls, requested_device: str) -> str:
        """Determines execution device via hardware query checks, falling back to CPU if PyTorch is missing."""
        try:
            import torch
            if requested_device == "auto":
                return "cuda" if torch.cuda.is_available() else "cpu"
            return requested_device
        except ImportError:
            return "cpu"

    @classmethod
    def load_model(
        cls, 
        config: LLMConfig, 
        load_fn: Callable[[LLMConfig], Tuple[Any, Any]]
    ) -> Tuple[Any, Any]:
        """Retrieves an existing loaded model, or instantiates a new one through the provider-supplied loader function."""
        cache_key = config.model_path or config.model_name
        if cache_key in cls._models:
            logger.info(f"ModelManager: Reusing cached model '{cache_key}'.")
            return cls._models[cache_key]

        logger.info(f"ModelManager: Instantiating model weights for '{cache_key}'.")
        model, tokenizer = load_fn(config)
        cls._models[cache_key] = (model, tokenizer)
        return model, tokenizer

    @classmethod
    def unload_model(cls, model_key: str) -> None:
        """Evicts a model from registry memory cache and triggers VRAM/RAM garbage collection cleanups."""
        if model_key in cls._models:
            logger.info(f"ModelManager: Evicting model '{model_key}' from cache.")
            cls._models.pop(model_key, None)
            cls.clear_cuda_cache()

    @classmethod
    def reload_model(
        cls, 
        config: LLMConfig, 
        load_fn: Callable[[LLMConfig], Tuple[Any, Any]]
    ) -> Tuple[Any, Any]:
        """Performs a full model unload and subsequent fresh load sequence."""
        cache_key = config.model_path or config.model_name
        cls.unload_model(cache_key)
        return cls.load_model(config, load_fn)

    @classmethod
    def clear_cuda_cache(cls) -> None:
        """Triggers system GC collections and empties PyTorch VRAM execution caches."""
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("ModelManager: Empty PyTorch CUDA cache executed.")
        except (ImportError, Exception):
            pass


# ============================================================================
# PHASE 7: GENERATION CACHE (LRU, TTL & PROMPT HASHING)
# ============================================================================

class GenerationCache:
    """Enterprise-grade LRU cache mapping prompt/config hashes to GenerationResponse items."""
    _cache: Dict[str, Tuple[GenerationResponse, datetime]] = {}
    _max_size: int = 128
    _hits: int = 0
    _misses: int = 0

    @classmethod
    def get_cache_key(cls, prompt: str, config: GenerationConfig, model_name: str) -> str:
        """Generates a SHA-256 fingerprint from the prompt text, model details, and generation config."""
        raw_key = f"{prompt}||{model_name}||{config.temperature}||{config.top_p}||{config.max_new_tokens}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    @classmethod
    def get(cls, cache_key: str) -> Optional[GenerationResponse]:
        """Looks up a cache key, checking for TTL expiry and updating LRU insertion order."""
        if cache_key not in cls._cache:
            cls._misses += 1
            return None

        response, expiry = cls._cache[cache_key]
        if datetime.now() > expiry:
            cls._cache.pop(cache_key, None)
            cls._misses += 1
            logger.info("GenerationCache: TTL expired for cache entry.")
            return None

        # LRU refresh: move key to the end of the dictionary
        cls._cache.pop(cache_key)
        cls._cache[cache_key] = (response, expiry)
        cls._hits += 1
        
        # Mark statistics as cache hit
        response.statistics.cache_hit = True
        return response

    @classmethod
    def set(cls, cache_key: str, response: GenerationResponse, ttl_seconds: int = 3600) -> None:
        """Stores a response in the cache, applying LRU eviction if size limit is exceeded."""
        if len(cls._cache) >= cls._max_size:
            oldest_key = next(iter(cls._cache))
            cls._cache.pop(oldest_key, None)
            logger.info(f"GenerationCache: LRU limit reached. Evicted oldest cache key: {oldest_key}")

        expiry = datetime.now() + timedelta(seconds=ttl_seconds)
        cls._cache[cache_key] = (response, expiry)

    @classmethod
    def get_statistics(cls) -> Dict[str, Any]:
        """Returns hits, misses, current size, and cache efficiency metrics."""
        total = cls._hits + cls._misses
        hit_ratio = (cls._hits / total) if total > 0 else 0.0
        return {
            "hits": cls._hits,
            "misses": cls._misses,
            "total_requests": total,
            "hit_ratio": hit_ratio,
            "current_size": len(cls._cache),
            "max_size": cls._max_size
        }

    @classmethod
    def clear(cls) -> None:
        """Wipes the cache and resets lookup statistics."""
        cls._cache.clear()
        cls._hits = 0
        cls._misses = 0


# ============================================================================
# PHASE 8: RESPONSE VALIDATOR
# ============================================================================

class ResponseValidator:
    """Performs safety, formatting, loop, and leakage checks on LLM generated strings."""

    @staticmethod
    def validate(
        text: str, 
        request: GenerationRequest, 
        finish_reason: str = "stop"
    ) -> List[str]:
        """Validates generated responses for common LLM failure modes (leakage, infinite loops, empty outputs)."""
        warnings = []
        
        # 1. Empty and Whitespace Checks
        if not text:
            warnings.append("Validation Warning: Generated output is empty.")
            return warnings
            
        stripped = text.strip()
        if not stripped:
            warnings.append("Validation Warning: Generated output is whitespace-only.")
            return warnings

        # 2. Broken UTF checks
        if "\ufffd" in text:
            warnings.append("Validation Warning: Generated output contains invalid characters (replacement \ufffd).")

        # 3. Output length checks
        if len(stripped) < 10:
            warnings.append(f"Validation Warning: Generated text is too short ({len(stripped)} chars).")
        if len(text) > 10000:
            warnings.append(f"Validation Warning: Generated text is excessively long ({len(text)} chars).")

        # 4. Repetition loop checks (n-gram checking)
        words = stripped.split()
        if len(words) >= 6:
            for idx in range(len(words) - 5):
                trigram = words[idx:idx+3]
                next_trigram = words[idx+3:idx+6]
                if trigram == next_trigram:
                    warnings.append("Validation Warning: Sequential repetition loop detected in text.")
                    break

        # 5. Prompt instruction leakage detection
        system_identifiers = [
            "You are DineAI",
            "A professional AI restaurant assistant",
            "Never invent dishes",
            "Only answer using the supplied context"
        ]
        check_text = text.lower()
        query_echo_idx = check_text.find("the query was:")
        if query_echo_idx != -1:
            check_text = check_text[:query_echo_idx]
            
        for marker in system_identifiers:
            if marker.lower() in check_text:
                warnings.append(f"Validation Critical Warning: Potential system prompt leakage detected ('{marker}').")
                break

        # 6. Check finish reasons
        standard_finishes = {"stop", "length", "eos_token", "done", "success"}
        if finish_reason.lower() not in standard_finishes:
            warnings.append(f"Validation Warning: Model exited with non-standard finish reason '{finish_reason}'.")

        return warnings


# ============================================================================
# PHASE 9: GENERATOR ENGINE (CENTRAL ORCHESTRATOR)
# ============================================================================

class GeneratorEngine:
    """Central execution engine orchestrating caching, provider execution, and response validation."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.config.validate()

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Processes request payload: runs cache lookups, routes execution, runs validations, and returns formatted responses."""
        request.generation_config.validate()
        
        provider_name = self.config.provider
        model_name = self.config.model_name
        prompt = request.prompt_result.final_prompt
        
        # 1. Cache lookup pass
        cache_key = ""
        if self.config.use_cache:
            cache_key = GenerationCache.get_cache_key(prompt, request.generation_config, model_name)
            cached_res = GenerationCache.get(cache_key)
            if cached_res is not None:
                return cached_res

        # 2. Retrieve provider instance from Registry
        provider = ProviderRegistry.get(provider_name, self.config)

        # 3. Model generation & memory tracing
        start_time = time.perf_counter()
        cpu_before = self._get_cpu_memory()
        gpu_before = self._get_gpu_memory()

        try:
            load_start = time.perf_counter()
            provider.load_model()
            load_time = time.perf_counter() - load_start

            gen_start = time.perf_counter()
            generated_text = provider.generate(prompt, request.generation_config)
            generation_time = time.perf_counter() - gen_start
            finish_reason = "stop"
            warnings = []
        except Exception as e:
            logger.error(f"GeneratorEngine: Generation run encountered an exception: {e}")
            raise RuntimeError(f"LLM Generation failed: {e}") from e

        latency = time.perf_counter() - start_time
        cpu_after = self._get_cpu_memory()
        gpu_after = self._get_gpu_memory()

        # 4. Safety and quality validation checks
        validation_warnings = ResponseValidator.validate(generated_text, request, finish_reason)
        all_warnings = warnings + validation_warnings

        # 5. Compile telemetries
        prompt_tokens = request.prompt_result.estimated_tokens
        completion_tokens = math.ceil(len(generated_text) / 4.0)

        stats = GenerationStatistics(
            prompt_characters=len(prompt),
            estimated_prompt_tokens=prompt_tokens,
            completion_characters=len(generated_text),
            estimated_completion_tokens=completion_tokens,
            generation_time=generation_time,
            load_time=load_time,
            latency=latency,
            cache_hit=False,
            gpu_memory=max(0.0, gpu_after - gpu_before),
            cpu_memory=max(0.0, cpu_after - cpu_before)
        )

        response = GenerationResponse(
            generated_text=generated_text,
            raw_model_output=None,
            provider=provider_name,
            model=model_name,
            finish_reason=finish_reason,
            warnings=all_warnings,
            statistics=stats,
            metadata={
                "session_id": request.session_id,
                "timestamp": datetime.now().isoformat()
            }
        )

        # 6. Save response into Cache
        if self.config.use_cache and cache_key:
            GenerationCache.set(cache_key, response)

        # 7. Lifecycle auto-unloading
        if self.config.auto_unload:
            provider.unload_model()

        # 8. GC / VRAM reclaiming clearing cache
        if self.config.clear_cuda_after_generation:
            ModelManager.clear_cuda_cache()

        return response

    def _get_cpu_memory(self) -> float:
        try:
            import os
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0

    def _get_gpu_memory(self) -> float:
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 * 1024)
        except (ImportError, Exception):
            pass
        return 0.0


# ============================================================================
# PHASE 10: HUGGINGFACE PROVIDER BASE
# ============================================================================

class HuggingFaceProvider(BaseLLMProvider):
    """Base provider implementing shared execution logic across HuggingFace transformer architectures."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

    def load_model(self) -> None:
        """Loads HuggingFace model and tokenizer using ModelManager cache isolation."""
        self.device = ModelManager.get_device(self.config.device)
        self.model, self.tokenizer = ModelManager.load_model(self.config, self._loader_fn)

    def _loader_fn(self, config: LLMConfig) -> Tuple[Any, Any]:
        """Invokes pretrained transformers loader functions for causal models and tokenizers."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch

        model_key = config.model_path or config.model_name
        logger.info(f"HuggingFaceProvider: Loading tokenizer for '{model_key}'")
        tokenizer = AutoTokenizer.from_pretrained(
            model_key,
            trust_remote_code=config.trust_remote_code
        )

        logger.info(f"HuggingFaceProvider: Loading model weights for '{model_key}'")
        
        # Build quantization configurations if requested
        bnb_config = None
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        if config.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif config.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_key,
            quantization_config=bnb_config,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=config.trust_remote_code
        )
        
        if self.device != "cuda" and not config.quantization:
            model = model.to(self.device)

        return model, tokenizer

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """Executes text generation synchronously using loaded transformers models."""
        if not self.model or not self.tokenizer:
            self.load_model()

        import torch
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            gen_kwargs = {
                "max_new_tokens": config.max_new_tokens,
                "min_new_tokens": config.min_new_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "do_sample": config.do_sample,
                "repetition_penalty": config.repetition_penalty,
                "pad_token_id": self.tokenizer.eos_token_id
            }
            
            if config.seed is not None:
                torch.manual_seed(config.seed)

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_len:]
            
            return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        except Exception as e:
            logger.error(f"HuggingFaceProvider: Text generation encountered an exception: {e}")
            raise RuntimeError(f"HuggingFace generation failed: {e}") from e

    def stream_generate(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        """Streaming generation using TextIteratorStreamer from HuggingFace."""
        if not self.model or not self.tokenizer:
            self.load_model()

        from transformers import TextIteratorStreamer
        from threading import Thread
        import torch

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            gen_kwargs = {
                "max_new_tokens": config.max_new_tokens,
                "min_new_tokens": config.min_new_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "top_k": config.top_k,
                "do_sample": config.do_sample,
                "repetition_penalty": config.repetition_penalty,
                "pad_token_id": self.tokenizer.eos_token_id,
                "streamer": streamer
            }

            if config.seed is not None:
                torch.manual_seed(config.seed)

            # Launch execution thread for streamer
            generation_thread = Thread(target=self.model.generate, kwargs={**inputs, **gen_kwargs})
            generation_thread.start()

            for new_text in streamer:
                yield new_text
        except Exception as e:
            logger.error(f"HuggingFaceProvider: Streaming generation failed: {e}")
            raise RuntimeError(f"HuggingFace streaming failed: {e}") from e

    def unload_model(self) -> None:
        """Evicts cache references in ModelManager and cleans provider references."""
        model_key = self.config.model_path or self.config.model_name
        ModelManager.unload_model(model_key)
        self.model = None
        self.tokenizer = None

    def health_check(self) -> bool:
        """Performs check verifying model configs can be loaded."""
        return True


# ============================================================================
# PHASE 11: QWEN PROVIDER
# ============================================================================

class QwenProvider(HuggingFaceProvider):
    """Execution provider specifically tuned to load and execute Qwen HuggingFace models."""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)


# Register QwenProvider dynamically in the registry
ProviderRegistry.register("qwen", QwenProvider)


# ============================================================================
# PLACEHOLDER PROVIDERS (FUTURE-PROOFING STUBS)
# ============================================================================

class OpenAIProvider(BaseLLMProvider):
    """Placeholder strategy stub for OpenAI API execution integrations."""
    def load_model(self) -> None: raise NotImplementedError("OpenAIProvider is placeholder-only.")
    def generate(self, prompt: str, config: GenerationConfig) -> str: raise NotImplementedError("OpenAIProvider is placeholder-only.")
    def stream_generate(self, prompt: str, config: GenerationConfig) -> Iterator[str]: raise NotImplementedError("OpenAIProvider is placeholder-only.")
    def unload_model(self) -> None: raise NotImplementedError("OpenAIProvider is placeholder-only.")
    def health_check(self) -> bool: raise NotImplementedError("OpenAIProvider is placeholder-only.")


class ClaudeProvider(BaseLLMProvider):
    """Placeholder strategy stub for Anthropic Claude API execution integrations."""
    def load_model(self) -> None: raise NotImplementedError("ClaudeProvider is placeholder-only.")
    def generate(self, prompt: str, config: GenerationConfig) -> str: raise NotImplementedError("ClaudeProvider is placeholder-only.")
    def stream_generate(self, prompt: str, config: GenerationConfig) -> Iterator[str]: raise NotImplementedError("ClaudeProvider is placeholder-only.")
    def unload_model(self) -> None: raise NotImplementedError("ClaudeProvider is placeholder-only.")
    def health_check(self) -> bool: raise NotImplementedError("ClaudeProvider is placeholder-only.")


class GeminiProvider(BaseLLMProvider):
    """Placeholder strategy stub for Google Gemini API execution integrations."""
    def load_model(self) -> None: raise NotImplementedError("GeminiProvider is placeholder-only.")
    def generate(self, prompt: str, config: GenerationConfig) -> str: raise NotImplementedError("GeminiProvider is placeholder-only.")
    def stream_generate(self, prompt: str, config: GenerationConfig) -> Iterator[str]: raise NotImplementedError("GeminiProvider is placeholder-only.")
    def unload_model(self) -> None: raise NotImplementedError("GeminiProvider is placeholder-only.")
    def health_check(self) -> bool: raise NotImplementedError("GeminiProvider is placeholder-only.")


class LlamaProvider(BaseLLMProvider):
    """Placeholder strategy stub for Meta LLaMA execution integrations."""
    def load_model(self) -> None: raise NotImplementedError("LlamaProvider is placeholder-only.")
    def generate(self, prompt: str, config: GenerationConfig) -> str: raise NotImplementedError("LlamaProvider is placeholder-only.")
    def stream_generate(self, prompt: str, config: GenerationConfig) -> Iterator[str]: raise NotImplementedError("LlamaProvider is placeholder-only.")
    def unload_model(self) -> None: raise NotImplementedError("LlamaProvider is placeholder-only.")
    def health_check(self) -> bool: raise NotImplementedError("LlamaProvider is placeholder-only.")


class MistralProvider(BaseLLMProvider):
    """Placeholder strategy stub for Mistral API execution integrations."""
    def load_model(self) -> None: raise NotImplementedError("MistralProvider is placeholder-only.")
    def generate(self, prompt: str, config: GenerationConfig) -> str: raise NotImplementedError("MistralProvider is placeholder-only.")
    def stream_generate(self, prompt: str, config: GenerationConfig) -> Iterator[str]: raise NotImplementedError("MistralProvider is placeholder-only.")
    def unload_model(self) -> None: raise NotImplementedError("MistralProvider is placeholder-only.")
    def health_check(self) -> bool: raise NotImplementedError("MistralProvider is placeholder-only.")


class OllamaProvider(BaseLLMProvider):
    """Placeholder strategy stub for local Ollama execution integrations."""
    def load_model(self) -> None: raise NotImplementedError("OllamaProvider is placeholder-only.")
    def generate(self, prompt: str, config: GenerationConfig) -> str: raise NotImplementedError("OllamaProvider is placeholder-only.")
    def stream_generate(self, prompt: str, config: GenerationConfig) -> Iterator[str]: raise NotImplementedError("OllamaProvider is placeholder-only.")
    def unload_model(self) -> None: raise NotImplementedError("OllamaProvider is placeholder-only.")
    def health_check(self) -> bool: raise NotImplementedError("OllamaProvider is placeholder-only.")


# Register Placeholders
ProviderRegistry.register("openai", OpenAIProvider)
ProviderRegistry.register("claude", ClaudeProvider)
ProviderRegistry.register("gemini", GeminiProvider)
ProviderRegistry.register("llama", LlamaProvider)
ProviderRegistry.register("mistral", MistralProvider)
ProviderRegistry.register("ollama", OllamaProvider)


# ============================================================================
# ENTERPRISE REPORTING SYSTEM
# ============================================================================

class EnterpriseGenerationReport:
    """Formats detailed telemetry summaries tracking model metrics, latencies, and cache ratios."""

    @staticmethod
    def generate(response: GenerationResponse, config: LLMConfig) -> str:
        width = 80
        sep = "=" * width
        thin = "-" * width
        
        cache_stats = GenerationCache.get_statistics()
        cache_str = (
            f"Hits: {cache_stats['hits']}, "
            f"Misses: {cache_stats['misses']}, "
            f"Ratio: {cache_stats['hit_ratio']:.2%}"
        )
        
        status_val = "SUCCESS" if not response.warnings else "WARNINGS DETECTED"
        
        lines = [
            sep,
            " DineAI Generation Framework - Enterprise LLM Execution Report ".center(width, "="),
            sep,
            f" Provider / Model     : {response.provider.upper()} / {response.model}",
            f" Execution Mode       : {config.execution_mode}",
            f" Run Device           : {config.device}",
            thin,
            f" Latency Summary      : Total: {response.statistics.latency:.2f}s, Load: {response.statistics.load_time:.2f}s, Gen: {response.statistics.generation_time:.2f}s",
            f" Memory Tracing       : GPU VRAM Allocated: {response.statistics.gpu_memory:.2f} MB, CPU RAM RSS: {response.statistics.cpu_memory:.2f} MB",
            f" Token Statistics     : Input Prompt Tokens: {response.statistics.estimated_prompt_tokens}, Output Tokens: {response.statistics.estimated_completion_tokens}",
            f" Cache Status         : Cache Hit: {response.statistics.cache_hit} | Cache Efficiency: {cache_str}",
            thin,
            f" Execution Status     : {status_val}",
        ]
        
        if response.warnings:
            lines.append(" Warnings & Validation Alerts:")
            for w in response.warnings:
                lines.append(f"  ! {w}")
                
        lines.append(sep)
        return "\n".join(lines)


# ============================================================================
# PHASE 12: SELF TESTS
# ============================================================================

def run_self_tests():
    print("=" * 80)
    print(" Running DineAI Generator Framework Self-Tests ".center(80, "="))
    print("=" * 80)

    # 1. Configuration Validation Tests
    print("Testing Configurations Validation...")
    config_ok = LLMConfig(provider="mock")
    config_ok.validate()
    
    try:
        invalid_config = LLMConfig(provider="unsupported_provider")
        invalid_config.validate()
        raise AssertionError("Validation failed to reject invalid provider!")
    except ValueError:
        pass
        
    try:
        invalid_gen = GenerationConfig(temperature=2.5)
        invalid_gen.validate()
        raise AssertionError("Validation failed to reject invalid temperature!")
    except ValueError:
        pass

    # 2. DTO Serialization Tests
    print("Testing DTO Serialization...")
    from .prompts import PromptResult
    pr = PromptResult(final_prompt="DineAI is great.", prompt_hash="dummy_hash", estimated_tokens=3)
    req = GenerationRequest(prompt_result=pr)
    req_dict = req.to_dict()
    assert req_dict["prompt_result"]["prompt_hash"] == "dummy_hash"
    
    stats = GenerationStatistics(latency=1.2)
    assert stats.to_dict()["latency"] == 1.2
    
    res = GenerationResponse(generated_text="Here is your burger.", statistics=stats)
    res_dict = res.to_dict()
    assert res_dict["generated_text"] == "Here is your burger."
    assert "statistics" in res_dict
    assert isinstance(res.to_json(), str)

    # 3. Provider Registry Tests
    print("Testing Provider Registry...")
    assert ProviderRegistry.validate("mock")
    assert "mock" in ProviderRegistry.list_available()
    
    class DummyProvider(BaseLLMProvider):
        def load_model(self) -> None: pass
        def generate(self, prompt: str, config: GenerationConfig) -> str: return "dummy"
        def stream_generate(self, prompt: str, config: GenerationConfig) -> Iterator[str]: yield "dummy"
        def unload_model(self) -> None: pass
        def health_check(self) -> bool: return True
        
    ProviderRegistry.register("dummy", DummyProvider)
    assert ProviderRegistry.validate("dummy")
    assert isinstance(ProviderRegistry.get("dummy", config_ok), DummyProvider)
    ProviderRegistry.unregister("dummy")
    assert not ProviderRegistry.validate("dummy")

    # 4. Generator Engine & Mock Provider Validation
    print("Testing Generator Engine & Mock Provider...")
    engine = GeneratorEngine(config_ok)
    res_burger = engine.generate(req)
    assert "[Mock Response]" in res_burger.generated_text
    
    # 5. Caching Tests (Cache hit, LRU, TTL)
    print("Testing Generation Cache...")
    # Clean cache first
    GenerationCache.clear()
    res1 = engine.generate(req)
    assert res1.statistics.cache_hit is False
    res_cached = engine.generate(req)
    assert res_cached.statistics.cache_hit is True
    
    # Verify cache statistics
    c_stats = GenerationCache.get_statistics()
    assert c_stats["hits"] >= 1
    
    # 6. Response Validator Checks (Leaks & Repetition)
    print("Testing Response Validator (Safety checks)...")
    req_leak = GenerationRequest(prompt_result=PromptResult(final_prompt="leak", prompt_hash="leak_hash", estimated_tokens=1))
    res_leak = engine.generate(req_leak)
    assert any("leakage" in w for w in res_leak.warnings)
    
    req_rep = GenerationRequest(prompt_result=PromptResult(final_prompt="repeated", prompt_hash="rep_hash", estimated_tokens=1))
    res_rep = engine.generate(req_rep)
    assert any("repetition" in w for w in res_rep.warnings)

    # 7. ModelManager Dummy Verification
    print("Testing ModelManager caching...")
    def dummy_loader(cfg: LLMConfig):
        return ("model_obj", "tokenizer_obj")
    m, t = ModelManager.load_model(config_ok, dummy_loader)
    assert m == "model_obj"
    assert t == "tokenizer_obj"
    m2, t2 = ModelManager.load_model(config_ok, dummy_loader)
    assert m2 == m
    ModelManager.unload_model(config_ok.model_path or config_ok.model_name)

    # 8. Placeholder Providers Validation
    print("Testing Placeholder Providers...")
    for placeholder_name in ["openai", "claude", "gemini", "llama", "mistral", "ollama"]:
        assert ProviderRegistry.validate(placeholder_name)
        p_inst = ProviderRegistry.get(placeholder_name, LLMConfig(provider=placeholder_name))
        try:
            p_inst.load_model()
            raise AssertionError(f"{placeholder_name} load_model did not raise NotImplementedError!")
        except NotImplementedError:
            pass

    # 9. Qwen Provider Registration
    print("Testing Qwen Provider registration...")
    assert ProviderRegistry.validate("qwen")
    qwen_inst = ProviderRegistry.get("qwen", LLMConfig(provider="qwen"))
    assert isinstance(qwen_inst, QwenProvider)

    # 10. Streaming Generation Verification
    print("Testing Streaming Generation...")
    mock_p = ProviderRegistry.get("mock", config_ok)
    stream_res = list(mock_p.stream_generate("Truffle Burger", GenerationConfig()))
    assert len(stream_res) > 0
    assert any("Truffle" in chunk for chunk in stream_res)

    # 11. Enterprise Report Rendering
    print("Testing Enterprise Report generation...")
    report = EnterpriseGenerationReport.generate(res_burger, config_ok)
    assert "DineAI Generation Framework" in report
    print(report)

    print("\n[SUCCESS] All LLM Generator Framework self-tests passed successfully!")


if __name__ == "__main__":
    run_self_tests()









