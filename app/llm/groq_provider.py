import os
import json
import logging
from typing import Type, Optional, List
from pydantic import BaseModel
from app.llm.base import BaseLLMProvider
from app.config import settings

logger = logging.getLogger(__name__)

# List of active, high-capacity Groq models to try in order
GROQ_MODEL_CANDIDATES = [
    "groq/compound-mini",
    "groq/compound",
    "qwen/qwen3.8-27b",
    "openai/gpt-oss-20b"
]

# Groq API provider implementation of BaseLLMProvider.
# Uses official Groq SDK with automatic candidate model fallbacks and JSON mode support.
class GroqProvider(BaseLLMProvider):
    """Groq LLM Provider Implementation."""

    # Initializes Groq provider settings with API keys and designated default model identifier.
    # Sets up internal client state for lazy initialization.
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self._api_key = api_key or settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        self._model_name = model_name or settings.GROQ_MODEL or "groq/compound-mini"
        self._client = None

    # Property returning the canonical provider identifier 'groq'.
    # Used for provider identification across system routing and fallbacks.
    @property
    def provider_name(self) -> str:
        return "groq"

    # Property returning the active model name string currently being used.
    # Keeps track of the model string selected after candidate resolution.
    @property
    def model_name(self) -> str:
        return self._model_name

    # Lazy-loads and returns the official Groq client instance using configured API key.
    # Throws explicit ValueError if no API key is supplied in configuration or environment.
    def _get_client(self):
        if not self._client:
            if not self._api_key:
                raise ValueError("GROQ_API_KEY is not configured.")
            from groq import Groq
            self._client = Groq(api_key=self._api_key)
        return self._client

    # Constructs an ordered list of candidate Groq model strings for fallback execution.
    # Ensures alternate high-performance Groq models are tried if primary model fails.
    def _get_model_candidates(self) -> List[str]:
        candidates = [self._model_name]
        # Iterate over preset Groq model names to build an un-duplicated candidate fallbacks list.
        # Guarantees multiple fallback models are available for API calls.
        for m in GROQ_MODEL_CANDIDATES:
            if m not in candidates:
                candidates.append(m)
        return candidates

    # Generates unstructured text completions using Groq chat completions API endpoint.
    # Iterates over candidate models to safely handle model deprecations or 404 errors.
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1500
    ) -> str:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_err = None
        # Loop through Groq model candidates attempting text completion calls until successful.
        # Catches model-not-found exceptions and retries with subsequent models in list.
        for model in self._get_model_candidates():
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                self._model_name = model
                return response.choices[0].message.content or ""
            except Exception as e:
                err_str = str(e)
                last_err = e
                if "404" in err_str or "model_not_found" in err_str:
                    logger.warning(f"Groq model '{model}' not found. Trying next candidate...")
                    continue
                raise e
        raise last_err

    # Generates structured JSON responses validated against a target Pydantic schema class.
    # Configures Groq json_object response format and parses output into Pydantic models.
    def generate_structured(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1500
    ) -> BaseModel:
        client = self._get_client()
        schema_json = json.dumps(schema_class.model_json_schema(), indent=2)

        enhanced_system = (system_prompt or "") + (
            f"\n\nIMPORTANT: Respond with ONLY a valid JSON object strictly matching this JSON schema:\n{schema_json}\n"
            "Do NOT include any markdown code blocks, backticks, or extra text outside the raw JSON object."
        )

        messages = [
            {"role": "system", "content": enhanced_system},
            {"role": "user", "content": prompt}
        ]

        last_err = None
        # Loop over candidate Groq models to execute structured JSON completions with schema checks.
        # Retries alternative models if specific Groq model fails or encounters 404 errors.
        for model in self._get_model_candidates():
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"}
                )
                self._model_name = model
                content = response.choices[0].message.content or "{}"
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.startswith("```"):
                    content = content[3:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

                data = json.loads(content)
                return schema_class.model_validate(data)
            except Exception as e:
                err_str = str(e)
                last_err = e
                if "404" in err_str or "model_not_found" in err_str:
                    logger.warning(f"Groq model '{model}' not found for structured output. Trying next candidate...")
                    continue
                raise e
        raise last_err
