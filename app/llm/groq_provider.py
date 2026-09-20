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

class GroqProvider(BaseLLMProvider):
    """Groq LLM Provider Implementation."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self._api_key = api_key or settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        self._model_name = model_name or settings.GROQ_MODEL or "groq/compound-mini"
        self._client = None

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_client(self):
        if not self._client:
            if not self._api_key:
                raise ValueError("GROQ_API_KEY is not configured.")
            from groq import Groq
            self._client = Groq(api_key=self._api_key)
        return self._client

    def _get_model_candidates(self) -> List[str]:
        candidates = [self._model_name]
        for m in GROQ_MODEL_CANDIDATES:
            if m not in candidates:
                candidates.append(m)
        return candidates

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
