import os
import json
import logging
from typing import Type, Optional, List
from pydantic import BaseModel
from app.llm.base import BaseLLMProvider
from app.config import settings

logger = logging.getLogger(__name__)

GEMINI_MODEL_CANDIDATES = [
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-flash-latest"
]

class GeminiProvider(BaseLLMProvider):
    """Gemini LLM Provider Implementation."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self._api_key = api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        self._model_name = model_name or settings.GEMINI_MODEL or "gemini-3.6-flash"
        self._client = None
        self._is_new_sdk = True

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_client(self):
        if not self._client:
            if not self._api_key:
                raise ValueError("GEMINI_API_KEY is not configured.")
            try:
                from google import genai
                self._client = genai.Client(api_key=self._api_key)
                self._is_new_sdk = True
            except ImportError:
                import google.generativeai as genai_legacy
                genai_legacy.configure(api_key=self._api_key)
                self._client = genai_legacy
                self._is_new_sdk = False
        return self._client

    def _get_model_candidates(self) -> List[str]:
        candidates = [self._model_name]
        for m in GEMINI_MODEL_CANDIDATES:
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
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        last_err = None
        for model in self._get_model_candidates():
            try:
                if self._is_new_sdk:
                    from google.genai import types
                    config = types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    )
                    response = client.models.generate_content(
                        model=model,
                        contents=full_prompt,
                        config=config
                    )
                    self._model_name = model
                    return response.text or ""
                else:
                    model_obj = client.GenerativeModel(model)
                    response = model_obj.generate_content(
                        full_prompt,
                        generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
                    )
                    self._model_name = model
                    return response.text or ""
            except Exception as e:
                err_str = str(e)
                last_err = e
                if "404" in err_str or "not found" in err_str:
                    logger.warning(f"Gemini model '{model}' not found. Trying next candidate...")
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
        schema_json = json.dumps(schema_class.model_json_schema(), indent=2)

        enhanced_system = (system_prompt or "") + (
            f"\n\nIMPORTANT: Respond with ONLY a valid JSON object strictly matching this JSON schema:\n{schema_json}\n"
            "Do NOT include any markdown code blocks, backticks, or extra text outside raw JSON object."
        )

        raw_text = self.generate(
            prompt=prompt,
            system_prompt=enhanced_system,
            temperature=temperature,
            max_tokens=max_tokens
        )

        content = raw_text.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        data = json.loads(content)
        return schema_class.model_validate(data)
