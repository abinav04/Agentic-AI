from abc import ABC, abstractmethod
from typing import Type, Any, Optional, Dict
from pydantic import BaseModel

# Abstract base class enforcing standard interfaces for all LLM backend integrations.
# Defines contract for text generation and structured Pydantic model response parsing.
class BaseLLMProvider(ABC):
    """Abstract Base Class for all LLM Providers.
    Making it modular and straightforward to add new models/providers.
    """

    # Abstract property method that returns the provider name (e.g., 'groq' or 'gemini').
    # Enables dynamic provider identification across the fallback routing logic.
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the canonical provider identifier (e.g., 'groq', 'gemini')."""
        pass

    # Abstract property method that returns the specific model designation string.
    # Used for logging, model tracking, and debugging provider calls.
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the active model name."""
        pass

    # Abstract method defining standard interface for generating raw text LLM responses.
    # Accepts user prompts, system prompts, temperature controls, and max token limits.
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1500
    ) -> str:
        """Generates plain text completion."""
        pass

    # Abstract method defining standard interface for generating structured Pydantic object outputs.
    # Enforces target JSON schema validation on the LLM's returned output.
    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1500
    ) -> BaseModel:
        """Generates JSON structured output conforming to pydantic schema_class."""
        pass
