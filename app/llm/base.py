from abc import ABC, abstractmethod
from typing import Type, Any, Optional, Dict
from pydantic import BaseModel

class BaseLLMProvider(ABC):
    """Abstract Base Class for all LLM Providers.
    Making it modular and straightforward to add new models/providers.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Returns the canonical provider identifier (e.g., 'groq', 'gemini')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Returns the active model name."""
        pass

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
