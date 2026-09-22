import logging
import time
import random
import re
from typing import Type, Optional, List, Tuple
from pydantic import BaseModel
from app.llm.base import BaseLLMProvider
from app.llm.groq_provider import GroqProvider
from app.llm.gemini_provider import GeminiProvider
from app.config import settings

logger = logging.getLogger(__name__)

# Helper function to parse recommended wait duration from rate limit and quota API error text.
# Returns calculated delay in seconds (with small padding) if retry instructions are found in message.
def parse_retry_after_seconds(err_str: str) -> Optional[float]:
    """Parses Retry-After duration or suggested wait time from API error messages."""
    try:
        # Match pattern like "try again in 2.5s" or "Retry-After: 3"
        match = re.search(r"try again in (\d+\.?\d*)s", err_str, re.IGNORECASE)
        if match:
            return float(match.group(1)) + 0.2
        match_min = re.search(r"try again in (\d+)m(\d+\.?\d*)s", err_str, re.IGNORECASE)
        if match_min:
            mins = float(match_min.group(1))
            secs = float(match_min.group(2))
            return mins * 60 + secs + 0.5
    except Exception:
        pass
    return None

# Resilience wrapper class for primary and secondary LLM providers with automatic fallback logic.
# Implements exponential backoff, rate limit cooldowns, and seamless provider switching on error.
class FallbackLLMProvider(BaseLLMProvider):
    """Production Fallback LLM Provider wrapper.
    Features:
    - Jittered Exponential Backoff (prevents stampede effect)
    - Retry-After header parsing from 429/503 responses
    - Primary/Secondary provider failover
    - Circuit Breaker / Cooldown for rate-limited providers
    - Immediate bypass on Daily Quota (TPD) exhaustion
    """

    # Initializes fallback manager with primary provider preference and available provider instances.
    # Sets up tracking state for cooldown timers, last used provider, and failure histories.
    def __init__(
        self,
        primary_name: Optional[str] = None,
        fallback_enabled: Optional[bool] = None
    ):
        self.primary_name = (primary_name or settings.LLM_PRIMARY_PROVIDER).lower()
        self.fallback_enabled = fallback_enabled if fallback_enabled is not None else settings.LLM_FALLBACK_ENABLED

        self.groq = GroqProvider()
        self.gemini = GeminiProvider()

        self.last_provider_used = self.primary_name
        self.last_fallback_occurred = False
        self.last_error = None
        self._cooldowns = {}  # {provider_name: timestamp_until_cooldown_expires}

    # Property returning the provider identifier string used in the most recent generation call.
    # Helps callers identify whether primary or secondary provider handled the request.
    @property
    def provider_name(self) -> str:
        return self.last_provider_used

    # Property returning the active model name of the provider currently selected for generation.
    # Routes model name query to either groq or gemini instance depending on active provider state.
    @property
    def model_name(self) -> str:
        if self.last_provider_used == "groq":
            return self.groq.model_name
        return self.gemini.model_name

    # Determines the ordered list of active providers based on priority preferences and cooldown status.
    # Filters out providers currently on cooldown unless all providers are currently cooling down.
    def _get_provider_chain(self) -> List[Tuple[str, BaseLLMProvider]]:
        now = time.time()
        if self.primary_name == "gemini":
            order = [("gemini", self.gemini), ("groq", self.groq)]
        else:
            order = [("groq", self.groq), ("gemini", self.gemini)]

        if not self.fallback_enabled:
            return [order[0]]

        # Filter out providers currently on cooldown
        active_chain = []
        # Loop through configured provider order to check if any provider is on active cooldown.
        # Excludes temporarily blocked rate-limited providers from the active provider chain.
        for name, provider in order:
            cooldown_until = self._cooldowns.get(name, 0)
            if cooldown_until < now:
                active_chain.append((name, provider))
            else:
                remaining = int(cooldown_until - now)
                logger.info(f"Skipping provider '{name}' (on cooldown for {remaining}s remaining).")

        # If all providers are on cooldown, ignore cooldown and attempt all as last resort
        return active_chain if active_chain else order

    # Registers a temporal cooldown block for a specified provider identifier.
    # Prevents retrying rate-limited or quota-exhausted providers for the given duration.
    def _mark_cooldown(self, name: str, duration: float = 60.0):
        self._cooldowns[name] = time.time() + duration
        logger.warning(f"Provider '{name}' placed on cooldown for {duration:.0f}s due to rate limit/quota.")

    # Checks whether an API error string indicates a transient failure like rate limit or timeout.
    # Used to determine if retry backoff should be applied before switching providers.
    def _is_transient_error(self, err_str: str) -> bool:
        err_lower = err_str.lower()
        return any(term in err_lower for term in [
            "429", "503", "rate limit", "timeout", "unavailable",
            "high demand", "overloaded", "tpm", "tpd", "rpm"
        ])

    # Checks if an error string explicitly signifies daily quota or token-per-day exhaustion.
    # Immediately triggers extended cooldown blocks to skip unnecessary retries.
    def _is_daily_quota_exhausted(self, err_str: str) -> bool:
        err_lower = err_str.lower()
        return any(term in err_lower for term in [
            "tpd", "daily limit", "quota exceeded", "exceeded your current quota", "insufficient_quota"
        ])

    # Calculates exponential backoff delay with random jitter to prevent API stampede effects.
    # Prioritizes Retry-After headers if present in the error message payload.
    def _calculate_backoff_delay(self, attempt: int, err_str: str) -> float:
        """Calculates backoff delay using Retry-After parsing + Jittered Exponential Backoff."""
        parsed_wait = parse_retry_after_seconds(err_str)
        if parsed_wait and parsed_wait <= 10.0:  # Cap parsed wait to reasonable 10s max
            return parsed_wait

        base_delay = 1.5 * (2 ** attempt)
        jitter = random.uniform(0.1, 0.7)  # Jitter prevents simultaneous retry collisions
        return min(12.0, base_delay + jitter)

    # Executes text generation with resilience retry loops and automatic multi-provider fallback.
    # Iterates across providers and retry attempts to ensure continuous application uptime.
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1500
    ) -> str:
        chain = self._get_provider_chain()
        errors = []

        # Outer loop iterates through ordered LLM providers in the fallback chain.
        # Handles seamless failover when primary provider fails or hits quota limits.
        for idx, (name, provider) in enumerate(chain):
            max_retries = 3
            # Inner loop executes retry attempts for the current provider using exponential backoff.
            # Handles transient failures and rate-limit retries before giving up on provider.
            for attempt in range(max_retries):
                try:
                    result = provider.generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    self.last_provider_used = name
                    self.last_fallback_occurred = (name != chain[0][0] or idx > 0)
                    if self.last_fallback_occurred:
                        logger.warning(f"Fallback active: Successfully generated answer using fallback provider '{name}'.")
                    return result
                except Exception as e:
                    err_str = str(e)
                    logger.warning(f"Provider {name} attempt {attempt+1}/{max_retries} failed: {err_str}")
                    
                    # If daily limit / TPD is exhausted, do not retry this provider at all
                    if self._is_daily_quota_exhausted(err_str):
                        logger.warning(f"Provider {name} hit DAILY quota (TPD). Placing on 5-min cooldown and bypassing retries.")
                        self._mark_cooldown(name, duration=300.0)
                        errors.append(f"{name} (Daily Quota): {err_str}")
                        break

                    if self._is_transient_error(err_str) and attempt < max_retries - 1:
                        delay = self._calculate_backoff_delay(attempt, err_str)
                        logger.info(f"Applying Jittered Backoff ({delay:.2f}s) for {name}...")
                        time.sleep(delay)
                        continue
                    
                    # If all retries fail for transient error, place provider on 60s cooldown
                    if self._is_transient_error(err_str):
                        self._mark_cooldown(name, duration=60.0)

                    errors.append(f"{name}: {err_str}")
                    break

        self.last_error = " | ".join(errors)
        logger.error(f"All providers failed in FallbackLLMProvider. Errors: {self.last_error}")
        raise RuntimeError(f"All configured LLM providers failed. {self.last_error}")

    # Executes structured JSON schema generation using resilient fallback provider chains.
    # Validates final output against Pydantic model schema while retrying across providers.
    def generate_structured(
        self,
        prompt: str,
        schema_class: Type[BaseModel],
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1500
    ) -> BaseModel:
        chain = self._get_provider_chain()
        errors = []

        # Outer loop iterates through ordered LLM providers to perform structured generation.
        # Manages failover between primary and secondary providers if schema generation fails.
        for idx, (name, provider) in enumerate(chain):
            max_retries = 3
            # Inner loop executes retry attempts for structured generation on the selected provider.
            # Applies exponential backoff for transient rate limits and quota handling.
            for attempt in range(max_retries):
                try:
                    result = provider.generate_structured(
                        prompt=prompt,
                        schema_class=schema_class,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    self.last_provider_used = name
                    self.last_fallback_occurred = (name != chain[0][0] or idx > 0)
                    if self.last_fallback_occurred:
                        logger.warning(f"Fallback active: Successfully generated structured output using fallback provider '{name}'.")
                    return result
                except Exception as e:
                    err_str = str(e)
                    logger.warning(f"Provider {name} structured attempt {attempt+1}/{max_retries} failed: {err_str}")
                    
                    if self._is_daily_quota_exhausted(err_str):
                        logger.warning(f"Provider {name} hit DAILY quota (TPD). Placing on 5-min cooldown and bypassing retries.")
                        self._mark_cooldown(name, duration=300.0)
                        errors.append(f"{name} (Daily Quota): {err_str}")
                        break

                    if self._is_transient_error(err_str) and attempt < max_retries - 1:
                        delay = self._calculate_backoff_delay(attempt, err_str)
                        logger.info(f"Applying Jittered Backoff ({delay:.2f}s) for structured {name}...")
                        time.sleep(delay)
                        continue

                    if self._is_transient_error(err_str):
                        self._mark_cooldown(name, duration=60.0)

                    errors.append(f"{name}: {err_str}")
                    break

        self.last_error = " | ".join(errors)
        logger.error(f"All providers failed structured output in FallbackLLMProvider. Errors: {self.last_error}")
        raise RuntimeError(f"All configured LLM providers failed structured output. {self.last_error}")
