import time
import logging
from typing import Callable, Any

logger = logging.getLogger(__name__)

# Utility function to execute a callable with automatic exponential backoff retry logic.
# Catches transient exceptions, logs warning attempts, and sleeps progressively before retrying.
def retry_with_backoff(
    func: Callable[[], Any],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0
) -> Any:
    """Executes func with exponential backoff on exceptions."""
    delay = initial_delay
    last_exception = None

    # Loop through retry attempts up to max_retries limit.
    # On failure, applies exponential backoff delay before executing the next attempt.
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_exception = e
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed with error: {e}. Retrying in {delay}s...")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= backoff_factor

    raise last_exception
