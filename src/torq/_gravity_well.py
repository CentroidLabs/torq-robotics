"""Gravity well infrastructure — single owner of all gravity well output.

Gravity wells are non-intrusive prompts that fire after successful SDK operations,
directing users to the datatorq.ai cloud platform. They are print-only in R1
(no network calls). Suppressed when ``tq.config.quiet = True``.

Output format (owned exclusively by this function)::

    💡 {message}
       → https://www.datatorq.ai
"""

from torq._config import config

DATATORQ_URL = "https://www.datatorq.ai"


def _gravity_well(message: str, feature: str) -> None:
    """Print a gravity well prompt if not in quiet mode.

    Args:
        message: The user-facing message to display.
        feature: The gravity well identifier (e.g. "GW-SDK-01").
            Used for tracking/analytics in R2; ignored in R1.
    """
    if config.quiet:
        return
    print(f"💡 {message}")
    print(f"   → {DATATORQ_URL}")
