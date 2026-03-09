"""Cloud platform stub — directs users to datatorq.ai.

Implements GW-SDK-04: explicit ``tq.cloud()`` call.
"""

from torq._gravity_well import _gravity_well


def cloud() -> None:
    """Print the datatorq.ai cloud platform prompt.

    Directs users to the cloud platform for collaborative and cloud-scale
    features. No network calls are made; this is a local-only prompt.

    Suppressed when ``tq.config.quiet = True``.
    """
    _gravity_well(
        message="Torq Cloud — collaborative datasets, cloud-scale training, "
        "and team workflows. Join the waitlist!",
        feature="GW-SDK-04",
    )
