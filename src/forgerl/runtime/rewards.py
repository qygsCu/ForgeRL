"""Validate a complete scoring call before accepting any of its results."""

import logging
from collections.abc import Sequence
from typing import cast

from forgerl.sdk.errors import (
    ForgeRLError,
    PluginExecutionError,
    PluginLoadError,
    ProtocolViolationError,
)
from forgerl.sdk.lifecycle import PluginLifecycle
from forgerl.sdk.rewards import RewardPlugin
from forgerl.sdk.schemas import RewardOutput, RewardSample, validate_reward_samples

logger = logging.getLogger(__name__)


def compute_rewards(
    plugin: PluginLifecycle,
    samples: Sequence[RewardSample],
    *,
    plugin_name: str,
) -> RewardOutput:
    """Call an already initialized plugin; its caller owns the lifecycle scope.

    Validate inputs before compute and the entire output before returning or
    reporting isolated failures. Input records are passed in a tuple to prevent
    ordinary container mutation. No training filtering or deep copy occurs.
    """
    validate_reward_samples(samples)
    inputs = tuple(samples) # use tuple to avoid rewriting
    expected_ids = {sample.request_id for sample in inputs}
    if not callable(getattr(plugin, "compute", None)):
        raise PluginLoadError(f"plugin reward:{plugin_name} must provide callable compute")

    scorer = cast(RewardPlugin, plugin)
    try:
        output = scorer.compute(inputs)
    except (ForgeRLError, MemoryError):
        raise
    except Exception as exc:
        raise PluginExecutionError(f"failed to compute plugin reward:{plugin_name}: {exc}") from exc

    # Keep protocol failures outside execution-error wrapping. Constructors may
    # already validate, but the plugin can still return a wrong or mutated object.
    if not isinstance(output, RewardOutput):
        raise ProtocolViolationError(f"plugin reward:{plugin_name} must return RewardOutput")
    try:
        output.validate()
    except ProtocolViolationError as exc:
        raise ProtocolViolationError(
            f"plugin reward:{plugin_name} returned invalid output: {exc}"
        ) from exc
    actual_ids = set(output.results)
    if actual_ids != expected_ids:
        raise ProtocolViolationError(
            f"plugin reward:{plugin_name} returned mismatched request IDs: "
            f"missing={sorted(expected_ids - actual_ids)}, "
            f"unexpected={sorted(actual_ids - expected_ids)}"
        )

    for request_id, result in output.results.items():
        if not result.valid:
            logger.warning(
                "plugin reward:%s request_id=%s stage=compute: %s",
                plugin_name,
                request_id,
                result.error_message,
            )
    return output
