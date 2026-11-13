"""Public sequence-level scoring interface (ADR-0010)."""

from collections.abc import Sequence
from typing import Protocol

from forgerl.sdk.lifecycle import PluginLifecycle
from forgerl.sdk.schemas import RewardOutput, RewardSample


class RewardPlugin(PluginLifecycle, Protocol):
    """Score read-only samples, preserving IDs and reporting isolated failures.

    Only explicitly recoverable per-sample failures become invalid results.
    Other exceptions propagate. Neither inputs nor returned results may be
    mutated by the plugin after return. Protocol hints do not enforce this.
    """

    def compute(self, samples: Sequence[RewardSample]) -> RewardOutput: ...
