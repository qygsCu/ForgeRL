"""Strict matching and exact comparison of numeric text, without evaluation."""

import re
from collections.abc import Sequence
from fractions import Fraction
from typing import Literal

from pydantic import BaseModel, ConfigDict

from forgerl.sdk import (
    SDK_API_VERSION,
    ConfigurationError,
    PluginContext,
    PluginExecutionError,
    PluginKind,
    PluginMetadata,
    ProtocolViolationError,
    RewardOutput,
    RewardResult,
    RewardSample,
    validate_reward_samples,
)

# ASCII digits only. fullmatch rejects all surrounding/internal whitespace,
# underscores, expressions, Unicode numerals, NaN and infinity before parsing.
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_FRACTION = re.compile(r"([+-]?[0-9]+)/([+-]?[0-9]+)")


def _parse_number(text: str) -> Fraction | None:
    """Return None for invalid numeric text; conversion/resource faults propagate."""
    fraction = _FRACTION.fullmatch(text)
    if fraction is not None:
        numerator, denominator = int(fraction[1]), int(fraction[2])
        if denominator == 0:
            return None
        return Fraction(numerator, denominator)
    if _NUMBER.fullmatch(text) is not None:
        return Fraction(text)
    return None


class MathRewardConfig(BaseModel):
    """Select the scoring rule without normalization or tolerance options."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    match_mode: Literal["strict", "numeric"] = "strict"


class MathReward:
    """Score one complete response with 0.0 or 1.0.

    Syntax-invalid responses are wrong answers, not scorer failures. Bad
    references are protocol errors. No artificial local-failure state is used.
    Shutdown is idempotent; repeated initialization before shutdown is rejected.
    """

    metadata = PluginMetadata(
        name="math_reward",
        kind=PluginKind.REWARD,
        version="0.1.0",
        api_version=SDK_API_VERSION,
        required_capabilities=frozenset(),
        optional_capabilities=frozenset(),
        supports_checkpoint=False,
        supports_multiprocessing=False,
        requires_gpu=False,
    )
    config_model = MathRewardConfig

    def __init__(self) -> None:
        self._match_mode: Literal["strict", "numeric"] | None = None

    def initialize(self, context: PluginContext) -> None:
        if self._match_mode is not None:
            raise PluginExecutionError("math_reward is already initialized")
        if not isinstance(context.config, MathRewardConfig):
            raise ConfigurationError("math_reward requires MathRewardConfig")
        self._match_mode = context.config.match_mode

    def compute(self, samples: Sequence[RewardSample]) -> RewardOutput:
        if self._match_mode is None:
            raise PluginExecutionError("math_reward must be initialized before compute")
        validate_reward_samples(samples)
        results: dict[str, RewardResult] = {}
        for sample in samples:
            if not sample.reference_answer:
                raise ProtocolViolationError(
                    f"math_reward request_id={sample.request_id!r}: reference_answer is empty"
                )
            if self._match_mode == "strict":
                correct = sample.response == sample.reference_answer
            else:
                reference = _parse_number(sample.reference_answer)
                if reference is None:
                    raise ProtocolViolationError(
                        f"math_reward request_id={sample.request_id!r}: "
                        "invalid numeric reference_answer"
                    )
                response = _parse_number(sample.response)
                correct = response is not None and response == reference
            results[sample.request_id] = RewardResult(reward=float(correct), valid=True)
        return RewardOutput(results=results)

    def shutdown(self) -> None:
        self._match_mode = None
