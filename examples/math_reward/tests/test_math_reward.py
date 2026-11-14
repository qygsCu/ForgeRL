"""Rule tests run against the installed, independent plugin package."""

import sys
from dataclasses import FrozenInstanceError

import pytest
from forgerl_math_reward import MathReward, MathRewardConfig
from pydantic import BaseModel, ValidationError

from forgerl.sdk import (
    ConfigurationError,
    PluginContext,
    PluginExecutionError,
    ProtocolViolationError,
    RewardPlugin,
    RewardSample,
)


class OtherConfig(BaseModel):
    pass


def initialized(mode: str = "strict") -> MathReward:
    plugin = MathReward()
    plugin.initialize(PluginContext(MathRewardConfig.model_validate({"match_mode": mode})))
    return plugin


@pytest.mark.parametrize(
    "response,reference,expected",
    [
        ("2", "2", 1.0),
        ("2.0", "2", 0.0),
        (" 2", "2", 0.0),
        ("2\n", "2", 0.0),
        ("", "2", 0.0),
        (" 2 ", " 2 ", 1.0),
        ("1/2", "0.5", 0.0),
    ],
)
def test_strict_compares_original_strings(response: str, reference: str, expected: float) -> None:
    output = initialized().compute([RewardSample("r1", response, reference)])
    assert output.results["r1"].reward == expected
    assert output.results["r1"].valid is True
    assert output.results["r1"].error_message is None


@pytest.mark.parametrize(
    "response,reference,expected",
    [
        ("2", "2.0", 1.0),
        ("2/4", "0.5", 1.0),
        ("1/2", "5e-1", 1.0),
        ("-3/4", "-7.5E-1", 1.0),
        ("1/-2", "-.5", 1.0),
        ("-1/-2", "+.5", 1.0),
        ("1/+2", "0.5", 1.0),
        ("2.", "2e+0", 1.0),
        ("+002", "2", 1.0),
        ("-0", "0e10", 1.0),
        ("1/3", "0.333333", 0.0),
        ("9007199254740993", "9007199254740992", 0.0),
        ("0.10000000000000000001", "0.1", 0.0),
        ("1.23e-5", "123/10000000", 1.0),
    ],
)
def test_numeric_compares_exact_rational_values(
    response: str, reference: str, expected: float
) -> None:
    output = initialized("numeric").compute([RewardSample("r1", response, reference)])
    assert output.results["r1"].reward == expected


INVALID_NUMBERS = [
    "",
    " 2",
    "2 ",
    "2\n",
    "\t2",
    "1 /2",
    "1/ 2",
    "1 2",
    "1\u00a02",
    "1/0",
    "1/-0",
    "0/0",
    "0.5/2",
    "1e0/2",
    "1 1/2",
    "1/2/3",
    "sqrt(4)",
    "1+1",
    "2*1",
    "2**1",
    "(2)",
    "NaN",
    "Infinity",
    "inf",
    "1_000",
    "２",
    "١",
    "2e",
    "1,000",
    "答案是2",
    "__import__('os')",
]


@pytest.mark.parametrize("response", INVALID_NUMBERS)
def test_invalid_numeric_answer_is_zero_not_a_scoring_failure(response: str) -> None:
    result = initialized("numeric").compute([RewardSample("r1", response, "2")]).results["r1"]
    assert result.reward == 0.0
    assert result.valid is True
    assert result.error_message is None


@pytest.mark.parametrize("reference", INVALID_NUMBERS)
def test_invalid_numeric_reference_is_a_protocol_error(reference: str) -> None:
    with pytest.raises(ProtocolViolationError, match="request_id='r1'"):
        initialized("numeric").compute([RewardSample("r1", "2", reference)])


def test_empty_reference_is_rejected_in_strict_mode() -> None:
    with pytest.raises(ProtocolViolationError, match="reference_answer is empty"):
        initialized().compute([RewardSample("r1", "", "")])


@pytest.mark.parametrize("field", ["response", "reference_answer"])
def test_conversion_limit_is_not_silently_scored_as_a_wrong_answer(field: str) -> None:
    limit = sys.get_int_max_str_digits()
    if limit == 0:
        pytest.skip("Python integer string conversion limit is disabled")
    huge_number = "1" * (limit + 1)
    sample = RewardSample(
        "r1",
        huge_number if field == "response" else "2",
        huge_number if field == "reference_answer" else "2",
    )
    # Legal grammar, but this interpreter cannot convert it: propagate the fault.
    with pytest.raises(ValueError, match="limit"):
        initialized("numeric").compute([sample])


@pytest.mark.parametrize("value", [2, 2.0, True, None])
def test_non_string_input_is_not_coerced(value: object) -> None:
    with pytest.raises(ProtocolViolationError):
        RewardSample("r1", value, "2")  # type: ignore[arg-type]
    with pytest.raises(ProtocolViolationError):
        RewardSample("r1", "2", value)  # type: ignore[arg-type]


@pytest.mark.parametrize("options", [{"match_mode": "approx"}, {"match_mode": 2}, {"strip": True}])
def test_config_rejects_unsupported_options(options: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        MathRewardConfig.model_validate(options)


def test_default_config_and_lifecycle_are_explicit() -> None:
    assert MathRewardConfig().match_mode == "strict"
    plugin = MathReward()
    samples = [RewardSample("r1", "2", "2.0")]
    with pytest.raises(PluginExecutionError, match="initialized before"):
        plugin.compute(samples)
    with pytest.raises(ConfigurationError, match="MathRewardConfig"):
        plugin.initialize(PluginContext(OtherConfig()))
    plugin.initialize(PluginContext(MathRewardConfig()))
    with pytest.raises(PluginExecutionError, match="already initialized"):
        plugin.initialize(PluginContext(MathRewardConfig(match_mode="numeric")))
    assert plugin.compute(samples).results["r1"].reward == 0.0
    plugin.shutdown()
    plugin.shutdown()
    with pytest.raises(PluginExecutionError):
        plugin.compute(samples)
    plugin.initialize(PluginContext(MathRewardConfig(match_mode="numeric")))
    assert plugin.compute(samples).results["r1"].reward == 1.0


def test_batch_identity_inputs_and_previous_outputs_are_preserved() -> None:
    plugin: RewardPlugin = initialized("numeric")
    samples = [RewardSample("r1", "1/2", "0.5"), RewardSample("r2", "bad", "2")]
    before = tuple(samples)
    first = plugin.compute(samples)
    second = plugin.compute([RewardSample("r3", "1", "1")])
    assert tuple(samples) == before
    assert set(first.results) == {"r1", "r2"}
    assert first.results["r1"].reward == 1.0
    assert first.results["r2"].reward == 0.0
    assert set(second.results) == {"r3"}
    with pytest.raises(FrozenInstanceError):
        first.results["r1"].reward = 0.0  # type: ignore[misc]
    with pytest.raises(ProtocolViolationError, match="duplicate"):
        plugin.compute([samples[0], samples[0]])
    with pytest.raises(ProtocolViolationError, match="empty"):
        plugin.compute([])
