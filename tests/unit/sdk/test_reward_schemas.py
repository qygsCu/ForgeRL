"""Reward data contracts reject ambiguity without repairing producer data."""

from dataclasses import FrozenInstanceError

import pytest

from forgerl.sdk import (
    ProtocolViolationError,
    RewardOutput,
    RewardResult,
    RewardSample,
    validate_reward_samples,
)


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_id", ""),
        ("request_id", " \t"),
        ("request_id", 1),
        ("response", None),
        ("response", 2),
        ("reference_answer", None),
        ("reference_answer", 2),
    ],
)
def test_sample_rejects_invalid_fields(field: str, value: object) -> None:
    fields = {"request_id": "r1", "response": "2", "reference_answer": "2"}
    fields[field] = value  # type: ignore[assignment]
    with pytest.raises(ProtocolViolationError, match=field):
        RewardSample(**fields)


def test_text_is_not_normalized_and_empty_text_is_structurally_valid() -> None:
    sample = RewardSample(" r1 ", " 2\n", "2")
    assert sample.request_id == " r1 "
    assert sample.response == " 2\n"
    assert RewardSample("r2", "", "").reference_answer == ""


@pytest.mark.parametrize("samples", [[], "r1", {}, [object()]])
def test_batch_input_rejects_invalid_container_or_items(samples: object) -> None:
    with pytest.raises(ProtocolViolationError):
        validate_reward_samples(samples)  # type: ignore[arg-type]


def test_duplicate_ids_are_rejected_before_dictionary_insertion() -> None:
    samples = [RewardSample("r1", "2", "2"), RewardSample("r1", "3", "2")]
    with pytest.raises(ProtocolViolationError, match="duplicate request_id"):
        validate_reward_samples(samples)


@pytest.mark.parametrize("reward", [-2.5, 0.0, 1.0, 100.0])
def test_general_reward_is_not_limited_to_exact_match_values(reward: float) -> None:
    assert RewardResult(reward, True).reward == reward


@pytest.mark.parametrize(
    "reward,valid,message",
    [
        (float("nan"), True, None),
        (float("inf"), True, None),
        (-float("inf"), True, None),
        (1, True, None),
        (True, True, None),
        (None, True, None),
        (1.0, 1, None),
        (1.0, True, "unexpected error"),
        (0.0, False, "failed"),
        (None, False, None),
        (None, False, ""),
        (None, False, " \t"),
        (None, False, 3),
    ],
)
def test_result_rejects_illegal_combinations(
    reward: object, valid: object, message: object
) -> None:
    with pytest.raises(ProtocolViolationError):
        RewardResult(reward, valid, message)  # type: ignore[arg-type]


def test_failed_result_preserves_diagnostics_without_a_fake_reward() -> None:
    result = RewardResult(None, False, "scorer unavailable")
    assert result.reward is None
    assert result.error_message == "scorer unavailable"


@pytest.mark.parametrize("version", ["", "1.1", 1.0, None])
def test_output_requires_exact_schema_version(version: object) -> None:
    with pytest.raises(ProtocolViolationError, match="schema_version"):
        RewardOutput({"r1": RewardResult(1.0, True)}, version)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "results", [[], {"": RewardResult(1.0, True)}, {"r1": {"reward": 1.0, "valid": True}}]
)
def test_output_requires_typed_records_and_valid_keys(results: object) -> None:
    with pytest.raises(ProtocolViolationError):
        RewardOutput(results)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "record,field,value",
    [
        (RewardSample("r1", "2", "2"), "response", "3"),
        (RewardResult(1.0, True), "reward", 0.0),
        (RewardOutput({}), "schema_version", "1.1"),
    ],
)
def test_frozen_records_prevent_ordinary_field_assignment(
    record: object, field: str, value: object
) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(record, field, value)


def test_output_revalidates_nested_records_after_boundary_crossing() -> None:
    result = RewardResult(1.0, True)
    output = RewardOutput({"r1": result})
    # Simulate a producer bypassing the frozen record, not normal supported usage.
    object.__setattr__(result, "reward", float("nan"))
    with pytest.raises(ProtocolViolationError, match="r1.*finite float"):
        output.validate()
