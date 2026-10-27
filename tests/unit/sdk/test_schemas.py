from dataclasses import replace

import pytest
import torch

from forgerl.sdk.schemas import TrajectoryBatch, validate_json_value


def make_valid_trajectory_batch() -> TrajectoryBatch:
    return TrajectoryBatch(
        schema_version="1.0",
        request_ids=["request-1", "request-2"],
        group_ids=torch.tensor([7, 7], dtype=torch.int64),
        prompt_ids=torch.tensor([[0, 11, 12], [21, 22, 23]], dtype=torch.int64),
        response_ids=torch.tensor(
            [[31, 32, 2, 0], [41, 2, 0, 0]], dtype=torch.int64
        ),
        attention_mask=torch.tensor(
            [
                [False, True, True, True, True, True, False],
                [True, True, True, True, True, False, False],
            ],
            dtype=torch.bool,
        ),
        response_mask=torch.tensor(
            [[True, True, True, False], [True, True, False, False]],
            dtype=torch.bool,
        ),
        old_logprobs=torch.tensor(
            [[-0.2, -0.4, -0.1, 0.0], [-0.3, -0.8, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        sequence_rewards=torch.tensor([1.0, -0.5], dtype=torch.float32),
        policy_version=3,
        metadata={"source": "unit-test", "temperature": 0.8},
    )

@pytest.mark.parametrize(
        "value",
        [
            None,
            "hpz",
            False,
            1,
            3.14,
            "",
            -12,
        ],
)

def test_validate_json_value_accepts_primitives(value: object) -> None:
    assert validate_json_value(value) is None

def test_validate_json_value_accepts_nested_value() -> None:
    value = {
        "metrics": {
            "values": [
                1.0,
                "hello world",
                False,
                None,
                {"loss": 0.25},
            ],
        },
        "tags": [],
        "extra": {},
    }

    assert validate_json_value(value) is None

@pytest.mark.parametrize(
    ("value", "expected_type"),
    [
        ((1, 2), "tuple"),
        ({1, 2}, "set"),
        (b"forgerl", "bytes"),
        (1 + 2j, "complex"),
        (object(), "object"),
        (torch.tensor([1, 2]), "Tensor"),
    ],
)
def test_validate_json_value_rejects_unsupported_type(
    value: object,
    expected_type: str,
) -> None:
    with pytest.raises(ValueError) as exc_info:
        validate_json_value(value)

    assert "JSON-compatible" in str(exc_info.value)
    assert expected_type in str(exc_info.value)

@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_validate_json_value_rejects_non_finite_float(value: float) -> None:
    with pytest.raises(ValueError) as exc_info:
        validate_json_value(value)

    assert "finite" in str(exc_info.value)
    assert "$" in str(exc_info.value)

def test_validate_json_value_rejects_non_string_dict_key() -> None:
    value = {
        42: "not allowed",
    }

    with pytest.raises(ValueError) as exc_info:
        validate_json_value(value)

    message = str(exc_info.value)
    assert "key" in message
    assert "str" in message
    assert "$" in message

def test_validate_json_value_reports_nested_error_path() -> None:
    value = {
        "metrics": {
            "values": [
                1.0,
                {"bad": {1, 2}},
            ],
        },
    }

    with pytest.raises(ValueError) as exc_info:
        validate_json_value(value)

    message = str(exc_info.value)
    assert "metrics" in message
    assert "values" in message
    assert "[1]" in message
    assert "bad" in message

def test_validate_json_value_does_not_modify_input() -> None:
    value = {
        "metrics": [1, 2.5, None],
    }
    original = {
        "metrics": [1, 2.5, None],
    }

    validate_json_value(value)

    assert value == original


def test_trajectory_batch_accepts_valid_data() -> None:
    batch = make_valid_trajectory_batch()

    assert batch.validate() is None

def test_trajectory_batch_rejects_unsupported_schema_version() -> None:
    batch = make_valid_trajectory_batch()
    with pytest.raises(ValueError, match="unsupported schema_version"):
        replace(batch, schema_version="2.0")

def test_trajectory_batch_reject_duplicate_request_ids() -> None:
    batch = make_valid_trajectory_batch()
    with pytest.raises(ValueError, match="unique within a batch"):
        replace(batch, request_ids=["duplicate", "duplicate"])

def test_trajectory_batch_rejects_wrong_ranks() -> None:
    batch = make_valid_trajectory_batch()
    with pytest.raises(ValueError, match="dimensional"):
        replace(batch, group_ids=torch.tensor([[1, 2], [3, 4]]))
