import json
from dataclasses import asdict, replace

import pytest
import torch

from forgerl.sdk.schemas import (
    RuntimeEvent,
    RuntimeEventType,
    TrajectoryBatch,
    WeightSnapshot,
    validate_json_value,
)


def make_valid_trajectory_batch() -> TrajectoryBatch:
    return TrajectoryBatch(
        schema_version="1.0",
        request_ids=["request-1", "request-2"],
        group_ids=torch.tensor([7, 7], dtype=torch.int64),
        prompt_ids=torch.tensor([[0, 11, 12], [21, 22, 23]], dtype=torch.int64),
        response_ids=torch.tensor([[31, 32, 2, 0], [41, 2, 0, 0]], dtype=torch.int64),
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


def make_valid_weight_snapshot() -> WeightSnapshot:
    return WeightSnapshot(
        schema_version="1.0",
        policy_version=3,
        format="lora",
        uri="checkpoints/policy_3/adapter.safetensors",
        checksum="sha256:" + "a" * 64,
        metadata={"step": 100},
    )


def make_valid_runtime_event() -> RuntimeEvent:
    return RuntimeEvent(
        schema_version="1.0",
        event_id="event-001",
        event_type=RuntimeEventType.TRAIN_STEP_COMPLETED,
        timestamp_ns=1_798_000_000_000_000_000,
        run_id="run-001",
        worker_id="trainer-0",
        policy_version=3,
        payload={"step": 100, "loss": 0.25},
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


def test_weight_snapshot_accepts_valid_data() -> None:
    snapshot = make_valid_weight_snapshot()

    assert snapshot.validate() is None


def test_weight_snapshot_accepts_in_memory_uri() -> None:
    snapshot = make_valid_weight_snapshot()

    in_memory_snapshot = replace(snapshot, uri=None)

    assert in_memory_snapshot.uri is None


def test_weight_snapshot_rejects_unsupported_schema_version() -> None:
    snapshot = make_valid_weight_snapshot()

    with pytest.raises(ValueError, match="unsupported schema_version"):
        replace(snapshot, schema_version="2.0")


@pytest.mark.parametrize("weight_format", ["full", "lora", "sharded"])
def test_weight_snapshot_accepts_supported_formats(weight_format: str) -> None:
    snapshot = make_valid_weight_snapshot()

    result = replace(snapshot, format=weight_format)  # type: ignore[arg-type]

    assert result.format == weight_format


def test_weight_snapshot_rejects_negative_policy_version() -> None:
    snapshot = make_valid_weight_snapshot()

    with pytest.raises(ValueError, match="policy_version must be non-negative"):
        replace(snapshot, policy_version=-1)


def test_weight_snapshot_rejects_unknown_format() -> None:
    snapshot = make_valid_weight_snapshot()

    with pytest.raises(ValueError, match="unsupported weight format"):
        replace(snapshot, format="archive")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "checksum",
    [
        "not-a-checksum",
        "sha256:" + "a" * 63,
        "sha256:" + "A" * 64,
    ],
)
def test_weight_snapshot_rejects_invalid_checksum(checksum: str) -> None:
    snapshot = make_valid_weight_snapshot()

    with pytest.raises(ValueError, match="checksum must have format"):
        replace(snapshot, checksum=checksum)


def test_weight_snapshot_rejects_invalid_metadata() -> None:
    snapshot = make_valid_weight_snapshot()

    with pytest.raises(ValueError, match="JSON-compatible"):
        replace(
            snapshot,
            metadata={"weights": torch.tensor([1.0])},  # type: ignore[dict-item]
        )


def test_runtime_event_accepts_valid_data() -> None:
    event = make_valid_runtime_event()

    assert event.validate() is None


def test_runtime_event_accepts_optional_fields_as_none() -> None:
    event = make_valid_runtime_event()

    result = replace(event, worker_id=None, policy_version=None)

    assert result.worker_id is None
    assert result.policy_version is None


def test_runtime_event_rejects_unsupported_schema_version() -> None:
    event = make_valid_runtime_event()

    with pytest.raises(ValueError, match="unsupported schema_version"):
        replace(event, schema_version="2.0")


def test_control_plane_objects_are_json_serializable() -> None:
    snapshot = make_valid_weight_snapshot()
    event = make_valid_runtime_event()

    serialized = json.dumps(
        {"snapshot": asdict(snapshot), "event": asdict(event)},
        allow_nan=False,
    )

    assert '"format": "lora"' in serialized
    assert '"event_type": "TRAIN_STEP_COMPLETED"' in serialized


def test_runtime_event_rejects_raw_event_type_string() -> None:
    event = make_valid_runtime_event()

    with pytest.raises(TypeError, match="event_type must be a RuntimeEventType"):
        replace(event, event_type="UNKNOWN")  # type: ignore[arg-type]


@pytest.mark.parametrize("field_name", ["event_id", "run_id"])
def test_runtime_event_rejects_empty_required_id(field_name: str) -> None:
    event = make_valid_runtime_event()

    with pytest.raises(ValueError, match=f"{field_name} must not be empty"):
        replace(event, **{field_name: ""})


def test_runtime_event_rejects_negative_timestamp() -> None:
    event = make_valid_runtime_event()

    with pytest.raises(ValueError, match="timestamp_ns must be non-negative"):
        replace(event, timestamp_ns=-1)


def test_runtime_event_rejects_negative_policy_version() -> None:
    event = make_valid_runtime_event()

    with pytest.raises(ValueError, match="policy_version must be non-negative"):
        replace(event, policy_version=-1)


def test_runtime_event_rejects_invalid_payload() -> None:
    event = make_valid_runtime_event()

    with pytest.raises(ValueError, match="JSON-compatible"):
        replace(
            event,
            payload={"tensor": torch.tensor([1.0])},  # type: ignore[dict-item]
        )
