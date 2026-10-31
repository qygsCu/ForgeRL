import json
from dataclasses import asdict, replace

import pytest

from forgerl.sdk import (
    CheckpointError,
    ConfigurationError,
    ErrorEnvelope,
    ForgeRLError,
    InvalidStateTransitionError,
    OutOfMemoryError,
    PluginCompatibilityError,
    PluginError,
    PluginExecutionError,
    PluginLoadError,
    ProtocolViolationError,
    WeightSyncError,
    WorkerCrashedError,
    WorkerError,
    WorkerTimeoutError,
)


def make_valid_error_envelope() -> ErrorEnvelope:
    return ErrorEnvelope(
        schema_version="1.0",
        error_type="WorkerTimeoutError",
        message="worker heartbeat timed out",
        worker_id="rollout-worker-0",
        stage="rollout",
        policy_version=3,
        traceback="Traceback (most recent call last):\n...",
    )


@pytest.mark.parametrize(
    "error_class",
    [
        ConfigurationError,
        ProtocolViolationError,
        InvalidStateTransitionError,
        PluginLoadError,
        PluginCompatibilityError,
        PluginExecutionError,
        WorkerTimeoutError,
        WorkerCrashedError,
        OutOfMemoryError,
        WeightSyncError,
        CheckpointError,
    ],
)
def test_expected_errors_share_common_base(
    error_class: type[ForgeRLError],
) -> None:
    with pytest.raises(ForgeRLError, match="expected failure"):
        raise error_class("expected failure")


def test_plugin_errors_share_plugin_base() -> None:
    assert issubclass(PluginLoadError, PluginError)
    assert issubclass(PluginCompatibilityError, PluginError)
    assert issubclass(PluginExecutionError, PluginError)


def test_worker_errors_share_worker_base() -> None:
    assert issubclass(WorkerTimeoutError, WorkerError)
    assert issubclass(WorkerCrashedError, WorkerError)


def test_error_envelope_accepts_valid_data() -> None:
    envelope = make_valid_error_envelope()

    assert envelope.validate() is None


def test_error_envelope_accepts_optional_fields_as_none() -> None:
    envelope = make_valid_error_envelope()

    result = replace(
        envelope,
        worker_id=None,
        policy_version=None,
        traceback=None,
    )

    assert result.worker_id is None
    assert result.policy_version is None
    assert result.traceback is None


def test_error_envelope_is_json_serializable() -> None:
    envelope = make_valid_error_envelope()

    restored = json.loads(json.dumps(asdict(envelope)))

    assert restored["error_type"] == "WorkerTimeoutError"
    assert restored["policy_version"] == 3


def test_error_envelope_rejects_unsupported_schema_version() -> None:
    envelope = make_valid_error_envelope()

    with pytest.raises(ValueError, match="unsupported schema_version"):
        replace(envelope, schema_version="2.0")


@pytest.mark.parametrize("field_name", ["error_type", "message", "stage"])
def test_error_envelope_rejects_empty_required_string(field_name: str) -> None:
    envelope = make_valid_error_envelope()

    with pytest.raises(ValueError, match=f"{field_name} must not be empty"):
        replace(envelope, **{field_name: ""})


@pytest.mark.parametrize("field_name", ["worker_id", "traceback"])
def test_error_envelope_rejects_empty_optional_string(field_name: str) -> None:
    envelope = make_valid_error_envelope()

    with pytest.raises(ValueError, match=f"{field_name} must not be empty"):
        replace(envelope, **{field_name: ""})


def test_error_envelope_rejects_negative_policy_version() -> None:
    envelope = make_valid_error_envelope()

    with pytest.raises(ValueError, match="policy_version must be non-negative"):
        replace(envelope, policy_version=-1)


def test_error_envelope_rejects_boolean_policy_version() -> None:
    envelope = make_valid_error_envelope()

    with pytest.raises(TypeError, match="policy_version must be int or None"):
        replace(envelope, policy_version=True)
