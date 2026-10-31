"""Behavioral tests for run lifecycle rules and transition event consistency."""

import json
from dataclasses import asdict
from uuid import UUID

import pytest

from forgerl.runtime.state import RuntimeState, RuntimeStateMachine
from forgerl.sdk import ForgeRLError, InvalidStateTransitionError
from forgerl.sdk.schemas import RUNTIME_EVENT_SCHEMA_VERSION, RuntimeEvent, RuntimeEventType

STARTUP = (RuntimeState.VALIDATING, RuntimeState.INITIALIZING, RuntimeState.RUNNING)
STATE_PATHS = [
    (),
    STARTUP[:1],
    STARTUP[:2],
    STARTUP,
    (*STARTUP, RuntimeState.CHECKPOINTING),
    (*STARTUP, RuntimeState.STOPPING),
    (*STARTUP, RuntimeState.STOPPING, RuntimeState.FINISHED),
    (*STARTUP, RuntimeState.DEGRADED),
    (*STARTUP, RuntimeState.DEGRADED, RuntimeState.RECOVERING),
    (RuntimeState.VALIDATING, RuntimeState.FAILED),
]


def follow_path(path: tuple[RuntimeState, ...]) -> RuntimeStateMachine:
    """Reach a lifecycle stage through public transitions rather than private writes."""
    machine = RuntimeStateMachine("run-001")
    for target in path:
        machine.transition(target, reason="Advance test workflow")
    return machine


def test_new_machine_has_independent_read_only_state() -> None:
    """A new run starts in CREATED and cannot change through the state property."""
    first = RuntimeStateMachine("run-001")
    second = RuntimeStateMachine("run-002")

    assert first.state is RuntimeState.CREATED
    first.transition(RuntimeState.VALIDATING, reason="Start validation")

    assert second.state is RuntimeState.CREATED
    with pytest.raises(AttributeError):
        first.state = RuntimeState.RUNNING  # type: ignore[misc]
    assert first.state is RuntimeState.VALIDATING


@pytest.mark.parametrize("run_id", [None, 42, False, []])
def test_constructor_rejects_non_string_run_id(run_id: object) -> None:
    """Invalid identifier types produce an actionable error before construction."""
    with pytest.raises(TypeError, match="Invalid run_id.*expected a non-empty string"):
        RuntimeStateMachine(run_id)  # type: ignore[arg-type]


@pytest.mark.parametrize("run_id", ["", " ", "\t\n"])
def test_constructor_rejects_blank_run_id(run_id: str) -> None:
    """Empty and whitespace-only identifiers are rejected."""
    with pytest.raises(ValueError, match="run identifier must not be empty or whitespace"):
        RuntimeStateMachine(run_id)


def test_can_transition_is_a_side_effect_free_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """Queries neither change state nor generate event identifiers."""
    machine = RuntimeStateMachine("run-001")

    def unexpected_uuid() -> UUID:
        """Fail if a query attempts to prepare an event."""
        raise AssertionError("A transition query must not generate an event")

    monkeypatch.setattr("forgerl.runtime.state.uuid4", unexpected_uuid)

    assert machine.can_transition(RuntimeState.VALIDATING)
    assert not machine.can_transition(RuntimeState.RUNNING)
    assert not machine.can_transition(RuntimeState.CREATED)
    assert machine.state is RuntimeState.CREATED


@pytest.mark.parametrize("target", ["VALIDATING", None, 1, []])
def test_target_must_be_an_enum_member(target: object) -> None:
    """Strings equal to StrEnum values must not bypass the typed API."""
    machine = RuntimeStateMachine("run-001")

    with pytest.raises(TypeError, match="run-001.*expected RuntimeState"):
        machine.can_transition(target)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="run-001.*expected RuntimeState"):
        machine.transition(target, reason="Invalid target")  # type: ignore[arg-type]

    assert machine.state is RuntimeState.CREATED


def test_transition_returns_a_complete_json_serializable_event() -> None:
    """The result records the committed transition and preserves caller inputs."""
    machine = RuntimeStateMachine(" run-001 ")
    reason = " Start validation "

    event = machine.transition(RuntimeState.VALIDATING, reason=reason, policy_version=0)

    assert machine.state is RuntimeState.VALIDATING
    assert isinstance(event, RuntimeEvent)
    assert event.schema_version == RUNTIME_EVENT_SCHEMA_VERSION == "1.0"
    assert event.event_type is RuntimeEventType.RUN_STATE_CHANGED
    assert event.run_id == " run-001 "
    assert event.worker_id is None
    assert event.policy_version == 0
    assert event.timestamp_ns > 0
    assert UUID(event.event_id).version == 4
    assert event.payload == {
        "from_state": "CREATED",
        "to_state": "VALIDATING",
        "reason": reason,
    }
    restored = json.loads(json.dumps(asdict(event), allow_nan=False))
    assert restored["event_type"] == "RUN_STATE_CHANGED"
    assert restored["payload"] == event.payload


def test_normal_workflow_can_checkpoint_and_finish() -> None:
    """A checkpoint cycle does not prevent normal shutdown and successful completion."""
    machine = RuntimeStateMachine("run-001")
    events = [
        machine.transition(target, reason="Advance normal workflow")
        for target in (
            *STARTUP,
            RuntimeState.CHECKPOINTING,
            RuntimeState.RUNNING,
            RuntimeState.STOPPING,
            RuntimeState.FINISHED,
        )
    ]

    assert machine.state is RuntimeState.FINISHED
    assert len({event.event_id for event in events}) == len(events)
    assert events[3].payload["from_state"] == "RUNNING"
    assert events[4].payload["to_state"] == "RUNNING"
    assert events[-1].payload["from_state"] == "STOPPING"


@pytest.mark.parametrize("path", STATE_PATHS)
def test_every_state_rejects_self_transitions(path: tuple[RuntimeState, ...]) -> None:
    """Repeated transition requests fail without changing the current stage."""
    machine = follow_path(path)
    original = machine.state

    assert not machine.can_transition(original)
    with pytest.raises(InvalidStateTransitionError, match="run-001"):
        machine.transition(original, reason="Duplicate transition")

    assert machine.state is original


def test_illegal_transition_reports_context_without_generating_an_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rejected requests identify the run, source, target, and valid alternatives."""
    machine = RuntimeStateMachine("run-001")

    def unexpected_uuid() -> UUID:
        """Fail if an illegal transition tries to generate an event."""
        raise AssertionError("An illegal transition must not generate an event")

    monkeypatch.setattr("forgerl.runtime.state.uuid4", unexpected_uuid)

    with pytest.raises(InvalidStateTransitionError) as exc_info:
        machine.transition(RuntimeState.RUNNING, reason="Skip startup")

    assert isinstance(exc_info.value, ForgeRLError)
    message = str(exc_info.value)
    for expected in ("run-001", "CREATED", "RUNNING", "Allowed target states: VALIDATING"):
        assert expected in message
    assert machine.state is RuntimeState.CREATED


@pytest.mark.parametrize("path", [STATE_PATHS[6], STATE_PATHS[9]])
def test_terminal_states_reject_all_targets(path: tuple[RuntimeState, ...]) -> None:
    """Neither successful nor failed runs can be restarted through a transition."""
    machine = follow_path(path)
    terminal = machine.state

    for target in RuntimeState:
        assert not machine.can_transition(target)
        with pytest.raises(InvalidStateTransitionError, match="terminal"):
            machine.transition(target, reason="Attempt to restart a terminated run")
        assert machine.state is terminal


@pytest.mark.parametrize(
    "path",
    [
        STATE_PATHS[0],
        STATE_PATHS[1],
        STATE_PATHS[2],
        STATE_PATHS[4],
        STATE_PATHS[7],
        STATE_PATHS[8],
    ],
)
def test_stopping_is_only_allowed_from_running(path: tuple[RuntimeState, ...]) -> None:
    """Startup, checkpointing, and recovery cannot be interrupted by an undefined stop."""
    machine = follow_path(path)
    original = machine.state

    with pytest.raises(InvalidStateTransitionError):
        machine.transition(RuntimeState.STOPPING, reason="Stop requested")

    assert machine.state is original


@pytest.mark.parametrize("path", [STARTUP, (*STARTUP, RuntimeState.CHECKPOINTING)])
def test_runtime_fault_can_recover(path: tuple[RuntimeState, ...]) -> None:
    """Both running and checkpointing faults can enter the approved recovery path."""
    machine = follow_path(path)
    machine.transition(RuntimeState.DEGRADED, reason="Fault detected")

    with pytest.raises(InvalidStateTransitionError):
        machine.transition(RuntimeState.RUNNING, reason="Attempt to skip recovery")

    machine.transition(RuntimeState.RECOVERING, reason="Begin recovery")
    event = machine.transition(RuntimeState.RUNNING, reason="Recovery verified", policy_version=7)

    assert machine.state is RuntimeState.RUNNING
    assert event.payload["from_state"] == "RECOVERING"
    assert event.policy_version == 7


@pytest.mark.parametrize("path", [(), STARTUP[:1], STARTUP[:2]])
def test_startup_cannot_bypass_initialization_via_degradation(
    path: tuple[RuntimeState, ...],
) -> None:
    """Recovery is unavailable until the run has successfully initialized."""
    machine = follow_path(path)
    original = machine.state

    for target in (RuntimeState.DEGRADED, RuntimeState.RECOVERING):
        with pytest.raises(InvalidStateTransitionError):
            machine.transition(target, reason="Attempt to bypass initialization")
        assert machine.state is original


@pytest.mark.parametrize(
    "path",
    [
        STARTUP[:1],
        STARTUP[:2],
        STARTUP,
        STATE_PATHS[4],
        STATE_PATHS[5],
        STATE_PATHS[7],
        STATE_PATHS[8],
    ],
)
def test_failures_are_valid_transitions_from_active_stages(
    path: tuple[RuntimeState, ...],
) -> None:
    """Startup, runtime, recovery, and cleanup errors can terminate the run as failed."""
    machine = follow_path(path)
    source = machine.state

    event = machine.transition(RuntimeState.FAILED, reason="Unrecoverable operation failure")

    assert machine.state is RuntimeState.FAILED
    assert event.payload["from_state"] == source.value
    assert event.payload["to_state"] == "FAILED"


@pytest.mark.parametrize("reason", [None, 1, False])
def test_invalid_reason_type_preserves_state(reason: object) -> None:
    """Reason types are validated before changing state."""
    machine = RuntimeStateMachine("run-001")

    with pytest.raises(TypeError, match="transition reason.*run-001"):
        machine.transition(RuntimeState.VALIDATING, reason=reason)  # type: ignore[arg-type]

    assert machine.state is RuntimeState.CREATED


@pytest.mark.parametrize("reason", ["", " ", "\n\t"])
def test_blank_reason_preserves_state(reason: str) -> None:
    """A reason must contain explanatory text rather than only whitespace."""
    machine = RuntimeStateMachine("run-001")

    with pytest.raises(ValueError, match="reason must not be empty or whitespace"):
        machine.transition(RuntimeState.VALIDATING, reason=reason)

    assert machine.state is RuntimeState.CREATED


@pytest.mark.parametrize("policy_version", [True, False, "1", 1.0])
def test_invalid_policy_version_type_preserves_state(policy_version: object) -> None:
    """Boolean values must not pass the integer policy version contract."""
    machine = RuntimeStateMachine("run-001")

    with pytest.raises(TypeError, match="policy_version.*run-001"):
        machine.transition(
            RuntimeState.VALIDATING,
            reason="Start validation",
            policy_version=policy_version,  # type: ignore[arg-type]
        )

    assert machine.state is RuntimeState.CREATED


def test_negative_policy_version_preserves_state() -> None:
    """Invalid policy version values cannot partially commit a transition."""
    machine = RuntimeStateMachine("run-001")

    with pytest.raises(ValueError, match="non-negative integer, received -1"):
        machine.transition(RuntimeState.VALIDATING, reason="Start validation", policy_version=-1)

    assert machine.state is RuntimeState.CREATED


def test_event_validation_failure_preserves_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """A generated event rejected by the protocol leaves the state untouched."""
    machine = RuntimeStateMachine("run-001")
    monkeypatch.setattr("forgerl.runtime.state.time_ns", lambda: -1)

    with pytest.raises(ValueError, match="timestamp_ns must be non-negative"):
        machine.transition(RuntimeState.VALIDATING, reason="Start validation")

    assert machine.state is RuntimeState.CREATED


def test_event_generation_failure_preserves_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Failures in event generation propagate without committing the requested state."""
    machine = RuntimeStateMachine("run-001")

    def unavailable_uuid() -> UUID:
        """Simulate an unavailable event identifier provider."""
        raise RuntimeError("Event identifier generation is unavailable")

    monkeypatch.setattr("forgerl.runtime.state.uuid4", unavailable_uuid)

    with pytest.raises(RuntimeError, match="Event identifier generation is unavailable"):
        machine.transition(RuntimeState.VALIDATING, reason="Start validation")

    assert machine.state is RuntimeState.CREATED
