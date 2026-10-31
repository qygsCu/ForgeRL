"""Run lifecycle transitions and their structured events.

The coordinator owns this state machine and serializes transition requests.
Transitions do not execute plugin operations, release resources, or publish events.
"""

from collections.abc import Mapping
from enum import StrEnum
from time import time_ns
from types import MappingProxyType
from typing import Final
from uuid import uuid4

from forgerl.sdk.errors import InvalidStateTransitionError
from forgerl.sdk.schemas import RUNTIME_EVENT_SCHEMA_VERSION, RuntimeEvent, RuntimeEventType


class RuntimeState(StrEnum):
    """Lifecycle stages of one run, independent of operating system processes."""

    CREATED = "CREATED"
    VALIDATING = "VALIDATING"
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    CHECKPOINTING = "CHECKPOINTING"
    STOPPING = "STOPPING"
    FINISHED = "FINISHED"
    DEGRADED = "DEGRADED"
    RECOVERING = "RECOVERING"
    FAILED = "FAILED"


_ALLOWED_TRANSITIONS: Final[Mapping[RuntimeState, frozenset[RuntimeState]]] = MappingProxyType(
    {
        RuntimeState.CREATED: frozenset({RuntimeState.VALIDATING}),
        RuntimeState.VALIDATING: frozenset({RuntimeState.INITIALIZING, RuntimeState.FAILED}),
        RuntimeState.INITIALIZING: frozenset({RuntimeState.RUNNING, RuntimeState.FAILED}),
        RuntimeState.RUNNING: frozenset(
            {
                RuntimeState.CHECKPOINTING,
                RuntimeState.STOPPING,
                RuntimeState.DEGRADED,
                RuntimeState.FAILED,
            }
        ),
        RuntimeState.CHECKPOINTING: frozenset(
            {RuntimeState.RUNNING, RuntimeState.DEGRADED, RuntimeState.FAILED}
        ),
        RuntimeState.STOPPING: frozenset({RuntimeState.FINISHED, RuntimeState.FAILED}),
        RuntimeState.FINISHED: frozenset(),
        RuntimeState.DEGRADED: frozenset({RuntimeState.RECOVERING, RuntimeState.FAILED}),
        RuntimeState.RECOVERING: frozenset({RuntimeState.RUNNING, RuntimeState.FAILED}),
        RuntimeState.FAILED: frozenset(),
    }
)


class RuntimeStateMachine:
    """Maintain one run's state and return an event for each successful transition.

    The coordinator establishes business preconditions, serializes calls, and
    collects or publishes returned events. Entering FAILED records a failed
    outcome; resource cleanup remains the coordinator's responsibility.
    """

    def __init__(self, run_id: str) -> None:
        """Create a state machine in CREATED for a non-empty run identifier.

        Raises:
            TypeError: run_id is not a string.
            ValueError: run_id is empty or contains only whitespace.
        """
        if not isinstance(run_id, str):
            raise TypeError(
                f"Invalid run_id: expected a non-empty string, received {type(run_id).__name__}."
            )
        if not run_id.strip():
            raise ValueError("Invalid run_id: the run identifier must not be empty or whitespace.")
        self._run_id = run_id
        self._state = RuntimeState.CREATED

    @property
    def state(self) -> RuntimeState:
        """Return the current lifecycle state without permitting direct assignment."""
        return self._state

    def can_transition(self, target: RuntimeState) -> bool:
        """Check a target against the current state without generating an event.

        Raises:
            TypeError: target is not a RuntimeState member.
        """
        if not isinstance(target, RuntimeState):
            raise TypeError(
                f"Invalid transition target for run {self._run_id!r}: "
                f"expected RuntimeState, received {type(target).__name__}."
            )
        return target in _ALLOWED_TRANSITIONS[self._state]

    def transition(
        self,
        target: RuntimeState,
        *,
        reason: str,
        policy_version: int | None = None,
    ) -> RuntimeEvent:
        """Commit a permitted transition and return its RUN_STATE_CHANGED event.

        All inputs and the event are validated before the current state changes.
        Failure leaves the original state intact. Event publication is delegated
        to the caller; returning does not guarantee durable delivery.

        Args:
            target: Lifecycle stage requested by the coordinator.
            reason: Non-empty explanation of the transition.
            policy_version: Current policy version, or None during startup.

        Raises:
            InvalidStateTransitionError: The current state does not permit target.
            TypeError: target, reason, or policy_version has an invalid type.
            ValueError: reason is blank, policy_version is negative, or generated
                event fields violate the RuntimeEvent contract.
        """
        if not self.can_transition(target):
            allowed = _ALLOWED_TRANSITIONS[self._state]
            explanation = (
                "Allowed target states: "
                + ", ".join(sorted(state.value for state in allowed))
                + "."
                if allowed
                else "The current state is terminal and does not permit further transitions."
            )
            raise InvalidStateTransitionError(
                f"Cannot transition run {self._run_id!r} from {self._state.value} "
                f"to {target.value}: the requested transition is not permitted. {explanation}"
            )

        if not isinstance(reason, str):
            raise TypeError(
                f"Invalid transition reason for run {self._run_id!r}: "
                f"expected a non-empty string, received {type(reason).__name__}."
            )
        if not reason.strip():
            raise ValueError(
                f"Invalid transition reason for run {self._run_id!r}: "
                "the reason must not be empty or whitespace."
            )
        if policy_version is not None:
            if type(policy_version) is not int:
                raise TypeError(
                    f"Invalid policy_version for run {self._run_id!r}: "
                    "expected a non-negative integer or None, "
                    f"received {type(policy_version).__name__}."
                )
            if policy_version < 0:
                raise ValueError(
                    f"Invalid policy_version for run {self._run_id!r}: "
                    f"expected a non-negative integer, received {policy_version}."
                )

        event = RuntimeEvent(
            schema_version=RUNTIME_EVENT_SCHEMA_VERSION,
            event_id=str(uuid4()),
            event_type=RuntimeEventType.RUN_STATE_CHANGED,
            timestamp_ns=time_ns(),
            run_id=self._run_id,
            worker_id=None,
            policy_version=policy_version,
            payload={
                "from_state": self._state.value,
                "to_state": target.value,
                "reason": reason,
            },
        )
        self._state = target
        return event
