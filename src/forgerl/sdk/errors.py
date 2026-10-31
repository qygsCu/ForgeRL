"""Standard exceptions and serializable error data for ForgeRL."""

from dataclasses import dataclass

ERROR_ENVELOPE_SCHEMA_VERSION = "1.0"


class ForgeRLError(Exception):
    """Base class for expected ForgeRL failures."""


class ConfigurationError(ForgeRLError):
    """Raised when runtime configuration is invalid."""


class ProtocolViolationError(ForgeRLError):
    """Raised when data violates a ForgeRL protocol contract."""


class PluginError(ForgeRLError):
    """Base class for plugin-related failures."""


class PluginLoadError(PluginError):
    """Raised when a plugin cannot be imported or constructed."""


class PluginCompatibilityError(PluginError):
    """Raised when a plugin is incompatible with the ForgeRL SDK."""


class PluginExecutionError(PluginError):
    """Raised when an initialized plugin fails during execution."""


class InvalidStateTransitionError(ForgeRLError):
    """Raised when a run requests a transition forbidden by its lifecycle rules."""


class WorkerError(ForgeRLError):
    """Base class for worker-related failures."""


class WorkerTimeoutError(WorkerError):
    """Raised when a worker misses a required timeout."""


class WorkerCrashedError(WorkerError):
    """Raised when a worker process exits unexpectedly."""


class OutOfMemoryError(ForgeRLError):
    """Raised when an operation cannot continue because memory is exhausted."""


class WeightSyncError(ForgeRLError):
    """Raised when publishing or loading policy weights fails."""


class CheckpointError(ForgeRLError):
    """Raised when saving, loading, or validating a checkpoint fails."""


@dataclass
class ErrorEnvelope:
    """Serializable description of a failure crossing a process boundary."""

    schema_version: str
    error_type: str
    message: str
    worker_id: str | None
    stage: str
    policy_version: int | None
    traceback: str | None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the envelope without changing its values."""
        if type(self.schema_version) is not str:
            raise TypeError(f"schema_version must be str, got {type(self.schema_version).__name__}")
        if self.schema_version != ERROR_ENVELOPE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {ERROR_ENVELOPE_SCHEMA_VERSION!r}"
            )

        self._validate_required_string("error_type", self.error_type)
        self._validate_required_string("message", self.message)
        self._validate_required_string("stage", self.stage)

        self._validate_optional_string("worker_id", self.worker_id)
        self._validate_optional_string("traceback", self.traceback)

        if self.policy_version is not None:
            if type(self.policy_version) is not int:
                raise TypeError(
                    f"policy_version must be int or None, got {type(self.policy_version).__name__}"
                )
            if self.policy_version < 0:
                raise ValueError(f"policy_version must be non-negative, got {self.policy_version}")

    @staticmethod
    def _validate_required_string(name: str, value: object) -> None:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be str, got {type(value).__name__}")
        if not value:
            raise ValueError(f"{name} must not be empty")

    @staticmethod
    def _validate_optional_string(name: str, value: object) -> None:
        if value is None:
            return
        if not isinstance(value, str):
            raise TypeError(f"{name} must be str or None, got {type(value).__name__}")
        if not value:
            raise ValueError(f"{name} must not be empty when provided")
