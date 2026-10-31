import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, TypeAlias

import torch

TRAJECTORY_BATCH_SCHEMA_VERSION = "1.0"
WEIGHT_SNAPSHOT_SCHEMA_VERSION = "1.0"
RUNTIME_EVENT_SCHEMA_VERSION = "1.0"

SUPPORTED_WEIGHT_FORMATS = frozenset({"full", "lora", "sharded"})
SHA256_CHECKSUM_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")

JsonPrimitive: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]

WeightFormat: TypeAlias = Literal["full", "lora", "sharded"]


def validate_json_value(value: object) -> None:
    """Validate that a value conforms to ForgeRL's strict JSON rules."""
    _validate_json_value(value=value, path="$")


def _validate_json_value(value: object, *, path: str) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must be finite, got {value}")
        return
    if isinstance(value, str):
        return
    if isinstance(value, list):
        for idx, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{idx!r}]")
        return
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                raise ValueError(f"{path} contains non-string key {k} of type {type(k).__name__}")
            _validate_json_value(v, path=f"{path}[{k!r}]")
        return
    raise ValueError(f"{path} must be JSON-compatible, got {type(value).__name__}")


@dataclass
class TrajectoryBatch:
    schema_version: str
    request_ids: list[str]
    group_ids: torch.Tensor
    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor
    old_logprobs: torch.Tensor
    sequence_rewards: torch.Tensor | None
    policy_version: int
    metadata: dict[str, JsonValue]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the batch against the ForgeRL trajectory data protocol."""
        if type(self.schema_version) is not str:
            raise TypeError(f"schema_version must be str, got {type(self.schema_version).__name__}")
        if self.schema_version != TRAJECTORY_BATCH_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {TRAJECTORY_BATCH_SCHEMA_VERSION!r}"
            )

        if type(self.policy_version) is not int:
            raise TypeError(f"policy_version must be int, got {type(self.policy_version).__name__}")
        if self.policy_version < 0:
            raise ValueError(f"policy_version must be non-negative, got {self.policy_version}")

        if not isinstance(self.request_ids, list):
            raise TypeError(f"request_ids must be a list, got {type(self.request_ids).__name__}")
        if not self.request_ids:
            raise ValueError("request_ids must not be empty")
        for index, request_id in enumerate(self.request_ids):
            if not isinstance(request_id, str):
                raise TypeError(
                    f"request_ids[{index}] must be str, got {type(request_id).__name__}"
                )
            if not request_id:
                raise ValueError(f"request_ids[{index}] must not be empty")
        if len(set(self.request_ids)) != len(self.request_ids):
            raise ValueError("request_ids must be unique within a batch")

        tensors = {
            "group_ids": self.group_ids,
            "prompt_ids": self.prompt_ids,
            "response_ids": self.response_ids,
            "attention_mask": self.attention_mask,
            "response_mask": self.response_mask,
            "old_logprobs": self.old_logprobs,
        }
        if self.sequence_rewards is not None:
            tensors["sequence_rewards"] = self.sequence_rewards

        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")

        expected_ranks = {
            "group_ids": 1,
            "prompt_ids": 2,
            "response_ids": 2,
            "attention_mask": 2,
            "response_mask": 2,
            "old_logprobs": 2,
        }
        if self.sequence_rewards is not None:
            expected_ranks["sequence_rewards"] = 1

        for name, expected_rank in expected_ranks.items():
            tensor = tensors[name]
            if tensor.ndim != expected_rank:
                raise ValueError(
                    f"{name} must be {expected_rank}-dimensional, got shape {tuple(tensor.shape)}"
                )

        batch_size, prompt_length = self.prompt_ids.shape
        response_batch_size, response_length = self.response_ids.shape
        if batch_size <= 0:
            raise ValueError("batch size B must be greater than zero")
        if prompt_length <= 0:
            raise ValueError("prompt length P must be greater than zero")
        if response_length <= 0:
            raise ValueError("response length R must be greater than zero")
        if response_batch_size != batch_size:
            raise ValueError(
                "response_ids must have the same batch size as prompt_ids; "
                f"expected {batch_size}, got {response_batch_size}"
            )

        expected_shapes = {
            "group_ids": (batch_size,),
            "prompt_ids": (batch_size, prompt_length),
            "response_ids": (batch_size, response_length),
            "attention_mask": (batch_size, prompt_length + response_length),
            "response_mask": (batch_size, response_length),
            "old_logprobs": (batch_size, response_length),
        }
        if self.sequence_rewards is not None:
            expected_shapes["sequence_rewards"] = (batch_size,)

        if len(self.request_ids) != batch_size:
            raise ValueError(
                f"request_ids must contain {batch_size} entries, got {len(self.request_ids)}"
            )
        for name, expected_shape in expected_shapes.items():
            actual_shape = tuple(tensors[name].shape)
            if actual_shape != expected_shape:
                raise ValueError(f"{name} must have shape {expected_shape}, got {actual_shape}")

        expected_dtypes = {
            "group_ids": torch.int64,
            "prompt_ids": torch.int64,
            "response_ids": torch.int64,
            "attention_mask": torch.bool,
            "response_mask": torch.bool,
            "old_logprobs": torch.float32,
        }
        if self.sequence_rewards is not None:
            expected_dtypes["sequence_rewards"] = torch.float32

        for name, expected_dtype in expected_dtypes.items():
            actual_dtype = tensors[name].dtype
            if actual_dtype != expected_dtype:
                raise TypeError(f"{name} must have dtype {expected_dtype}, got {actual_dtype}")

        expected_device = self.prompt_ids.device
        for name, tensor in tensors.items():
            if tensor.device != expected_device:
                raise ValueError(
                    f"all tensors must be on device {expected_device}; {name} is on {tensor.device}"
                )

        prompt_mask = self.attention_mask[:, :prompt_length]
        response_attention_mask = self.attention_mask[:, prompt_length:]
        if not torch.equal(response_attention_mask, self.response_mask):
            raise ValueError("the last R columns of attention_mask must equal response_mask")

        prompt_has_right_padding = prompt_mask[:, :-1] & ~prompt_mask[:, 1:]
        if torch.any(prompt_has_right_padding).item():
            raise ValueError("prompt attention mask must use left padding")

        response_has_left_padding = ~self.response_mask[:, :-1] & self.response_mask[:, 1:]
        if torch.any(response_has_left_padding).item():
            raise ValueError("response_mask must use right padding")

        masked_logprobs = self.old_logprobs.masked_select(~self.response_mask)
        if torch.any(masked_logprobs != 0.0).item():
            raise ValueError("old_logprobs must be 0.0 where response_mask is False")

        active_logprobs = self.old_logprobs.masked_select(self.response_mask)
        if not torch.all(torch.isfinite(active_logprobs)).item():
            raise ValueError("old_logprobs must be finite at active response positions")
        if (
            self.sequence_rewards is not None
            and not torch.all(torch.isfinite(self.sequence_rewards)).item()
        ):
            raise ValueError("sequence_rewards must contain only finite values")

        if not isinstance(self.metadata, dict):
            raise TypeError(f"metadata must be a dict, got {type(self.metadata).__name__}")
        validate_json_value(self.metadata)


@dataclass
class WeightSnapshot:
    schema_version: str
    policy_version: int
    format: WeightFormat
    uri: str | None
    checksum: str
    metadata: dict[str, JsonValue]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the weight snapshot descriptor."""
        if type(self.schema_version) is not str:
            raise TypeError(f"schema_version must be str, got {type(self.schema_version).__name__}")
        if self.schema_version != WEIGHT_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {WEIGHT_SNAPSHOT_SCHEMA_VERSION!r}"
            )

        if type(self.policy_version) is not int:
            raise TypeError(f"policy_version must be int, got {type(self.policy_version).__name__}")
        if self.policy_version < 0:
            raise ValueError(f"policy_version must be non-negative, got {self.policy_version}")

        if not isinstance(self.format, str):
            raise TypeError(f"format must be str, got {type(self.format).__name__}")
        if self.format not in SUPPORTED_WEIGHT_FORMATS:
            supported_formats = ", ".join(sorted(SUPPORTED_WEIGHT_FORMATS))
            raise ValueError(
                f"unsupported weight format {self.format!r}; expected one of: {supported_formats}"
            )

        if self.uri is not None:
            if not isinstance(self.uri, str):
                raise TypeError(f"uri must be str or None, got {type(self.uri).__name__}")
            if not self.uri:
                raise ValueError("uri must not be empty when provided")

        if not isinstance(self.checksum, str):
            raise TypeError(f"checksum must be str, got {type(self.checksum).__name__}")
        if SHA256_CHECKSUM_PATTERN.fullmatch(self.checksum) is None:
            raise ValueError(
                "checksum must have format 'sha256:' followed by "
                "64 lowercase hexadecimal characters"
            )

        if not isinstance(self.metadata, dict):
            raise TypeError(f"metadata must be a dict, got {type(self.metadata).__name__}")
        validate_json_value(self.metadata)


class RuntimeEventType(StrEnum):
    """Controlled event categories shared by runtime producers and consumers."""

    RUN_STATE_CHANGED = "RUN_STATE_CHANGED"
    RUN_STARTED = "RUN_STARTED"
    PLUGIN_LOADED = "PLUGIN_LOADED"
    WORKER_STARTED = "WORKER_STARTED"
    WORKER_HEARTBEAT = "WORKER_HEARTBEAT"
    ROLLOUT_COMPLETED = "ROLLOUT_COMPLETED"
    TRAJECTORY_DROPPED = "TRAJECTORY_DROPPED"
    TRAIN_STEP_COMPLETED = "TRAIN_STEP_COMPLETED"
    WEIGHT_PUBLISHED = "WEIGHT_PUBLISHED"
    CHECKPOINT_STARTED = "CHECKPOINT_STARTED"
    CHECKPOINT_COMPLETED = "CHECKPOINT_COMPLETED"
    WORKER_FAILED = "WORKER_FAILED"
    RUN_RECOVERED = "RUN_RECOVERED"
    RUN_FINISHED = "RUN_FINISHED"


@dataclass
class RuntimeEvent:
    schema_version: str
    event_id: str
    event_type: RuntimeEventType
    timestamp_ns: int
    run_id: str
    worker_id: str | None
    policy_version: int | None
    payload: dict[str, JsonValue]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate a structured runtime event."""
        if type(self.schema_version) is not str:
            raise TypeError(f"schema_version must be str, got {type(self.schema_version).__name__}")
        if self.schema_version != RUNTIME_EVENT_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {self.schema_version!r}; "
                f"expected {RUNTIME_EVENT_SCHEMA_VERSION!r}"
            )

        if not isinstance(self.event_id, str):
            raise TypeError(f"event_id must be str, got {type(self.event_id).__name__}")
        if not self.event_id:
            raise ValueError("event_id must not be empty")

        if not isinstance(self.event_type, RuntimeEventType):
            raise TypeError(
                f"event_type must be a RuntimeEventType, got {type(self.event_type).__name__}"
            )

        if type(self.timestamp_ns) is not int:
            raise TypeError(f"timestamp_ns must be int, got {type(self.timestamp_ns).__name__}")
        if self.timestamp_ns < 0:
            raise ValueError(f"timestamp_ns must be non-negative, got {self.timestamp_ns}")

        if not isinstance(self.run_id, str):
            raise TypeError(f"run_id must be str, got {type(self.run_id).__name__}")
        if not self.run_id:
            raise ValueError("run_id must not be empty")

        if self.worker_id is not None:
            if not isinstance(self.worker_id, str):
                raise TypeError(
                    f"worker_id must be str or None, got {type(self.worker_id).__name__}"
                )
            if not self.worker_id:
                raise ValueError("worker_id must not be empty when provided")

        if self.policy_version is not None:
            if type(self.policy_version) is not int:
                raise TypeError(
                    f"policy_version must be int or None, got {type(self.policy_version).__name__}"
                )
            if self.policy_version < 0:
                raise ValueError(f"policy_version must be non-negative, got {self.policy_version}")

        if not isinstance(self.payload, dict):
            raise TypeError(f"payload must be a dict, got {type(self.payload).__name__}")
        validate_json_value(self.payload)
