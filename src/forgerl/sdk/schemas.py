import math
from dataclasses import dataclass
from typing import TypeAlias

import torch

TRAJECTORY_BATCH_SCHEMA_VERSION = "1.0"

JsonPrimitive: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonPrimitive | list["JsonValue"] | dict[str, "JsonValue"]


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
