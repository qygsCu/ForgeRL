"""Pydantic models defining ForgeRL's core configuration structure."""

from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
)
from pydantic import (
    JsonValue as PydanticJsonValue,
)

from forgerl.sdk.schemas import validate_json_value

NonEmptyStr = Annotated[StrictStr, Field(min_length=1)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]


class StrictConfigModel(BaseModel):
    """Base model that rejects unknown configuration fields."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ExecutionMode(StrEnum):
    """Supported runtime execution layouts."""

    MOCK = "mock"
    COLOCATED_SEQUENTIAL = "colocated_sequential"
    SEPARATED_SYNC = "separated_sync"
    BOUNDED_ASYNC = "bounded_async"


class ModelDType(StrEnum):
    """Supported model computation dtypes."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class RunConfig(StrictConfigModel):
    """Settings identifying and locating one ForgeRL run."""

    name: NonEmptyStr
    seed: NonNegativeInt = 42
    output_dir: NonEmptyStr
    execution_mode: ExecutionMode = ExecutionMode.MOCK


class ModelConfig(StrictConfigModel):
    """Shared model selection settings."""

    name_or_path: NonEmptyStr
    dtype: ModelDType = ModelDType.FLOAT32
    use_lora: StrictBool = True


class PluginSelection(StrictConfigModel):
    """Select a plugin and carry JSON-only plugin-owned options."""

    plugin: NonEmptyStr
    options: dict[str, PydanticJsonValue] = Field(default_factory=dict)

    @field_validator("options", mode="before")
    @classmethod
    def options_must_be_strict_json(cls, value: object) -> object:
        """Reject values that cannot cross configuration boundaries safely."""
        if not isinstance(value, dict):
            raise ValueError("options must be a dictionary")
        validate_json_value(value)
        return value


class SchedulerConfig(PluginSelection):
    """Core queue and policy-staleness scheduling settings."""

    plugin: NonEmptyStr = "fifo"
    queue_capacity: PositiveInt = 2
    max_staleness: NonNegativeInt = 0


class CheckpointConfig(StrictConfigModel):
    """Checkpoint retention and frequency settings."""

    every_steps: PositiveInt = 10
    keep_last: PositiveInt = 3


class ObservabilityConfig(StrictConfigModel):
    """Structured event and profiling switches."""

    jsonl_events: StrictBool = True
    nvtx: StrictBool = False


class ForgeRLConfig(StrictConfigModel):
    """Top-level validated configuration consumed by the runtime."""

    run: RunConfig
    model: ModelConfig
    algorithm: PluginSelection
    rollout: PluginSelection
    reward: PluginSelection
    trainer: PluginSelection
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
