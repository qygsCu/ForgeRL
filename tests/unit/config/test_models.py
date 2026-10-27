import json
from copy import deepcopy
from typing import Any

import pytest
import torch
from pydantic import ValidationError

from forgerl.config import (
    ExecutionMode,
    ForgeRLConfig,
    ModelDType,
    PluginSelection,
)


def make_valid_config_data() -> dict[str, Any]:
    return {
        "run": {
            "name": "cpu-smoke-test",
            "seed": 42,
            "output_dir": "outputs/cpu-smoke-test",
            "execution_mode": "mock",
        },
        "model": {
            "name_or_path": "tiny-random-model",
            "dtype": "float32",
            "use_lora": True,
        },
        "algorithm": {
            "plugin": "grpo",
            "options": {"group_size": 4, "clip_ratio": 0.2},
        },
        "rollout": {
            "plugin": "mock-rollout",
            "options": {"max_new_tokens": 16},
        },
        "reward": {
            "plugin": "exact-match",
        },
        "trainer": {
            "plugin": "mock-trainer",
        },
    }


def test_forgerl_config_accepts_valid_data_and_applies_defaults() -> None:
    config = ForgeRLConfig.model_validate(make_valid_config_data())

    assert config.run.execution_mode is ExecutionMode.MOCK
    assert config.model.dtype is ModelDType.FLOAT32
    assert config.scheduler.plugin == "fifo"
    assert config.scheduler.queue_capacity == 2
    assert config.scheduler.max_staleness == 0
    assert config.checkpoint.every_steps == 10
    assert config.observability.jsonl_events is True
    assert config.observability.nvtx is False


def test_forgerl_config_rejects_unknown_top_level_field() -> None:
    data = make_valid_config_data()
    data["unknown_section"] = {}

    with pytest.raises(ValidationError, match="unknown_section"):
        ForgeRLConfig.model_validate(data)


def test_forgerl_config_rejects_unknown_nested_field() -> None:
    data = make_valid_config_data()
    data["run"]["typo_seed"] = 7

    with pytest.raises(ValidationError, match="typo_seed"):
        ForgeRLConfig.model_validate(data)


def test_forgerl_config_rejects_missing_required_section() -> None:
    data = make_valid_config_data()
    del data["model"]

    with pytest.raises(ValidationError, match="model"):
        ForgeRLConfig.model_validate(data)


@pytest.mark.parametrize(
    ("path", "invalid_value"),
    [
        (("run", "seed"), -1),
        (("scheduler", "queue_capacity"), 0),
        (("scheduler", "max_staleness"), -1),
        (("checkpoint", "every_steps"), 0),
        (("checkpoint", "keep_last"), 0),
    ],
)
def test_forgerl_config_rejects_invalid_numeric_range(
    path: tuple[str, str],
    invalid_value: int,
) -> None:
    data = make_valid_config_data()
    data.setdefault("scheduler", {})
    data.setdefault("checkpoint", {})
    data[path[0]][path[1]] = invalid_value

    with pytest.raises(ValidationError, match=path[1]):
        ForgeRLConfig.model_validate(data)


def test_forgerl_config_rejects_invalid_execution_mode() -> None:
    data = make_valid_config_data()
    data["run"]["execution_mode"] = "unbounded_async"

    with pytest.raises(ValidationError, match="execution_mode"):
        ForgeRLConfig.model_validate(data)


def test_forgerl_config_does_not_coerce_strict_integer() -> None:
    data = make_valid_config_data()
    data["run"]["seed"] = "42"

    with pytest.raises(ValidationError, match="seed"):
        ForgeRLConfig.model_validate(data)


def test_plugin_options_reject_non_json_value() -> None:
    data = make_valid_config_data()
    data["algorithm"]["options"] = {"tensor": torch.tensor([1.0])}

    with pytest.raises(ValidationError, match="JSON-compatible"):
        ForgeRLConfig.model_validate(data)


def test_plugin_option_defaults_are_not_shared() -> None:
    first = PluginSelection(plugin="first")
    second = PluginSelection(plugin="second")

    first.options["enabled"] = True

    assert second.options == {}


def test_configuration_snapshot_contains_defaults_and_is_json_serializable() -> None:
    config = ForgeRLConfig.model_validate(make_valid_config_data())

    snapshot = config.model_dump(mode="json")
    serialized = json.dumps(snapshot, allow_nan=False)

    assert snapshot["scheduler"]["queue_capacity"] == 2
    assert snapshot["checkpoint"]["keep_last"] == 3
    assert '"execution_mode": "mock"' in serialized


def test_configuration_validation_does_not_modify_input() -> None:
    data = make_valid_config_data()
    original = deepcopy(data)

    ForgeRLConfig.model_validate(data)

    assert data == original


def test_configuration_assignment_is_validated() -> None:
    config = ForgeRLConfig.model_validate(make_valid_config_data())

    with pytest.raises(ValidationError, match="queue_capacity"):
        config.scheduler.queue_capacity = 0
