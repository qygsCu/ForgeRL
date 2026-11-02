from typing import Annotated

import pytest
from pydantic import Field, StrictInt, ValidationError

from forgerl.config.models import PluginSelection, StrictConfigModel
from forgerl.runtime.registry import PluginRegistry
from forgerl.runtime.validation import validate_plugin_options
from forgerl.sdk.errors import ConfigurationError
from forgerl.sdk.plugins import (
    SDK_API_VERSION,
    PluginDescriptor,
    PluginKind,
    PluginMetadata,
)
from forgerl.sdk.schemas import JsonValue


class MockRolloutConfig(StrictConfigModel):
    max_new_tokens: Annotated[StrictInt, Field(gt=0)] = 16


def unexpected_factory() -> object:
    raise AssertionError("配置校验不应创建插件")


def make_rollout_descriptor(
    name: str = "mock-rollout",
    kind: PluginKind = PluginKind.ROLLOUT,
    api_version: str = SDK_API_VERSION,
) -> PluginDescriptor:
    return PluginDescriptor(
        metadata=PluginMetadata(
            name=name,
            kind=kind,
            version="0.1.0",
            api_version=api_version,
            required_capabilities=frozenset(),
            optional_capabilities=frozenset(),
            supports_checkpoint=False,
            supports_multiprocessing=True,
            requires_gpu=False,
        ),
        config_model=MockRolloutConfig,
        factory=unexpected_factory,
    )


def make_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register(make_rollout_descriptor())
    return registry


@pytest.mark.parametrize(
    ("options", "expected"),
    [
        ({}, 16),
        ({"max_new_tokens": 32}, 32),
    ],
)
def test_validate_plugin_options_accepts_valid_options(
    options: dict[str, JsonValue],
    expected: int,
) -> None:
    registry = make_registry()
    selection = PluginSelection(
        plugin="mock-rollout",
        options=options,
    )

    result = validate_plugin_options(
        selection=selection,
        kind=PluginKind.ROLLOUT,
        registry=registry,
    )

    assert isinstance(result, MockRolloutConfig)
    assert result.max_new_tokens == expected


@pytest.mark.parametrize(
    "options",
    [
        {"max_new_tokens": -1},
        {"max_new_tokens": "32"},
        {"max_new_token": 32},
    ],
    ids=["negative value", "wrong type", "uknown-field"],
)
def test_validate_plugin_options_rejects_invalid_options(
    options: dict[str, JsonValue],
) -> None:
    registry = make_registry()
    selection = PluginSelection(
        plugin="mock-rollout",
        options=options,
    )

    with pytest.raises(ConfigurationError) as exc_info:
        validate_plugin_options(selection, kind=PluginKind.ROLLOUT, registry=registry)

    assert "rollout:mock-rollout" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, ValidationError)


def test_validate_plugin_options_rejects_unknown_plugin() -> None:
    registry = PluginRegistry()
    selection = PluginSelection(plugin="missing-rollout")

    with pytest.raises(
        ConfigurationError,
        match="rollout:missing-rollout",
    ):
        validate_plugin_options(
            selection,
            kind=PluginKind.ROLLOUT,
            registry=registry,
        )

