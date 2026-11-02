from importlib import metadata

import pytest
from pydantic import BaseModel, ConfigDict

from forgerl.runtime import discovery
from forgerl.sdk.errors import PluginLoadError
from forgerl.sdk.plugins import (
    SDK_API_VERSION,
    PluginKind,
    PluginMetadata,
)


class MockConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MockReward:
    metadata = PluginMetadata(
        name="mock_reward",
        kind=PluginKind.REWARD,
        version="0.1.0",
        api_version=SDK_API_VERSION,
        required_capabilities=frozenset(),
        optional_capabilities=frozenset(),
        supports_checkpoint=False,
        supports_multiprocessing=False,
        requires_gpu=False,
    )
    config_model = MockConfig

    def __init__(self) -> None:
        raise AssertionError("发现插件时不应创建实例")


@pytest.fixture
def reward_entry_point() -> metadata.EntryPoint:
    return metadata.EntryPoint(
        name="mock_reward",
        value="example_package.reward:MockReward",
        group="forgerl.reward",
    )


def test_load_entry_point_preserves_declaration_without_instantiation(
    monkeypatch: pytest.MonkeyPatch,
    reward_entry_point: metadata.EntryPoint,
) -> None:
    monkeypatch.setattr(
        metadata.EntryPoint,
        "load",
        lambda self: MockReward,
    )

    descriptor = discovery.load_entry_point(
        reward_entry_point,
        kind=PluginKind.REWARD,
    )

    assert descriptor.metadata is MockReward.metadata
    assert descriptor.config_model is MockConfig
    assert descriptor.factory is MockReward

def test_load_entry_point_preserves_import_error(
    monkeypatch: pytest.MonkeyPatch,
    reward_entry_point: metadata.EntryPoint,
) -> None:
    original_error = ImportError("missing optional dependency")

    def failing_load(self: metadata.EntryPoint) -> object:
        raise original_error

    monkeypatch.setattr(metadata.EntryPoint, "load", failing_load)

    with pytest.raises(PluginLoadError) as exc_info:
        discovery.load_entry_point(
            reward_entry_point,
            kind=PluginKind.REWARD,
        )

    assert "forgerl.reward:mock_reward" in str(exc_info.value)
    assert exc_info.value.__cause__ is original_error

def test_discover_plugins_registers_entry_points(
    monkeypatch: pytest.MonkeyPatch,
    reward_entry_point: metadata.EntryPoint,
) -> None:
    entry_points = metadata.EntryPoints([reward_entry_point])

    monkeypatch.setattr(
        discovery.metadata,
        "entry_points",
        lambda: entry_points,
    )
    monkeypatch.setattr(
        metadata.EntryPoint,
        "load",
        lambda self: MockReward,
    )

    registry = discovery.discover_plugins()

    descriptor = registry.get(PluginKind.REWARD, "mock_reward")
    assert descriptor.factory is MockReward
    assert registry.list_plugins() == [descriptor]