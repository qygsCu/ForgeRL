from dataclasses import replace

import pytest
from pydantic import BaseModel, ConfigDict

from forgerl.runtime.registry import PluginRegistry
from forgerl.sdk.errors import ConfigurationError, PluginCompatibilityError, PluginLoadError
from forgerl.sdk.plugins import SDK_API_VERSION, PluginDescriptor, PluginKind, PluginMetadata


class MockConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


def unexpected_factory() -> object:
    raise AssertionError("Registry operations must not create plugin instances")


def make_descriptor(
    name: str = "mock",
    kind: PluginKind = PluginKind.REWARD,
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
        config_model=MockConfig,
        factory=unexpected_factory,
    )


def test_register_and_get_preserve_descriptor_without_creating_plugin() -> None:
    registry = PluginRegistry()
    descriptor = make_descriptor()

    registry.register(descriptor)

    assert registry.get(PluginKind.REWARD, "mock") is descriptor
    assert registry.list_plugins() == [descriptor]


def test_duplicate_registration_preserves_original_descriptor() -> None:
    registry = PluginRegistry()
    original = make_descriptor()
    replacement = replace(original, metadata=replace(original.metadata, version="0.2.0"))
    registry.register(original)

    with pytest.raises(PluginLoadError, match="duplicate plugin reward:mock"):
        registry.register(replacement)

    assert registry.get(PluginKind.REWARD, "mock") is original
    assert registry.list_plugins() == [original]


def test_same_name_can_be_registered_in_different_categories() -> None:
    registry = PluginRegistry()
    reward = make_descriptor(kind=PluginKind.REWARD)
    rollout = make_descriptor(kind=PluginKind.ROLLOUT)

    registry.register(reward)
    registry.register(rollout)

    assert registry.get(PluginKind.REWARD, "mock") is reward
    assert registry.get(PluginKind.ROLLOUT, "mock") is rollout


def test_incompatible_api_is_rejected_without_changing_registry() -> None:
    registry = PluginRegistry()
    original = make_descriptor()
    registry.register(original)

    with pytest.raises(PluginCompatibilityError) as exc_info:
        registry.register(make_descriptor(name="future", api_version="2.0"))

    message = str(exc_info.value)
    assert "reward:future" in message
    assert "'2.0'" in message
    assert repr(SDK_API_VERSION) in message
    assert registry.list_plugins() == [original]


def test_get_does_not_fall_back_to_another_category() -> None:
    registry = PluginRegistry()
    registry.register(make_descriptor(kind=PluginKind.REWARD))

    with pytest.raises(ConfigurationError, match="rollout:mock"):
        registry.get(PluginKind.ROLLOUT, "mock")


def test_list_plugins_filters_sorts_and_returns_a_separate_list() -> None:
    registry = PluginRegistry()
    beta = make_descriptor(name="beta")
    alpha = make_descriptor(name="alpha")
    algorithm = make_descriptor(name="zeta", kind=PluginKind.ALGORITHM)
    for descriptor in (beta, algorithm, alpha):
        registry.register(descriptor)

    assert registry.list_plugins() == [algorithm, alpha, beta]
    assert registry.list_plugins(PluginKind.REWARD) == [alpha, beta]
    assert registry.list_plugins(PluginKind.TRAINER) == []

    result = registry.list_plugins()
    result.clear()

    assert registry.list_plugins() == [algorithm, alpha, beta]


def test_registries_do_not_share_entries() -> None:
    first = PluginRegistry()
    second = PluginRegistry()

    first.register(make_descriptor())

    assert second.list_plugins() == []
    with pytest.raises(ConfigurationError, match="reward:mock"):
        second.get(PluginKind.REWARD, "mock")


def test_registry_starts_empty() -> None:
    registry = PluginRegistry()

    assert registry.list_plugins() == []
    assert registry.list_plugins(PluginKind.REWARD) == []
