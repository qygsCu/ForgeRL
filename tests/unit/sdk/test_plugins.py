from dataclasses import replace

import pytest
from pydantic import BaseModel, ConfigDict

from forgerl.sdk import (
    SDK_API_VERSION,
    PluginDescriptor,
    PluginKind,
    PluginLoadError,
    PluginMetadata,
)


class MockConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


def make_metadata() -> PluginMetadata:
    return PluginMetadata(
        name="mock-reward",
        kind=PluginKind.REWARD,
        version="0.1.0",
        api_version=SDK_API_VERSION,
        required_capabilities=frozenset(),
        optional_capabilities=frozenset({"batching"}),
        supports_checkpoint=False,
        supports_multiprocessing=True,
        requires_gpu=False,
    )


def test_metadata_accepts_valid_declaration_and_leaves_compatibility_to_registry() -> None:
    metadata = make_metadata()

    assert metadata.validate() is None
    assert replace(metadata, api_version="2.0").api_version == "2.0"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("name", "  "),
        ("kind", "reward"),
        ("required_capabilities", {"batching"}),
        ("optional_capabilities", frozenset({""})),
        ("requires_gpu", 1),
    ],
)
def test_metadata_rejects_invalid_declaration(field_name: str, value: object) -> None:
    with pytest.raises(PluginLoadError, match=field_name):
        replace(make_metadata(), **{field_name: value})


def test_descriptor_does_not_call_factory_during_validation() -> None:
    calls = 0

    def factory() -> object:
        nonlocal calls
        calls += 1
        return object()

    descriptor = PluginDescriptor(make_metadata(), MockConfig, factory)

    assert descriptor.validate() is None
    assert calls == 0
    descriptor.factory()
    assert calls == 1


def test_descriptor_rejects_config_model_that_ignores_unknown_fields() -> None:
    with pytest.raises(PluginLoadError, match="extra='forbid'"):
        PluginDescriptor(make_metadata(), BaseModel, object)


def test_descriptor_rejects_non_callable_factory() -> None:
    descriptor = PluginDescriptor(make_metadata(), MockConfig, object)

    with pytest.raises(PluginLoadError, match="factory"):
        replace(descriptor, factory=None)
