"""Inspection-only fixture proving CLI commands never construct a plugin."""

from pydantic import BaseModel, ConfigDict, Field

from forgerl.sdk.plugins import SDK_API_VERSION, PluginKind, PluginMetadata


class FixtureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scale: float = Field(default=1.0, gt=0)


class FixtureReward:
    metadata = PluginMetadata(
        name="cli_fixture_reward",
        kind=PluginKind.REWARD,
        version="0.0.1",
        api_version=SDK_API_VERSION,
        required_capabilities=frozenset(),
        optional_capabilities=frozenset(),
        supports_checkpoint=False,
        supports_multiprocessing=False,
        requires_gpu=False,
    )
    config_model = FixtureConfig

    def __init__(self) -> None:
        raise AssertionError("CLI commands must not construct the fixture plugin")
