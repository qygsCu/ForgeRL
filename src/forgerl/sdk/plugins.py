"""Plugin declarations shared by plugins and the runtime registry."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from forgerl.sdk.errors import PluginLoadError

SDK_API_VERSION = "1.0"


class PluginKind(StrEnum):
    """Plugin categories, matching their entry point group suffixes."""

    ALGORITHM = "algorithm"
    REWARD = "reward"
    ROLLOUT = "rollout"
    TRAINER = "trainer"
    SCHEDULER = "scheduler"
    CALLBACK = "callback"


@dataclass(frozen=True)
class PluginMetadata:
    """Immutable plugin identity and capabilities, without runtime resources."""

    name: str
    kind: PluginKind
    version: str
    api_version: str
    required_capabilities: frozenset[str]
    optional_capabilities: frozenset[str]
    supports_checkpoint: bool
    supports_multiprocessing: bool
    requires_gpu: bool

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Reject malformed declarations; API compatibility is checked by the registry."""
        for field_name in ("name", "version", "api_version"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise PluginLoadError(f"{field_name} must be a non-empty string")

        if not isinstance(self.kind, PluginKind):
            raise PluginLoadError("kind must be a PluginKind")

        for field_name in ("required_capabilities", "optional_capabilities"):
            capabilities = getattr(self, field_name)
            if not isinstance(capabilities, frozenset):
                raise PluginLoadError(f"{field_name} must be a frozenset of non-empty strings")
            if any(not isinstance(item, str) or not item.strip() for item in capabilities):
                raise PluginLoadError(f"{field_name} must contain only non-empty strings")

        for field_name in ("supports_checkpoint", "supports_multiprocessing", "requires_gpu"):
            if type(getattr(self, field_name)) is not bool:
                raise PluginLoadError(f"{field_name} must be bool")


@dataclass(frozen=True)
class PluginDescriptor:
    """In-process registration data; constructing it never calls the plugin factory.

    The factory constructs an instance without arguments. Validated configuration
    is passed later by the lifecycle manager during plugin initialization.
    """

    metadata: PluginMetadata
    config_model: type[BaseModel]
    factory: Callable[[], object]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate the declaration without constructing a model or plugin instance."""
        if not isinstance(self.metadata, PluginMetadata):
            raise PluginLoadError("metadata must be a PluginMetadata")
        self.metadata.validate()

        if not isinstance(self.config_model, type) or not issubclass(self.config_model, BaseModel):
            raise PluginLoadError("config_model must be a Pydantic BaseModel subclass")
        if self.config_model.model_config.get("extra") != "forbid":
            raise PluginLoadError("config_model must reject unknown fields with extra='forbid'")
        if not callable(self.factory):
            raise PluginLoadError("factory must be callable")
