"""Instance-local registry of plugin declarations, without plugin initialization."""

from forgerl.sdk.errors import ConfigurationError, PluginCompatibilityError, PluginLoadError
from forgerl.sdk.plugins import SDK_API_VERSION, PluginDescriptor, PluginKind


class PluginRegistry:
    """Store plugin descriptors by category and name without calling their factories."""

    def __init__(self) -> None:
        """Create an empty registry with storage independent of other instances."""
        self._plugins: dict[tuple[PluginKind, str], PluginDescriptor] = {}

    def register(self, descriptor: PluginDescriptor) -> None:
        """Register a constructed descriptor, rejecting incompatible APIs and duplicate keys.

        Descriptor construction validates its declaration. Registration only checks
        SDK compatibility and uniqueness; failed registration leaves existing entries intact.

        Raises:
            PluginCompatibilityError: The declared SDK API version is unsupported.
            PluginLoadError: A plugin with the same category and name is registered.
        """
        metadata = descriptor.metadata
        plugin_id = f"{metadata.kind.value}:{metadata.name}"
        if metadata.api_version != SDK_API_VERSION:
            raise PluginCompatibilityError(
                f"plugin {plugin_id} declares unsupported SDK API version "
                f"{metadata.api_version!r}; expected {SDK_API_VERSION!r}"
            )
        key = (metadata.kind, metadata.name)
        if key in self._plugins:
            raise PluginLoadError(f"duplicate plugin {plugin_id}")
        self._plugins[key] = descriptor

    def get(self, kind: PluginKind, name: str) -> PluginDescriptor:
        """Return the original descriptor identified by category and name.

        Raises:
            ConfigurationError: No plugin is registered under the requested key.
        """
        key = (kind, name)
        if key not in self._plugins:
            raise ConfigurationError(f"plugin {kind.value}:{name} is not registered")
        return self._plugins[key]

    def list_plugins(self, kind: PluginKind | None = None) -> list[PluginDescriptor]:
        """Return descriptors sorted by category value and name, optionally filtered.

        Passing None includes all categories. The returned list is separate from
        registry storage, while its entries are the original descriptor objects.
        """
        descriptors = [
            descriptor
            for descriptor in self._plugins.values()
            if kind is None or descriptor.metadata.kind == kind
        ]
        return sorted(
            descriptors,
            key=lambda descriptor: (
                descriptor.metadata.kind.value,
                descriptor.metadata.name,
            ),
        )
