"""Discover installed plugins without creating plugin instances."""

from importlib import metadata

from forgerl.runtime.registry import PluginRegistry
from forgerl.sdk.errors import PluginLoadError
from forgerl.sdk.plugins import PluginDescriptor, PluginKind


def load_entry_point(
    entry_point: metadata.EntryPoint,
    *,
    kind: PluginKind,
) -> PluginDescriptor:
    """Load a plugin class and validate its declaration without instantiating it.

    The entry point group and name must match the declared category and name.
    Import and declaration failures raise PluginLoadError with entry point context.
    SDK API compatibility is checked later when the descriptor is registered.
    """
    plugin_id = f"{entry_point.group}:{entry_point.name}"
    expected_group = f"forgerl.{kind.value}"
    if entry_point.group != expected_group:
        raise PluginLoadError(
            f"plugin {plugin_id} uses group {entry_point.group!r}; "
            f"expected {expected_group!r}"
        )

    try:
        plugin_class: object = entry_point.load()
    except Exception as exc:
        raise PluginLoadError(f"failed to load plugin {plugin_id}: {exc}") from exc

    if not isinstance(plugin_class, type):
        raise PluginLoadError(f"plugin {plugin_id} entry point must reference a class")

    try:
        descriptor = PluginDescriptor(
            metadata=getattr(plugin_class, "metadata", None),
            config_model=getattr(plugin_class, "config_model", None),
            factory=plugin_class,
        )
    except (AttributeError, PluginLoadError) as exc:
        raise PluginLoadError(f"invalid declaration for plugin {plugin_id}: {exc}") from exc

    if descriptor.metadata.kind != kind:
        raise PluginLoadError(
            f"plugin {plugin_id} declares kind {descriptor.metadata.kind.value!r}; "
            f"expected {kind.value!r}"
        )
    if descriptor.metadata.name != entry_point.name:
        raise PluginLoadError(
            f"plugin {plugin_id} declares name {descriptor.metadata.name!r}; "
            f"expected {entry_point.name!r}"
        )

    return descriptor


def discover_plugins() -> PluginRegistry:
    """Register installed ForgeRL declarations in a new, independent registry.

    Only SDK-defined entry point groups are inspected. The first invalid entry,
    duplicate name, or incompatible SDK API aborts discovery; no partial registry
    is returned. Plugin factories are retained without being called.
    """
    registry = PluginRegistry()
    entry_points = metadata.entry_points()

    for kind in PluginKind:
        group = f"forgerl.{kind.value}"
        entries = sorted(
            entry_points.select(group=group),
            key=lambda entry_point: (entry_point.name, entry_point.value),
        )
        for entry_point in entries:
            descriptor = load_entry_point(entry_point, kind=kind)
            registry.register(descriptor)

    return registry
