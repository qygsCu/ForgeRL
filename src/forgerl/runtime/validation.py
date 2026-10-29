from pydantic import BaseModel

from forgerl.config.models import PluginSelection
from forgerl.runtime.registry import PluginRegistry
from forgerl.sdk.plugins import PluginKind


def validate_plugin_options(
    selection: PluginSelection,
    *,
    kind: PluginKind,
    registry: PluginRegistry,
) -> BaseModel:
    """Resolve a plugin and validate its options without creating an instance."""
    descriptor = registry.get(kind, selection.plugin)
    descriptor.config_model.model_validate(selection.options)
