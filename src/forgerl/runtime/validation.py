"""Resolve registered plugins and validate their options before plugin creation."""

from pydantic import BaseModel, ValidationError

from forgerl.config.models import ForgeRLConfig, PluginSelection
from forgerl.runtime.registry import PluginRegistry
from forgerl.sdk.errors import ConfigurationError
from forgerl.sdk.plugins import PluginKind, ValidatedPluginConfigs


def validate_plugin_options(
    selection: PluginSelection,
    *,
    kind: PluginKind,
    registry: PluginRegistry,
) -> BaseModel:
    """Validate one selection with the registered plugin's configuration model.

    Args:
        selection: Plugin name and JSON-compatible options supplied by the caller.
        kind: Category used with the name to look up the plugin descriptor.
        registry: Registry containing the available plugin descriptors.

    Returns:
        An instance of the plugin's configuration model, including its defaults.
        The concrete type is determined by the registered descriptor.

    Raises:
        ConfigurationError: The plugin is not registered or its options are invalid.
            Invalid options retain the original ValidationError as their cause.

    This function does not call the plugin factory or check runtime capabilities.
    """
    descriptor = registry.get(kind=kind, name=selection.plugin)
    try:
        return descriptor.config_model.model_validate(selection.options)
    except ValidationError as exc:
        raise ConfigurationError(
            f"invalid options for plugin {kind.value}:{selection.plugin}: {exc}"
        ) from exc


def validate_run_plugins(
    config: ForgeRLConfig,
    *,
    registry: PluginRegistry,
) -> ValidatedPluginConfigs:
    """Validate the five plugin selections in an already parsed run configuration.

    Args:
        config: Structurally validated ForgeRL configuration containing the selections.
        registry: Registry containing every selected plugin, including the scheduler.

    Returns:
        Configuration model instances for the algorithm, rollout, reward, trainer,
        and scheduler plugins, ready for later plugin initialization.

    Raises:
        ConfigurationError: The first missing plugin or invalid options encountered,
            in algorithm, rollout, reward, trainer, then scheduler order.

    Only each selection's options are passed to its plugin configuration model.
    Scheduler queue and staleness settings remain in the original run configuration.
    No plugin factory is called, and no partial result is returned on failure.
    """
    selections: dict[PluginKind, PluginSelection] = {
        PluginKind.ALGORITHM: config.algorithm,
        PluginKind.ROLLOUT: config.rollout,
        PluginKind.REWARD: config.reward,
        PluginKind.TRAINER: config.trainer,
        PluginKind.SCHEDULER: config.scheduler,
    }
    validated = {
        kind: validate_plugin_options(
            selection,
            kind=kind,
            registry=registry,
        )
        for kind, selection in selections.items()
    }
    return ValidatedPluginConfigs(
        algorithm=validated[PluginKind.ALGORITHM],
        rollout=validated[PluginKind.ROLLOUT],
        reward=validated[PluginKind.REWARD],
        trainer=validated[PluginKind.TRAINER],
        scheduler=validated[PluginKind.SCHEDULER],
    )
