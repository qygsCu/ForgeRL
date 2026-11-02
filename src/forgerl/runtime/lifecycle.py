"""Manage initialization and cleanup for a single plugin instance."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast

from forgerl.sdk.errors import ConfigurationError, PluginExecutionError, PluginLoadError
from forgerl.sdk.lifecycle import PluginContext, PluginLifecycle
from forgerl.sdk.plugins import PluginDescriptor

logger = logging.getLogger(__name__)


@contextmanager
def managed_plugin(
    descriptor: PluginDescriptor,
    context: PluginContext,
) -> Iterator[PluginLifecycle]:
    """Yield an initialized plugin and attempt cleanup once when its scope exits.

    Configuration must already be validated; only its model type is checked here.
    Ordinary initialization and standalone shutdown failures are wrapped with
    PluginExecutionError, retaining their cause. Body exceptions propagate unchanged.
    If cleanup also fails, log that failure and preserve the primary exception.
    """
    plugin_id = f"{descriptor.metadata.kind.value}:{descriptor.metadata.name}"
    if not isinstance(context.config, descriptor.config_model):
        raise ConfigurationError(
            f"plugin {plugin_id} "
            f"expects config {descriptor.config_model.__name__}, "
            f"got {type(context.config).__name__}"
        )

    try:
        instance = descriptor.factory()
    except Exception as exc:
        raise PluginLoadError(f"failed to create plugin {plugin_id}: {exc}") from exc

    for method_name in ("initialize", "shutdown"):
        if not callable(getattr(instance, method_name, None)):
            raise PluginLoadError(f"plugin {plugin_id} must provide callable {method_name}")

    # This informs static checking; the checks above do not validate full signatures.
    plugin = cast(PluginLifecycle, instance)
    try:
        try:
            plugin.initialize(context)
        except Exception as exc:
            raise PluginExecutionError(f"failed to initialize plugin {plugin_id}: {exc}") from exc
        yield plugin
    except BaseException:
        # Also release partially initialized resources and clean up on interruption.
        try:
            plugin.shutdown()
        except BaseException:
            logger.exception("failed to shut down plugin %s while handling an error", plugin_id)
        raise
    else:
        try:
            plugin.shutdown()
        except Exception as exc:
            raise PluginExecutionError(f"failed to shut down plugin {plugin_id}: {exc}") from exc
