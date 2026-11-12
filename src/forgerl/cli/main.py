import argparse
import sys
from collections.abc import Sequence
from json import JSONDecodeError, dumps, loads
from pathlib import Path

from pydantic import ValidationError

from forgerl.config.models import PluginSelection
from forgerl.runtime.discovery import discover_plugins
from forgerl.runtime.registry import PluginRegistry
from forgerl.runtime.validation import validate_plugin_options
from forgerl.sdk.errors import ConfigurationError, ForgeRLError
from forgerl.sdk.plugins import PluginKind


def build_parser() -> argparse.ArgumentParser:
    """Define plugin commands without discovering or initializing plugins.

    Parsed arguments include command, plugin_command, and kind. Inspect and
    validate also provide name; validate provides options as a Path. Kind is a
    string (or None for an unfiltered list), converted by the command handler.
    """
    parser = argparse.ArgumentParser(
        prog="forgerl",
        description="ForgeRL plugin discovery and configuration tools",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plugin_parser = commands.add_parser("plugin", help="Inspect installed plugins")
    plugin_commands = plugin_parser.add_subparsers(dest="plugin_command", required=True)
    kinds = [kind.value for kind in PluginKind]

    list_parser = plugin_commands.add_parser("list", help="List available plugins")
    list_parser.add_argument("--kind", choices=kinds, help="Filter by plugin category")

    inspect_parser = plugin_commands.add_parser("inspect", help="Show metadata and config schema")
    inspect_parser.add_argument("name", help="Registered plugin name")
    inspect_parser.add_argument("--kind", choices=kinds, required=True, help="Plugin category")

    validate_parser = plugin_commands.add_parser("validate", help="Validate plugin options")
    validate_parser.add_argument("name", help="Registered plugin name")
    validate_parser.add_argument("--kind", choices=kinds, required=True, help="Plugin category")
    validate_parser.add_argument(
        "--options",
        type=Path,
        required=True,
        metavar="FILE",
        help="JSON file containing plugin options",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, dispatch plugin commands, and report expected failures.

    Successful handlers return exit code 0; expected operation failures return 1.
    Argparse handles help and usage errors with SystemExit codes 0 and 2.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        registry = discover_plugins()

        if args.plugin_command == "list":
            kind = PluginKind(args.kind) if args.kind is not None else None
            _list_plugins(registry, kind)
        elif args.plugin_command == "inspect":
            _inspect_plugin(registry, PluginKind(args.kind), args.name)
        elif args.plugin_command == "validate":
            _validate_plugin(registry, PluginKind(args.kind), args.name, args.options)
        else:
            parser.error(f"unsupported plugin command: {args.plugin_command}")

    except (ForgeRLError, OSError, UnicodeError, JSONDecodeError) as exc:
        print(f"forgerl: {exc}", file=sys.stderr)
        return 1

    return 0


def _list_plugins(registry: PluginRegistry, kind: PluginKind | None) -> None:
    """List registry descriptors, optionally filtered by category."""
    plugins = registry.list_plugins(kind)

    if not plugins:
        print("No plugin found")
        return

    print("KIND\tNAME\tVERSION\tAPI_VERSION")
    for descriptor in plugins:
        metadata = descriptor.metadata
        print(f"{metadata.kind.value}\t{metadata.name}\t{metadata.version}\t{metadata.api_version}")


def _inspect_plugin(registry: PluginRegistry, kind: PluginKind, name: str) -> None:
    """Display plugin metadata and configuration schema as JSON without creating it.

    Missing plugins raise ConfigurationError, handled by main(). Capability sets
    are sorted for stable, JSON-compatible output.
    """
    descriptor = registry.get(kind, name)
    metadata = descriptor.metadata
    result = {
        "metadata": {
            "name": metadata.name,
            "kind": metadata.kind.value,
            "version": metadata.version,
            "api_version": metadata.api_version,
            "required_capabilities": sorted(metadata.required_capabilities),
            "optional_capabilities": sorted(metadata.optional_capabilities),
            "supports_checkpoint": metadata.supports_checkpoint,
            "supports_multiprocessing": metadata.supports_multiprocessing,
            "requires_gpu": metadata.requires_gpu,
        },
        "config_schema": descriptor.config_model.model_json_schema(),
    }
    print(dumps(result, ensure_ascii=False, indent=2))


def _validate_plugin(
    registry: PluginRegistry,
    kind: PluginKind,
    name: str,
    options_path: Path,
) -> None:
    """Read plugin options, validate them, and display the resolved configuration."""
    options = loads(options_path.read_text(encoding="utf-8"))

    if not isinstance(options, dict):
        raise ConfigurationError(f"options for plugin {kind.value}:{name} must be a JSON object")

    try:
        selection = PluginSelection(plugin=name, options=options)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid options for plugin {kind.value}:{name}: {exc}") from exc

    validated = validate_plugin_options(selection, kind=kind, registry=registry)
    print(validated.model_dump_json(indent=2))
