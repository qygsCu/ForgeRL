"""Load, score, and check rewards using a registry owned by the calling runtime."""

import logging
from collections.abc import Sequence
from json import JSONDecodeError, loads
from pathlib import Path
from typing import cast

from pydantic import ValidationError

from forgerl.config.models import PluginSelection
from forgerl.runtime.lifecycle import managed_plugin
from forgerl.runtime.registry import PluginRegistry
from forgerl.runtime.validation import validate_plugin_options
from forgerl.sdk.errors import (
    ConfigurationError,
    ForgeRLError,
    PluginCompatibilityError,
    PluginExecutionError,
    PluginLoadError,
    ProtocolViolationError,
)
from forgerl.sdk.lifecycle import PluginContext, PluginLifecycle
from forgerl.sdk.plugins import PluginKind
from forgerl.sdk.rewards import RewardPlugin
from forgerl.sdk.schemas import JsonValue, RewardOutput, RewardSample, validate_reward_samples

logger = logging.getLogger(__name__)


def load_reward_samples(samples_path: Path) -> list[RewardSample]:
    """Load JSON records and reject invalid batches before resource initialization."""
    try:
        source = samples_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ConfigurationError(f"cannot read samples file {samples_path}: {exc}") from exc

    def unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
        record: dict[str, object] = {}
        for key, value in pairs:
            if key in record:
                raise ProtocolViolationError(
                    f"samples file {samples_path}: duplicate field {key!r}"
                )
            record[key] = value
        return record

    try:
        raw_samples = loads(source, object_pairs_hook=unique_fields)
    except JSONDecodeError as exc:
        raise ProtocolViolationError(
            f"samples file {samples_path}: invalid JSON at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}"
        ) from exc
    if not isinstance(raw_samples, list):
        raise ProtocolViolationError(f"samples file {samples_path}: $ must be a JSON array")

    required = {"request_id", "response", "reference_answer"}
    samples: list[RewardSample] = []
    for index, item in enumerate(raw_samples):
        location = f"samples file {samples_path}: $[{index}]"
        if not isinstance(item, dict):
            raise ProtocolViolationError(f"{location} must be a JSON object")
        missing, unknown = required - item.keys(), item.keys() - required
        if missing or unknown:
            raise ProtocolViolationError(
                f"{location}: missing fields={sorted(missing)}, unknown fields={sorted(unknown)}"
            )
        try:
            samples.append(RewardSample(**item))
        except ProtocolViolationError as exc:
            raise ProtocolViolationError(f"{location}: {exc}") from exc
    try:
        validate_reward_samples(samples)
    except ProtocolViolationError as exc:
        raise ProtocolViolationError(f"samples file {samples_path}: {exc}") from exc
    return samples


def compute_rewards(
    plugin: PluginLifecycle,
    samples: Sequence[RewardSample],
    *,
    plugin_name: str,
) -> RewardOutput:
    """Call an already initialized plugin; its caller owns the lifecycle scope.

    Validate inputs before compute and the entire output before returning or
    reporting isolated failures. Input records are passed in a tuple to prevent
    ordinary container mutation. No training filtering or deep copy occurs.
    """
    validate_reward_samples(samples)
    inputs = tuple(samples)  # use tuple to avoid rewriting
    expected_ids = {sample.request_id for sample in inputs}
    if not callable(getattr(plugin, "compute", None)):
        raise PluginLoadError(f"plugin reward:{plugin_name} must provide callable compute")

    scorer = cast(RewardPlugin, plugin)
    try:
        output = scorer.compute(inputs)
    except (ForgeRLError, MemoryError):
        raise
    except Exception as exc:
        raise PluginExecutionError(f"failed to compute plugin reward:{plugin_name}: {exc}") from exc

    # Keep protocol failures outside execution-error wrapping. Constructors may
    # already validate, but the plugin can still return a wrong or mutated object.
    if not isinstance(output, RewardOutput):
        raise ProtocolViolationError(f"plugin reward:{plugin_name} must return RewardOutput")
    try:
        output.validate()
    except ProtocolViolationError as exc:
        raise ProtocolViolationError(
            f"plugin reward:{plugin_name} returned invalid output: {exc}"
        ) from exc
    actual_ids = set(output.results)
    if actual_ids != expected_ids:
        raise ProtocolViolationError(
            f"plugin reward:{plugin_name} returned mismatched request IDs: "
            f"missing={sorted(expected_ids - actual_ids)}, "
            f"unexpected={sorted(actual_ids - expected_ids)}"
        )

    for request_id, result in output.results.items():
        if not result.valid:
            logger.warning(
                "plugin reward:%s request_id=%s stage=compute: %s",
                plugin_name,
                request_id,
                result.error_message,
            )
    return output


def run_reward_test(
    name: str, samples_path: Path, registry: PluginRegistry, options_path: Path | None = None
) -> dict[str, JsonValue]:
    """Validate, score, and close before returning a sample-call protocol report.

    The caller discovers plugins once and supplies its registry; this function
    neither discovers plugins nor owns a global registry. Legal scoring failures
    are reported, including an all-invalid batch. Business accuracy and untested
    capabilities are not certified by a passed report.
    """
    descriptor = registry.get(PluginKind.REWARD, name)
    metadata = descriptor.metadata
    if metadata.requires_gpu or metadata.supports_checkpoint or metadata.required_capabilities:
        raise PluginCompatibilityError(
            f"plugin reward:{name}: this check supports CPU rewards without Checkpoint "
            "or required runtime capabilities"
        )
    try:
        options = {} if options_path is None else loads(options_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, JSONDecodeError) as exc:
        raise ConfigurationError(f"cannot read options file {options_path}: {exc}") from exc
    if not isinstance(options, dict):
        raise ConfigurationError(f"options for plugin reward:{name} must be a JSON object")
    try:
        selection = PluginSelection(plugin=name, options=options)
    except ValidationError as exc:
        raise ConfigurationError(f"invalid options for plugin reward:{name}: {exc}") from exc
    config = validate_plugin_options(selection, kind=PluginKind.REWARD, registry=registry)
    samples = load_reward_samples(samples_path)
    resolved_config = loads(config.model_dump_json())

    with managed_plugin(descriptor, PluginContext(config)) as plugin:
        output = compute_rewards(plugin, samples, plugin_name=name)
        failures: dict[str, JsonValue] = {
            request_id: result.error_message
            for request_id, result in output.results.items()
            if not result.valid
        }
        report: dict[str, JsonValue] = {
            "status": "passed",
            "plugin": {
                "kind": metadata.kind.value,
                "name": metadata.name,
                "version": metadata.version,
                "api_version": metadata.api_version,
            },
            "config": resolved_config,
            "sample_count": len(samples),
            "valid_count": len(samples) - len(failures),
            "invalid_count": len(failures),
            "failures": failures,
        }
    # A shutdown error must prevent successful report delivery.
    return report
