"""Score acceptance, diagnostic delivery, and lifecycle failure integration."""

import logging
from collections.abc import Callable, Sequence
from dataclasses import FrozenInstanceError
from typing import cast

import pytest
from pydantic import BaseModel, ConfigDict

from forgerl.runtime.lifecycle import managed_plugin
from forgerl.runtime.rewards import compute_rewards
from forgerl.sdk import (
    SDK_API_VERSION,
    PluginContext,
    PluginDescriptor,
    PluginExecutionError,
    PluginKind,
    PluginLoadError,
    PluginMetadata,
    ProtocolViolationError,
    RewardOutput,
    RewardResult,
    RewardSample,
)


class RecoverableScoringError(Exception):
    pass


class RecordingReward:
    """Test scorer: strict matching plus an explicitly injected local failure."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.fail_ids: set[str] = set()
        self.compute_error: BaseException | None = None
        self.shutdown_error: BaseException | None = None
        self.output: RewardOutput | None = None
        self.received: Sequence[RewardSample] | None = None

    def initialize(self, context: PluginContext) -> None:
        self.events.append("initialize")

    def shutdown(self) -> None:
        self.events.append("shutdown")
        if self.shutdown_error is not None:
            raise self.shutdown_error

    def compute(self, samples: Sequence[RewardSample]) -> RewardOutput:
        self.received = samples
        if self.compute_error is not None:
            raise self.compute_error
        if self.output is not None:
            return self.output
        results: dict[str, RewardResult] = {}
        for sample in samples:
            self.events.append(sample.request_id)
            try:
                if sample.request_id in self.fail_ids:
                    raise RecoverableScoringError("scorer unavailable")
                results[sample.request_id] = RewardResult(
                    float(sample.response == sample.reference_answer), True
                )
            except RecoverableScoringError as exc:
                results[sample.request_id] = RewardResult(None, False, str(exc))
        return RewardOutput(results)


class RewardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


def descriptor(factory: Callable[[], object]) -> PluginDescriptor:
    return PluginDescriptor(
        metadata=PluginMetadata(
            name="test_reward",
            kind=PluginKind.REWARD,
            version="0.1.0",
            api_version=SDK_API_VERSION,
            required_capabilities=frozenset(),
            optional_capabilities=frozenset(),
            supports_checkpoint=False,
            supports_multiprocessing=False,
            requires_gpu=False,
        ),
        config_model=RewardConfig,
        factory=factory,
    )


@pytest.fixture
def samples() -> list[RewardSample]:
    return [
        RewardSample("r1", "2", "2"),
        RewardSample("r2", " 2", "2"),
        RewardSample("r3", "", "2"),
    ]


def test_strict_matching_preserves_inputs_and_cleans_up(samples: list[RewardSample]) -> None:
    plugin = RecordingReward()
    original = tuple(samples)
    with managed_plugin(descriptor(lambda: plugin), PluginContext(RewardConfig())) as managed:
        output = compute_rewards(managed, samples, plugin_name="test_reward")
    assert [item.reward for item in output.results.values()] == [1.0, 0.0, 0.0]
    assert tuple(samples) == original
    assert isinstance(plugin.received, tuple)
    assert plugin.events == ["initialize", "r1", "r2", "r3", "shutdown"]


@pytest.mark.parametrize("failed_ids", [{"r2"}, {"r1", "r2", "r3"}])
def test_local_failures_return_reasons_and_continue_later_samples(
    samples: list[RewardSample], failed_ids: set[str], caplog: pytest.LogCaptureFixture
) -> None:
    plugin = RecordingReward()
    plugin.fail_ids = failed_ids
    with caplog.at_level(logging.WARNING, logger="forgerl.runtime.rewards"):
        output = compute_rewards(plugin, samples, plugin_name="test_reward")
    assert plugin.events == ["r1", "r2", "r3"]
    assert set(output.results) == {"r1", "r2", "r3"}
    for request_id in failed_ids:
        assert output.results[request_id] == RewardResult(None, False, "scorer unavailable")
        assert f"request_id={request_id} stage=compute: scorer unavailable" in caplog.text
    assert "reward:test_reward" in caplog.text
    assert len(caplog.records) == len(failed_ids)


def test_duplicate_input_prevents_plugin_call(samples: list[RewardSample]) -> None:
    plugin = RecordingReward()
    with pytest.raises(ProtocolViolationError, match="duplicate"):
        compute_rewards(plugin, [samples[0], samples[0]], plugin_name="test_reward")
    assert plugin.received is None


@pytest.mark.parametrize("keys", [{"r1", "r2"}, {"r1", "r2", "foreign"}])
def test_missing_or_foreign_ids_reject_entire_output_before_logging(
    samples: list[RewardSample], keys: set[str], caplog: pytest.LogCaptureFixture
) -> None:
    plugin = RecordingReward()
    plugin.output = RewardOutput(
        {key: RewardResult(None, False, "scorer unavailable") for key in keys}
    )
    with pytest.raises(ProtocolViolationError, match="mismatched request IDs"):
        compute_rewards(plugin, samples, plugin_name="test_reward")
    assert not caplog.records


def test_return_order_does_not_affect_acceptance(samples: list[RewardSample]) -> None:
    plugin = RecordingReward()
    plugin.output = RewardOutput(
        {sample.request_id: RewardResult(1.0, True) for sample in reversed(samples)}
    )
    assert compute_rewards(plugin, samples, plugin_name="test_reward") is plugin.output


def test_wrong_return_type_is_protocol_error(samples: list[RewardSample]) -> None:
    plugin = RecordingReward()
    plugin.output = cast(RewardOutput, {"r1": 1.0})
    with pytest.raises(ProtocolViolationError, match="must return RewardOutput"):
        compute_rewards(plugin, samples, plugin_name="test_reward")


def test_boundary_rejects_corrupted_output_and_still_cleans_up(
    samples: list[RewardSample],
) -> None:
    plugin = RecordingReward()
    plugin.output = RewardOutput({sample.request_id: RewardResult(1.0, True) for sample in samples})
    object.__setattr__(plugin.output.results["r2"], "reward", float("nan"))
    with pytest.raises(ProtocolViolationError, match="r2.*finite float"):
        with managed_plugin(descriptor(lambda: plugin), PluginContext(RewardConfig())) as managed:
            compute_rewards(managed, samples, plugin_name="test_reward")
    assert plugin.events == ["initialize", "shutdown"]


@pytest.mark.parametrize("shutdown_fails", [False, True])
def test_unhandled_error_is_wrapped_with_cause_and_cleanup_preserves_it(
    samples: list[RewardSample], shutdown_fails: bool, caplog: pytest.LogCaptureFixture
) -> None:
    plugin = RecordingReward()
    original = ValueError("scorer bug")
    plugin.compute_error = original
    if shutdown_fails:
        plugin.shutdown_error = RuntimeError("cleanup failed")
    with pytest.raises(PluginExecutionError, match="reward:test_reward") as caught:
        with managed_plugin(descriptor(lambda: plugin), PluginContext(RewardConfig())) as managed:
            compute_rewards(managed, samples, plugin_name="test_reward")
    assert caught.value.__cause__ is original
    assert plugin.events == ["initialize", "shutdown"]
    assert ("failed to shut down" in caplog.text) == shutdown_fails


@pytest.mark.parametrize(
    "error",
    [ProtocolViolationError("bad contract"), MemoryError("out of memory"), KeyboardInterrupt()],
)
def test_classified_errors_resource_failure_and_interruption_are_preserved(
    samples: list[RewardSample], error: BaseException
) -> None:
    plugin = RecordingReward()
    plugin.compute_error = error
    with pytest.raises(type(error)) as caught:
        with managed_plugin(descriptor(lambda: plugin), PluginContext(RewardConfig())) as managed:
            compute_rewards(managed, samples, plugin_name="test_reward")
    assert caught.value is error
    assert plugin.events[-1] == "shutdown"


def test_ordinary_record_mutation_is_not_a_local_scoring_failure(
    samples: list[RewardSample],
) -> None:
    class MutatingReward(RecordingReward):
        def compute(self, samples: Sequence[RewardSample]) -> RewardOutput:
            samples[0].response = "changed"  # type: ignore[misc]
            return super().compute(samples)

    with pytest.raises(PluginExecutionError) as caught:
        compute_rewards(MutatingReward(), samples, plugin_name="test_reward")
    assert isinstance(caught.value.__cause__, FrozenInstanceError)
    assert samples[0].response == "2"


def test_missing_compute_is_reported_as_load_failure(samples: list[RewardSample]) -> None:
    class LifecycleOnly:
        def initialize(self, context: PluginContext) -> None:
            pass

        def shutdown(self) -> None:
            pass

    with pytest.raises(PluginLoadError, match="callable compute"):
        compute_rewards(LifecycleOnly(), samples, plugin_name="test_reward")
