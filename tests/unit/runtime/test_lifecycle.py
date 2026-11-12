import logging
from collections.abc import Callable

import pytest
from pydantic import BaseModel, ConfigDict

from forgerl.runtime.lifecycle import managed_plugin
from forgerl.sdk.errors import ConfigurationError, PluginExecutionError, PluginLoadError
from forgerl.sdk.lifecycle import PluginContext
from forgerl.sdk.plugins import (
    SDK_API_VERSION,
    PluginDescriptor,
    PluginKind,
    PluginMetadata,
)


class MockConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OtherConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RecordingPlugin:
    def __init__(
        self,
        events: list[str],
        *,
        initialize_error: BaseException | None = None,
        shutdown_error: BaseException | None = None,
    ) -> None:
        self.events = events
        self.initialize_error = initialize_error
        self.shutdown_error = shutdown_error
        self.received_context: PluginContext | None = None

    def initialize(self, context: PluginContext) -> None:
        self.events.append("initialize")
        self.received_context = context
        if self.initialize_error is not None:
            raise self.initialize_error

    def shutdown(self) -> None:
        self.events.append("shutdown")
        if self.shutdown_error is not None:
            raise self.shutdown_error


def make_descriptor(
    factory: Callable[[], object],
) -> PluginDescriptor:
    return PluginDescriptor(
        metadata=PluginMetadata(
            name="mock_reward",
            kind=PluginKind.REWARD,
            version="0.1.0",
            api_version=SDK_API_VERSION,
            required_capabilities=frozenset(),
            optional_capabilities=frozenset(),
            supports_checkpoint=False,
            supports_multiprocessing=False,
            requires_gpu=False,
        ),
        config_model=MockConfig,
        factory=factory,
    )


def unexpected_factory() -> object:
    raise AssertionError("The plugin factory must not be called")


@pytest.fixture
def events() -> list[str]:
    return []


@pytest.fixture
def context() -> PluginContext:
    return PluginContext(config=MockConfig())


def test_normal_lifecycle_preserves_order_and_closes_once(
    events: list[str],
    context: PluginContext,
) -> None:
    plugin = RecordingPlugin(events)
    descriptor = make_descriptor(lambda: plugin)

    assert events == []

    with managed_plugin(descriptor, context) as managed:
        assert managed is plugin
        assert plugin.received_context is context
        assert events == ["initialize"]
        events.append("body")
        assert events == ["initialize", "body"]

    assert events == ["initialize", "body", "shutdown"]


def test_config_type_mismatch_does_not_call_factory() -> None:
    """Reject the wrong configuration model before calling the factory."""
    context = PluginContext(config=OtherConfig())
    descriptor = make_descriptor(unexpected_factory)

    with pytest.raises(ConfigurationError) as exc_info:
        with managed_plugin(descriptor, context):
            pytest.fail("Invalid configuration must prevent entry into the body")

    message = str(exc_info.value)
    assert "reward:mock_reward" in message
    assert "MockConfig" in message
    assert "OtherConfig" in message


def test_factory_failure_preserves_cause(
    context: PluginContext,
) -> None:
    """Wrap construction failures without losing their original cause."""
    original_error = RuntimeError("construction failed")
    calls: list[str] = []

    def failing_factory() -> object:
        calls.append("factory")
        raise original_error

    descriptor = make_descriptor(failing_factory)

    with pytest.raises(PluginLoadError, match="reward:mock_reward") as exc_info:
        with managed_plugin(descriptor, context):
            pytest.fail("A failed factory must prevent entry into the body")

    assert exc_info.value.__cause__ is original_error
    assert calls == ["factory"]


def test_missing_lifecycle_methods_are_rejected(
    context: PluginContext,
) -> None:
    """Reject an object without lifecycle methods before entering the body."""
    descriptor = make_descriptor(object)

    with pytest.raises(PluginLoadError, match="reward:mock_reward") as exc_info:
        with managed_plugin(descriptor, context):
            pytest.fail("An invalid plugin must prevent entry into the body")

    assert "initialize" in str(exc_info.value)


@pytest.mark.parametrize("method_name", ["initialize", "shutdown"])
def test_non_callable_lifecycle_method_is_rejected_before_initialization(
    method_name: str,
    monkeypatch: pytest.MonkeyPatch,
    events: list[str],
    context: PluginContext,
) -> None:
    """Validate both lifecycle methods before attempting initialization."""
    plugin = RecordingPlugin(events)
    monkeypatch.setattr(plugin, method_name, None)
    descriptor = make_descriptor(lambda: plugin)

    with pytest.raises(PluginLoadError, match="reward:mock_reward") as exc_info:
        with managed_plugin(descriptor, context):
            pytest.fail("A non-callable lifecycle method must prevent entry into the body")

    assert method_name in str(exc_info.value)
    assert plugin.received_context is None
    assert events == []


def test_initialization_failure_still_closes_once(
    events: list[str],
    context: PluginContext,
) -> None:
    """Attempt cleanup for a partially initialized plugin and retain the cause."""
    original_error = RuntimeError("initialization failed")
    plugin = RecordingPlugin(events, initialize_error=original_error)
    descriptor = make_descriptor(lambda: plugin)

    with pytest.raises(PluginExecutionError, match="reward:mock_reward") as exc_info:
        with managed_plugin(descriptor, context):
            pytest.fail("Failed initialization must prevent entry into the body")

    assert exc_info.value.__cause__ is original_error
    assert plugin.received_context is context
    assert events == ["initialize", "shutdown"]


def test_body_failure_propagates_original_exception_and_closes_once(
    events: list[str],
    context: PluginContext,
) -> None:
    """Propagate the original business exception after attempting cleanup."""
    original_error = ValueError("business operation failed")
    plugin = RecordingPlugin(events)
    descriptor = make_descriptor(lambda: plugin)

    with pytest.raises(ValueError) as exc_info:
        with managed_plugin(descriptor, context):
            events.append("body")
            raise original_error

    assert exc_info.value is original_error
    assert events == ["initialize", "body", "shutdown"]


def test_shutdown_failure_preserves_cause(
    events: list[str],
    context: PluginContext,
) -> None:
    """Report a standalone cleanup failure without retrying shutdown."""
    original_error = RuntimeError("shutdown failed")
    plugin = RecordingPlugin(events, shutdown_error=original_error)
    descriptor = make_descriptor(lambda: plugin)

    with pytest.raises(PluginExecutionError, match="reward:mock_reward") as exc_info:
        with managed_plugin(descriptor, context):
            events.append("body")

    assert exc_info.value.__cause__ is original_error
    assert events == ["initialize", "body", "shutdown"]


def test_initialization_and_shutdown_failure_preserves_initialization_error(
    events: list[str],
    context: PluginContext,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep the initialization error primary and log the cleanup traceback."""
    initialize_error = RuntimeError("initialization failed")
    shutdown_error = RuntimeError("shutdown failed")
    plugin = RecordingPlugin(
        events,
        initialize_error=initialize_error,
        shutdown_error=shutdown_error,
    )
    descriptor = make_descriptor(lambda: plugin)

    with caplog.at_level(logging.ERROR, logger="forgerl.runtime.lifecycle"):
        with pytest.raises(PluginExecutionError) as exc_info:
            with managed_plugin(descriptor, context):
                pytest.fail("Failed initialization must prevent entry into the body")

    assert exc_info.value.__cause__ is initialize_error
    assert events == ["initialize", "shutdown"]
    records = [record for record in caplog.records if record.name == "forgerl.runtime.lifecycle"]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.ERROR
    assert "reward:mock_reward" in record.getMessage()
    assert record.exc_info is not None
    assert record.exc_info[1] is shutdown_error
    assert record.exc_info[2] is not None


def test_body_and_shutdown_failure_preserves_body_error(
    events: list[str],
    context: PluginContext,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Keep the business error primary and log the cleanup traceback."""
    body_error = ValueError("business operation failed")
    shutdown_error = RuntimeError("shutdown failed")
    plugin = RecordingPlugin(events, shutdown_error=shutdown_error)
    descriptor = make_descriptor(lambda: plugin)

    with caplog.at_level(logging.ERROR, logger="forgerl.runtime.lifecycle"):
        with pytest.raises(ValueError) as exc_info:
            with managed_plugin(descriptor, context):
                events.append("body")
                raise body_error

    assert exc_info.value is body_error
    assert events == ["initialize", "body", "shutdown"]
    records = [record for record in caplog.records if record.name == "forgerl.runtime.lifecycle"]
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.ERROR
    assert "reward:mock_reward" in record.getMessage()
    assert record.exc_info is not None
    assert record.exc_info[1] is shutdown_error
    assert record.exc_info[2] is not None


@pytest.mark.parametrize("stage", ["initialize", "body"])
def test_keyboard_interrupt_still_closes_and_propagates(
    stage: str,
    events: list[str],
    context: PluginContext,
) -> None:
    """Attempt cleanup on Python interruption and propagate it unchanged."""
    interruption = KeyboardInterrupt("interrupted")
    plugin = RecordingPlugin(
        events,
        initialize_error=interruption if stage == "initialize" else None,
    )
    descriptor = make_descriptor(lambda: plugin)

    with pytest.raises(KeyboardInterrupt) as exc_info:
        with managed_plugin(descriptor, context):
            if stage == "initialize":
                pytest.fail("Interrupted initialization must prevent entry into the body")
            events.append("body")
            raise interruption

    assert exc_info.value is interruption
    if stage == "initialize":
        assert events == ["initialize", "shutdown"]
    else:
        assert events == ["initialize", "body", "shutdown"]
