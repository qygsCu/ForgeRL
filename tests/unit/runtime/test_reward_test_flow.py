import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, ConfigDict, Field

import forgerl.cli.main as cli
import forgerl.runtime.rewards as flow
from forgerl.runtime.registry import PluginRegistry
from forgerl.sdk import (
    SDK_API_VERSION,
    ConfigurationError,
    PluginCompatibilityError,
    PluginContext,
    PluginDescriptor,
    PluginExecutionError,
    PluginKind,
    PluginMetadata,
    ProtocolViolationError,
    RewardOutput,
    RewardResult,
    RewardSample,
)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scale: float = Field(default=1.0, gt=0)


class Scorer:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.fail_ids: set[str] = set()
        self.errors: dict[str, Exception] = {}
        self.output: RewardOutput | None = None

    def initialize(self, context: PluginContext) -> None:
        self.events.append("initialize")
        if "initialize" in self.errors:
            raise self.errors["initialize"]

    def compute(self, samples: Sequence[RewardSample]) -> RewardOutput:
        self.events.append("compute")
        if "compute" in self.errors:
            raise self.errors["compute"]
        return (
            self.output
            if self.output is not None
            else RewardOutput(
                {
                    sample.request_id: RewardResult(None, False, "cannot score")
                    if sample.request_id in self.fail_ids
                    else RewardResult(0.0, True)
                    for sample in samples
                }
            )
        )

    def shutdown(self) -> None:
        self.events.append("shutdown")
        if "shutdown" in self.errors:
            raise self.errors["shutdown"]


@pytest.fixture
def setup_flow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    scorer = Scorer()
    factory = Mock(return_value=scorer)
    descriptor = PluginDescriptor(
        PluginMetadata(
            "demo",
            PluginKind.REWARD,
            "0.1.0",
            SDK_API_VERSION,
            frozenset(),
            frozenset(),
            False,
            False,
            False,
        ),
        Config,
        factory,
    )
    registry = PluginRegistry()
    registry.register(descriptor)
    monkeypatch.setattr(cli, "discover_plugins", Mock(return_value=registry))
    samples = tmp_path / "samples.json"
    samples.write_text(
        json.dumps(
            [
                {"request_id": name, "response": "wrong", "reference_answer": "2"}
                for name in ("r1", "r2")
            ]
        ),
        encoding="utf-8",
    )
    return scorer, factory, descriptor, samples, registry


@pytest.mark.parametrize("failed", [set(), {"r1"}, {"r1", "r2"}])
def test_python_entry_returns_report_after_cleanup(setup_flow, failed, capsys) -> None:
    scorer, factory, _, samples, registry = setup_flow
    scorer.fail_ids = failed
    report = flow.run_reward_test("demo", samples, registry)
    assert scorer.events == ["initialize", "compute", "shutdown"]
    factory.assert_called_once_with()
    assert report["status"] == "passed"
    assert report["sample_count"] == 2
    assert report["valid_count"] == 2 - len(failed)
    assert report["invalid_count"] == len(failed)
    assert report["failures"] == {key: "cannot score" for key in failed}
    assert report["config"] == {"scale": 1.0}
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("bad", ["config", "samples", "missing_config", "missing_samples"])
def test_bad_files_do_not_create_plugin(setup_flow, tmp_path, bad) -> None:
    scorer, factory, _, samples, registry = setup_flow
    options = tmp_path / "options.json"
    options.write_text('{"scale": 0}', encoding="utf-8")
    if bad == "samples":
        samples.write_text("[]", encoding="utf-8")
    if bad == "missing_samples":
        samples = tmp_path / "missing_samples.json"
    selected_options = options if bad == "config" else None
    if bad == "missing_config":
        selected_options = tmp_path / "missing_options.json"
    with pytest.raises((ConfigurationError, ProtocolViolationError)):
        flow.run_reward_test("demo", samples, registry, selected_options)
    factory.assert_not_called()
    assert scorer.events == []


def test_config_override_is_resolved_without_printing(setup_flow, tmp_path) -> None:
    _, _, _, samples, registry = setup_flow
    options = tmp_path / "options.json"
    options.write_text('{"scale": 2.5}', encoding="utf-8")
    assert flow.run_reward_test("demo", samples, registry, options)["config"] == {"scale": 2.5}


def test_shared_registry_is_reused_without_rediscovery(setup_flow, monkeypatch) -> None:
    scorer, factory, descriptor, samples, registry = setup_flow
    discover = Mock(side_effect=AssertionError("callee must use the supplied registry"))
    monkeypatch.setattr("forgerl.runtime.discovery.discover_plugins", discover)
    for _ in range(2):
        assert flow.run_reward_test("demo", samples, registry)["status"] == "passed"
    assert factory.call_count == 2
    assert registry.get(PluginKind.REWARD, "demo") is descriptor
    assert scorer.events == ["initialize", "compute", "shutdown"] * 2
    discover.assert_not_called()


def test_cli_reports_legal_scoring_failure_as_protocol_success(setup_flow, capsys) -> None:
    scorer, _, _, samples, registry = setup_flow
    scorer.fail_ids = {"r1", "r2"}
    assert cli.main(["plugin", "test", "demo", "--kind", "reward", "--samples", str(samples)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["invalid_count"] == 2
    assert report["valid_count"] == 0
    assert report["status"] == "passed"


def test_missing_compute_still_closes_plugin(setup_flow, monkeypatch) -> None:
    scorer, _, _, samples, registry = setup_flow
    monkeypatch.setattr(scorer, "compute", None)
    from forgerl.sdk import PluginLoadError

    with pytest.raises(PluginLoadError, match="callable compute"):
        flow.run_reward_test("demo", samples, registry)
    assert scorer.events == ["initialize", "shutdown"]


@pytest.mark.parametrize("phase", ["initialize", "compute", "shutdown"])
def test_cli_failure_has_no_success_report_and_cleanup_is_attempted(
    setup_flow, capsys, phase
) -> None:
    scorer, _, _, samples, registry = setup_flow
    original = RuntimeError(f"{phase} failed")
    scorer.errors[phase] = original
    assert cli.main(["plugin", "test", "demo", "--kind", "reward", "--samples", str(samples)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"{phase} failed" in captured.err
    assert scorer.events[-1] == "shutdown"


def test_execution_and_cleanup_failures_preserve_primary_cause(setup_flow) -> None:
    scorer, _, _, samples, registry = setup_flow
    original = ValueError("primary")
    scorer.errors = {"compute": original, "shutdown": RuntimeError("secondary")}
    with pytest.raises(PluginExecutionError) as caught:
        flow.run_reward_test("demo", samples, registry)
    assert caught.value.__cause__ is original
    assert scorer.events[-1] == "shutdown"


@pytest.mark.parametrize("bad_output", ["missing_id", "nan"])
def test_invalid_output_aborts_and_cleans_up(setup_flow, bad_output) -> None:
    scorer, _, _, samples, registry = setup_flow
    scorer.output = RewardOutput({"r1": RewardResult(1.0, True)})
    if bad_output == "nan":
        object.__setattr__(scorer.output.results["r1"], "reward", float("nan"))
    with pytest.raises(ProtocolViolationError):
        flow.run_reward_test("demo", samples, registry)
    assert scorer.events[-1] == "shutdown"


@pytest.mark.parametrize("flag", ["requires_gpu", "supports_checkpoint", "required_capabilities"])
def test_unsupported_scope_fails_before_factory(setup_flow, monkeypatch, flag) -> None:
    _, factory, descriptor, samples, registry = setup_flow
    value = frozenset({"extra"}) if flag == "required_capabilities" else True
    registry = PluginRegistry()
    registry.register(replace(descriptor, metadata=replace(descriptor.metadata, **{flag: value})))
    with pytest.raises(PluginCompatibilityError):
        flow.run_reward_test("demo", samples, registry)
    factory.assert_not_called()


def test_cli_only_delegates_and_redirects_plugin_prints(tmp_path, monkeypatch, capsys) -> None:
    samples = tmp_path / "samples.json"
    options = tmp_path / "options.json"
    registry = PluginRegistry()

    def run(name, samples_path, supplied_registry, options_path):
        assert (name, samples_path, options_path) == ("demo", samples, options)
        assert supplied_registry is registry
        print("plugin diagnostic")
        return {"status": "passed"}

    monkeypatch.setattr(cli, "run_reward_test", run)
    discover = Mock(return_value=registry)
    monkeypatch.setattr(cli, "discover_plugins", discover)
    assert (
        cli.main(
            [
                "plugin",
                "test",
                "demo",
                "--kind",
                "reward",
                "--samples",
                str(samples),
                "--options",
                str(options),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "passed"}
    assert captured.err.strip() == "plugin diagnostic"
    discover.assert_called_once_with()


@pytest.mark.parametrize(
    "args,code",
    [
        (["plugin", "test", "--help"], 0),
        (["plugin", "test", "demo", "--kind", "reward"], 2),
        (["plugin", "test", "demo", "--samples", "samples.json"], 2),
        (["plugin", "test", "demo", "--kind", "trainer", "--samples", "samples.json"], 2),
    ],
)
def test_help_and_usage_errors_do_not_run_system(monkeypatch, args, code) -> None:
    run = Mock(side_effect=AssertionError("must not execute"))
    monkeypatch.setattr(cli, "run_reward_test", run)
    monkeypatch.setattr(cli, "discover_plugins", run)
    with pytest.raises(SystemExit) as caught:
        cli.main(args)
    assert caught.value.code == code
    run.assert_not_called()
