import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, ConfigDict, Field

import forgerl.cli.main as cli
from forgerl.runtime.registry import PluginRegistry
from forgerl.sdk.errors import ConfigurationError, PluginCompatibilityError, PluginLoadError
from forgerl.sdk.plugins import SDK_API_VERSION, PluginDescriptor, PluginKind, PluginMetadata


class RewardConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", title="奖励配置")

    scale: float = Field(default=1.0, gt=0)


def make_descriptor(
    name: str = "demo",
    kind: PluginKind = PluginKind.REWARD,
) -> PluginDescriptor:
    return PluginDescriptor(
        metadata=PluginMetadata(
            name=name,
            kind=kind,
            version="0.1.0",
            api_version=SDK_API_VERSION,
            required_capabilities=frozenset({"z", "a"}),
            optional_capabilities=frozenset(),
            supports_checkpoint=False,
            supports_multiprocessing=True,
            requires_gpu=False,
        ),
        config_model=RewardConfig,
        factory=Mock(side_effect=AssertionError("CLI commands must not create plugins")),
    )


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> PluginRegistry:
    result = PluginRegistry()
    monkeypatch.setattr(cli, "discover_plugins", lambda: result)
    return result


@pytest.mark.parametrize("kind_args", [[], ["--kind", "reward"]])
def test_list_empty_registry_succeeds(
    registry: PluginRegistry,
    kind_args: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert cli.main(["plugin", "list", *kind_args]) == 0

    captured = capsys.readouterr()
    assert captured.out.strip() == "No plugin found"
    assert captured.err == ""


def test_list_sorts_plugins_without_constructing_them(
    registry: PluginRegistry,
    capsys: pytest.CaptureFixture[str],
) -> None:
    descriptors = [
        make_descriptor("beta"),
        make_descriptor("zeta", PluginKind.ALGORITHM),
        make_descriptor("alpha"),
    ]
    for descriptor in descriptors:
        registry.register(descriptor)

    assert cli.main(["plugin", "list"]) == 0

    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        "KIND\tNAME\tVERSION\tAPI_VERSION",
        f"algorithm\tzeta\t0.1.0\t{SDK_API_VERSION}",
        f"reward\talpha\t0.1.0\t{SDK_API_VERSION}",
        f"reward\tbeta\t0.1.0\t{SDK_API_VERSION}",
    ]
    assert captured.err == ""
    for descriptor in descriptors:
        descriptor.factory.assert_not_called()


def test_list_filters_category(
    registry: PluginRegistry,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry.register(make_descriptor("same", PluginKind.REWARD))
    registry.register(make_descriptor("same", PluginKind.ROLLOUT))

    assert cli.main(["plugin", "list", "--kind", "reward"]) == 0

    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        "KIND\tNAME\tVERSION\tAPI_VERSION",
        f"reward\tsame\t0.1.0\t{SDK_API_VERSION}",
    ]
    assert captured.err == ""


def test_inspect_outputs_metadata_and_schema_without_constructing_plugin(
    registry: PluginRegistry,
    capsys: pytest.CaptureFixture[str],
) -> None:
    descriptor = make_descriptor()
    registry.register(descriptor)

    assert cli.main(["plugin", "inspect", "demo", "--kind", "reward"]) == 0

    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["metadata"] == {
        "name": "demo",
        "kind": "reward",
        "version": "0.1.0",
        "api_version": SDK_API_VERSION,
        "required_capabilities": ["a", "z"],
        "optional_capabilities": [],
        "supports_checkpoint": False,
        "supports_multiprocessing": True,
        "requires_gpu": False,
    }
    assert result["config_schema"] == RewardConfig.model_json_schema()
    assert captured.err == ""
    descriptor.factory.assert_not_called()


@pytest.mark.parametrize("command", ["inspect", "validate"])
def test_unknown_plugin_reports_failure(
    registry: PluginRegistry,
    command: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["plugin", command, "missing", "--kind", "reward"]
    if command == "validate":
        options_path = tmp_path / "options.json"
        options_path.write_text("{}", encoding="utf-8")
        argv.extend(["--options", str(options_path)])

    assert cli.main(argv) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "reward:missing" in captured.err


@pytest.mark.parametrize("options, expected", [("{}", 1.0), ('{"scale": 2.5}', 2.5)])
def test_validate_outputs_resolved_options_without_constructing_plugin(
    registry: PluginRegistry,
    options: str,
    expected: float,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    descriptor = make_descriptor()
    registry.register(descriptor)
    options_path = tmp_path / "options.json"
    options_path.write_text(options, encoding="utf-8")

    assert (
        cli.main(
            [
                "plugin",
                "validate",
                "demo",
                "--kind",
                "reward",
                "--options",
                str(options_path),
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"scale": expected}
    assert captured.err == ""
    descriptor.factory.assert_not_called()


@pytest.mark.parametrize(
    "options, error_fragment",
    [
        ('{"unknown": 1}', "unknown"),
        ('{"scale": 0}', "scale"),
        ('{"scale": "invalid"}', "scale"),
        ("[]", "JSON object"),
        ("null", "JSON object"),
        ('"text"', "JSON object"),
        ("42", "JSON object"),
        ('{"scale": NaN}', "finite"),
        ('{"scale": Infinity}', "finite"),
    ],
)
def test_validate_rejects_invalid_options(
    registry: PluginRegistry,
    options: str,
    error_fragment: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry.register(make_descriptor())
    options_path = tmp_path / "options.json"
    options_path.write_text(options, encoding="utf-8")

    assert (
        cli.main(
            [
                "plugin",
                "validate",
                "demo",
                "--kind",
                "reward",
                "--options",
                str(options_path),
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "reward:demo" in captured.err
    assert error_fragment in captured.err


@pytest.mark.parametrize("file_state", ["missing", "invalid_json", "invalid_encoding"])
def test_validate_reports_file_errors(
    registry: PluginRegistry,
    file_state: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry.register(make_descriptor())
    options_path = tmp_path / "options.json"
    if file_state == "invalid_json":
        options_path.write_text("{", encoding="utf-8")
    elif file_state == "invalid_encoding":
        options_path.write_bytes(b"\xff")

    assert (
        cli.main(
            [
                "plugin",
                "validate",
                "demo",
                "--kind",
                "reward",
                "--options",
                str(options_path),
            ]
        )
        == 1
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("forgerl: ")
    assert captured.err.strip() != "forgerl:"


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        ["plugin", "--help"],
        ["plugin", "list", "--help"],
        ["plugin", "inspect", "--help"],
        ["plugin", "validate", "--help"],
    ],
)
def test_help_succeeds_without_discovery(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    discover = Mock(side_effect=AssertionError("Help must not discover plugins"))
    monkeypatch.setattr(cli, "discover_plugins", discover)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(argv)

    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out
    assert captured.err == ""
    discover.assert_not_called()


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["plugin"],
        ["plugin", "unknown"],
        ["plugin", "list", "--kind", "invalid"],
        ["plugin", "inspect", "demo"],
        ["plugin", "inspect", "--kind", "reward"],
        ["plugin", "validate", "demo", "--kind", "reward"],
        ["plugin", "list", "--unknown"],
    ],
)
def test_usage_errors_exit_before_discovery(
    argv: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    discover = Mock(side_effect=AssertionError("Invalid usage must not discover plugins"))
    monkeypatch.setattr(cli, "discover_plugins", discover)

    with pytest.raises(SystemExit) as exc_info:
        cli.main(argv)

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "usage:" in captured.err
    discover.assert_not_called()


@pytest.mark.parametrize(
    "error_type",
    [PluginLoadError, PluginCompatibilityError, ConfigurationError],
)
def test_discovery_errors_report_failure(
    error_type: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "discover_plugins", Mock(side_effect=error_type("broken plugin")))

    assert cli.main(["plugin", "list"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "broken plugin" in captured.err


def test_unexpected_errors_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    original_error = RuntimeError("unexpected bug")
    monkeypatch.setattr(cli, "discover_plugins", Mock(side_effect=original_error))

    with pytest.raises(RuntimeError) as exc_info:
        cli.main(["plugin", "list"])

    assert exc_info.value is original_error
