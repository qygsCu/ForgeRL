import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import forgerl.runtime.rewards as reward_test
from forgerl.sdk import ConfigurationError, ProtocolViolationError, RewardSample


def write_samples(tmp_path: Path, value: object) -> Path:
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


def record() -> dict[str, str]:
    return {"request_id": "r1", "response": "2", "reference_answer": "2"}


def test_loader_preserves_text_and_order_without_discovering_plugins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    discover = Mock(side_effect=AssertionError("sample loading must not discover plugins"))
    monkeypatch.setattr("forgerl.runtime.discovery.discover_plugins", discover)
    path = write_samples(
        tmp_path,
        [
            {"request_id": " r2 ", "response": " 中文\n", "reference_answer": "2"},
            {"request_id": "r1", "response": "", "reference_answer": ""},
        ],
    )
    result = reward_test.load_reward_samples(path)
    assert result == [RewardSample(" r2 ", " 中文\n", "2"), RewardSample("r1", "", "")]
    discover.assert_not_called()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("value", [{}, None, "text", 3, False])
def test_top_level_must_be_array(tmp_path: Path, value: object) -> None:
    with pytest.raises(ProtocolViolationError, match="must be a JSON array"):
        reward_test.load_reward_samples(write_samples(tmp_path, value))


@pytest.mark.parametrize("value", [None, [], "text", 3])
def test_each_record_must_be_object(tmp_path: Path, value: object) -> None:
    with pytest.raises(ProtocolViolationError, match=r"\$\[1\].*JSON object"):
        reward_test.load_reward_samples(write_samples(tmp_path, [record(), value]))


@pytest.mark.parametrize("field", ["request_id", "response", "reference_answer"])
def test_missing_fields_are_located(tmp_path: Path, field: str) -> None:
    value = record()
    del value[field]
    with pytest.raises(ProtocolViolationError, match=rf"\$\[0\].*missing fields=.*{field}"):
        reward_test.load_reward_samples(write_samples(tmp_path, [value]))


def test_unknown_fields_are_not_ignored(tmp_path: Path) -> None:
    with pytest.raises(ProtocolViolationError, match="unknown fields=.*response_typo"):
        reward_test.load_reward_samples(
            write_samples(tmp_path, [{**record(), "response_typo": "2"}])
        )


@pytest.mark.parametrize("field", ["request_id", "response", "reference_answer"])
@pytest.mark.parametrize("value", [None, 2, True, [], {}])
def test_field_types_are_not_coerced(tmp_path: Path, field: str, value: object) -> None:
    with pytest.raises(ProtocolViolationError, match=rf"\$\[0\].*{field}.*string"):
        reward_test.load_reward_samples(write_samples(tmp_path, [{**record(), field: value}]))


@pytest.mark.parametrize("request_id", ["", " \t"])
def test_invalid_id_is_located(tmp_path: Path, request_id: str) -> None:
    with pytest.raises(ProtocolViolationError, match=r"\$\[0\].*request_id"):
        reward_test.load_reward_samples(
            write_samples(tmp_path, [{**record(), "request_id": request_id}])
        )


def test_duplicate_ids_are_rejected_before_overwrite(tmp_path: Path) -> None:
    with pytest.raises(ProtocolViolationError, match="duplicate request_id: 'r1'"):
        reward_test.load_reward_samples(
            write_samples(tmp_path, [record(), {**record(), "response": "3"}])
        )


def test_empty_array_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ProtocolViolationError, match="must not be empty"):
        reward_test.load_reward_samples(write_samples(tmp_path, []))


def test_duplicate_json_fields_are_not_silently_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "samples.json"
    path.write_text(
        '[{"request_id":"r1","request_id":"r2","response":"2","reference_answer":"2"}]',
        encoding="utf-8",
    )
    with pytest.raises(ProtocolViolationError, match="duplicate field 'request_id'"):
        reward_test.load_reward_samples(path)


def test_json_error_reports_file_line_and_column(tmp_path: Path) -> None:
    path = tmp_path / "samples.json"
    path.write_text('[\n{"request_id":\n]', encoding="utf-8")
    with pytest.raises(ProtocolViolationError, match="invalid JSON at line 3, column 1") as caught:
        reward_test.load_reward_samples(path)
    assert str(path) in str(caught.value)
    assert isinstance(caught.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize("state", ["missing", "directory", "invalid_encoding"])
def test_unreadable_file_has_path_and_cause(tmp_path: Path, state: str) -> None:
    path = tmp_path / "samples.json"
    if state == "directory":
        path.mkdir()
    elif state == "invalid_encoding":
        path.write_bytes(b"\xff")
    with pytest.raises(ConfigurationError) as caught:
        reward_test.load_reward_samples(path)
    assert str(path) in str(caught.value)
    assert caught.value.__cause__ is not None


def test_example_file_uses_existing_reward_contract() -> None:
    root = Path(__file__).resolve().parents[3]
    result = reward_test.load_reward_samples(root / "examples" / "math_reward" / "samples.json")
    assert [sample.request_id for sample in result] == ["r1", "r2", "r3"]
