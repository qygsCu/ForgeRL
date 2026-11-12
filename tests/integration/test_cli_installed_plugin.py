"""Exercise installed console scripts and entry points without mocking discovery."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_cli_discovers_and_validates_installed_plugin(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    environment = tmp_path / "venv"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PIP_NO_INDEX"] = "1"
    # Reuse installed CPU/GPU dependencies; install project packages only in this venv.
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(environment)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    scripts = environment / ("Scripts" if os.name == "nt" else "bin")
    python = scripts / ("python.exe" if os.name == "nt" else "python")
    install = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "--disable-pip-version-check",
            "install",
            "--no-index",
            "--no-deps",
            "--no-build-isolation",
            str(root),
            str(root / "tests" / "fixtures" / "cli_plugin"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        env=env,
    )
    assert install.returncode == 0, install.stdout + install.stderr
    executable = scripts / ("forgerl.exe" if os.name == "nt" else "forgerl")

    def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(executable), *args],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=60,
            env=env,
        )

    help_result = run_cli("--help")
    assert help_result.returncode == 0, help_result.stderr
    assert "usage:" in help_result.stdout

    listed = run_cli("plugin", "list", "--kind", "reward")
    assert listed.returncode == 0, listed.stderr
    assert "cli_fixture_reward" in listed.stdout

    inspected = run_cli("plugin", "inspect", "cli_fixture_reward", "--kind", "reward")
    assert inspected.returncode == 0, inspected.stderr
    result = json.loads(inspected.stdout)
    assert result["metadata"]["name"] == "cli_fixture_reward"
    assert result["config_schema"]["properties"]["scale"]["default"] == 1.0

    options_path = tmp_path / "options.json"
    options_path.write_text("{}", encoding="utf-8")
    validated = run_cli(
        "plugin",
        "validate",
        "cli_fixture_reward",
        "--kind",
        "reward",
        "--options",
        str(options_path),
    )
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout) == {"scale": 1.0}

    options_path.write_text('{"unknown": 1}', encoding="utf-8")
    rejected = run_cli(
        "plugin",
        "validate",
        "cli_fixture_reward",
        "--kind",
        "reward",
        "--options",
        str(options_path),
    )
    assert rejected.returncode == 1
    assert rejected.stdout == ""
    assert "unknown" in rejected.stderr
