"""Install both distributions and score through real discovery from outside the repo."""

import json
import os
import subprocess
import sys
from pathlib import Path


def test_installed_math_reward_rules_and_runtime(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    package = root / "examples" / "math_reward"
    environment = tmp_path / "venv"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PIP_NO_INDEX"] = "1"
    env["PIP_CONFIG_FILE"] = os.devnull

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            args,
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    run(sys.executable, "-m", "venv", "--system-site-packages", str(environment))
    scripts = environment / ("Scripts" if os.name == "nt" else "bin")
    python = str(scripts / ("python.exe" if os.name == "nt" else "python"))
    executable = str(scripts / ("forgerl.exe" if os.name == "nt" else "forgerl"))
    run(
        python,
        "-m",
        "pip",
        "--disable-pip-version-check",
        "install",
        "--no-index",
        "--no-deps",
        "--no-build-isolation",
        str(root),
        str(package),
    )
    assert "math_reward" in run(executable, "plugin", "list", "--kind", "reward").stdout
    inspected = json.loads(
        run(executable, "plugin", "inspect", "math_reward", "--kind", "reward").stdout
    )
    assert inspected["metadata"]["requires_gpu"] is False
    schema = inspected["config_schema"]["properties"]["match_mode"]
    assert schema["default"] == "strict"
    assert schema["enum"] == ["strict", "numeric"]

    options = tmp_path / "options.json"
    for mode in ("strict", "numeric"):
        options.write_text(json.dumps({"match_mode": mode}), encoding="utf-8")
        result = run(
            executable,
            "plugin",
            "validate",
            "math_reward",
            "--kind",
            "reward",
            "--options",
            str(options),
        )
        assert json.loads(result.stdout) == {"match_mode": mode}

    check = tmp_path / "check_scoring.py"
    check.write_text(
        """from pathlib import Path
import sys
import torch
import forgerl_math_reward
from forgerl.config.models import PluginSelection
from forgerl.runtime.discovery import discover_plugins
from forgerl.runtime.lifecycle import managed_plugin
from forgerl.runtime.rewards import compute_rewards
from forgerl.runtime.validation import validate_plugin_options
from forgerl.sdk import PluginContext, PluginKind, RewardSample

assert Path(forgerl_math_reward.__file__).resolve().is_relative_to(Path(sys.prefix))
def reject_cuda(*args, **kwargs):
    raise AssertionError("CPU reward must not access CUDA")
torch.cuda._lazy_init = reject_cuda
torch.cuda.is_available = reject_cuda
registry = discover_plugins()
descriptor = registry.get(PluginKind.REWARD, "math_reward")
for mode, expected in [("strict", [0.0, 1.0, 0.0]), ("numeric", [1.0, 1.0, 0.0])]:
    selection = PluginSelection(plugin="math_reward", options={"match_mode": mode})
    config = validate_plugin_options(selection, kind=PluginKind.REWARD, registry=registry)
    samples = [RewardSample("r1", "2/4", "5e-1"),
               RewardSample("r2", "2", "2"), RewardSample("r3", "1+1", "2")]
    with managed_plugin(descriptor, PluginContext(config)) as plugin:
        output = compute_rewards(plugin, samples, plugin_name="math_reward")
    assert [output.results[s.request_id].reward for s in samples] == expected
    assert all(r.valid and r.error_message is None for r in output.results.values())
print("Installed math reward verified in both modes without CUDA")
""",
        encoding="utf-8",
    )
    assert "both modes without CUDA" in run(python, str(check)).stdout
    # Run the package's own rule tests against its installed wheel, not a sys.path shim.
    rule_tests = run(
        python,
        "-m",
        "pytest",
        str(package / "tests"),
        "-c",
        str(package / "pyproject.toml"),
        "-q",
        "-p",
        "no:cacheprovider",
    )
    assert "passed" in rule_tests.stdout
