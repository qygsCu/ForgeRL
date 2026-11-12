# ForgeRL

ForgeRL 是面向 LLM 强化学习训练的可扩展运行时。目前已实现插件声明、
自动发现、配置校验、单插件生命周期与 Runtime 状态机；训练闭环尚未实现。

## 安装

使用 Python 3.11 或以上版本，在项目目录运行：

```bash
python -m pip install -e ".[dev]"
forgerl --help
```

## 插件管理

```bash
forgerl plugin list
forgerl plugin list --kind reward
forgerl plugin inspect NAME --kind reward
forgerl plugin validate NAME --kind reward --options options.json
```

- `list` 按类别和名称排序；没有插件时正常返回。
- `inspect` 输出插件元数据和配置 JSON Schema。
- `validate` 读取仅包含插件参数的 JSON 对象，输出校验后包含默认值的配置。
- `--kind` 支持 `algorithm/reward/rollout/trainer/scheduler/callback`。
- 管理命令不创建插件实例，也不验证 GPU 可用性、健康状态或业务接口。

成功退出码为 `0`；插件或参数文件处理失败为 `1`，原因写到标准错误；
命令用法错误为 `2`。新安装的环境没有插件时，空列表属于正常结果。

## 安装样例验证

仓库提供仅用于 CLI 验证的插件包；其构造函数会直接报错，以确认管理命令
不会创建插件实例。这不是可用于训练的 Reward 实现。

```bash
python -m pip install ./tests/fixtures/cli_plugin
forgerl plugin inspect cli_fixture_reward --kind reward
```

创建 UTF-8 编码的 `options.json`，内容为 `{}`，然后运行：

```bash
forgerl plugin validate cli_fixture_reward --kind reward --options options.json
```

预期输出 `{"scale": 1.0}`（带缩进）。将内容改为 `{"unknown": 1}` 或
`{"scale": 0}` 后，命令应失败并返回退出码 `1`。
样例用完后可通过 `python -m pip uninstall forgerl-cli-fixture` 移除。

## 开发检查

```bash
python -m pytest tests/unit/cli/test_main.py -q
python -m pytest tests/unit/runtime/test_lifecycle.py -q
python -m pytest tests/integration/test_cli_installed_plugin.py -q
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
python -m mypy src
python -m build
```

安装集成测试在临时虚拟环境中复用当前环境依赖，离线安装 ForgeRL 和样例包，
通过真正的 `forgerl` 控制台入口验证发现、查看和参数校验；不模拟 entry points。
运行该测试前，当前环境需具备项目依赖、pip、setuptools >= 77.0.3 和 wheel。
测试不会将样例插件安装到当前开发环境。

开发学习待办：插件生命周期完成后学习插件校验、创建、使用逻辑。
