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

## 数学奖励插件

独立可安装的 [数学奖励示例包](examples/math_reward/README.md) 支持严格字符串匹配
和精确数值匹配；数值模式接受整数、小数、整数分数及科学计数法，拒绝空白和
一般表达式。包拥有独立的 `pyproject.toml`、入口点与规则测试，可原样迁到独立仓库。

```bash
python -m pip install ./examples/math_reward
forgerl plugin inspect math_reward --kind reward
python -m pytest tests/integration/test_math_reward_installed.py -q
```

通过命令行执行协议检查：

```bash
forgerl plugin test math_reward --kind reward --samples examples/math_reward/samples.json
```

可选 `--options config.json` 指定插件配置，省略时使用默认值。Runtime 自动读取并
校验数据、初始化、评分和关闭，CLI 只输出 JSON 报告；Python 也可直接调用
`forgerl.runtime.rewards.run_reward_test(name, samples_path, registry, options_path=None)`，
由调用方先执行一次 `discover_plugins()` 并传入共享 Registry；评分检查内部不重复发现。
成功退出码为0，处理失败为1，用法错误为2。合法的 `valid=False` 仍属于协议通过，
报告包含 `valid_count`、`invalid_count` 和 `failures`，不能用协议通过判断业务准确性。
首版只检查 CPU、无 Checkpoint、无额外必需能力的 Reward 插件，其他范围明确拒绝。
具体配置、评分调用和错误语义见包内 README。

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

## 第四周本地工程验收（2026-11-20）

环境：Windows、Python 3.11.16、CPU PyTorch 2.14.1。按CI对应步骤执行：

| 检查 | 结果 |
| --- | --- |
| `python -m ruff check .` | 通过 |
| `python -m ruff format --check .` | 50个文件全部符合格式 |
| `python -m mypy src` | 18个源文件全部通过 |
| `python -m pytest -q -p no:cacheprovider` | 353项通过，含真实安装与plugin test验证 |
| `python -m build --no-isolation` | 主包sdist与wheel构建通过，产物位于dist/ |
| `python -m build --no-isolation examples/math_reward` | 插件sdist与wheel构建通过，产物位于examples/math_reward/dist/ |

构建复用当前环境已有依赖，测试关闭本地pytest缓存；本轮未触发远端GitHub Actions，
未验证Ubuntu环境。当前支持CPU序列级评分、配置校验、合法部分失败报告及异常清理；
Checkpoint、GPU、多进程、训练闭环和完整能力一致性验证尚未支持。
独立插件Git仓库与跨仓库验收按负责人要求暂缓（Deferred）；上述结果覆盖本地工程检查，
不代替原计划中的独立仓库验收。
