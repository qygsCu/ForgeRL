# ForgeRL Math Reward

独立发行的 CPU 序列级奖励插件，通过 `forgerl.reward` entry point 注册为
`math_reward`。当前作为示例包存放在 ForgeRL 仓库内，可以将本目录原样迁到
独立仓库；源码仅依赖公开 SDK，不需要向 ForgeRL 增加具体插件分支。

## 安装与配置

先安装当前 ForgeRL，然后在主仓库目录运行：

```bash
python -m pip install ./examples/math_reward
forgerl plugin inspect math_reward --kind reward
```

配置只接受 `match_mode`，默认 `strict`；未知参数（包括 `strip`）会报错。
创建 UTF-8 的 `options.json`，内容为 `{"match_mode": "numeric"}`，再运行：

```bash
forgerl plugin validate math_reward --kind reward --options options.json
```

## 评分规则

- `strict`：原始字符串严格相等得 `1.0`，否则 `0.0`；不去空格或解释数值。
- `numeric`：先完整检查文本语法，再用标准库 `Fraction` 精确比较有理数；
  不经过二进制浮点，不使用容差，不执行表达式。
- 输入必须是字符串，整数 `2` 不会自动转换成 `"2"`。
- 每个输出以原始 `request_id` 为键，包含 SDK `RewardResult` 的
  `reward`、`valid`、`error_message`，外层为 `RewardOutput`。

数值模式允许 ASCII 数字和可选正负号：

| 形式 | 允许示例 |
| --- | --- |
| 整数 | `2`、`-2`、`+002` |
| 小数 | `2.0`、`.5`、`2.` |
| 整数分数 | `1/2`、`2/4`、`1/-2`、`-1/-2` |
| 科学计数法 | `5e-1`、`1.5E+3` |

分数允许非最简形式，分子与分母只能是整数，分母不能为零。
`1/2`、`0.5`、`5e-1`相等；`1/3`与`0.333333`不相等。
拒绝任何空白（包括首尾空白、换行）、Unicode 数字、下划线、千位分隔符、
NaN、Infinity、混合分数、`0.5/2`、`sqrt(4)`、`1+1`、括号和其他表达式。
这些语法限制仅属于数值模式；严格模式逐字符比较。

非法或空回答正常判错，`valid=True, reward=0.0, error_message=None`。
空参考答案在两种模式都报 `ProtocolViolationError`；数值模式的非法参考答案
同样报协议错误，整批不返回部分结果。插件不人为制造可恢复评分故障，
不将编程错误、中断或资源故障转成 `valid=False`。

## 调用示例

```python
from forgerl.config.models import PluginSelection
from forgerl.runtime.discovery import discover_plugins
from forgerl.runtime.lifecycle import managed_plugin
from forgerl.runtime.rewards import compute_rewards
from forgerl.runtime.validation import validate_plugin_options
from forgerl.sdk import PluginContext, PluginKind, RewardSample

registry = discover_plugins()
descriptor = registry.get(PluginKind.REWARD, "math_reward")
config = validate_plugin_options(
    PluginSelection(plugin="math_reward", options={"match_mode": "numeric"}),
    kind=PluginKind.REWARD,
    registry=registry,
)
with managed_plugin(descriptor, PluginContext(config)) as plugin:
    output = compute_rewards(
        plugin,
        [RewardSample("r1", "2/4", "5e-1")],
        plugin_name="math_reward",
    )
assert output.results["r1"].reward == 1.0
```

构造函数不取得运行资源；未初始化和关闭后均拒绝评分。重复初始化明确报错，
关闭可重复执行，关闭后可以重新初始化。输入记录和已返回结果按只读契约使用，
后续调用不会复用或修改以前的结果字典。

## 验证

安装插件后可运行包内规则测试：

```bash
python -m pytest examples/math_reward/tests -q
```

主仓库提供离线安装集成测试：

```bash
python -m pytest tests/integration/test_math_reward_installed.py -q
```

该测试在临时虚拟环境中安装 ForgeRL 和本包，从仓库外运行 CLI、真实发现与
评分，并运行包内规则测试。复用当前环境依赖，不安装到当前开发环境；需已有
pytest、Pydantic、PyTorch、pip、setuptools >= 77.0.3 和 wheel。

`Fraction` 使用任意精度整数，但仍受 Python 整数字符串转换限制及可用资源约束；
不承诺支持无限长度数值或无限指数。格式合法但因转换限制或资源故障无法评分时，
异常向外传播，不静默判为零分。当前无 GPU、Checkpoint、多进程能力。
可通过 ForgeRL 的协议检查命令调用：

```bash
forgerl plugin test math_reward --kind reward --samples examples/math_reward/samples.json
forgerl plugin test math_reward --kind reward --samples examples/math_reward/samples.json --options options.json
```

第二条命令使用前面创建的配置文件。报告只验证协议，不验证预期业务奖励；
评分正确性由本包规则测试验证。关闭成功后才输出成功报告。
