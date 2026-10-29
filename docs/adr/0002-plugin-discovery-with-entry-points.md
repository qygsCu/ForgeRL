
# ADR-0002：使用 Entry Points 发现插件，通过 Registry 统一管理

- Status: Proposed
- Date: 2026-10-29
- Deciders: ForgeRL 项目负责人

## Context

ForgeRL 需要允许第三方 Python 包扩展算法、奖励函数、Rollout、
Trainer、调度器和 Callback，且无需修改主仓库。

Runtime Core 应依赖 SDK 定义的协议，通过统一入口获取插件，
避免直接导入具体实现，或根据插件名称编写条件分支。

插件还可能存在名称冲突、SDK API 版本不兼容和配置错误。
这些问题应在训练初始化前被发现。

第二周先实现元数据、手动注册和配置校验；第三周再实现自动发现、
生命周期和 CLI。两阶段应共享同一套注册契约。

## Decision

### 1. 使用 Python Entry Points 声明第三方插件

第三方插件通过其 Python 包的 pyproject.toml 声明入口：

```toml
[project.entry-points."forgerl.reward"]
example_math_reward = "forgerl_example.reward:MathReward"
```

插件安装到当前 Python 环境后，ForgeRL 使用
importlib.metadata.entry_points() 查找对应入口。

各插件类别使用独立的 entry point group：

| 插件类别  | Entry Point Group |
| --------- | ----------------- |
| Algorithm | forgerl.algorithm |
| Reward    | forgerl.reward    |
| Rollout   | forgerl.rollout   |
| Trainer   | forgerl.trainer   |
| Scheduler | forgerl.scheduler |
| Callback  | forgerl.callback  |

不通过扫描工作目录、修改 sys.path 或维护第三方插件名称列表发现插件。

### 2. 使用 SDK 类型描述插件

SDK 定义 PluginKind 和 PluginMetadata。

PluginMetadata 包含：

- name：插件名称。
- kind：插件类别。
- version：插件自身版本。
- api_version：插件遵循的 ForgeRL SDK API 版本。
- required_capabilities：运行所必需的能力。
- optional_capabilities：可选能力。
- supports_checkpoint：是否支持保存和恢复插件状态。
- supports_multiprocessing：是否支持多进程使用。
- requires_gpu：是否要求 GPU。

元数据只描述插件，不持有模型、优化器或运行中的资源。

插件版本、SDK API 版本、数据 Schema 版本和 Policy 版本相互独立。

### 3. 使用类型化的注册描述对象

PluginDescriptor 持有：

- PluginMetadata。
- 插件拥有的 Pydantic 配置模型。
- 延迟创建插件实例的工厂。

工厂签名为 `Callable[[], object]`，只负责无参构造；已校验的配置在后续
生命周期的初始化阶段传入。构造插件描述对象时不调用工厂。
配置模型必须是 Pydantic `BaseModel` 子类，且声明 `extra="forbid"`。
元数据和描述对象构造时执行校验，非法声明抛出 `PluginLoadError`；
`api_version` 的支持情况由 Registry 判断，而非元数据对象自身判断。

第三方入口初期指向插件类。插件类通过类属性暴露元数据和配置模型，
加载器将其适配为 PluginDescriptor。

配置模型和工厂属于进程内描述对象，不写入 JSON 元数据快照。

完整的插件执行接口由各类别的 Protocol 定义；
元数据合法不代表插件行为已经通过一致性测试。

### 4. Registry 统一管理注册结果

PluginRegistry 提供注册、查询和列表操作，内部以
(kind, name) 作为唯一键。

规则如下：

- 同一类别内名称必须唯一，重复注册失败，不覆盖已有项。
- 不同类别允许使用相同名称。
- 查询不存在的插件时明确失败，不回退到其他插件。
- 列表按 `(kind.value, name)` 排序，并返回新列表，便于 CLI 展示和测试。
- Registry 使用独立实例，不使用全局可变单例。
- 注册和查询不创建插件实例，也不分配 GPU 等运行资源。
- 注册接收正常构造并已通过自身校验的 PluginDescriptor，不重复执行其字段校验。
  Registry 负责 SDK API 兼容性和名称唯一性检查。
- 注册失败不得改变已有条目；查询不存在的插件抛出 ConfigurationError。

内置插件、测试 Mock 和第三方插件最终都通过同一个注册入口，
遵守相同的冲突与兼容性规则。

自动发现时，entry point group 必须与 metadata.kind 一致，
entry point name 必须与 metadata.name 一致。

### 5. 将发现、加载、配置校验和初始化分开

处理流程为：

```text
发现入口记录
  → 导入声明对象，读取元数据和配置模型
  → 检查元数据及 SDK API 版本
  → 注册描述对象
  → 根据运行配置选择插件
  → 校验插件参数及运行环境能力
  → 创建实例
  → 初始化与健康检查
  → 执行
  → 关闭
```

发现入口记录不等于初始化插件。

加载入口对象会导入插件模块，因此插件模块应避免在导入时加载模型、
创建 Worker 或访问 GPU。

第二周只实现手动注册、兼容性检查和配置校验；
自动发现与完整生命周期在第三周实现。

### 6. 在初始化前检查版本和配置

SDK 初始 API 版本为字符串 "1.0"。
当前只接受精确匹配，不推测版本兼容关系。

PluginSelection.options 负责承载严格 JSON 参数；
插件自己的 Pydantic 配置模型负责验证参数名称、类型、范围和默认值。

插件配置拒绝未知字段。校验失败时不得调用插件工厂或初始化方法。

必需能力不满足时拒绝运行；可选能力缺失时，
插件必须按照其公开约定运行。

### 7. 使用明确的错误类型

- 导入失败、非法注册描述和名称冲突：PluginLoadError。
- SDK API 不兼容或缺少必需能力：PluginCompatibilityError。
- 配置引用不存在的插件或参数不合法：ConfigurationError。

底层异常使用异常链保留，错误信息包含插件类别、名称和失败原因。

不得通过静默跳过冲突、自动替换插件或忽略参数来继续训练。

## Alternatives Considered

### 在 Runtime 中直接导入具体插件

实现简单，但每增加一个第三方插件都可能需要修改核心代码，
违背独立安装和扩展的目标，因此不采用。

### 扫描目录并自动导入 Python 文件

依赖目录结构和导入路径，容易产生意外导入和命名冲突，
也无法直接复用已安装 Python 包的声明信息，因此不采用。

### 只提供手动注册

适合单元测试和第二周的增量开发，但不能完成第三方包安装后
自动发现的目标，因此保留为基础机制，并增加 entry points 适配层。

### 通过装饰器修改全局 Registry

写法简短，但注册依赖导入顺序，全局状态也会干扰测试隔离，
因此不作为核心注册机制。

## Consequences

### Positive

- 第三方插件可以独立打包和安装。
- Runtime 无需依赖具体插件实现。
- Mock 和外部插件共享注册规则。
- 名称冲突、版本和配置问题可以提前失败。
- 手动注册便于 CPU 测试，自动发现可在其上增量实现。

### Negative

- 需要维护元数据、描述对象和加载适配层。
- 加载插件声明仍会执行 Python 模块导入。
- 精确 API 版本匹配要求接口升级时同步更新插件。
- 元数据和配置校验不能替代插件行为一致性测试。

## Compatibility

当前尚未发布稳定的插件 SDK，本决策新增插件契约，
不修改现有 TrajectoryBatch、WeightSnapshot 和 RuntimeEvent。

将来修改插件声明格式、入口对象契约或 SDK API 兼容规则时，
必须更新 ADR，并明确迁移方式。

## Validation

第二周验收：

- 合法 Mock 描述能够注册和查询。
- 同类别重名失败，不同类别同名成功。
- 未知插件查询失败。
- 不支持的 SDK API 版本被拒绝。
- 非法元数据被拒绝。
- Registry 实例之间相互隔离。
- 插件配置默认值、未知字段和非法参数均有测试。
- 配置校验失败时工厂未被调用。

第三周验收：

- 已安装示例包的 entry point 能被发现并注册。
- 入口名称或类别与元数据不一致时失败。
- 导入失败保留原始异常信息。
- 自动发现过程中不创建插件实例。

主体测试必须能够在无 GPU 的 CPU 环境运行。

## Review Conditions

出现以下需求时重新评审：

- SDK 需要支持多个兼容版本。
- 插件需要远程运行或跨语言加载。
- 插件需要动态卸载、热更新或运行时替换。
- 现有描述对象无法满足真实第三方插件的需要。

## References

- [项目设计：插件系统](../../ForgeRL_PROJECT_SPEC.md#7-插件系统)
- [ADR-0003：类型化数据协议](0003-typed-data-protocol.md)
