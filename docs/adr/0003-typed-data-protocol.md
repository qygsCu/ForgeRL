# ADR-0003：使用带版本的类型化数据协议

- Status: Proposed
- Date: 2026-10-21
- Deciders: ForgeRL 项目负责人

## Context

ForgeRL 的 Rollout、Reward、Algorithm、Trainer、Scheduler 和 Runtime 由不同模块或插件实现，后续还会运行在不同进程中。模型输入、生成结果、轨迹、权重快照和运行事件需要跨越这些边界传输。

如果模块之间传递无约束的 `dict[str, Any]`，字段拼写、Tensor Shape、dtype、device、Mask 或策略版本错误无法在数据产生时被可靠发现。错误可能在进入 Buffer 或训练数步后才暴露，也可能因为 Tensor 广播等行为没有立即报错，却产生错误的训练结果。

ForgeRL 还需要支持第三方插件。插件实现不能依赖其他插件的内部对象，也不能要求 Runtime 为具体算法或后端增加特殊分支。因此，跨模块数据必须由 SDK 提供稳定、可检查的公共契约。

异步训练进一步要求明确区分三类版本：

- ForgeRL 包版本描述软件发布版本。
- Schema 版本描述数据结构和字段语义版本。
- Policy 版本描述生成轨迹或权重快照所对应的模型参数版本。

项目的 v0.1 数据传输以 Python 进程和 CPU Tensor 为正确性基线。共享内存、跨 GPU 传输以及跨语言协议不属于本决策的当前范围。

## Decision

### 1. 使用 SDK 中定义的类型化对象

**所有跨模块、跨插件、跨进程或需要持久化的顶层数据，都必须使用 `forgerl.sdk` 中定义的类型化对象。核心路径不得传递无约束的 `dict[str, Any]`。**

Tensor 密集的数据对象使用 Python `dataclass`，并提供显式的运行时验证。配置对象使用 Pydantic，由配置系统单独负责。插件的行为接口使用 `typing.Protocol`，不与数据对象混合。

v0.1 首先定义以下对象：

- `TrajectoryBatch`：保存 Rollout 生成并供 Reward、Algorithm 和 Trainer 使用的轨迹数据。
- `WeightSnapshot`：描述已发布权重的位置、格式、版本和完整性信息，不直接携带完整权重 Tensor。
- `RuntimeEvent`：描述运行状态变化、Worker 活动、训练步骤和故障等结构化事件。

后续的 `GenerateRequest`、`GenerationBatch`、`RewardOutput` 和 `TrainBatch` 应在其生产者与消费者语义明确后，通过相同原则定义，不在本 ADR 中提前猜测字段。

### 2. 显式验证且不进行隐式修复

类型提示不能表达 Tensor 的全部运行时约束，因此每个协议对象必须提供明确的验证入口。对象构造时执行验证；从进程队列、文件或插件边界恢复数据后，消费者也应再次验证。

验证器只检查并报告错误，不得自动执行以下操作：

- 转换 dtype；
- 移动 device；
- 调整 Shape；
- 添加或删除 Padding；
- 修复 Mask；
- 修改 Schema 或 Policy 版本。

隐式修复会掩盖生产者错误，并可能产生不可见的 Tensor 拷贝和性能开销。不合法数据应通过 ForgeRL 的标准协议错误明确失败。

协议对象默认不深拷贝 Tensor。调用者应将构造后的协议对象视为只读；需要修改时创建新对象。`frozen=True` 不能阻止 Tensor 的原地修改，因此不将其视为完整的不可变性保证。

### 3. 每种跨边界对象独立维护 Schema 版本

`TrajectoryBatch`、`WeightSnapshot` 和 `RuntimeEvent` 都必须包含字符串类型的 `schema_version`。三种对象独立演化，初始版本均为字符串 `"1.0"`。

v0.1 只接受各对象明确支持的精确版本。缺少版本或遇到未知版本时必须在使用数据前失败，不进行猜测或静默兼容。将来如需支持向前或向后兼容，应通过新的 ADR 定义版本范围、字段默认值和迁移策略。

Schema 版本不得使用浮点数，也不得与 ForgeRL 包版本或 Policy 版本共用。

### 4. 限制可扩展元数据为严格 JSON 值

协议允许通过 `metadata` 或 `payload` 保存少量扩展信息，但其值必须符合递归的 `JsonValue` 定义：

- `None`；
- `bool`；
- `int`；
- 有限的 `float`；
- `str`；
- `list[JsonValue]`；
- `dict[str, JsonValue]`。

不允许 Tensor、tuple、set、文件句柄、自定义对象、非字符串字典键、NaN 或正负无穷。大 Tensor 不得通过 `metadata` 或事件 `payload` 绕过数据面传输。

验证错误应尽可能包含具体字段路径，以便定位嵌套数据中的非法值。

### 5. 规范化 TrajectoryBatch 的 Tensor 语义

设：

- `B` 为 Batch 中的采样轨迹数量；
- `P` 为 Padding 后的 Prompt 长度；
- `R` 为 Padding 后的 Response 长度。

`B`、`P` 和 `R` 必须大于零。字段约束如下：

| 字段 | Shape | dtype | 语义 |
| --- | --- | --- | --- |
| `request_ids` | 长度 `[B]` | 非空字符串 | 每条采样轨迹的唯一标识 |
| `group_ids` | `[B]` | `torch.int64` | 标识来自同一 Prompt 的轨迹组 |
| `prompt_ids` | `[B, P]` | `torch.int64` | 左 Padding 后的 Prompt Token |
| `response_ids` | `[B, R]` | `torch.int64` | 右 Padding 后的 Response Token |
| `attention_mask` | `[B, P + R]` | `torch.bool` | 完整序列中的有效 Token |
| `response_mask` | `[B, R]` | `torch.bool` | 参与 RL Loss 的 Response Token |
| `old_logprobs` | `[B, R]` | `torch.float32` | 旧策略对实际采样 Token 的 Log Probability |
| `sequence_rewards` | `[B]` 或 `None` | `torch.float32` | 每条回答的序列级奖励 |

所有实际存在的 Tensor 必须位于同一个 device。`sequence_rewards=None` 表示奖励尚未计算，此时不参与 dtype、Shape 和 device 检查；需要奖励的下游阶段必须拒绝 `None`。

`request_ids` 在同一个 Batch 内必须唯一，生产者应尽量保证其在整个 Run 内唯一。多个采样来自同一 Prompt 的关系仅由 `group_ids` 表达。`group_ids` 不要求连续、排序或从零开始，通用协议也不检查每组的样本数量；组大小属于具体算法的约束。

### 6. 统一 Mask 与 Padding 语义

SDK 中所有表示有效性的 Mask 均使用 `torch.bool`：

- `True` 表示有效、可见或参与计算；
- `False` 表示 Padding、不可见或不参与计算。

ForgeRL 的规范化批数据使用以下 Padding 规则：

- Prompt 左 Padding；
- Response 右 Padding；
- 模型实际生成的第一个 EOS Token 参与 Loss；
- EOS 之后和 Padding 位置不参与 Loss；
- `response_mask` 必须表现为有效 Token 在前、Padding 在后的形式；
- `attention_mask` 的最后 `R` 列必须与 `response_mask` 相同；
- `response_mask=False` 的位置上，`old_logprobs` 必须为 `0.0`。

协议以 Mask 判断位置是否有效，不根据 Token ID 推断 Padding。实际 `pad_token_id` 由模型和 tokenizer 配置管理，因为部分模型可能共用 PAD 和 EOS Token ID。

### 7. WeightSnapshot 使用带算法标识的校验和

`WeightSnapshot.policy_version` 必须为非负整数。`format` 只允许 `full`、`lora` 或 `sharded`。

`checksum` 使用以下规范格式：

```text
sha256:<64 个小写十六进制字符>
```

单文件快照对文件原始字节计算 SHA-256。分片权重以后通过规范化 Manifest 记录各分片的名称、大小和校验和，并对 Manifest 计算主校验和；该 Manifest 的具体结构由权重发布或 Checkpoint ADR 定义。

加载方必须在校验格式、内容完整性和 Policy 版本后，才能更新自身已加载的 Policy 版本。Checksum 用于检测损坏或错误文件，不作为发布者身份认证或数字签名。

当 `uri=None` 表示内存传输时，权重内容指纹的规范化计算方式留到权重发布阶段定义。在该规则确定前，不得生成虚假或占位校验和。

### 8. RuntimeEvent 使用结构化、可序列化数据

核心事件类型使用受控的字符串枚举，避免同一事件出现不同拼写。`event_id` 和 `run_id` 必须为非空字符串，`timestamp_ns` 必须为非负整数，`policy_version` 必须为 `None` 或非负整数。

按 [ADR-0009](0009-runtime-state-machine.md)，新增 `RUN_STATE_CHANGED` 事件，payload 包含非空字符串 `from_state`、`to_state` 和 `reason`。该 payload 由状态机按转换契约构造；通用 RuntimeEvent 验证器仍负责字段类型和严格 JSON 约束，不负责检查状态转换规则。这是尚未发布稳定协议时的开发阶段契约修订，RuntimeEvent Schema 保持 `"1.0"`；使用封闭枚举的生产者和消费者必须同步更新，不保证旧消费者接受新增事件值。

`timestamp_ns` 表示用于跨进程关联事件的 Unix wall-clock 纳秒时间。性能耗时应使用单独记录的 monotonic duration，不依赖两个 wall-clock 时间戳相减。

事件 `payload` 仅允许严格 `JsonValue`，不得携带大 Tensor、异常对象或不可序列化资源。

### 9. 区分数据面和控制面序列化

`TrajectoryBatch` 属于数据面，可以包含 Tensor；`WeightSnapshot` 和 `RuntimeEvent` 属于小型控制或观测记录，不直接包含模型权重等大对象。

v0.1 要求协议对象能够在 CPU Mock 环境完成 Python 进程传输所需的序列化往返，并在恢复后重新通过验证。Tensor 不强制转换为 JSON，RuntimeEvent 和元数据则必须可以转换为 JSON 表示。

Python 序列化只用于 ForgeRL 管理的可信进程之间，不用于加载不可信输入。Checkpoint 的长期存储格式和原子性由独立 ADR 决定。

## Alternatives Considered

### 1. 使用无约束字典

字典实现简单、修改字段方便，但无法可靠发现拼写错误、缺失字段或错误类型。它也不能表达 Tensor Shape、dtype、device 和 Mask 语义，容易把问题推迟到训练阶段，因此不采用。

### 2. 仅使用 TypedDict

`TypedDict` 可以改善静态类型检查，但运行时仍然是普通字典，不能自动验证第三方插件输出或跨进程恢复的数据，也不能检查 Tensor 的内部约束，因此不足以单独承担数据协议。

### 3. 所有数据对象都使用 Pydantic

Pydantic 适合配置和普通 JSON 数据，但 Tensor 仍需启用任意类型并编写自定义验证器。对高频 Tensor 数据对象而言，它没有消除主要验证工作，还会引入额外依赖行为和开销。因此配置层使用 Pydantic，Tensor 数据面使用 dataclass 和显式验证。

### 4. 使用 Protobuf、Arrow 或其他跨语言协议

这些方案具有更正式的 Schema 和跨语言能力，但 Tensor、device 和进程内对象仍需额外适配。ForgeRL v0.1 只支持 Python 单机运行，引入跨语言 IDL、代码生成和转换层会显著增加复杂度，因此暂不采用。

### 5. 在消费者内部按需转换错误数据

消费者自动转换 dtype、移动 device 或修复 Shape 能让部分输入继续运行，但会隐藏生产者错误、增加不可见复制，并导致不同消费者采取不一致的修复策略。因此选择在协议边界失败，由生产者修正数据。

## Consequences

### Positive

- 模块和插件之间具有统一、可检查的公共契约。
- Shape、dtype、device、版本和 Mask 错误能够尽早暴露。
- Runtime 不需要了解具体算法或后端的内部对象。
- CPU Mock、单 GPU 和多进程模式可以共享相同协议。
- 策略版本和 Schema 版本的含义清晰，便于实现陈旧度控制。
- 可以独立编写协议单元测试和插件一致性测试。
- 结构化事件可以稳定用于日志、指标、故障分析和性能报告。

### Negative

- 每个协议对象都需要维护显式验证代码和测试。
- 构造和跨边界验证会产生少量运行时开销。
- 后端可能需要在明确位置转换 Mask 或模型输入 dtype。
- dataclass 不提供深层不可变性，调用方仍需遵守所有权约定。
- 严格版本检查意味着协议升级必须同步更新生产者、消费者和测试。
- 当前 Python dataclass 方案不能直接支持跨语言 Worker。

## Validation

本决策通过以下测试落实：

### 正常路径

- 三种协议对象可使用合法字段构造。
- `TrajectoryBatch` 同时支持奖励存在和 `sequence_rewards=None`。
- 合法 metadata 和事件 payload 可以进行 JSON 往返。
- CPU Tensor 协议对象可以序列化、反序列化并重新验证。

### 边界条件

- 最小合法的 `B`、`P` 和 `R`。
- Prompt 左 Padding、Response 右 Padding及 EOS 边界。
- 可选的 Worker、Policy 版本和 URI 字段为 `None`。
- 合法的深层 JsonValue。

### 失败路径

- 缺少或不支持的 Schema 版本。
- Shape、rank、dtype 或 device 不一致。
- 非法 Mask、Padding 或 Mask 之外的非零 Log Probability。
- 空或重复的 `request_ids`。
- 负 Policy 版本或时间戳。
- 非法权重格式或 checksum。
- 包含 Tensor、set、tuple、非字符串键、NaN 或无穷大的 metadata/payload。

所有主体测试必须能够在无 GPU 的 CPU CI 环境运行。涉及多 device 的额外测试可以使用可用设备执行或明确标记为可跳过，但不能让主体协议验证依赖 CUDA。

## Review Conditions

出现以下情况时重新评审本决策：

- 需要支持非 Python 或远程 Worker；
- 共享内存、零拷贝或 GPU-to-GPU 传输需要改变对象所有权；
- Profiling 表明协议验证成为显著性能瓶颈；
- 出现真实的 Schema 向前或向后兼容需求；
- WeightSnapshot 需要支持分片 Manifest 或内存权重的稳定内容指纹；
- 第三方插件需要扩展核心事件类型；
- Checkpoint 长期存储格式需要与运行时传输协议统一。

## References

- [ForgeRL 项目设计：数据协议](../../ForgeRL_PROJECT_SPEC.md#8-数据协议)
- [ForgeRL 项目设计：总体架构](../../ForgeRL_PROJECT_SPEC.md#5-总体架构)
- [ForgeRL 项目设计：测试计划](../../ForgeRL_PROJECT_SPEC.md#14-测试计划)
