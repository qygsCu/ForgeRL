
# 异步 RL Runtime 三个核心数据类的设计哲学

## 1. 核心思想

`TrajectoryBatch`、`WeightSnapshot` 和 `RuntimeEvent` 分别描述异步 RL 系统中的**训练数据、模型状态和运行事件**。

其设计遵循一个核心原则：

> 将复杂系统拆分为职责独立的组件，通过明确、稳定、可验证的数据契约（Data Contract）实现协作。

| 数据类              | 本质抽象 | 核心职责                         |
| ------------------- | -------- | -------------------------------- |
| `TrajectoryBatch` | Data     | 传递 Rollout 生成的训练数据      |
| `WeightSnapshot`  | State    | 描述和分发特定版本的模型权重     |
| `RuntimeEvent`    | Event    | 记录系统运行过程中发生的重要事实 |

## 2. 六个核心设计原则

### 2.1 关注点分离（Separation of Concerns）

不同信息具有不同的职责和生命周期，应当分别抽象。

- **TrajectoryBatch**：关心生成了什么训练数据，不关心模型如何分发。
- **WeightSnapshot**：关心使用哪个版本的模型，不关心具体训练数据。
- **RuntimeEvent**：关心系统发生了什么，不直接承担训练数据和模型权重的传输。

**设计价值：高内聚、低耦合，使算法、权重分发和运行时管理能够独立演化。**

### 2.2 显式状态与版本化（Explicit State & Versioning）

同步程序可以通过执行顺序隐式保证状态一致性，但异步系统中各组件具有独立的执行进度。

因此，必须通过 `policy_version` 显式标识模型状态。

关键不变量：

- `WeightSnapshot.policy_version` 唯一对应一份确定的模型权重。
- `TrajectoryBatch.policy_version` 必须对应实际生成轨迹的 Policy。
- `old_logprobs` 必须由该 Policy 计算得到。
- Worker 必须成功加载并验证新权重，才能切换采样版本。

**设计价值：正确性不依赖偶然的执行顺序，而依赖明确的数据不变量（Invariants）。**

### 2.3 数据与元数据分离（Data vs Metadata）

大型数据不应直接通过控制消息传递。

例如：

- `WeightSnapshot` 保存权重的 URI、版本和 Checksum，而非完整 Tensor。
- `RuntimeEvent` 保存事件类型及必要的 Payload，而非完整 `TrajectoryBatch`。

将系统通信分为：

- **Data Plane**：传递实际训练数据和模型权重。
- **Control Plane**：传递控制信息、状态描述及运行事件。

**设计价值：减少不必要的数据复制与通信，降低组件耦合，提高扩展性。**

### 2.4 事实与命令分离（Events vs Commands）

- **Command**：要求执行某个操作，例如 `StartRollout`。
- **Event**：描述已经发生的事实，例如 `ROLLOUT_COMPLETED`。

Event 的 Producer 只负责声明事实，Consumer 决定如何处理。

例如：

```text
ROLLOUT_COMPLETED
       │
       ├── Metrics：更新吞吐量指标
       ├── Logger：记录执行日志
       └── Coordinator：更新任务状态
```

**设计价值：通过事件发布与订阅解耦生产者和消费者。**

但并非所有操作都需要事件化。直接调用、任务队列与事件机制可以共同存在。

### 2.5 数据契约与组件自治（Data Contract & Component Autonomy）

异步组件不应过度依赖彼此的内部实现，而应依赖稳定的数据协议。

一个完整的 Data Contract 不仅规定字段，还包括：

- 字段的类型、形状和语义。
- 必须满足的数据不变量。
- 版本兼容性和 Schema 演化规则。
- 数据合法性校验。
- 扩展字段的使用约定。

例如：

- `schema_version`：支持协议演化。
- `validate()`：检查数据合法性。
- `metadata` / `payload`：承载可选的扩展信息。

**设计价值：稳定核心接口，让组件可以独立开发、替换和扩展。**

### 2.6 为失败而设计（Design for Failure）

异步系统必须考虑组件失败、消息延迟、数据重复和状态不一致。

三个类分别提供基础支持：

| 数据类              | 可靠性机制                                         |
| ------------------- | -------------------------------------------------- |
| `TrajectoryBatch` | Request ID 去重、Group 完整性、Policy Version 校验 |
| `WeightSnapshot`  | Checksum 校验、版本管理、原子发布与切换            |
| `RuntimeEvent`    | Event ID 去重、运行追踪、故障诊断                  |

**设计价值：系统不应假设操作永远成功，而应具备检测、隔离和恢复失败的能力。**

这些数据类提供必要信息，但真正的可靠性仍需要持久化、幂等处理和恢复机制配合。

## 3. 三个数据类的整体关系

```text
              WeightSnapshot
              (Policy State)
                     │
                     ▼
               Rollout Worker
                     │
                     ▼
              TrajectoryBatch
              (Training Data)
                     │
                     ▼
                GRPO Trainer
                     │
                     ▼
              New Policy State
                     │
                     └──────────→ 下一轮权重发布

各组件在运行过程中产生 RuntimeEvent
                     │
                     ▼
              Event Dispatcher
                     │
             ┌───────┼───────┐
             ▼       ▼       ▼
           Metrics  Logger  Coordinator
```

其中：

- `TrajectoryBatch` 连接 Rollout 与 Training。
- `WeightSnapshot` 连接 Training 与下一轮 Rollout。
- `RuntimeEvent` 提供运行过程的事件通知与可观测性。

## 4. 进一步的架构原则

### 机制与策略分离（Mechanism vs Policy）

- **Mechanism**：如何发布权重、传递轨迹、分发事件。
- **Policy**：何时发布权重、允许多少 Policy Staleness、失败后是否重试。

数据契约不应包含过多调度策略。

### 避免过度设计（Avoid Overengineering）

抽象应先于复杂实现，但不意味着必须立即引入分布式基础设施。

例如，第一版完全可以使用：

- 内存 Queue 传递 `TrajectoryBatch`。
- 本地文件保存 `WeightSnapshot` 对应的权重。
- 进程内 Event Dispatcher 处理 `RuntimeEvent`。

未来扩展时替换实现，而不是推翻核心接口。

## 5. 最终总结

三个数据类体现了异步 RL Runtime 的三个基本维度：

\[
\boxed{
\text{Runtime}=
\text{Data}+\text{State}+\text{Events}
}
\]

其核心设计哲学可以归纳为：

1. **职责分离**：每个抽象只负责一种核心信息。
2. **显式契约**：组件之间通过明确的数据协议协作。
3. **版本一致性**：用状态标识和不变量保证异步执行的正确性。
4. **松耦合**：隐藏组件内部实现，依赖稳定接口。
5. **可扩展性**：稳定核心抽象，允许外围机制独立演化。
6. **可靠性**：主动考虑失败、重复、延迟和恢复。

> **异步系统设计的本质，不是让所有组件共享同一个执行顺序，而是让它们在独立运行的条件下，依然能够通过明确的契约实现正确、可靠、可扩展的协作。**
