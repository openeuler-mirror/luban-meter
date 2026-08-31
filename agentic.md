# LuBan-Meter Agent 开发指南

本文档面向参与 LuBan-Meter 开发的团队成员和 Agent，用于快速建立项目认知、定位
代码入口，并按照一致的方式完成开发与验证。

本文档只保留开始开发时需要掌握的内容。详细架构、接口、指标公式和使用方式以对应
专题文档为准：

- [架构说明](docs/architecture.md)
- [Benchmark 脚本开发指南](docs/develop-benchmark.md)
- [生成式推理指标说明](docs/metrics.md)
- [Inference 评测指标说明](docs/inference.md)
- [使用说明](docs/usage.md)
- [第一阶段项目进展及规划](docs/luban-meter第一阶段项目进展及规划.md)

## 1. 当前状态与开发计划

### 1.1 当前已有能力

LuBan-Meter 当前已经具备以下基础能力：

- 按 `module + benchmark` 自动发现 Benchmark；
- 通过 YAML 加载 Benchmark 配置；
- 运行单个 Benchmark；
- 通过 Suite 顺序编排多个 Benchmark；
- 将一次运行划分为原始数据采集和指标处理两个阶段；
- 输出 `raw_result.json`、`result.json` 和 Suite 汇总结果；
- 保存运行身份、参数、指标、环境、产物和错误信息。
- 可配置在线 SLO（TTFT、TPOT、E2EL）与服务 Goodput 有效吞吐量计算；
- 可配置 Engine 内部 SLO 与独立的 Engine Goodput 计算；
- P99 熔断机制，Case 超阈值时自动跳过后续 Case。

当前已经实现两个 `generate` Benchmark：

- `serving-online`：通过 OpenAI-compatible HTTP 流式接口测试在线生成服务；
- `vllm-engine-offline`：调用 vLLM Engine 执行离线推理并采集内部阶段时间和吞吐数据，可按
  Engine 内部 SLO 判定 Engine Goodput。

`benchmark/generate/common/` 当前提供流式响应处理、Token 计数和通用统计能力。

`inference` 模块已部分落地，面向模型任务效果评测：

- `inference/common/` 提供 OpenAI-compatible 在线服务调用（chat、completions +
  echo logprobs）、本地数据集确定性加载、Prompt 模板渲染、答案解析和指标计算
  （Accuracy、Token F1、ROUGE、Pass@k、Perplexity，其中后几类计算能力已具备，
  数据集评测待接入）等公共能力；
- 已实现三个 Benchmark：`ceval` 和 `cmmlu`（四选一题目 Accuracy，支持
  ppl/logprob 打分和 gen 生成抽取两种评测模式）以及 `gsm8k`（生成模式 +
  数值答案 Exact Match 判分）；
- 数据集只使用本地 json/jsonl 文件，离线准备脚本位于
  `benchmark/inference/scripts/`（prepare_ceval、prepare_cmmlu、prepare_gsm8k）。

### 1.2 计划开发内容

当前计划主要包括：

- 补充 Queue、Scheduler 和 KV Cache 等服务端运行指标；
- 增加 GPU 利用率、显存峰值、功耗和温度等设备数据采集；
- 增加超时、OOM 和长时间稳定性测试；
- 完善 Engine forward 阶段测试；
- 建设跨运行结果的汇总报告；
- 继续建设 `inference` 下的模型任务级评测（已完成 ceval、cmmlu、gsm8k），
  补齐 HumanEval、SQuAD、摘要类和语言建模（WikiText）任务；
- 为新增数据集接入 Token F1、ROUGE、Pass@k 和 Perplexity 等指标的端到端评测；
- 建设 `inference` 任务的 Suite 编排与跨运行汇总报告；
- 后续扩展幻觉、事实一致性、安全拒答和 Prompt Injection 等评测。

开始新任务前，应先检查对应能力是否已经存在，并以当前代码和专题文档为准确认实现
状态。

## 2. 架构与目录

### 2.1 单任务执行链路

```text
CLI
→ BenchmarkRegistry
→ CoreEngine
→ ExecutionSession
→ benchmark.py
→ raw_result.json
→ result.py
→ result.json
```

`CoreEngine` 是一次运行的统一边界。它负责解析 Benchmark、创建执行会话、运行采集
脚本、处理结果，并将各阶段异常转换为标准失败结果。

### 2.2 Suite 执行链路

```text
Suite YAML
→ SuiteLoader
→ SuiteRunner
→ 多个 RunRequest
→ CoreEngine
→ 每个任务独立输出结果
→ suite_result.json
```

### 2.3 核心目录

```text
src/luban_meter/
├── benchmark/
│   ├── generate/
│   │   ├── common/
│   │   ├── serving-online/
│   │   └── vllm-engine-offline/
│   └── inference/
│       ├── common/
│       ├── scripts/
│       ├── ceval/
│       ├── cmmlu/
│       └── gsm8k/
├── core/
├── execution/
├── result/
├── suite/
│   └── definitions/
└── utils/
```

各目录的主要职责如下：

- `benchmark/`：具体评测场景、配置和指标处理实现；
- `core/`：请求模型、配置解析、Benchmark 发现和运行编排；
- `execution/`：执行命令、会话和运行过程管理；
- `result/`：原始结果处理、标准结果构造和写入；
- `suite/`：Suite 定义加载、校验和任务编排；
- `utils/`：JSON 和运行标识等通用工具。

## 3. Benchmark 开发原则

### 3.1 按测试场景组织

一个 Benchmark 对应一个边界清晰的测试场景。开发前应先确定：

- 被测对象和调用方式；
- 测试 Case 的变量和固定条件；
- 需要采集的原始事实；
- 可以从原始数据派生的指标；
- 指标的观察边界和统计窗口。

同一场景需要的原始数据应尽量一次完整采集，再由结果处理阶段计算多个指标。

### 3.2 标准目录

新增 Benchmark 使用以下结构：

```text
src/luban_meter/benchmark/<module>/<benchmark>/
├── benchmark.py
├── result.py
└── config.example.yaml
```

`benchmark.py` 和 `result.py` 同时存在后，`BenchmarkRegistry` 才会发现该 Benchmark。
Benchmark 名称使用小写字母、数字、连字符或下划线，并以字母或数字开头。

### 3.3 采集与计算

`benchmark.py` 负责：

- 读取并校验运行请求和配置；
- 调用在线服务、Engine 或数据集任务；
- 记录逐请求、逐 Case 或逐样本原始数据；
- 记录环境、日志、产物和错误信息；
- 写出符合 `luban-meter.raw/v1` 的结果。

`result.py` 负责：

- 校验原始结果结构和数据范围；
- 从原始数据计算派生指标；
- 按 Request、Service、Engine、Batch 或 Task 视角组织指标；
- 计算 Mean、P50、P90、P99 等统计值；
- 输出指标单位、样本数量和统计条件。

### 3.4 配置和公共能力

配置应明确字段名称、类型、范围以及字段之间的组合关系。服务地址、模型、请求速率、
并发度、Batch Size、输入输出长度和超时等测试条件均通过配置或运行请求表达。

多个生成式 Benchmark 需要复用的流式解析、Token 计数和统计逻辑放在
`benchmark/generate/common/`。只服务于单个场景的逻辑保留在对应 Benchmark 目录。

多个 `inference` Benchmark 需要复用的在线服务调用、数据集加载、Prompt 渲染、
答案解析和指标计算逻辑放在 `benchmark/inference/common/`；数据集离线准备脚本
放在 `benchmark/inference/scripts/`，Benchmark 运行时只读取本地数据集文件。

完整请求、原始结果和最终结果协议参见
[Benchmark 脚本开发指南](docs/develop-benchmark.md)。

## 4. 必要开发流程

### 4.1 每次开发都需要

1. 阅读任务相关文档和最接近的现有实现。
2. 明确测试场景、输入条件、原始数据、目标指标和结果结构。
3. 在现有架构内完成范围明确的代码修改。
4. 验证成功路径、失败路径和配置边界。
5. 运行与修改范围相关的测试和静态检查。
6. 检查最终差异，确认代码、测试和文档保持一致。

### 4.2 根据任务执行

- 新增 Benchmark 时，创建标准目录、示例配置和对应测试；
- 编排多个 Benchmark 时，增加或修改 Suite 定义；
- 多个生成式场景出现相同逻辑时，将稳定能力提取到 `generate/common/`；
- 修改公开接口、配置或指标语义时，更新对应专题文档；
- 功能依赖真实服务、模型或 Engine 时，在目标环境完成冒烟测试；
- 任务包含代码交付时，再执行提交、推送和交付检查。

## 5. 指标与结果注意事项

### 5.1 数据分层

指标实现应保持以下数据关系清晰：

```text
原始采集字段
→ 单请求或单样本派生指标
→ Case 聚合指标
→ 整次运行结果
```

在线生成测试通常需要保留：

- 请求计划时间、实际开始时间和完成时间；
- 首个有效输出时间；
- 每次流式输出事件的时间；
- 输入和输出 Token 数；
- 请求成功、失败和超时状态；
- Case 和整次运行的起止时间；
- 并发变化事件；
- 请求参数、服务信息和模型信息。

这些原始数据可以支持 TTFT、ITL、TPOT、E2EL、请求吞吐量、Token 吞吐量、调度
偏差、并发度以及成功失败统计。

### 5.2 统计条件

每个 Case 应保存输入长度、输出长度、请求速率、并发度、Batch Size、缓存和精度等
会影响结果的条件。聚合指标应同时给出单位、样本数、测试窗口和失败样本处理方式。

在线服务指标按照客户端时间线计算；Engine 指标按照 Engine 提供的内部时间戳计算。
指标的定义、公式和两类观察边界参见[生成式推理指标说明](docs/metrics.md)。

## 6. 验证要求

验证范围应与修改风险相匹配，至少覆盖与本次修改直接相关的项目：

- Benchmark 能被 `luban-meter benchmarks list` 正确发现；
- 配置正常值、边界值和非法值行为符合预期；
- 成功运行能够生成完整的原始结果和最终结果；
- 请求失败、处理失败和超时能够生成可诊断结果；
- 原始字段足以复算最终指标；
- 指标单位、样本数和观察边界明确；
- 相关单元测试通过；
- 静态检查和差异检查通过。

常用本地验证命令：

```bash
luban-meter benchmarks list
pytest -q
ruff check src tests
git diff --check
```

真实环境验证还应记录：

- 代码版本和运行时间；
- 服务、模型和关键配置；
- 执行命令；
- 原始结果、最终结果和日志位置；
- 已验证的能力和仍待验证的范围。

## 7. 完成标准

任务完成时应确认：

- 功能实现与任务目标一致；
- 成功、失败和边界场景已验证；
- 原始数据能够解释并复算最终指标；
- 指标公式、单位、样本数和测试条件明确；
- 相关测试和静态检查通过；
- 公开行为变化已同步到相应文档；
- 需要真实环境验证的功能已留下可追溯证据；
- 最终差异只包含本任务需要的修改。
