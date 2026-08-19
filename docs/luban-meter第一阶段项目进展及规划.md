# LuBan-Meter 大模型评测工具阶段工作汇报

## 一、项目概述

LuBan-Meter 面向异构 AI 硬件环境提供统一、模块化、可扩展的 AI Benchmark 能力。
项目不负责安装或适配驱动、框架和模型，而是在用户已经准备好的可运行环境中，
统一完成 Benchmark 发现、配置加载、任务执行、指标计算和结果落盘。

本阶段围绕大模型生成式推理，完成了从框架设计到在线服务与 vLLM Benchmark 落地的
第一阶段建设，重点解决以下问题：

- 同一套评测脚本如何在不同硬件环境中复用；
- 在线服务性能与 Engine 内部性能如何分开测量；
- TTFT、ITL、TPOT、E2EL 和吞吐量如何保证计时边界清晰；
- 一次测试如何复用原始请求时间线计算多个指标；
- 不同输入长度、输出长度和负载条件如何形成可比较的测试矩阵；
- 单任务和 Suite 多任务组合如何使用统一执行链路并输出标准结果。

## 二、阶段目标与完成情况

| 阶段目标 | 当前状态 | 阶段成果 |
|---|---|---|
| 建立统一 Benchmark 框架 | 已完成 | 形成 CLI、Core、Execution、Result、Suite 分层架构 |
| 建立统一 Benchmark 分类 | 已完成 | 使用 `benchmark/<module>/<benchmark>/` 管理评测脚本 |
| 支持单任务执行 | 已完成 | 通过 `module + benchmark + config` 定位并执行测试 |
| 支持多任务编排 | 已完成 | Suite 顺序执行多个任务，每个任务独立输出结果 |
| 在线服务生成性能测试 | 已完成 | 实现 `serving-online` 开放式固定 Request Rate 负载测试 |
| vLLM Engine 阶段测试 | 已完成 | 实现 `vllm-engine-stage` Prefill、Decode 和内部 TTFT 测试 |
| 统一统计方法 | 已完成 | 输出 Mean、Median、P50、P90、P99、Min、Max、Stddev 和 Count |
| 建设多硬件评价体系 | 规划中 | 以在线自回归推理为基础，在不同硬件环境复用同一 Benchmark 和测试语义 |
| 模型任务精度评测 | 待建设 | 架构已预留 `inference`，尚未实现 Accuracy、F1、ROUGE 等脚本 |
| 设备和服务内部监控 | 待建设 | GPU 利用率、显存、功耗、服务端队列和 KV Cache 实际使用率尚未采集 |

## 三、总体架构方案

LuBan-Meter 采用“统一框架、场景组织、跨硬件复用、结果标准化”的设计。

```text
CLI
  → Core Engine
  → 按 module/benchmark 发现统一脚本
  → 当前宿主机 Python 环境执行 benchmark.py
  → raw_result.json
  → result.py 计算和整理指标
  → result.json
```

工程目录如下：

```text
src/luban_meter/
├── core/                         # 单次 Run 编排、配置与脚本发现
├── execution/                    # 宿主机执行、日志和会话管理
├── result/                       # 原始结果处理与标准结果输出
├── suite/                        # 多个 Benchmark 顺序编排
└── benchmark/
    ├── generate/                 # 自回归生成引擎与在线服务性能
    └── inference/                # 基于在线服务的模型任务效果评测（规划）
```

架构方案具有以下特点：

1. **跨硬件复用**：同一 Benchmark 在不同硬件环境运行，公共 Core 不包含硬件品牌路由。
2. **场景驱动**：Benchmark 按测试场景组织，而不是为每个指标单独建立脚本。
3. **采集与计算分离**：`benchmark.py` 采集原始事实，`result.py` 校验并计算指标。
4. **环境边界明确**：框架复用用户准备好的 Python、驱动和硬件运行时，不承担适配。
5. **结果可追溯**：每次运行保存请求、原始结果、标准结果、标准输出和错误日志。
6. **可组合扩展**：新增场景或 Suite 不需要修改核心执行流程。

当前框架只负责任务独立执行和结果落盘，后续可在标准 `result.json` 之上增加独立的汇总报告层，按统一 Case 对齐结果，生成
“硬件环境 × 模型 × 场景 × 指标”矩阵，同时保留原始结果用于复核。

## 四、已实现的生成式推理评测

### 4.1 在线服务 Benchmark：`serving-online`

在线测试从 OpenAI-compatible API 客户端观察完整请求，包含 HTTP、API Server、
服务排队、调度、Prefill、Decode 和流式传输等开销，回答“用户实际等待多久，
服务在指定负载下能够处理多少请求”。

负载模型为：

```text
input_lengths × output_lengths × request_rates
```

每个组合形成独立 Case，分别预热、施加固定 Request Rate 并聚合结果。
`max_concurrency` 只作为客户端安全上限，不替代 Request Rate；如果请求不能按计划
启动，结果会通过 Dispatch Delay 和 Achieved Request Start Rate 暴露偏差。

已实现的在线指标包括：

- 用户体验：TTFT、事件级 ITL、TPOT、E2EL；
- 单请求性能：输出 Token 吞吐量、Decode Token 吞吐量；
- 服务能力：Request、Input Token、Output Token、Total Token Throughput；
- 负载质量：Offered/Achieved Request Start Rate、Dispatch Delay；
- 并发情况：配置上限、峰值并发、平均并发；
- 可靠性：成功请求数、失败请求数；
- 统计结果：Mean、Median、P50、P90、P99、Min、Max、Stddev、Count。

### 4.2 Engine 阶段 Benchmark：`vllm-engine-stage`

Engine 测试直接调用 vLLM Engine，不包含 HTTP 和网络，重点观察请求被引擎调度后
的 Prefill、Decode 和 Token 处理能力，回答“引擎内部各阶段耗时如何”。

测试矩阵为：

```text
input_lengths × output_lengths × request_batch_sizes
```

每个 Case 独立预热并执行多个正式 Round。当前固定随机种子、关闭 Prefix Cache（计算Prefill阶段关闭缓存）、
关闭 Chunked Prefill 和 Detokenize，减少测试语义漂移。

已实现的 Engine 指标包括：

- Engine Internal TTFT；
- Prefill Latency 和单请求 Prefill Token Throughput；
- Decode Latency 和 Mean Decode Step Latency；
- Per-sequence Decode Rate；
- Engine Execution Latency；
- Batch Aggregate Prefill/Decode Token Throughput；
- KV Cache Block 数量、Block Size、Token 容量和静态最大并发估算（vllm参数设置）。

### 4.3 两类测试的边界

| 测试视角 | 在线服务 | Engine 阶段 |
|---|---|---|
| 观察位置 | API 客户端 | vLLM Engine 内部时间戳 |
| HTTP/网络 | 包含 | 不包含 |
| 服务排队/API处理 | 包含 | 不完整或由引擎字段定义 |
| Prefill/Decode | 包含完整链路影响 | 观察引擎内部阶段窗口 |
| 主要用途 | 用户体验、服务容量和开放式负载 | 引擎阶段分析和批量 Token 能力 |

在线 TTFT 与 Engine Internal TTFT 的时间边界不同，只能在相同条件下做趋势对照或
辅助定位额外开销，不能当作同一个指标直接比较。

## 五、对照 LLM 评价指标体系的覆盖情况

参考 LLM 评价指标概览，完整评测可分为模型效果、语言模型自身指标、生成质量、
推理性能、系统资源与可靠性、安全可信六类。当前覆盖情况如下：

| 指标类别 | 代表指标 | 当前覆盖 | 说明 |
|---|---|---|---|
| 模型效果 | Accuracy、Precision、Recall、F1、EM、Pass@k | 未实现 | 规划由 `inference` 按任务和数据集实现 |
| 语言模型自身 | Cross-Entropy Loss、Perplexity | 未实现 | 需要模型与标准语料评测 |
| 生成质量 | BLEU、ROUGE、BERTScore、Judge Score | 未实现 | 模型生成质量 |
| 生成推理性能 | TTFT、ITL、TPOT、E2EL、Prefill/Decode | 已实现 | 覆盖在线客户端和 vLLM推理引擎两个观察边界 |
| 服务能力 | request/s、token/s、负载速率、并发 | 已实现 | 支持精确输入/输出长度和固定 Request Rate 矩阵 |
| Goodput | 满足 SLO 的有效吞吐 | 未实现 | 需要新增可配置 TTFT、TPOT、E2EL SLO 判定（设置高质量门槛） |
| 缓存指标 | KV 容量、使用率、命中率、Eviction | 部分实现 | 当前只有 vllm推理引擎启动的参数设置KV Cache 静态容量信息 |
| 设备资源 | GPU利用率、显存、带宽、功耗、能耗 | 未实现 | 需要独立 Device Collector |
| 服务内部 | Queue Time、Scheduler Time、KV实际使用率 | 未实现 | 需要接入 vLLM `/metrics` 等服务端数据源，调用服务端接口采取数据 |
| 可靠性 | 成功率、超时率、OOM、长稳 | 部分实现 | 当前记录成功/失败请求次数来计算成功率等 |
| 安全可信 | 幻觉率、安全拒答、鲁棒性 | 未实现 | 属于后续模型质量和安全评测范围 |

阶段结论是：当前已经形成较完整的**生成式推理性能与服务能力评测基线**，但尚不能
宣称具备完整的“LLM全链路综合评价”能力。模型效果、生成质量、资源效率和安全可信仍需
按独立数据来源与测试场景逐步建设。

## 六、指标与脚本组织原则

本阶段确定了以下指标工程原则：

1. **按测试场景组织脚本，不按指标拆脚本。**
   在线一次完整采集请求时间线，即可同时计算 TTFT、ITL、TPOT、E2EL 和吞吐量。
2. **同一原始数据只采集一次。**
   Mean、P50、P90、P99 等统计量统一由同一批样本计算，避免重复发压造成条件变化。
3. **改变工作负载必须形成独立 Case。**
   输入长度、输出长度、Request Rate、Batch Size、模型精度或缓存配置发生变化时，
   必须独立测试，不能混合样本。
4. **计时边界必须随指标输出。**
   禁止只发布“TTFT = 10 ms”而不说明客户端、服务端或 Engine 内部观察位置。
5. **性能、质量和资源指标分组。**
   客户端不能推导的服务端或设备指标，必须由独立 Collector 真实采集。
6. **比较必须绑定测试上下文。**
   结果需要同时记录模型版本、硬件数量、精度、输入/输出长度、负载、预热与采样数。

## 七、阶段成果

本阶段已经形成以下可复用资产：

- 一套支持跨硬件复用的 Benchmark 框架与统一 CLI；
- 单任务与 Suite 共用 CoreEngine 的执行机制；
- Benchmark 自动发现和标准目录协议；
- `raw_result.json` 与 `result.json` 两阶段结果协议；
- 通用在线服务测试和 vLLM Engine 阶段测试；
- 精确输入/输出 Token 长度与开放式固定 Request Rate 负载矩阵；
- Request View、Service View、Engine Request/Batch Metrics 指标分层；
- 通用统计聚合、状态管理、日志和结果归档；
- 架构说明、使用说明、指标说明和 Benchmark 开发指南文档；

## 八、下一阶段工作计划

### 第一优先级：补齐生成服务关键能力

- 实现可配置 SLO 与 Goodput；
- 接入 vLLM 服务端指标，补充 Queue、Scheduler 和 KV Cache 实际使用情况；
- 增加 GPU 利用率、显存峰值、功耗和温度等 Device Collector；
- 增加超时、OOM 和长时间稳定性测试；
- 在不同硬件环境中验证同一 `serving-online`，保持客户端观察边界一致；
- 根据实际 Engine 接口逐步实现引擎 forward 阶段测试，无法获得的内部字段明确
  标记为不支持，不使用客户端估算值替代；
- 建设独立汇总报告层，按统一 Case 生成多硬件覆盖度和指标对比矩阵。

### 第二优先级：建设 `inference` 模块

- 建设完整模型任务级评测，统一采集任务结果、端到端延迟和模型核心执行时间；
- 首批支持 MMLU/C-Eval 等 Accuracy 任务以及摘要类 ROUGE 任务；
- 按任务扩展 F1、EM、Pass@k、Perplexity 等指标；
- 基于之前的框架，对接脚本执行接口，开发属于模型生成结果评测的架构体系。

`inference` 统一通过在线推理服务调用模型，优先复用 OpenAI-compatible HTTP 接口；
数据集、Prompt、答案解析和评分逻辑不按硬件环境复制。

### 暂不规划：算子层 Benchmark

当前阶段不创建算子模块或算子脚本，优先完成生成性能和在线服务模型效果两条主线。

### 后续扩展：模型安全与可信评测

- 建设幻觉、事实一致性、安全拒答、过度拒答和 Prompt Injection 测试；
- 将模型质量、安全性与性能测试作为独立 Benchmark，通过 Suite 组合执行；
- 保持每项指标的数据来源和判定规则可审计。





## 附录：模型评测数据集调用方式及评测指标

| 任务类型       | 数据集                     | 主要指标                            | 调用方式                   |
| -------------- | -------------------------- | ----------------------------------- | -------------------------- |
| 中文知识与考试 | C-Eval                     | Accuracy、学科平均准确率            | OpenAI接口或离线模型       |
| 英文综合知识   | MMLU                       | Accuracy、分类准确率                | OpenAI接口或离线模型       |
| 数学推理       | GSM8K、MATH                | Final Answer Accuracy、Exact Match  | OpenAI接口                 |
| 开放域问答     | Natural Questions、SQuAD   | EM、Token F1、Contains Match        | OpenAI接口                 |
| 文本摘要       | CNN/DailyMail、XSum、LCSTS | ROUGE-1/2/L、BERTScore、Judge Score | OpenAI接口                 |
| 机器翻译       | WMT                        | BLEU、chrF、BERTScore、COMET        | OpenAI接口                 |
| 代码生成       | HumanEval、MBPP            | Pass@1、Pass@k、编译率、单测通过率  | OpenAI接口+沙箱执行        |
| 语言建模       | WikiText-2、WikiText-103   | Cross-Entropy Loss、Perplexity      | 必须离线模型或完整logprobs |
| 开放式对话     | MT-Bench                   | Judge Score、多轮一致性             | OpenAI接口+Judge模型       |
