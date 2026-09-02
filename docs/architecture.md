# LuBan-Meter 架构说明

## 1. 设计目标

LuBan-Meter 在用户已经准备好的硬件与软件环境中执行统一 Benchmark。框架不安装、
切换或修改驱动和运行时，只负责定位脚本、加载配置、执行测试、计算指标和输出结果。

核心约定：

- Benchmark 按评测目标组织，不按硬件厂商复制；
- 单任务和 Suite 共用同一个 `CoreEngine`；
- `benchmark.py` 采集原始事实，`result.py` 校验并计算指标；
- 每个 Suite 任务独立输出结果；
- 硬件差异通过当前运行环境和 Benchmark 配置体现；
- 当前只建设 `generate` 和 `inference`，不实现算子层评测。

## 2. 分层架构

```text
CLI
├── benchmarks list
├── run
└── suite
     │
     ▼
Core / Suite
├── BenchmarkRegistry
├── CoreEngine
├── SuiteLoader
└── SuiteRunner
     │
     ▼
Execution
├── 当前宿主机 Python
├── 当前 Shell 环境
└── benchmark.py
     │
     ▼
Result
├── raw_result.json
├── result.py
└── result.json
```

### 单任务链路

```text
RunRequest(module, benchmark, config)
→ benchmark/<module>/<benchmark>/
→ benchmark.py
→ raw_result.json
→ result.py
→ result.json
```

### Suite 链路

```text
SuiteRequest(suite)
→ suite/definitions/<suite>.yaml
→ SuiteRunner
→ RunRequest 1 ... RunRequest N
→ 每个任务独立 result.json
→ suite_result.json
```

## 3. 工程目录

```text
src/luban_meter/
├── benchmark/
│   ├── generate/
│   │   ├── common/
│   │   │   ├── prometheus.py
│   │   │   └── statistics.py
│   │   ├── serving-online/
│   │   │   ├── benchmark.py
│   │   │   ├── result.py
│   │   │   └── serving_online.yaml
│   │   ├── vllm-engine-offline/
│   │   │   ├── benchmark.py
│   │   │   ├── result.py
│   │   │   └── vllm_engine_offline.yaml
│   │   └── vllm_metrics/
│   │       ├── benchmark.py
│   │       ├── result.py
│   │       └── vllm_metrics.yaml
│   └── inference/
│       ├── common/
│       ├── scripts/
│       ├── data/                 # 随包内置样例数据集（ceval/cmmlu/gsm8k jsonl）
│       ├── ceval/
│       ├── cmmlu/
│       └── gsm8k/
├── core/
│   ├── engine.py
│   ├── registry.py
│   ├── models.py
│   └── config.py
├── execution/
├── result/
└── suite/
    └── definitions/
```

`benchmark/` 是所有评测实现的唯一根目录：

- `generate`：生成式推理性能与服务能力评测；
- `inference`：通过在线推理服务执行数据集任务，评测模型效果；

本阶段不创建 `operation` 或类似算子目录。

## 4. Benchmark 发现协议

Benchmark 固定使用以下结构：

```text
benchmark/<module>/<benchmark>/
├── benchmark.py
├── result.py
└── *.yaml
```

`BenchmarkRegistry` 只接受 `generate`、`inference` 两个模块，并要求
`benchmark.py` 与 `result.py` 同时存在。列表命令直接返回 Benchmark 名称：

```text
generate    serving-online,vllm-engine-offline,vllm-metrics,device-monitor
inference   ceval,cmmlu,gsm8k
```

公共层不包含硬件品牌字段。相同 Benchmark 应在不同硬件环境中执行同一份脚本和
同语义配置，以保证比较边界一致；只有引擎专属能力才在名称中体现，例如
`vllm-engine-offline`。

### 硬件检测

所有 generate 模块的 Benchmark 在执行前自动调用 `print_hardware_info()` 检测
当前主机硬件并输出摘要信息（设备数量、厂商、型号），方便复现问题。该功能由
`common/device_monitor.py` 提供，支持 NVIDIA / 华为昇腾 / AMD / 寒武纪 /
摩尔线程 / 壁仞 / 燧原等厂商。

## 5. Generate 与 Inference 的边界

### generate

回答“推理链路有多快、服务能处理多少负载”，主要包括：

- TTFT、ITL、TPOT、E2EL；
- Prefill、Decode 与 Engine 内部时延；
- Request/Token Throughput；
- 固定请求速率、调度偏差与并发；
- KV Cache 静态容量环境。

### inference

回答“在线推理服务返回的模型结果是否正确或质量如何”，通过 HTTP 服务执行标准
数据集和任务。已实现：

- C-Eval、CMMLU 选择题 Accuracy（ppl logprob 打分与 gen 生成抽取两种模式；ppl
  模式仅允许 `prompt_format=base`，组合 ppl + chat 会被配置校验拒绝）；
- GSM8K 数学题 Exact Match。

数据集默认随包内置在 `benchmark/inference/data/`，相对路径优先按 CWD 解析，
未命中时回退到包内置数据，使同一份脚本可从任意目录运行。

后续规划：

- 问答 EM、F1（SQuAD）；
- 摘要 ROUGE（LCSTS）；
- 代码生成 Pass@k（HumanEval）；
- 语言建模 Perplexity（WikiText）；
- 任务级端到端时延。

`inference` 不直接加载硬件专属模型接口；首选统一的在线推理服务协议，使同一套
题目、Prompt、解析和评分逻辑可以复用于不同硬件环境。

## 6. 执行与结果协议

框架通过标准请求文件调用：

```bash
python benchmark.py --request <request.json> --output <raw_result.json>
```

请求包含：

- `run_id`；
- `module`；
- `benchmark`；
- `config`；
- `model_path` 或 `model_name`；
- `parameters`；
- `output_dir`。

最终 `result.json` 包含相同的运行身份信息，但不包含厂商字段：

```json
{
  "schema_version": "luban-meter.result/v1",
  "run_id": "generate-...",
  "status": "success",
  "module": "generate",
  "benchmark": "serving-online",
  "config": "...",
  "model": {},
  "environment": {},
  "parameters": {},
  "metrics": {},
  "artifacts": {},
  "metadata": {},
  "error": null
}
```

硬件、驱动、引擎和服务版本属于环境事实，后续由统一环境采集器写入
`environment`，而不是用于脚本路由。

## 7. Suite

Suite 定义统一存放在：

```text
src/luban_meter/suite/definitions/<suite>.yaml
```

示例：

```yaml
name: generation-basic
tasks:
  - name: serving-online
    module: generate
    benchmark: serving-online
    config: configs/serving-online.yaml

  - name: vllm-engine-offline
    module: generate
    benchmark: vllm-engine-offline
    config: configs/vllm-engine-offline.yaml
```

Suite 只编排任务，不改变运行环境，也不在任务间比较指标。

## 8. 扩展边界

- 新增生成性能场景：增加 `benchmark/generate/<benchmark>/`；
- 新增模型效果任务：增加 `benchmark/inference/<benchmark>/`；
- 新增任务组合：增加 `suite/definitions/<suite>.yaml`；
- 新增硬件支持：验证现有 Benchmark 能在目标环境运行，必要差异通过配置表达；
- 新增引擎内部测试：仅在确有稳定内部接口时增加带引擎名称的 Benchmark；
- 算子微基准：不在当前阶段范围内。
