# LuBan-Meter

面向异构 AI 硬件环境的模块化 Benchmark 工具集，统一完成 Benchmark 发现、配置加载、任务执行、指标计算和结果落盘。

## 核心能力

- **生成式推理性能评测**：测量在线服务和 vLLM 引擎的 TTFT、TPOT、吞吐量等性能指标
- **模型服务质量评测**：通过标准数据集（C-Eval、CMMLU、GSM8K、HumanEval、LCSTS、SQuAD、WikiText）评估推理质量
- **Suite 多任务编排**：一次执行多个 Benchmark，统一输出结果
- **跨硬件复用**：同一套 Benchmark 在不同硬件环境运行，通过统一化、通用化、标准化的多硬件测评工具框架，完成从配置、执行到报告交付的端到端闭环

## 快速开始

### 安装

```bash
python3.12 -m pip install -e ".[dev]"
```

要求：Python >= 3.12。模型、驱动、推理引擎和在线服务由用户提前准备。

### 查看 Benchmark

```bash
luban-meter benchmarks list
```

输出示例：

```text
generate                serving-online,vllm-engine-offline,vllm_metrics    Large-model generation benchmarks
model_service_quality   ceval,cmmlu,gsm8k,humaneval,lcsts,squad,wikitext   Model service quality benchmarks
```

### 运行第一个测试

```bash
luban-meter run \
  --module generate \
  --benchmark serving-online \
  --config src/luban_meter/benchmarking/generation_performance/online_serving/serving_online.yaml \
  --model-name <served-model-name>
```

## 使用指南

### 运行单个 Benchmark

```bash
luban-meter run \
  --module <module> \
  --benchmark <benchmark> \
  --config <config.yaml> \
  [--model-path <path>] \
  [--model-name <name>] \
  [--output runs] \
  [--timeout 3600]
```

参数说明：

| 参数 | 必填 | 说明 |
|---|---:|---|
| `--module` | 是 | `generate` 或 `model_service_quality` |
| `--benchmark` | 是 | 场景名称（如 `serving-online`、`ceval`） |
| `--config` | 是 | 本次测试参数 YAML |
| `--model-path` | 否 | 本地模型路径 |
| `--model-name` | 否 | 逻辑模型名或在线服务模型名 |
| `--output` | 否 | 结果根目录，默认 `runs` |
| `--timeout` | 否 | 执行超时秒数 |
| `--name` | 否 | 保存到结果中的易读运行名称 |
| `--format` | 否 | 控制台输出 `text`（默认）或 `json` |
| `--monitor-url` | 否 | Prometheus exporter 地址，启用硬件监控 |
| `--monitor-interval` | 否 | 硬件监控采样间隔秒数，默认 `1.0` |

### 常用命令示例

**在线生成服务性能测试**：

```bash
luban-meter run \
  --module generate \
  --benchmark serving-online \
  --config src/luban_meter/benchmarking/generation_performance/online_serving/serving_online.yaml \
  --model-name <served-model-name>
```

配置示例（`serving_online.yaml`）：

```yaml
service_url: http://127.0.0.1:8000
input_lengths: [128, 1024]
output_lengths: [1, 128]
request_rates: [1.0, 5.0]
```

每个 `input_length × output_length × request_rate` 组合形成独立 Case。

**vLLM 离线引擎测试**：

```bash
CUDA_VISIBLE_DEVICES=0 luban-meter run \
  --module generate \
  --benchmark vllm-engine-offline \
  --config src/luban_meter/benchmarking/generation_performance/offline_vllm_engine/vllm_engine_offline.yaml \
  --model-path /data/models/<model>
```

**模型服务质量评测**：

```bash
# C-Eval 选择题（ppl 模式）
luban-meter run \
  --module model_service_quality \
  --benchmark ceval \
  --config src/luban_meter/benchmarking/model_service_quality/ceval/ceval.yaml \
  --model-name <served-model-name>

# HumanEval 代码补全（Pass@1，需先构建沙箱镜像）
docker build \
  -f src/luban_meter/benchmarking/model_service_quality/humaneval/Containerfile \
  -t luban-meter-humaneval-sandbox:v1 \
  src/luban_meter/benchmarking/model_service_quality/humaneval

luban-meter run \
  --module model_service_quality \
  --benchmark humaneval \
  --config src/luban_meter/benchmarking/model_service_quality/humaneval/humaneval.yaml \
  --model-name <served-model-name>
```

### 硬件监控（可选）

通过 `--monitor-url` 指定 Prometheus exporter 地址后，框架在 Benchmark 运行期间周期采集硬件指标（GPU 利用率、功耗、温度、显存 + CPU 利用率、内存）。

```bash
luban-meter run \
  --module generate \
  --benchmark serving-online \
  --config config.yaml \
  --monitor-url http://43.138.110.236:9400 \
  --monitor-interval 1.0
```

支持的 exporter：

- **DCGM exporter**（NVIDIA GPU）— 默认端口 9400
- **node_exporter**（CPU/内存）— 默认端口 9100

监控数据注入 `result.json` 的 `environment.device_monitoring`，包含静态硬件环境、设备统计、时间序列和 6 张折线图 PNG。

### 报告生成

执行完成后自动在结果目录的 `report/` 中生成 Markdown、CSV 和静态 PNG。

重新导出已保存的结果：

```bash
luban-meter report --input runs/<run-id>/result.json
luban-meter report --input runs/<suite-id>/suite_result.json
```

### Suite 多任务编排

Suite 用于一次执行多个 Benchmark，统一输出结果。

**定义文件**（`src/luban_meter/suite/definitions/model_service_quality_standard.yaml`）：

```yaml
name: model_service_quality_standard
tasks:
  - name: ceval
    module: model_service_quality
    benchmark: ceval
    config: ../../benchmarking/model_service_quality/ceval/ceval.yaml
  - name: cmmlu
    module: model_service_quality
    benchmark: cmmlu
    config: ../../benchmarking/model_service_quality/cmmlu/cmmlu.yaml
  - name: gsm8k
    module: model_service_quality
    benchmark: gsm8k
    config: ../../benchmarking/model_service_quality/gsm8k/gsm8k.yaml
  - name: humaneval
    module: model_service_quality
    benchmark: humaneval
    config: ../../benchmarking/model_service_quality/humaneval/humaneval.yaml
```

**运行 Suite**：

```bash
luban-meter suite \
  --suite model_service_quality_standard \
  --model-name <served-model-name>
```

**输出结构**：

```text
runs/<suite-id>/
├── suite_request.json      # 运行请求和定义
├── suite_result.json       # 汇总结果
└── tasks/
    ├── <task-run-id-1>/
    │   └── result.json     # ceval 完整结果
    ├── <task-run-id-2>/
    │   └── result.json     # cmmlu 完整结果
    └── ...
```

**特性**：
- 按顺序执行每个 task，每个 task 生成独立的 `result.json`
- `suite_result.json` 汇总所有 task 的指标和状态
- 支持 `--fail-fast`：某个 task 失败时停止后续 task
- 支持 `--task-config`：为特定 task 覆盖配置（如 `--task-config ceval=ceval_custom.yaml`）


## 功能特性

### 生成式推理性能评测

| Benchmark | 工作负载 | 核心指标 | 用途 |
|---|---|---|---|
| `serving-online` | random / dataset | TTFT、ITL、TPOT、E2EL、吞吐量 | 在线服务性能与容量评估 |
| `vllm-engine-offline` | 矩阵遍历 | 内部 TTFT、Prefill/Decode 时延、Engine 吞吐 | vLLM 引擎内部性能分析 |
| `vllm_metrics` | 服务端采集 | 请求/Token 吞吐、TTFT/TPOT 分位数、KV Cache 使用率 | vLLM 服务端指标监控 |

**serving-online** 支持两种模式：
- **random**：精确长度 Token ID Prompt，遍历输入/输出长度和请求速率矩阵
- **dataset**：ShareGPT 真实对话，支持 Poisson/Gamma/恒定到达过程

### 模型服务质量评测

| 数据集 | 任务类型 | 评测模式 | 核心指标 |
|---|---|---|---|
| C-Eval | 中文知识问答 | ppl / gen | Accuracy |
| CMMLU | 英文综合知识 | ppl / gen | Accuracy |
| GSM8K | 数学推理 | gen | Exact Match |
| HumanEval | 代码生成 | gen + 沙箱执行 | Pass@1 |
| LCSTS | 中文摘要 | gen | ROUGE-1/2/L |
| SQuAD | 阅读理解 | gen | Exact Match、Token F1 |
| WikiText | 语言建模 | loss | Perplexity、Bits-per-Byte |

数据集默认随包内置在 `benchmarking/model_service_quality/data/`，开箱即用。


## 架构与开发

### 执行流程

```text
CLI
→ BenchmarkRegistry
→ RunCoordinator
→ ExecutionSession
→ collect_raw.py
→ raw_result.json
→ calculate_metrics.py
→ result.json
```

单任务由 `module + benchmark + config` 三个参数确定。

### 目录结构

```text
src/luban_meter/
├── benchmarking/
│   ├── generation_performance/        # 生成式推理性能评测
│   │   ├── common/
│   │   ├── online_serving/
│   │   ├── offline_vllm_engine/
│   │   └── vllm_service_metrics/
│   └── model_service_quality/         # 模型服务质量评测
│       ├── common/                    # 公共层
│       ├── scripts/                   # 数据集准备脚本
│       ├── data/                      # 随包内置数据集
│       ├── ceval/
│       ├── cmmlu/
│       ├── gsm8k/
│       ├── humaneval/
│       ├── lcsts/
│       ├── squad/
│       └── wikitext/
├── core/
├── execution/
├── result/
├── reporting/
├── suite/
│   └── definitions/
└── utils/
```

### Benchmark 发现协议

每个 Benchmark 目录必须包含：

```text
benchmarking/<category>/<benchmark>/
├── collect_raw.py
├── calculate_metrics.py
└── config.example.yaml
```

`BenchmarkRegistry` 自动发现 `collect_raw.py` 和 `calculate_metrics.py` 同时存在的目录。

### 核心原则

1. **按测试场景组织**：一次采集原始事实，计算多个指标
2. **跨硬件复用**：同一 Benchmark 在不同环境运行，不按硬件品牌复制脚本
3. **采集与计算分离**：`collect_raw.py` 采集原始数据，`calculate_metrics.py` 校验并计算指标
4. **差异通过配置表达**：服务地址、模型、并发度等写入配置，不进入代码分支
5. **结果可审计**：保留原始记录、参数、日志和失败原因

### 新增 Benchmark 步骤

1. 创建目录 `benchmarking/<category>/<benchmark>/`
2. 实现 `collect_raw.py`：接收 `--request` 和 `--output`，输出 `luban-meter.raw/v1`
3. 实现 `calculate_metrics.py`：定义 `process(raw_result)`，输出 `luban-meter.result/v2`
4. 提供 `config.example.yaml`：包含配置字段说明和示例值
5. 校验配置：必填字段、类型、范围、边界
6. 验证：`luban-meter benchmarks list` 能发现新 Benchmark


### 验证清单

- [ ] `luban-meter benchmarks list` 正确发现
- [ ] 配置正常值、边界值和非法值行为符合预期
- [ ] 成功运行生成完整的原始结果和最终结果
- [ ] 失败和超时时生成可诊断结果
- [ ] 指标单位、样本数和观察边界明确
- [ ] `pytest -q` 通过
- [ ] `ruff check src tests` 通过

## 指标说明

### 生成式推理指标

| 指标 | 定义 | 单位 |
|---|---|---|
| TTFT | 首 Token 时延（Time To First Token） | ms |
| ITL | inter-Token Latency，排除首 Token | ms |
| TPOT | 每输出 Token 时延，包含首 Token 等待 | ms/token |
| E2EL | 端到端时延（End-to-End Latency） | ms |
| Request Throughput | 每秒处理的请求数 | req/s |
| Token Throughput | 每秒生成的 Token 数 | token/s |

在线结果统一采用 GuideLLM 的 Token 指标口径：TPOT 包含首 Token 等待，ITL 排除首 Token；两者按对应 Token 数加权。

### 模型服务质量指标

| 指标 | 定义 | 适用数据集 |
|---|---|---|
| Accuracy | 正确预测比例 | C-Eval、CMMLU |
| Exact Match | 答案完全匹配比例 | GSM8K、SQuAD |
| Token F1 | 基于 Token 的 F1 分数 | SQuAD |
| Pass@1 | 单次采样通过单元测试比例 | HumanEval |
| ROUGE-1/2/L | 基于 n-gram 重叠的摘要质量 | LCSTS |
| Perplexity | 语言模型预测的不确定性 | WikiText |
| Bits-per-Byte | 每字节的平均信息量 | WikiText |

## 验证与测试

```bash
luban-meter benchmarks list
pytest -q
ruff check src tests
git diff --check
```

## 当前范围与边界

**当前范围**：
- 生成式推理性能评测（在线服务 + vLLM 引擎 + 服务端指标）
- 模型服务质量评测（7 个数据集，3 种评测模式）
- Suite 多任务编排
- 硬件监控（可选，通过 Prometheus exporter）
- 报告生成（Markdown、CSV、PNG）

**不在当前范围**：
- 算子层 Benchmark
- 硬件驱动安装和适配
- 跨运行对比和优劣分析
- 模型安全与可信评测

硬件差异由运行时环境和配置体现，框架复用用户已准备好的 Python、驱动、推理引擎和在线服务环境。
