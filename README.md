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
generation_performance  offline_vllm_engine,online_serving,vllm_service_metrics  Large-model generation benchmarks
model_service_quality   ceval,cmmlu,gsm8k,humaneval,lcsts,squad,wikitext   Model service quality benchmarks
```

### 运行第一个测试

```bash
luban-meter run \
  --module generation_performance \
  --benchmark online_serving \
  --config src/luban_meter/benchmarking/generation_performance/online_serving/online_serving.yaml \
  --model-name <served_model_name>
```

## 使用指南

### 运行单个 Benchmark

```bash
luban-meter run \
  --module <category_name> \
  --benchmark <benchmark_name> \
  --config <config_path> \
  [--model-path <model_path>] \
  [--model-name <model_name>] \
  [--output runs] \
  [--timeout 3600]
```

以上为命令语法示意：尖括号表示需要替换的值，方括号表示可选参数，实际执行时不输入这些括号。

参数说明：

| 参数 | 必填 | 说明 |
|---|---:|---|
| `--module` | 是 | `generation_performance` 或 `model_service_quality` |
| `--benchmark` | 是 | 场景名称（如 `online_serving`、`ceval`） |
| `--config` | 是 | 本次测试参数 YAML |
| `--model-path` | 否 | 本地模型路径 |
| `--model-name` | 否 | 逻辑模型名或在线服务模型名 |
| `--output` | 否 | 结果根目录，默认 `runs` |
| `--timeout` | 否 | 执行超时秒数，默认 `3600` |
| `--name` | 否 | 保存到结果中的易读运行名称 |
| `--format` | 否 | 控制台输出 `text`（默认）或 `json` |
| `--monitor-url` | 否 | Prometheus exporter 地址，启用硬件监控 |
| `--monitor-interval` | 否 | 硬件监控采样间隔秒数，默认 `1.0` |

### 常用命令示例

**在线生成服务性能测试**：

```bash
luban-meter run \
  --module generation_performance \
  --benchmark online_serving \
  --config src/luban_meter/benchmarking/generation_performance/online_serving/online_serving.yaml \
  --model-name <served_model_name>
```

配置示例（`online_serving.yaml`）：

```yaml
service_url: http://127.0.0.1:8000
input_lengths: [128, 1024]
output_lengths: [1, 128]
request_rates: [1.0, 5.0]
```

`workload_mode: random` 为默认模式；配置中的 `input_lengths`、`output_lengths` 和 `request_rates` 分别提供输入长度、输出长度和请求速率列表。每组取值形成独立 Case，Case中的字段名为 `input_length`、`output_length` 和 `request_rate`。

**vLLM 离线引擎测试**：

```bash
CUDA_VISIBLE_DEVICES=0 luban-meter run \
  --module generation_performance \
  --benchmark offline_vllm_engine \
  --config src/luban_meter/benchmarking/generation_performance/offline_vllm_engine/offline_vllm_engine.yaml \
  --model-path /data/models/<model_directory>
```

**模型服务质量评测**：

```bash
# C-Eval 选择题（ppl 模式）
luban-meter run \
  --module model_service_quality \
  --benchmark ceval \
  --config src/luban_meter/benchmarking/model_service_quality/ceval/ceval.yaml \
  --model-name <served_model_name>

# HumanEval 代码补全（Pass@1，需先构建沙箱镜像）
docker build \
  -f src/luban_meter/benchmarking/model_service_quality/humaneval/Containerfile \
  -t luban-meter-humaneval-sandbox:v1 \
  src/luban_meter/benchmarking/model_service_quality/humaneval

luban-meter run \
  --module model_service_quality \
  --benchmark humaneval \
  --config src/luban_meter/benchmarking/model_service_quality/humaneval/humaneval.yaml \
  --model-name <served_model_name>
```

### 硬件监控（可选）

通过 `--monitor-url` 指定 Prometheus exporter 地址后，框架在 Benchmark 运行期间周期采集硬件指标（GPU 利用率、功耗、温度、显存 + CPU 利用率、内存）。

```bash
luban-meter run \
  --module generation_performance \
  --benchmark online_serving \
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
luban-meter report --input runs/<run_id>/result.json
luban-meter report --input runs/<suite_id>/suite_result.json
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
  --model-name <served_model_name>
```

**输出结构**：

```text
runs/<suite_id>/
├── suite_request.json      # 运行请求和定义
├── suite_result.json       # 汇总结果
└── tasks/
    ├── <task_run_id_1>/
    │   └── result.json     # ceval 完整结果
    ├── <task_run_id_2>/
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
| `online_serving` | random / dataset | TTFT、ITL、TPOT、E2EL、吞吐量 | 在线服务性能与容量评估 |
| `offline_vllm_engine` | 矩阵遍历 | 内部 TTFT、Prefill/Decode 时延、Engine 吞吐 | vLLM 引擎内部性能分析 |
| `vllm_service_metrics` | 服务端采集 | 请求/Token 吞吐、TTFT/TPOT 分位数、KV Cache 使用率 | vLLM 服务端指标监控 |

**online_serving** 支持两种模式：
- **`workload_mode: random`**：精确长度 Token ID Prompt，遍历输入/输出长度和请求速率矩阵
- **`workload_mode: dataset`**：通过 `dataset_path` 和 `dataset_format` 配置数据；`arrival_process` 支持 `poisson`、`gamma`、`constant`，`burstiness` 控制Gamma到达间隔分布

### 模型服务质量评测

| 数据集 | 任务类型 | 评测模式 | 核心指标 |
|---|---|---|---|
| C-Eval | 中文知识问答 | ppl / gen | Accuracy |
| CMMLU | 中文综合知识 | ppl / gen | Accuracy |
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

单任务通过 `--module`、`--benchmark` 和 `--config` 指定。CLI、文件协议与Python内部字段的对应关系如下：

| CLI选项 | 请求JSON或Suite任务YAML字段 | Python请求字段 |
|---|---|---|
| `--module` | `module` | `category_name` |
| `--benchmark` | `benchmark` | `benchmark_name` |
| `--config` | `config` | `config_path` |
| `--model-path` | 请求JSON中的 `model_path` | `model_path` |
| `--model-name` | 请求JSON中的 `model_name` | `model_name` |
| `--output` | 请求JSON中的 `output_dir` | `output_dir` |
| `--timeout` | `timeout` | `timeout` |
| `--name` | 请求JSON中的 `display_name` | `display_name` |

`--format`控制CLI输出，不属于`RunRequest`字段；`--monitor-url`和`--monitor-interval`通过`LUBAN_MONITOR_URL`、`LUBAN_MONITOR_INTERVAL`传给执行层。Suite定义中的`name`是套件或任务名称，与运行时的`display_name`不同。

### 核心接口

| 接口 | 参数与返回值 |
|---|---|
| `BenchmarkRegistry.list_categories()` | 返回类别名称与说明 |
| `BenchmarkRegistry.list_benchmarks(category_name)` | 返回指定类别的Benchmark名称 |
| `BenchmarkRegistry.resolve(request)` | 接收`RunRequest`，返回`ResolvedRun` |
| `RunCoordinator.run(request)` | 接收`RunRequest`，返回`RunResult` |
| `ExecutionSession.execute(run)` | 接收`ResolvedRun`，返回`RawRunArtifacts` |
| `ResultManager.process(run, artifacts)` | 接收已解析请求与原始产物，返回`RunResult` |
| `SuiteLoader.load(suite)` | 接收套件名称，返回`SuiteDefinition` |
| `SuiteRunner.run(request, definition)` | 接收`SuiteRequest`和`SuiteDefinition`，返回`SuiteResult` |

`ResolvedRun.benchmark_definition`保存`BenchmarkDefinition`，其中`collector_path`和`processor_path`分别指向采集脚本和指标处理脚本。

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

新增Benchmark推荐使用以下结构：

```text
benchmarking/<category_name>/<benchmark_name>/
├── collect_raw.py
├── calculate_metrics.py
└── <benchmark_name>.yaml
```

`BenchmarkRegistry`自动发现`collect_raw.py`和`calculate_metrics.py`同时存在的目录。配置文件通过`--config`指定，不参与发现；内置配置按Benchmark名称命名，例如`online_serving.yaml`和`ceval.yaml`。

### 核心原则

1. **按测试场景组织**：一次采集原始事实，计算多个指标
2. **跨硬件复用**：同一 Benchmark 在不同环境运行，不按硬件品牌复制脚本
3. **采集与计算分离**：`collect_raw.py` 采集原始数据，`calculate_metrics.py` 校验并计算指标
4. **差异通过配置表达**：服务地址、模型、并发度等写入配置，不进入代码分支
5. **结果可审计**：保留原始记录、参数、日志和失败原因

### 新增 Benchmark 步骤

1. 创建目录 `benchmarking/<category_name>/<benchmark_name>/`
2. 实现 `collect_raw.py`：接收 `--request` 和 `--output`，输出 `luban-meter.raw/v1`
3. 实现 `calculate_metrics.py`：定义 `process(raw_result)`，返回含`metrics`等字段的处理结果；`ResultManager`封装为`RunResult`，并写出`luban-meter.result/v2`
4. 提供 `<benchmark_name>.yaml`：包含配置字段说明和示例值，通过`--config`传入
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
ruff check --select E,W,F,N --line-length 79 src tests
ruff check src tests
git diff --check
```

硬件差异由运行时环境和配置体现，框架复用用户已准备好的 Python、驱动、推理引擎和在线服务环境。
