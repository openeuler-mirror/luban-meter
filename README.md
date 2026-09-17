# LuBan-Meter

面向异构 AI 硬件环境的模块化 Benchmark 工具集。

## 文档

- [架构说明](docs/architecture.md)
- [使用说明](docs/usage.md)
- [Benchmark 脚本开发指南](docs/develop-benchmark.md)
- [生成式推理指标说明](docs/metrics.md)
- [模型服务质量评测指标说明](docs/model_service_quality.md)
- [第一阶段项目进展及规划](docs/luban-meter第一阶段项目进展及规划.md)

## 当前范围

LuBan-Meter 复用用户已经准备好的 Python、驱动、推理引擎、模型和在线服务环境，
负责 Benchmark 发现、配置加载、任务执行、指标计算和结果落盘。框架不安装或切换
硬件运行时，Benchmark 使用启动 `luban-meter` 的当前 Python，并继承当前 Shell 环境。

要求：

```text
Python >= 3.12
```

## 目录与分类

源码命名与CLI名称的对应关系见[命名约定与迁移说明](docs/naming.md)。
领域术语见[术语表](CONTEXT.md)。

公共架构不再按硬件厂商复制脚本。Benchmark 直接按评测目标分类：

```text
src/luban_meter/
├── benchmarking/
│   ├── generation_performance/                 # 生成式推理性能评测
│   │   ├── common/
│   │   ├── online_serving/
│   │   └── offline_vllm_engine/
│   └── model_service_quality/                # 基于在线推理服务的模型服务质量评测
│       ├── common/               # 公共层：client / dataset / prompts / parsers / metrics / choice
│       ├── scripts/              # 数据集离线准备脚本（官方格式 → 本地 jsonl）
│       ├── data/                 # 随包内置的标准数据集（含 HumanEval）
│       ├── ceval/
│       ├── cmmlu/
│       ├── gsm8k/
│       ├── humaneval/
│       ├── lcsts/
│       └── wikitext/
├── core/
├── execution/
├── result/
└── suite/
    └── definitions/              # 可选的多任务 Suite YAML
```

当前不建设算子层 Benchmark。硬件差异由运行时环境和配置体现，不进入公共 CLI、
目录层级或结果协议。

单任务由三个核心参数确定：

```text
module + benchmark + config
```

执行流程：

```text
CLI
→ Core Engine
→ benchmarking/<category_directory>/<benchmark_directory>/collect_raw.py
→ raw_result.json
→ calculate_metrics.py
→ result.json
```

每个 Benchmark 目录遵循统一协议：

```text
benchmarking/<category_directory>/<benchmark_directory>/
├── collect_raw.py
├── calculate_metrics.py
└── config.example.yaml
```

## 已实现 Benchmark

`generation_performance/serving-online` 通过 OpenAI-compatible HTTP 流式接口评测在线服务性能，
支持两种工作负载模式：

- **random**：通过 `/v1/completions` 发送精确长度 Token ID Prompt，遍历
  `input_lengths × output_lengths × request_rates` 矩阵，输出 TTFT、ITL、TPOT、
  E2EL、吞吐量、调度偏差、并发和成功/失败请求统计；
- **dataset**：通过 `/v1/chat/completions` 发送 ShareGPT 真实对话 Prompt，
  支持 Poisson、Gamma 和恒定到达过程调度请求，输出变长输入/输出的分布统计、
  到达过程指标和服务容量评估。

在线结果统一采用 GuideLLM 的 Token 指标口径：TPOT 包含首 Token 等待，ITL
排除首 Token；两者按对应 Token 数加权。SLO 中的 `tpot_ms` 判定 ITL。
具体公式和历史原始文件重算规则见
[指标说明](docs/metrics.md#4-在线服务指标)。

`generation_performance/vllm-engine-offline` 直接调用 vLLM Engine 进行离线推理，遍历输入长度、输出长度和请求
批量矩阵，输出内部 TTFT、Prefill/Decode 时延与吞吐量、Engine Execution Latency，
并记录 KV Cache 静态容量环境；可选的 `engine_slo` 根据 Engine 内部时间线计算
与在线服务边界分离的 Engine Goodput。

`model_service_quality` 通过在线推理服务评测模型任务效果，已端到端实现 `ceval`、`cmmlu`（选择题
Accuracy，支持 ppl / gen 两种评测模式）、`gsm8k`（数学题 Exact Match，gen 模式）、
`humaneval`（代码补全 Pass@1，强制 Docker 沙箱执行）、`lcsts`（中文摘要
ROUGE-1/2/L，gen 模式）和 `wikitext`（语言建模
Perplexity / Bits-per-Byte，loss 模式）。
其中 ppl / loss 模式走 `/v1/completions` 的 `echo + logprobs` 打分，要求
`prompt_format=base`；gen 模式可走 chat 或 base 传输。默认数据集随包内置在
`benchmarking/model_service_quality/data/`，相对路径优先按 CWD 解析，未命中时回退到包内置数据。

## CLI 示例

查看模块和已发现的 Benchmark：

```bash
luban-meter benchmarks list
```

运行在线生成服务性能测试：

```bash
luban-meter run \
  --module generate \
  --benchmark serving-online \
  --config src/luban_meter/benchmarking/generation_performance/online_serving/serving_online.yaml \
  --model-name <served-model-name>
```

运行 vLLM 离线引擎测试：

```bash
CUDA_VISIBLE_DEVICES=0 luban-meter run \
  --module generate \
  --benchmark vllm-engine-offline \
  --config src/luban_meter/benchmarking/generation_performance/offline_vllm_engine/vllm_engine_offline.yaml \
  --model-path /data/models/<model>
```

运行 model_service_quality 模型服务质量评测（C-Eval 选择题 ppl 打分）：

```bash
luban-meter run \
  --module model_service_quality \
  --benchmark ceval \
  --config src/luban_meter/benchmarking/model_service_quality/ceval/ceval.yaml \
  --model-name <served-model-name>
```

HumanEval 官方 164 题已随包内置；运行 Pass@1 前只需构建专用沙箱镜像：

```bash
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

完整执行与安全协议见 [HumanEval 协议说明](docs/humaneval-protocol.md)。

运行 WikiText 语言建模评测（loss 模式，Perplexity + Bits-per-Byte）：

```bash
luban-meter run \
  --module model_service_quality \
  --benchmark wikitext \
  --config src/luban_meter/benchmarking/model_service_quality/wikitext/wikitext.yaml \
  --model-name <served-model-name>
```

运行 LCSTS 中文摘要评测（gen 模式，ROUGE-1/2/L F-measure）：

```bash
luban-meter run \
  --module model_service_quality \
  --benchmark lcsts \
  --config src/luban_meter/benchmarking/model_service_quality/lcsts/lcsts.yaml \
  --model-name <served-model-name>
```

运行 Suite：

```bash
luban-meter suite \
  --suite model_service_quality_standard \
  --model-name <served-model-name>
```

该 Suite 顺序执行 C-Eval、CMMLU、GSM8K 和 HumanEval；每个任务的指标同时写入
`suite_result.json`，并保留各自完整的 `result.json`。

## 安装与验证

```bash
python3.12 -m pip install -e ".[dev]"
luban-meter benchmarks list
pytest -q
ruff check src tests
```
