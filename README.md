# LuBan-Meter

面向异构 AI 硬件环境的模块化 Benchmark 工具集。

## 文档

- [架构说明](docs/architecture.md)
- [使用说明](docs/usage.md)
- [Benchmark 脚本开发指南](docs/develop-benchmark.md)
- [生成式推理指标说明](docs/metrics.md)
- [Inference 评测指标说明](docs/inference.md)
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

公共架构不再按硬件厂商复制脚本。Benchmark 直接按评测目标分类：

```text
src/luban_meter/
├── benchmark/
│   ├── generate/                 # 生成式推理性能评测
│   │   ├── common/
│   │   ├── serving-online/
│   │   └── vllm-engine-stage/
│   └── inference/                # 基于在线推理服务的模型效果评测
│       ├── common/               # 公共层：client / dataset / prompts / parsers / metrics / choice
│       ├── scripts/              # 数据集离线准备脚本（官方格式 → 本地 jsonl）
│       ├── data/                 # 随包内置的样例数据集（ceval / cmmlu / gsm8k）
│       ├── ceval/
│       ├── cmmlu/
│       └── gsm8k/
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
→ benchmark/<module>/<benchmark>/benchmark.py
→ raw_result.json
→ result.py
→ result.json
```

每个 Benchmark 目录遵循统一协议：

```text
benchmark/<module>/<benchmark>/
├── benchmark.py
├── result.py
└── config.example.yaml
```

## 已实现 Benchmark

`generate/serving-online` 通过 OpenAI-compatible HTTP 流式接口遍历精确输入长度、
输出长度和固定请求速率矩阵，输出 TTFT、ITL、TPOT、E2EL、吞吐量、调度偏差、
并发和成功/失败请求统计。

`generate/vllm-engine-stage` 直接调用 vLLM Engine，遍历输入长度、输出长度和请求
批量矩阵，输出内部 TTFT、Prefill/Decode 时延与吞吐量、Engine Execution Latency，
并记录 KV Cache 静态容量环境。

`inference` 通过在线推理服务评测模型任务效果，已端到端实现 `ceval`、`cmmlu`（选择题
Accuracy，支持 ppl / gen 两种评测模式）和 `gsm8k`（数学题 Exact Match，gen 模式）。
其中 ppl 模式走 `/v1/completions` 的 `echo + logprobs` 打分，要求 `prompt_format=base`；
gen 模式可走 chat 或 base 传输。默认数据集随包内置在
`benchmark/inference/data/`，相对路径优先按 CWD 解析，未命中时回退到包内置数据。

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
  --config src/luban_meter/benchmark/generate/serving-online/serving_online.yaml \
  --model-name <served-model-name>
```

运行 vLLM Engine 阶段测试：

```bash
CUDA_VISIBLE_DEVICES=0 luban-meter run \
  --module generate \
  --benchmark vllm-engine-stage \
  --config src/luban_meter/benchmark/generate/vllm-engine-stage/vllm_engine_stage.yaml \
  --model-path /data/models/<model>
```

运行 inference 模型效果评测（C-Eval 选择题 ppl 打分）：

```bash
luban-meter run \
  --module inference \
  --benchmark ceval \
  --config src/luban_meter/benchmark/inference/ceval/ceval.yaml \
  --model-name <served-model-name>
```

运行 Suite：

```bash
luban-meter suite \
  --suite generation-basic \
  --model-path /data/models/<model>
```

## 安装与验证

```bash
python3.12 -m pip install -e ".[dev]"
luban-meter benchmarks list
pytest -q
ruff check src tests
```
