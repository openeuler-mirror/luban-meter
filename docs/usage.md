# LuBan-Meter 使用说明

## 1. 安装与运行环境

```bash
python3.12 -m pip install -e ".[dev]"
```

模型、驱动、Python 依赖、推理引擎和在线服务由用户提前准备。LuBan-Meter 使用
当前 Python 和 Shell 环境，不读取或切换额外环境配置。

## 2. 查看 Benchmark

```bash
luban-meter benchmarks list
```

当前模块：

```text
generate    serving-online,vllm-engine-offline,vllm-metrics  Large-model generation benchmarks
inference   ceval,cmmlu,gsm8k                   Online-service model evaluation benchmarks
```

`generate` 测量生成式推理性能；`inference` 用于基于在线推理服务的模型效果评测。
当前不提供算子层模块。

## 3. 运行单个 Benchmark

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

参数：

| 参数 | 必填 | 说明 |
|---|---:|---|
| `--module` | 是 | `generate` 或 `inference` |
| `--benchmark` | 是 | `benchmark/<module>/` 下的目录名称 |
| `--config` | 是 | 本次测试参数 YAML |
| `--model-path` | 否 | 本地模型路径 |
| `--model-name` | 否 | 逻辑模型名或在线服务模型名 |
| `--output` | 否 | 结果根目录，默认 `runs` |
| `--timeout` | 否 | 执行超时秒数 |

框架按以下路径解析脚本：

```text
src/luban_meter/benchmark/<module>/<benchmark>/benchmark.py
src/luban_meter/benchmark/<module>/<benchmark>/result.py
```

所有 generate 模块的 Benchmark 在执行前会自动检测硬件并输出摘要信息，例如：

```text
[Device Monitor] Detected 8 x 910B2 (ascend) via npu-smi
  [ascend] Device 0: 910B2
  [ascend] Device 1: 910B2
  ...
```

## 4. 在线生成服务测试

先启动兼容 OpenAI Completions API 的流式服务，然后执行：

```bash
luban-meter run \
  --module generate \
  --benchmark serving-online \
  --config src/luban_meter/benchmark/generate/serving-online/serving_online.yaml \
  --model-name <served-model-name>
```

配置矩阵：

```yaml
service_url: http://127.0.0.1:8000
request_timeout: 120
warmup: 2
rounds: 10
input_lengths: [128, 1024]
output_lengths: [1, 128]
request_rates: [1.0, 5.0]
max_concurrency: 32
seed_prompt: "Explain the benchmark methodology."
temperature: 0.0
ignore_eos: true
seed: 0
```

每个 `input_length × output_length × request_rate` 组合形成独立 Case。脚本通过
`/tokenize` 构造精确长度输入，通过 `min_tokens = max_tokens` 固定输出长度，并以
开放式固定 Request Rate 调度请求。

## 5. vLLM 离线引擎测试

进入已安装 vLLM 的 Python 环境后执行：

```bash
CUDA_VISIBLE_DEVICES=0 luban-meter run \
  --module generate \
  --benchmark vllm-engine-offline \
  --config src/luban_meter/benchmark/generate/vllm-engine-offline/vllm_engine_offline.yaml \
  --model-path /data/models/<model>
```

该 Benchmark 直接调用 vLLM Engine，因此属于带引擎约束的实现；它不代表某个
硬件厂商，能否运行由当前环境是否支持 vLLM 决定。

如需判定满足 Engine 内部时延目标的有效吞吐，可在配置中增加 `engine_slo`。该
结果只覆盖 vLLM Engine 内部调度至 Token 生成的时间窗口，不包含 HTTP、网络和
客户端排队，不能与 `serving-online` 的服务 Goodput 直接比较。字段定义和计算公式
见 [生成式推理指标说明](metrics.md#55-engine-内部-slo-与-goodput)。

## 6. vLLM 服务端指标采集

对正在运行的 vLLM 推理服务，通过 `/metrics` 端点采集 Prometheus 格式指标，
用于瓶颈定位与服务容量评估：

```bash
luban-meter run \
  --module generate \
  --benchmark vllm-metrics \
  --config src/luban_meter/benchmark/generate/vllm_metrics/vllm_metrics.yaml \
  --model-name <served-model-name>
```

配置参数：

```yaml
service_url: http://127.0.0.1:8000
api_key: ""
request_timeout: 10.0
collect_interval: 1.0
collect_duration: 60.0
```

- `collect_interval`：每次 HTTP 请求 `/metrics` 的间隔（秒）
- `collect_duration`：总采集时长（秒），到达后停止采集并聚合

该 Benchmark 不发起推理请求，只读取服务端已有指标，适合接入已运行的 vLLM 服务。

## 8. Suite

Suite YAML 位于：

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
    timeout: 1800

  - name: engine-offline
    module: generate
    benchmark: vllm-engine-offline
    config: configs/vllm-engine-offline.yaml
```

相对 `config` 路径以 Suite YAML 所在目录为基准。运行命令：

```bash
luban-meter suite \
  --suite generation-basic \
  --model-path /data/models/<model> \
  --output runs
```

失败后立即停止后续任务：

```bash
luban-meter suite --suite generation-basic --fail-fast
```

Suite 参数：

| 参数 | 必填 | 说明 |
|---|---:|---|
| `--suite` | 是 | `suite/definitions/` 下的逻辑名称 |
| `--model-path` | 否 | 所有任务共享的模型路径 |
| `--model-name` | 否 | 所有任务共享的模型名称 |
| `--output` | 否 | Suite 输出根目录 |
| `--timeout` | 否 | 单任务默认超时 |
| `--fail-fast` | 否 | 首个失败后停止调度 |

## 9. 结果目录

单任务：

```text
runs/<run-id>/
├── request.json
├── raw/
│   ├── raw_result.json
│   ├── stdout.log
│   ├── stderr.log
│   └── artifacts/
└── result.json
```

Suite：

```text
runs/<suite-id>/
├── suite_request.json
├── suite_result.json
└── tasks/<run-id>/...
```

运行请求和最终结果不包含硬件厂商路由字段。硬件、驱动和引擎版本等事实后续统一
写入 `environment`。

## 10. 常见问题

### Benchmark 不存在

确认目录中同时存在：

```text
benchmark/<module>/<benchmark>/benchmark.py
benchmark/<module>/<benchmark>/result.py
```

### 运行依赖不可用

确认启动 LuBan-Meter 前已激活包含所需引擎与依赖的 Python 环境，并设置必要的
设备可见性等环境变量。

### 结果失败

依次检查 `result.json.error`、`raw/stderr.log`、`raw/stdout.log` 和
`raw/raw_result.json`。

## 11. 开发验证

```bash
ruff check src tests
pytest -q
python -m luban_meter benchmarks list
```

## 12. inference 模型任务效果测试

`inference` Benchmark 基于本地数据集调用在线推理服务。安装包内置了 `ceval`、
`cmmlu`、`gsm8k` 的样例数据集，位于
`src/luban_meter/benchmark/inference/data/`，开箱即用。若需替换为完整官方
数据集，先用离线准备脚本将官方格式转换为本地 jsonl（运行时不下载数据）：

```bash
python src/luban_meter/benchmark/inference/scripts/prepare_ceval.py \
  --source /path/to/ceval --out data/ceval
python src/luban_meter/benchmark/inference/scripts/prepare_cmmlu.py \
  --source /path/to/cmmlu --out data/cmmlu
python src/luban_meter/benchmark/inference/scripts/prepare_gsm8k.py \
  --source /path/to/gsm8k --out data/gsm8k
```

配置中的 `dataset_path` 为相对路径时按以下顺序解析：先相对当前工作目录
（CWD），未命中时回退到包内置的 `benchmark/inference/data/` 目录。因此默认
配置不指定 `dataset_path` 即使用内置样例数据，从 `/tmp` 等任意目录运行也可
正常加载。

运行示例（C-Eval 选择题 Accuracy，ppl 模式）：

```bash
luban-meter run \
  --module inference \
  --benchmark ceval \
  --config src/luban_meter/benchmark/inference/ceval/ceval.yaml \
  --model-name <name>
```

当前可用 Benchmark：`ceval`、`cmmlu`（选择题 Accuracy，支持 ppl/gen 两种评测
模式）和 `gsm8k`（数学题 Exact Match，gen 模式）。其中 ppl / loss 模式依赖
`/v1/completions` 的 `echo + logprobs` 回显，且仅允许 `prompt_format=base`
（对话格式层会注入特殊 Token 破坏 ppl 续写打分，组合 ppl + chat 会被配置校验
拒绝）；gen 模式可使用 chat 或 base 传输。配置字段、评测模式和指标口径参见
[Inference 评测指标说明](inference.md)。
